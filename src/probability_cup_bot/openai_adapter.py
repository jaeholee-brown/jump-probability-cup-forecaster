from __future__ import annotations

import json
import logging
import time
from typing import Any, TypeVar

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


class ModelOutputError(RuntimeError):
    pass


def _is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError, ModelOutputError, TimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500
    return False


def _status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make a Pydantic JSON schema friendlier to OpenAI strict structured outputs."""
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
        props = schema.get("properties", {})
        schema["required"] = list(props.keys())
        for value in props.values():
            if isinstance(value, dict):
                _strict_schema(value)
    if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        _strict_schema(schema["items"])
    for key in ("anyOf", "oneOf", "allOf"):
        for value in schema.get(key, []) or []:
            if isinstance(value, dict):
                _strict_schema(value)
    defs = schema.get("$defs", {})
    for value in defs.values():
        if isinstance(value, dict):
            _strict_schema(value)
    return schema


class OpenAIAdapter:
    def __init__(self, api_key: str, *, base_url: str | None = None, provider: str = "openai") -> None:
        if not api_key:
            raise ModelOutputError(f"{provider} API key is required")
        self.provider = provider
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @retry(
        retry=retry_if_exception(_is_retryable_exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def structured_response(
        self,
        *,
        model: str,
        instructions: str,
        user_input: str,
        schema_model: type[T],
        schema_name: str,
        reasoning_effort: str = "medium",
        tools: list[dict[str, Any]] | None = None,
    ) -> T:
        started_at = time.perf_counter()
        tool_count = len(tools or [])
        reasoning_payload = self._reasoning_payload(model, reasoning_effort)
        logger.info(
            "Model call start provider=%s model=%s schema=%s tools=%d reasoning=%s",
            self.provider,
            model,
            schema_name,
            tool_count,
            reasoning_effort if reasoning_payload is not None else "provider_default",
        )
        try:
            schema = _strict_schema(schema_model.model_json_schema())
            request: dict[str, Any] = {
                "model": model,
                "instructions": instructions,
                "input": user_input,
                "tools": tools or [],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    }
                },
            }
            if reasoning_payload is not None:
                request["reasoning"] = reasoning_payload
            response = await self.client.responses.create(**request)
            self._log_usage(response, model=model, schema_name=schema_name)
            text = getattr(response, "output_text", None) or self._extract_text(response)
            data = json.loads(text)
            parsed = schema_model.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning(
                "Model call failed provider=%s model=%s schema=%s error_type=ModelOutputError elapsed=%.1fs",
                self.provider,
                model,
                schema_name,
                time.perf_counter() - started_at,
            )
            raise ModelOutputError(f"Could not parse {schema_name}: {exc}\n{text[:2000]}") from exc
        except Exception as exc:
            logger.warning(
                "Model call failed provider=%s model=%s schema=%s error_type=%s status=%s elapsed=%.1fs",
                self.provider,
                model,
                schema_name,
                type(exc).__name__,
                _status_code(exc),
                time.perf_counter() - started_at,
            )
            raise
        logger.info(
            "Model call end provider=%s model=%s schema=%s elapsed=%.1fs",
            self.provider,
            model,
            schema_name,
            time.perf_counter() - started_at,
        )
        return parsed

    def _log_usage(self, response: Any, *, model: str, schema_name: str) -> None:
        usage = getattr(response, "usage", None)
        server_side_tool_usage = getattr(response, "server_side_tool_usage", None)
        if usage is None and server_side_tool_usage is None:
            return
        logger.info(
            "Model call usage provider=%s model=%s schema=%s usage=%s server_side_tool_usage=%s",
            self.provider,
            model,
            schema_name,
            self._jsonable(usage),
            self._jsonable(server_side_tool_usage),
        )

    def _reasoning_payload(self, model: str, reasoning_effort: str) -> dict[str, str] | None:
        if self.provider == "xai" and model.startswith("grok-4.20-0309-"):
            return None
        return {"effort": reasoning_effort}

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, dict):
            return value
        return str(value)

    @staticmethod
    def _extract_text(response: Any) -> str:
        output = getattr(response, "output", None) or []
        chunks: list[str] = []
        for item in output:
            content = getattr(item, "content", None) or []
            for part in content:
                text = getattr(part, "text", None)
                if text:
                    chunks.append(text)
        if chunks:
            return "\n".join(chunks)
        return str(response)
