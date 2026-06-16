from __future__ import annotations

import json
import logging
import time
from typing import Any, TypeVar

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


class ModelOutputError(RuntimeError):
    pass


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
        retry=retry_if_exception_type(
            (APIConnectionError, APIStatusError, APITimeoutError, ModelOutputError, RateLimitError, TimeoutError)
        ),
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
        logger.info(
            "Model call start provider=%s model=%s schema=%s tools=%d reasoning=%s",
            self.provider,
            model,
            schema_name,
            tool_count,
            reasoning_effort,
        )
        try:
            schema = _strict_schema(schema_model.model_json_schema())
            response = await self.client.responses.create(
                model=model,
                instructions=instructions,
                input=user_input,
                tools=tools or [],
                reasoning={"effort": reasoning_effort},
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
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
                "Model call failed provider=%s model=%s schema=%s error_type=%s elapsed=%.1fs",
                self.provider,
                model,
                schema_name,
                type(exc).__name__,
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
