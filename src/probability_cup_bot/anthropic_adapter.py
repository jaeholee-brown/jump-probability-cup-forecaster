from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, TypeVar

from anthropic import APIConnectionError, APIStatusError, APITimeoutError, AsyncAnthropic, RateLimitError
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from probability_cup_bot.openai_adapter import ModelOutputError
from probability_cup_bot.usage import record_usage


T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)
DEFAULT_ANTHROPIC_MAX_TOKENS = 8192


def _anthropic_max_tokens() -> int:
    raw_value = os.getenv("ANTHROPIC_MAX_TOKENS", "").strip()
    if not raw_value:
        return DEFAULT_ANTHROPIC_MAX_TOKENS
    return max(1024, int(raw_value))


def _is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError, TimeoutError)):
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


class AnthropicAdapter:
    def __init__(self, api_key: str, *, provider: str = "anthropic") -> None:
        if not api_key:
            raise ModelOutputError(f"{provider} API key is required")
        self.provider = provider
        self.client = AsyncAnthropic(api_key=api_key)

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
        logger.info(
            "Model call start provider=%s model=%s schema=%s tools=%d reasoning=%s",
            self.provider,
            model,
            schema_name,
            len(tools or []),
            reasoning_effort,
        )
        del reasoning_effort, tools
        try:
            schema = schema_model.model_json_schema()
            response = await self.client.messages.create(
                model=model,
                max_tokens=_anthropic_max_tokens(),
                system=(
                    f"{instructions}\n\nReturn one structured {schema_name} tool call. "
                    "Do not add commentary outside the tool call."
                ),
                tools=[
                    {
                        "name": schema_name,
                        "description": f"Structured {schema_name} output.",
                        "input_schema": schema,
                    }
                ],
                tool_choice={"type": "tool", "name": schema_name},
                messages=[{"role": "user", "content": user_input}],
            )
            self._log_usage(response, model=model, schema_name=schema_name)
            record_usage(
                provider=self.provider,
                model=model,
                schema_name=schema_name,
                usage=getattr(response, "usage", None),
            )
            data = self._extract_tool_input(response, schema_name)
            text = "" if data is not None else self._extract_text(response)
            if data is None:
                data = _load_json_object(text)
            if isinstance(data, dict) and isinstance(data.get(schema_name), dict):
                data = data[schema_name]
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

    @staticmethod
    def _extract_tool_input(response: Any, schema_name: str) -> Any | None:
        for item in getattr(response, "content", []) or []:
            if getattr(item, "type", "") != "tool_use":
                continue
            if getattr(item, "name", "") != schema_name:
                continue
            tool_input = getattr(item, "input", None)
            if tool_input is not None:
                return tool_input
        return None

    def _log_usage(self, response: Any, *, model: str, schema_name: str) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        logger.info(
            "Model call usage provider=%s model=%s schema=%s usage=%s",
            self.provider,
            model,
            schema_name,
            usage.model_dump() if hasattr(usage, "model_dump") else str(usage),
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        chunks: list[str] = []
        for item in getattr(response, "content", []) or []:
            text = getattr(item, "text", None)
            if text:
                chunks.append(text)
        if chunks:
            return "\n".join(chunks).strip()
        return str(response)


def _load_json_object(text: str) -> Any:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        return data
    raise json.JSONDecodeError("No JSON object found", stripped, 0)
