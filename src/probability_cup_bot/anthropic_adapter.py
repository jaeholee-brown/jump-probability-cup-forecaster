from __future__ import annotations

import json
import logging
import time
from typing import Any, TypeVar

from anthropic import APIConnectionError, APIStatusError, APITimeoutError, AsyncAnthropic, RateLimitError
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from probability_cup_bot.openai_adapter import ModelOutputError


T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


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
                max_tokens=4096,
                system=(
                    f"{instructions}\n\n"
                    f"Return only a JSON object named {schema_name} matching this JSON Schema. "
                    "Do not wrap it in markdown and do not add commentary.\n"
                    f"{json.dumps(schema, ensure_ascii=True)}"
                ),
                messages=[{"role": "user", "content": user_input}],
            )
            text = self._extract_text(response)
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
