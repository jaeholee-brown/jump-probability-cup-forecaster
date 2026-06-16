from __future__ import annotations

import json
from typing import Any, TypeVar

from anthropic import AsyncAnthropic
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from probability_cup_bot.openai_adapter import ModelOutputError


T = TypeVar("T", bound=BaseModel)


class AnthropicAdapter:
    def __init__(self, api_key: str, *, provider: str = "anthropic") -> None:
        if not api_key:
            raise ModelOutputError(f"{provider} API key is required")
        self.provider = provider
        self.client = AsyncAnthropic(api_key=api_key)

    @retry(
        retry=retry_if_exception_type((ModelOutputError, TimeoutError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
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
        del reasoning_effort, tools
        schema = schema_model.model_json_schema()
        response = await self.client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0.2,
            system=(
                f"{instructions}\n\n"
                f"Return only a JSON object named {schema_name} matching this JSON Schema. "
                "Do not wrap it in markdown and do not add commentary.\n"
                f"{json.dumps(schema, ensure_ascii=True)}"
            ),
            messages=[{"role": "user", "content": user_input}],
        )
        text = self._extract_text(response)
        try:
            data = json.loads(text)
            return schema_model.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ModelOutputError(f"Could not parse {schema_name}: {exc}\n{text[:2000]}") from exc

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
