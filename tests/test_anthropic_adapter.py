from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from probability_cup_bot.anthropic_adapter import AnthropicAdapter, _is_retryable_exception, _load_json_object
from probability_cup_bot.openai_adapter import ModelOutputError


def test_load_json_object_accepts_markdown_wrapped_json() -> None:
    text = 'Here is the JSON:\n```json\n{"ok": true}\n```'

    assert _load_json_object(text) == {"ok": True}


def test_load_json_object_accepts_named_schema_wrapper() -> None:
    text = '{"forecast_batch": {"match_id": "match"}}'

    assert _load_json_object(text) == {"forecast_batch": {"match_id": "match"}}


def test_anthropic_parse_errors_are_not_retried() -> None:
    assert not _is_retryable_exception(ModelOutputError("bad json"))


def test_extract_tool_input_uses_named_tool_call() -> None:
    class ToolUse:
        type = "tool_use"
        name = "forecast_batch"
        input = {"match_id": "match"}

    class Response:
        content = [ToolUse()]

    assert AnthropicAdapter._extract_tool_input(Response(), "forecast_batch") == {"match_id": "match"}


async def test_anthropic_structured_response_allows_larger_tool_outputs() -> None:
    class TinySchema(BaseModel):
        ok: bool

    class ToolUse:
        type = "tool_use"
        name = "tiny_schema"
        input = {"ok": True}

    class Response:
        content = [ToolUse()]
        usage = None

    class Messages:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        async def create(self, **kwargs: Any) -> Response:
            self.kwargs = kwargs
            return Response()

    class Client:
        def __init__(self) -> None:
            self.messages = Messages()

    adapter = AnthropicAdapter("test-key")
    client = Client()
    adapter.client = client

    result = await adapter.structured_response(
        model="claude-opus-4-6",
        instructions="Return JSON.",
        user_input="{}",
        schema_model=TinySchema,
        schema_name="tiny_schema",
    )

    assert result.ok is True
    assert client.messages.kwargs["max_tokens"] == 8192
