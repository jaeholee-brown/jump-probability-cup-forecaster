from __future__ import annotations

from probability_cup_bot.anthropic_adapter import _is_retryable_exception, _load_json_object
from probability_cup_bot.openai_adapter import ModelOutputError


def test_load_json_object_accepts_markdown_wrapped_json() -> None:
    text = 'Here is the JSON:\n```json\n{"ok": true}\n```'

    assert _load_json_object(text) == {"ok": True}


def test_load_json_object_accepts_named_schema_wrapper() -> None:
    text = '{"forecast_batch": {"match_id": "match"}}'

    assert _load_json_object(text) == {"forecast_batch": {"match_id": "match"}}


def test_anthropic_parse_errors_are_not_retried() -> None:
    assert not _is_retryable_exception(ModelOutputError("bad json"))
