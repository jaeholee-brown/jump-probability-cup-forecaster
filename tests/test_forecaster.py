from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from probability_cup_bot.config import Settings
from probability_cup_bot.forecaster import MatchForecaster


T = TypeVar("T", bound=BaseModel)


class FakeAdapter:
    def __init__(self, provider: str) -> None:
        self.provider = provider

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
        raise AssertionError("not called")


def test_forecaster_builds_cross_provider_specs() -> None:
    settings = Settings(
        sportspredict_api_key="sportspredict_test_key",
        openai_api_key="openai_test_key",
        xai_api_key="xai_test_key",
        anthropic_api_key="anthropic_test_key",
    )
    forecaster = MatchForecaster(
        settings,
        openai=FakeAdapter("openai"),
        grok=FakeAdapter("xai"),
        anthropic=FakeAdapter("anthropic"),
    )

    specs = forecaster._forecast_model_specs()

    assert [spec.provider for spec in specs] == ["openai", "grok", "grok", "claude", "claude"]
    assert [spec.model for spec in specs] == [
        "gpt-5",
        "grok-4.3",
        "grok-4.20-0309-reasoning",
        "claude-opus-4-8",
        "claude-opus-4-6",
    ]
    assert [spec.variants for spec in specs] == [
        ("base_rate_frequency",),
        ("base_rate_frequency",),
        ("base_rate_frequency",),
        ("base_rate_frequency",),
        ("base_rate_frequency",),
    ]
    assert [spec.weight for spec in specs] == [1.0, 0.4, 0.6, 0.7, 0.8]


def test_forecaster_applies_calibration_multipliers() -> None:
    settings = Settings(
        sportspredict_api_key="sportspredict_test_key",
        openai_api_key="openai_test_key",
        xai_api_key="xai_test_key",
        anthropic_api_key="anthropic_test_key",
    )
    forecaster = MatchForecaster(
        settings,
        openai=FakeAdapter("openai"),
        grok=FakeAdapter("xai"),
        anthropic=FakeAdapter("anthropic"),
        calibration_multipliers={"gpt-5": 0.9, "grok-4.20-0309-reasoning": 1.1},
    )

    specs = forecaster._forecast_model_specs()

    assert [spec.weight for spec in specs] == [0.9, 0.4, 0.66, 0.7, 0.8]
