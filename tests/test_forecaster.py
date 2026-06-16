from __future__ import annotations

import logging
from typing import Any, TypeVar

from pydantic import BaseModel

from probability_cup_bot.config import Settings
from probability_cup_bot.forecaster import MatchForecaster
from probability_cup_bot.models import ForecastBatch, Market, MarketForecast, MarketMatch, Match, MatchEvidence


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


class SuccessfulAdapter:
    provider = "openai"

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
        return schema_model.model_validate(
            {
                "match_id": "match",
                "match_name": "A vs B",
                "model": model,
                "prompt_variant": "base_rate_frequency",
                "provider": self.provider,
                "forecasts": [
                    {
                        "market_id": "market",
                        "question": "Will A win?",
                        "reference_class": "Even soccer match favorite win rates.",
                        "probability": 0.58,
                        "confidence": "medium",
                        "evidence_quality": "medium",
                    }
                ],
            }
        )


async def test_forecast_match_logs_variant_progress_without_evidence_text(caplog) -> None:
    settings = Settings(
        sportspredict_api_key="sportspredict_test_key",
        openai_api_key="openai_test_key",
        use_grok_forecast=False,
        use_claude_forecast=False,
    )
    forecaster = MatchForecaster(settings, openai=SuccessfulAdapter())
    match = Match(id="match", name="A vs B", closing_time="2026-06-20T12:00:00Z")
    market = Market(
        id="market",
        question="Will A win?",
        status="open",
        match=MarketMatch(id="match", name="A vs B", closing_time="2026-06-20T12:00:00Z"),
        lobby_id="lobby",
    )
    evidence = MatchEvidence(
        match_id="match",
        match_name="A vs B",
        generated_at="2026-06-16T00:00:00Z",
        query_summary="private evidence summary",
        key_facts=["private evidence fact"],
    )
    caplog.set_level(logging.INFO, logger="probability_cup_bot.forecaster")

    forecasts = await forecaster.forecast_match(match=match, markets=[market], evidence=evidence)

    assert forecasts[0].market_id == "market"
    assert "Forecast variant start match_id=match provider=openai model=gpt-5" in caplog.text
    assert "Forecast variant end match_id=match provider=openai model=gpt-5" in caplog.text
    assert "private evidence" not in caplog.text


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


def test_aggregate_preserves_component_reasoning_metadata() -> None:
    settings = Settings(
        sportspredict_api_key="sportspredict_test_key",
        openai_api_key="openai_test_key",
    )
    forecaster = MatchForecaster(settings)
    market = Market(
        id="market",
        question="Will A win?",
        status="open",
        match=MarketMatch(id="match", name="A vs B", closing_time="2026-06-20T12:00:00Z"),
        lobby_id="lobby",
    )
    batch = ForecastBatch(
        match_id="match",
        match_name="A vs B",
        model="gpt-5",
        prompt_variant="base_rate_frequency",
        provider="openai",
        weight=1.0,
        forecasts=[
            MarketForecast(
                market_id="market",
                question="Will A win?",
                resolution_interpretation="A must win in regulation.",
                reference_class="Even soccer match favorite win rates.",
                base_rate=0.52,
                base_rate_rationale="Similar favorites win slightly more often than not.",
                yes_reasons=["A rates higher."],
                no_reasons=["B has upset paths."],
                probability_rationale="Start near 52%, then move up for team strength.",
                probability=0.58,
                confidence="medium",
                evidence_quality="medium",
                calibration_notes="Avoid overconfidence.",
                consistency_notes="Consistent with draw risk.",
            )
        ],
    )

    [forecast] = forecaster._aggregate([market], [batch])

    assert forecast.metadata["probability_rationales"] == [
        "Start near 52%, then move up for team strength."
    ]
    assert forecast.metadata["base_rate_rationales"] == [
        "Similar favorites win slightly more often than not."
    ]
