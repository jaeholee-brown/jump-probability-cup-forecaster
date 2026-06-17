from __future__ import annotations

import logging
from typing import Any, TypeVar

import pytest
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
    def __init__(self, provider: str = "openai") -> None:
        self.provider = provider
        self.reasoning_efforts: list[str] = []

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
        self.reasoning_efforts.append(reasoning_effort)
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


async def test_forecast_match_uses_no_reasoning_label_for_claude() -> None:
    settings = Settings(
        sportspredict_api_key="sportspredict_test_key",
        openai_api_key="",
        anthropic_api_key="anthropic_test_key",
        use_openai_forecast=False,
        use_grok_forecast=False,
        claude_forecast_models=("claude-opus-4-8",),
    )
    adapter = SuccessfulAdapter("anthropic")
    forecaster = MatchForecaster(settings, anthropic=adapter)
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
        query_summary="summary",
    )

    forecasts = await forecaster.forecast_match(match=match, markets=[market], evidence=evidence)

    assert forecasts[0].market_id == "market"
    assert adapter.reasoning_efforts == ["none"]


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
    assert [spec.weight for spec in specs] == [0.5, 0.225, 0.2, 1.35, 0.6]


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

    assert [spec.weight for spec in specs] == pytest.approx([0.45, 0.225, 0.22, 1.35, 0.6])


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


def test_aggregate_repairs_grok_boundary_probability_from_rationale() -> None:
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
        model="grok-4.3",
        prompt_variant="base_rate_frequency",
        provider="grok",
        weight=1.0,
        forecasts=[
            MarketForecast(
                market_id="market",
                question="Will A win?",
                reference_class="Similar favorites.",
                probability_rationale="Base 0.55 lifted by team strength. Final probability: 0.61",
                probability=0.01,
                confidence="medium",
                evidence_quality="medium",
            )
        ],
    )

    [forecast] = forecaster._aggregate([market], [batch])

    assert forecast.component_probabilities == [0.61]
    assert forecast.metadata["raw_component_probabilities"] == [0.01]
    assert forecast.metadata["probability_repairs"][0]["raw_probability"] == 0.01
    assert forecast.metadata["probability_repairs"][0]["probability"] == 0.61
    assert forecast.probability_int > 55


def test_aggregate_leaves_grok_boundary_probability_when_rationale_has_no_number() -> None:
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
        model="grok-4.3",
        prompt_variant="base_rate_frequency",
        provider="grok",
        weight=1.0,
        forecasts=[
            MarketForecast(
                market_id="market",
                question="Will A win?",
                reference_class="Unavailable player prop.",
                probability_rationale="Player excluded from the squad; probability near minimum.",
                probability=0.01,
                confidence="high",
                evidence_quality="high",
            )
        ],
    )

    [forecast] = forecaster._aggregate([market], [batch])

    assert forecast.component_probabilities == [0.01]
    assert forecast.metadata["probability_repairs"] == []
