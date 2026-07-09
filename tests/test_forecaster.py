from __future__ import annotations

import json
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
        self.user_inputs: list[dict[str, Any]] = []

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
        self.user_inputs.append(json.loads(user_input))
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
    adapter = SuccessfulAdapter()
    forecaster = MatchForecaster(settings, openai=adapter)
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
    assert adapter.user_inputs[0]["markets"][0]["profile"]["family"] == "match_result"
    assert adapter.user_inputs[0]["market_profiles"][0]["family"] == "match_result"
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
        "grok-4.5",
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
    assert [spec.weight for spec in specs] == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert [spec.independent for spec in specs] == [False, False, True, False, False]
    grok_45 = specs[2]
    assert grok_45.tools == [{"type": "web_search"}, {"type": "x_search"}]


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

    assert [spec.weight for spec in specs] == pytest.approx([0.9, 1.0, 1.0, 1.0, 1.0])


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


def test_aggregate_applies_conservative_penalty_taker_coherence_floor() -> None:
    settings = Settings(
        sportspredict_api_key="sportspredict_test_key",
        openai_api_key="openai_test_key",
        # Disabled by default (hurt on all 3 settled firings); test the flag path.
        enable_coherence_adjustments=True,
    )
    forecaster = MatchForecaster(settings)
    penalty_market = Market(
        id="penalty",
        question="Will a penalty be awarded in Switzerland vs Bosnia?",
        status="open",
        match=MarketMatch(
            id="match",
            name="Switzerland vs Bosnia",
            closing_time="2026-06-20T12:00:00Z",
        ),
        lobby_id="lobby",
    )
    xhaka_market = Market(
        id="xhaka_sot",
        question="Will Granit Xhaka have 1+ shot on target?",
        status="open",
        match=MarketMatch(
            id="match",
            name="Switzerland vs Bosnia",
            closing_time="2026-06-20T12:00:00Z",
        ),
        lobby_id="lobby",
    )
    batch = ForecastBatch(
        match_id="match",
        match_name="Switzerland vs Bosnia",
        model="gpt-5",
        prompt_variant="base_rate_frequency",
        provider="openai",
        weight=1.0,
        forecasts=[
            MarketForecast(
                market_id="penalty",
                question=penalty_market.question,
                reference_class="Broad penalty-award rates.",
                probability_rationale="Penalty award path is material. Final probability: 0.30",
                probability=0.30,
                confidence="medium",
                evidence_quality="medium",
            ),
            MarketForecast(
                market_id="xhaka_sot",
                question=xhaka_market.question,
                reference_class="Midfielder SOT rates plus penalty role.",
                yes_reasons=["Xhaka is the likely penalty taker."],
                probability_rationale="Open-play SOT is low, but penalty role adds a path. Final probability: 0.10",
                probability=0.10,
                confidence="medium",
                evidence_quality="medium",
            ),
        ],
    )

    forecasts = forecaster._aggregate([penalty_market, xhaka_market], [batch])
    xhaka = next(forecast for forecast in forecasts if forecast.market_id == "xhaka_sot")

    assert xhaka.probability >= 0.13
    assert xhaka.metadata["coherence_adjustments"][0]["old_probability"] < xhaka.probability
    assert "penalty-taker channel" in xhaka.metadata["coherence_adjustments"][0]["reason"]


def test_aggregate_applies_family_correction() -> None:
    settings = Settings(
        sportspredict_api_key="sportspredict_test_key",
        openai_api_key="openai_test_key",
    )
    corrections = {
        "enabled": True,
        "shifts": {"cards": -0.45},
        "intercept": 0.0,
        "slope": 1.2,
        "family_stats": {"cards": {"count": 42, "yes_rate": 0.29, "mean_forecast": 0.52}},
    }
    forecaster = MatchForecaster(settings, family_corrections=corrections)
    market = Market(
        id="cards_market",
        question="Will there be 4 or more total cards shown?",
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
                market_id="cards_market",
                question=market.question,
                reference_class="Tournament card rates.",
                probability_rationale="Cards estimate. Final probability: 0.52",
                probability=0.52,
                confidence="medium",
                evidence_quality="medium",
            ),
        ],
    )

    forecasts = forecaster._aggregate([market], [batch])
    result = forecasts[0]

    # logit(0.52) + (-0.45) scaled by slope 1.2 => noticeably below 0.52
    assert result.probability < 0.45
    note = result.metadata["family_correction"]
    assert note["family"] == "cards"
    assert note["shift"] == -0.45
    assert note["raw_probability"] == 0.52


def test_aggregate_without_corrections_is_plain_log_odds_mean() -> None:
    settings = Settings(
        sportspredict_api_key="sportspredict_test_key",
        openai_api_key="openai_test_key",
    )
    forecaster = MatchForecaster(settings)
    market = Market(
        id="m",
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
                market_id="m",
                question=market.question,
                reference_class="odds",
                probability_rationale="Final probability: 0.60",
                probability=0.60,
                confidence="medium",
                evidence_quality="medium",
            ),
        ],
    )

    forecasts = forecaster._aggregate([market], [batch])

    # extremize_alpha=1.0 and base_shrinkage=0.0 defaults: no distortion.
    assert forecasts[0].probability == pytest.approx(0.60, abs=1e-9)


def test_tournament_context_includes_family_rate_and_tie_note() -> None:
    settings = Settings(
        sportspredict_api_key="sportspredict_test_key",
        openai_api_key="openai_test_key",
    )
    corrections = {
        "enabled": True,
        "shifts": {},
        "family_stats": {"cards": {"count": 42, "yes_rate": 0.2857, "mean_forecast": 0.52}},
    }
    forecaster = MatchForecaster(settings, family_corrections=corrections)

    context = forecaster._tournament_context(
        "cards", "Will Curaçao receive more cards than Ivory Coast?"
    )

    assert context["family_settled_count"] == 42
    assert "resolved YES 29%" in context["note"]
    assert "tie resolves NO" in context["comparison_note"]

    no_stats = forecaster._tournament_context("shots", "Will X have 2+ shots?")
    assert no_stats is None or "note" not in (no_stats or {})


async def test_independent_forecaster_does_not_receive_shared_evidence() -> None:
    settings = Settings(
        sportspredict_api_key="sportspredict_test_key",
        openai_api_key="",
        xai_api_key="xai_test_key",
        use_openai_forecast=False,
        use_claude_forecast=False,
        use_grok_forecast=False,
        use_grok_independent_forecast=True,
    )
    adapter = SuccessfulAdapter("xai")
    forecaster = MatchForecaster(settings, grok=adapter)
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
        query_summary="secret shared research summary",
        key_facts=["secret shared research fact"],
        odds_context="Bookmaker consensus: A 1.80, B 4.50",
    )

    forecasts = await forecaster.forecast_match(match=match, markets=[market], evidence=evidence)

    assert forecasts[0].market_id == "market"
    payload = adapter.user_inputs[0]
    serialized = json.dumps(payload)
    assert "secret shared research" not in serialized
    assert payload["evidence"]["odds_context"] == "Bookmaker consensus: A 1.80, B 4.50"
    assert "research independently" in payload["evidence"]["note"]


def test_divergent_independent_component_is_dropped() -> None:
    settings = Settings(
        sportspredict_api_key="sportspredict_test_key",
        openai_api_key="openai_test_key",
    )
    forecaster = MatchForecaster(settings)
    market = Market(
        id="m",
        question="Will A win?",
        status="open",
        match=MarketMatch(id="match", name="A vs B", closing_time="2026-06-20T12:00:00Z"),
        lobby_id="lobby",
    )

    def make_batch(model: str, variant: str, p: float) -> ForecastBatch:
        return ForecastBatch(
            match_id="match",
            match_name="A vs B",
            model=model,
            prompt_variant=variant,
            provider="grok" if "grok" in model else "openai",
            weight=1.0,
            forecasts=[
                MarketForecast(
                    market_id="m",
                    question=market.question,
                    reference_class="odds",
                    probability_rationale=f"Final probability: {p}",
                    probability=p,
                    confidence="medium",
                    evidence_quality="medium",
                )
            ],
        )

    batches = [
        make_batch("gpt-5", "openai_gpt_5_base_rate_frequency", 0.55),
        make_batch("claude-opus-4-8", "claude_claude_opus_4_8_base_rate_frequency", 0.57),
        # rogue: thinks the match already resolved
        make_batch("grok-4.5", "grok_grok_4_5_independent_base_rate_frequency", 0.01),
    ]
    forecasts = forecaster._aggregate([market], batches)

    assert len(forecasts) == 1
    # rogue dropped: aggregate stays near the sane components
    assert forecasts[0].probability > 0.45
    dropped = forecasts[0].metadata["dropped_independent_components"]
    assert len(dropped) == 1 and dropped[0]["model"] == "grok-4.5"

    # sane disagreement survives (0.30 vs ~0.56 is well under 3 logits)
    batches[2] = make_batch("grok-4.5", "grok_grok_4_5_independent_base_rate_frequency", 0.30)
    forecasts = forecaster._aggregate([market], batches)
    assert forecasts[0].metadata["dropped_independent_components"] == []
    assert 0.40 < forecasts[0].probability < 0.55
