import json
import logging
from datetime import timedelta, timezone

from probability_cup_bot.config import Settings
from probability_cup_bot.models import AggregatedForecast, Market, MarketMatch, Match, NewsCheck, Prediction, utcnow
from probability_cup_bot.runner import ForecastRunner


class FakeNewsMonitor:
    async def check_match(
        self,
        *,
        match: Match,
        markets: list[Market],
        match_history: dict,
        cached_news: dict,
        firecrawl_context: str = "",
    ) -> NewsCheck:
        return NewsCheck(
            match_id=match.id,
            match_name=match.name,
            checked_at=utcnow().isoformat(),
            should_reforecast=True,
            estimated_delta_points=4,
            materiality="medium",
            evidence_quality="high",
            reason="Confirmed lineup materially changes player prop.",
            summary="Player X is confirmed out.",
            new_developments=["Player X is confirmed out."],
            sources=[],
        )


def test_plan_writes_creates_updates_and_skips(caplog) -> None:
    settings = Settings(
        sportspredict_api_key="sportspredict_test_key",
        openai_api_key="openai_test_key",
        update_threshold_points=2,
    )
    runner = ForecastRunner(settings)
    forecasts = [
        AggregatedForecast(
            market_id="new",
            question="new?",
            probability=0.61,
            probability_int=61,
            component_probabilities=[0.61],
            confidence="medium",
            evidence_quality="medium",
        ),
        AggregatedForecast(
            market_id="update",
            question="update?",
            probability=0.66,
            probability_int=66,
            component_probabilities=[0.66],
            confidence="medium",
            evidence_quality="medium",
        ),
        AggregatedForecast(
            market_id="skip",
            question="skip?",
            probability=0.51,
            probability_int=51,
            component_probabilities=[0.51],
            confidence="medium",
            evidence_quality="medium",
        ),
    ]
    existing = [
        Prediction(id="pred_update", market_id="update", lobby_id="lobby", probability=62),
        Prediction(id="pred_skip", market_id="skip", lobby_id="lobby", probability=50),
    ]
    caplog.set_level(logging.INFO, logger="probability_cup_bot.runner")
    plan = runner._plan_writes(
        forecasts=forecasts,
        existing_predictions=existing,
        lobby_id="lobby",
    )
    assert len(plan["creates"]) == 1
    assert plan["creates"][0]["market_id"] == "new"
    assert len(plan["updates"]) == 1
    assert plan["updates"][0]["prediction_id"] == "pred_update"
    assert len(plan["skips"]) == 1
    assert "Plan writes ready creates=1 updates=1 skips=1" in caplog.text


def test_select_matches_restricts_to_forced_target_match() -> None:
    settings = Settings(
        sportspredict_api_key="sportspredict_test_key",
        openai_api_key="openai_test_key",
    )
    runner = ForecastRunner(settings)
    closing_time = (utcnow() + timedelta(hours=2)).isoformat()
    matches = [
        Match(id="target", name="Target FC vs Other FC", closing_time=closing_time),
        Match(id="other", name="Other FC vs Third FC", closing_time=closing_time),
    ]
    markets = [
        Market(
            id="target_market",
            question="Will Target FC win?",
            status="open",
            match=MarketMatch(id="target", name="Target FC vs Other FC", closing_time=closing_time),
            lobby_id="lobby",
        ),
        Market(
            id="other_market",
            question="Will Other FC win?",
            status="open",
            match=MarketMatch(id="other", name="Other FC vs Third FC", closing_time=closing_time),
            lobby_id="lobby",
        ),
    ]
    existing = [
        Prediction(id="pred_target", market_id="target_market", lobby_id="lobby", probability=50),
        Prediction(id="pred_other", market_id="other_market", lobby_id="lobby", probability=50),
    ]

    selected = runner._select_matches(
        matches,
        markets,
        existing,
        {"matches": {}},
        target_match_ids={"target"},
        force_target_matches=True,
    )

    assert [match.id for match, _markets in selected] == ["target"]


def test_forecast_checkpoint_writes_partial_telemetry(tmp_path) -> None:
    settings = Settings(
        sportspredict_api_key="sportspredict_test_key",
        openai_api_key="openai_test_key",
        state_dir=tmp_path / "state",
        logs_dir=tmp_path / "logs",
    )
    runner = ForecastRunner(settings)
    forecast = AggregatedForecast(
        market_id="market",
        question="Will A win?",
        probability=0.61,
        probability_int=61,
        component_probabilities=[0.6, 0.62],
        confidence="medium",
        evidence_quality="medium",
    )

    runner._write_forecast_checkpoint(
        [forecast],
        completed_matches=1,
        failed_matches=0,
        total_matches=3,
        status="running",
        stage="forecasting",
        latest_match_id="match",
        latest_match_name="A vs B",
        elapsed_seconds=12.5,
    )

    checkpoint = json.loads((tmp_path / "state" / "in-progress-run.json").read_text())
    assert checkpoint["status"] == "running"
    assert checkpoint["stage"] == "forecasting"
    assert checkpoint["completed_matches"] == 1
    assert checkpoint["forecast_count"] == 1
    assert checkpoint["elapsed_seconds"] == 12.5
    assert checkpoint["forecasts"][0]["market_id"] == "market"


def test_select_matches_skips_fresh_existing_predictions() -> None:
    settings = Settings(
        sportspredict_api_key="sportspredict_test_key",
        openai_api_key="openai_test_key",
        max_prediction_age_hours=12,
        force_reforecast_within_hours=6,
    )
    runner = ForecastRunner(settings)
    closing_time = (utcnow() + timedelta(days=2)).astimezone(timezone.utc).isoformat()
    updated_date = (utcnow() - timedelta(hours=2)).astimezone(timezone.utc).isoformat()
    match = Match(
        id="match",
        name="A vs B",
        closing_time=closing_time,
        open_market_count=1,
    )
    markets = [
        Market(
            id="market",
            question="Will A win?",
            status="open",
            match=MarketMatch(id="match", name="A vs B", closing_time=closing_time),
            lobby_id="lobby",
        )
    ]
    existing = [
        Prediction(
            id="prediction",
            market_id="market",
            lobby_id="lobby",
            probability=51,
            market_status="open",
            updated_date=updated_date,
        )
    ]

    selected = runner._select_matches([match], markets, existing)

    assert selected == []


def test_select_matches_forces_missing_and_close_markets() -> None:
    settings = Settings(
        sportspredict_api_key="sportspredict_test_key",
        openai_api_key="openai_test_key",
        max_prediction_age_hours=12,
        force_reforecast_within_hours=6,
    )
    runner = ForecastRunner(settings)
    closing_time = (utcnow() + timedelta(hours=2)).astimezone(timezone.utc).isoformat()
    updated_date = (utcnow() - timedelta(hours=1)).astimezone(timezone.utc).isoformat()
    match = Match(
        id="match",
        name="A vs B",
        closing_time=closing_time,
        open_market_count=2,
    )
    markets = [
        Market(
            id="existing_market",
            question="Will A win?",
            status="open",
            match=MarketMatch(id="match", name="A vs B", closing_time=closing_time),
            lobby_id="lobby",
        ),
        Market(
            id="new_market",
            question="Will both teams score?",
            status="open",
            match=MarketMatch(id="match", name="A vs B", closing_time=closing_time),
            lobby_id="lobby",
        ),
    ]
    existing = [
        Prediction(
            id="prediction",
            market_id="existing_market",
            lobby_id="lobby",
            probability=51,
            market_status="open",
            updated_date=updated_date,
        )
    ]

    selected = runner._select_matches([match], markets, existing)

    assert len(selected) == 1
    assert selected[0][0].id == "match"


def test_select_matches_uses_history_to_accelerate_volatile_updates() -> None:
    settings = Settings(
        sportspredict_api_key="sportspredict_test_key",
        openai_api_key="openai_test_key",
        max_prediction_age_hours=12,
        force_reforecast_within_hours=6,
        stale_reforecast_without_news=True,
    )
    runner = ForecastRunner(settings)
    closing_time = (utcnow() + timedelta(days=2)).astimezone(timezone.utc).isoformat()
    updated_date = (utcnow() - timedelta(hours=5)).astimezone(timezone.utc).isoformat()
    match = Match(
        id="match",
        name="A vs B",
        closing_time=closing_time,
        open_market_count=1,
    )
    markets = [
        Market(
            id="market",
            question="Will player X score a goal?",
            status="open",
            match=MarketMatch(id="match", name="A vs B", closing_time=closing_time),
            lobby_id="lobby",
        )
    ]
    existing = [
        Prediction(
            id="prediction",
            market_id="market",
            lobby_id="lobby",
            probability=51,
            market_status="open",
            updated_date=updated_date,
        )
    ]
    history = {
        "matches": {
            "match": {
                "last_forecast_at": updated_date,
                "max_component_spread_points": 25,
                "worst_evidence_quality": "medium",
                "worst_confidence": "medium",
            }
        }
    }

    selected = runner._select_matches([match], markets, existing, history)

    assert len(selected) == 1


def test_latest_only_mode_waits_for_final_window_or_news() -> None:
    settings = Settings(
        sportspredict_api_key="sportspredict_test_key",
        openai_api_key="openai_test_key",
        max_prediction_age_hours=12,
        force_reforecast_within_hours=1.5,
        stale_reforecast_without_news=False,
    )
    runner = ForecastRunner(settings)
    closing_time = (utcnow() + timedelta(hours=12)).astimezone(timezone.utc).isoformat()
    updated_date = (utcnow() - timedelta(hours=6)).astimezone(timezone.utc).isoformat()
    match = Match(id="match", name="A vs B", closing_time=closing_time, open_market_count=1)
    markets = [
        Market(
            id="market",
            question="Will player X score a goal?",
            status="open",
            match=MarketMatch(id="match", name="A vs B", closing_time=closing_time),
            lobby_id="lobby",
        )
    ]
    existing = [
        Prediction(
            id="prediction",
            market_id="market",
            lobby_id="lobby",
            probability=51,
            market_status="open",
            updated_date=updated_date,
        )
    ]

    selected = runner._select_matches([match], markets, existing)

    assert selected == []


def test_final_window_respects_minimum_interval() -> None:
    settings = Settings(
        sportspredict_api_key="sportspredict_test_key",
        openai_api_key="openai_test_key",
        force_reforecast_within_hours=1.5,
        final_reforecast_min_interval_minutes=30,
    )
    runner = ForecastRunner(settings)
    closing_time = (utcnow() + timedelta(hours=1)).astimezone(timezone.utc).isoformat()
    fresh_update = (utcnow() - timedelta(minutes=10)).astimezone(timezone.utc).isoformat()
    stale_update = (utcnow() - timedelta(minutes=40)).astimezone(timezone.utc).isoformat()
    match = Match(id="match", name="A vs B", closing_time=closing_time, open_market_count=1)
    markets = [
        Market(
            id="market",
            question="Will A win?",
            status="open",
            match=MarketMatch(id="match", name="A vs B", closing_time=closing_time),
            lobby_id="lobby",
        )
    ]

    fresh = [
        Prediction(id="prediction", market_id="market", lobby_id="lobby", probability=51, updated_date=fresh_update)
    ]
    stale = [
        Prediction(id="prediction", market_id="market", lobby_id="lobby", probability=51, updated_date=stale_update)
    ]

    assert runner._select_matches([match], markets, fresh) == []
    assert len(runner._select_matches([match], markets, stale)) == 1


def test_firecrawl_gate_targets_close_or_volatile_matches() -> None:
    settings = Settings(
        sportspredict_api_key="sportspredict_test_key",
        openai_api_key="openai_test_key",
        firecrawl_mode="targeted",
        firecrawl_force_within_hours=2,
        firecrawl_volatile_within_hours=24,
    )
    runner = ForecastRunner(settings)
    close_time = (utcnow() + timedelta(hours=1)).astimezone(timezone.utc).isoformat()
    later_time = (utcnow() + timedelta(hours=12)).astimezone(timezone.utc).isoformat()
    close_match = Match(id="close", name="A vs B", closing_time=close_time)
    later_match = Match(id="later", name="C vs D", closing_time=later_time)
    close_markets = [
        Market(
            id="close_market",
            question="Will A win?",
            status="open",
            match=MarketMatch(id="close", name="A vs B", closing_time=close_time),
            lobby_id="lobby",
        )
    ]
    volatile_markets = [
        Market(
            id="volatile_market",
            question="Will player X score a goal?",
            status="open",
            match=MarketMatch(id="later", name="C vs D", closing_time=later_time),
            lobby_id="lobby",
        )
    ]

    assert runner._should_use_firecrawl(close_match, close_markets, {}, {}, utcnow())
    assert runner._should_use_firecrawl(later_match, volatile_markets, {}, {}, utcnow())


def test_forecast_firecrawl_context_is_cached_for_audit() -> None:
    news_cache = {"matches": {}}
    match = Match(id="match", name="A vs B", closing_time="2026-06-20T12:00:00Z")
    context = "Firecrawl credits used: 14\nFirecrawl query: A vs B lineup\n- Official: Team news"

    ForecastRunner._record_forecast_firecrawl_context(news_cache, match, context)

    entry = news_cache["matches"]["match"]
    assert entry["forecast_firecrawl_credits"] == 14
    assert entry["forecast_firecrawl_context"] == context
    assert entry["forecast_firecrawl_history"][0]["context"] == context
    cached_context = ForecastRunner._cached_news_context(entry)
    assert "Cached full-research Firecrawl snippets" in cached_context
    assert "Official: Team news" in cached_context


def test_affected_markets_limits_news_monitor_promotion() -> None:
    markets = [
        Market(
            id="affected",
            question="Will player X score?",
            status="open",
            match=MarketMatch(id="match", name="A vs B", closing_time="2026-06-20T12:00:00Z"),
            lobby_id="lobby",
        ),
        Market(
            id="unaffected",
            question="Will A win?",
            status="open",
            match=MarketMatch(id="match", name="A vs B", closing_time="2026-06-20T12:00:00Z"),
            lobby_id="lobby",
        ),
    ]

    affected = ForecastRunner._affected_markets(markets, ["affected"])

    assert [market.id for market in affected] == ["affected"]


def test_component_coverage_reports_missing_configured_models() -> None:
    settings = Settings(
        sportspredict_api_key="sportspredict_test_key",
        openai_api_key="openai_test_key",
        xai_api_key="xai_test_key",
        anthropic_api_key="anthropic_test_key",
    )
    runner = ForecastRunner(settings)
    forecasts = [
        AggregatedForecast(
            market_id="market",
            question="Will A win?",
            probability=0.61,
            probability_int=61,
            component_probabilities=[0.62, 0.60],
            confidence="medium",
            evidence_quality="medium",
            metadata={"models": ["gpt-5", "grok-4.3"]},
        )
    ]

    coverage = runner._component_coverage(forecasts)

    assert coverage["forecast_count"] == 1
    assert coverage["full_coverage_market_count"] == 0
    assert coverage["partial_coverage_market_count"] == 1
    assert coverage["missing_by_model"] == {
        "grok-4.20-0309-reasoning": 1,
        "claude-opus-4-8": 1,
        "claude-opus-4-6": 1,
    }
    assert coverage["markets_missing_components"][0]["missing_models"] == [
        "grok-4.20-0309-reasoning",
        "claude-opus-4-8",
        "claude-opus-4-6",
    ]


async def test_news_monitor_promotes_skipped_match() -> None:
    settings = Settings(
        sportspredict_api_key="sportspredict_test_key",
        openai_api_key="openai_test_key",
        stale_reforecast_without_news=False,
        news_monitor_materiality_threshold_points=2,
    )
    runner = ForecastRunner(settings)
    closing_time = (utcnow() + timedelta(hours=12)).astimezone(timezone.utc).isoformat()
    updated_date = (utcnow() - timedelta(hours=1)).astimezone(timezone.utc).isoformat()
    match = Match(id="match", name="A vs B", closing_time=closing_time, open_market_count=1)
    markets = [
        Market(
            id="market",
            question="Will player X score a goal?",
            status="open",
            match=MarketMatch(id="match", name="A vs B", closing_time=closing_time),
            lobby_id="lobby",
        )
    ]
    existing = [
        Prediction(
            id="prediction",
            market_id="market",
            lobby_id="lobby",
            probability=51,
            market_status="open",
            updated_date=updated_date,
        )
    ]

    selected = runner._select_matches([match], markets, existing)
    selected, news_cache, checks = await runner._augment_selected_with_news_monitor(
        selected=selected,
        matches=[match],
        markets=markets,
        existing_predictions=existing,
        history={},
        news_cache={"matches": {}},
        news_monitor=FakeNewsMonitor(),
        firecrawl=None,
    )

    assert len(selected) == 1
    assert selected[0][0].id == "match"
    assert news_cache["matches"]["match"]["estimated_delta_points"] == 4
    assert checks[0]["should_reforecast"] is True


async def test_write_predictions_continues_after_update_failure() -> None:
    settings = Settings(
        sportspredict_api_key="sportspredict_test_key",
        openai_api_key="openai_test_key",
        submit=True,
        sportspredict_update_interval_seconds=0,
    )
    runner = ForecastRunner(settings)

    class FakeSportsPredict:
        async def submit_batch(self, predictions):
            return {"total": len(predictions), "succeeded": len(predictions), "failed": 0}

        async def update_prediction(self, prediction_id, probability):
            if prediction_id == "first":
                raise RuntimeError("rate limited")
            return Prediction(id=prediction_id, market_id="second_market", lobby_id="lobby", probability=probability)

    result = await runner._write_predictions(
        FakeSportsPredict(),
        {
            "creates": [],
            "updates": [
                {"prediction_id": "first", "market_id": "first_market", "probability": 42},
                {"prediction_id": "second", "market_id": "second_market", "probability": 57},
            ],
            "skips": [],
        },
    )

    assert result["mode"] == "submitted_with_errors"
    assert len(result["updates"]) == 1
    assert result["updates"][0]["id"] == "second"
    assert len(result["failed_updates"]) == 1
    assert result["failed_updates"][0]["market_id"] == "first_market"
