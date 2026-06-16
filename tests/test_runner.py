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


def test_plan_writes_creates_updates_and_skips() -> None:
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
