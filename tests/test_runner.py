from datetime import timedelta, timezone

from probability_cup_bot.config import Settings
from probability_cup_bot.models import AggregatedForecast, Market, MarketMatch, Match, Prediction, utcnow
from probability_cup_bot.runner import ForecastRunner


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
