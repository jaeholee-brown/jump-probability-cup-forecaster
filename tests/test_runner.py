from probability_cup_bot.config import Settings
from probability_cup_bot.models import AggregatedForecast, Prediction
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
