from __future__ import annotations

from probability_cup_bot.dashboard import _json_script_payload, build_dashboard_data
from probability_cup_bot.models import Market, MarketMatch, Match, Prediction
from probability_cup_bot.state import write_json


def test_dashboard_data_joins_platform_predictions_to_bot_forecasts(tmp_path) -> None:
    write_json(
        tmp_path / "latest-run.json",
        {
            "forecasts": [
                {
                    "market_id": "market",
                    "probability_int": 63,
                    "confidence": "high",
                    "evidence_quality": "medium",
                    "component_probabilities": [0.6, 0.65],
                }
            ]
        },
    )
    match = Match(
        id="match",
        name="A vs B",
        closing_time="2026-06-20T12:00:00Z",
        open_market_count=1,
    )
    market = Market(
        id="market",
        question="Will A win?",
        status="open",
        match=MarketMatch(id="match", name="A vs B", closing_time="2026-06-20T12:00:00Z"),
        lobby_id="lobby",
    )
    prediction = Prediction(
        id="prediction",
        market_id="market",
        lobby_id="lobby",
        probability=61,
        market_status="open",
    )

    data = build_dashboard_data(
        matches=[match],
        markets=[market],
        predictions=[prediction],
        state_dir=tmp_path,
    )

    assert data["summary"]["prediction_count"] == 1
    assert data["summary"]["bot_probability_mismatch_count"] == 1
    assert data["rows"][0]["question"] == "Will A win?"
    assert data["rows"][0]["probability"] == 61
    assert data["rows"][0]["latest_bot_probability"] == 63


def test_json_script_payload_is_parseable_json_without_html_entities() -> None:
    payload = _json_script_payload({"question": 'A "quoted" <script> & value'})

    assert "&quot;" not in payload
    assert "\\u003cscript\\u003e" in payload
    assert payload.startswith("{")
