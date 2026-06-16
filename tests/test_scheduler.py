from __future__ import annotations

from datetime import datetime, timezone

from probability_cup_bot.models import Match, Market
from probability_cup_bot.scheduler import build_due_actions


def test_opening_time_is_prediction_lock_before_closing_time() -> None:
    match = Match(
        id="match",
        name="FRA vs SEN",
        opening_time="2026-06-16T19:00:00Z",
        closing_time="2026-06-16T21:30:00Z",
    )
    market = Market(
        id="market",
        question="Will France win?",
        status="open",
        lobby_id="lobby",
        match={
            "id": "match",
            "name": "FRA vs SEN",
            "opening_time": "2026-06-16T19:00:00Z",
            "closing_time": "2026-06-16T21:30:00Z",
        },
    )

    assert match.closes_at == datetime(2026, 6, 16, 19, 0, tzinfo=timezone.utc)
    assert market.closes_at == datetime(2026, 6, 16, 19, 0, tzinfo=timezone.utc)


def test_due_actions_forecasts_at_thirty_minutes_before_close() -> None:
    now = datetime(2026, 6, 16, 12, 30, tzinfo=timezone.utc)
    schedule = {
        "matches": {
            "due": {
                "closing_time": "2026-06-16T13:00:00Z",
            },
            "not_due": {
                "closing_time": "2026-06-16T13:01:00Z",
            },
        }
    }

    due = build_due_actions(schedule, now=now)

    assert due.forecast_match_ids == ["due"]
    assert due.news_match_ids == []


def test_due_actions_use_opening_time_before_closing_time() -> None:
    now = datetime(2026, 6, 16, 18, 30, tzinfo=timezone.utc)
    schedule = {
        "matches": {
            "due": {
                "opening_time": "2026-06-16T19:00:00Z",
                "closing_time": "2026-06-16T21:30:00Z",
            },
        }
    }

    due = build_due_actions(schedule, now=now)

    assert due.forecast_match_ids == ["due"]
    assert due.news_match_ids == []


def test_due_actions_news_checks_at_ten_minutes_after_late_forecast() -> None:
    now = datetime(2026, 6, 16, 12, 50, tzinfo=timezone.utc)
    schedule = {
        "matches": {
            "match": {
                "closing_time": "2026-06-16T13:00:00Z",
                "late_forecast_completed_at": "2026-06-16T12:31:00Z",
            }
        }
    }

    due = build_due_actions(schedule, now=now)

    assert due.forecast_match_ids == []
    assert due.news_match_ids == ["match"]


def test_due_actions_runs_forecast_instead_of_news_if_late_forecast_missing() -> None:
    now = datetime(2026, 6, 16, 12, 52, tzinfo=timezone.utc)
    schedule = {
        "matches": {
            "match": {
                "closing_time": "2026-06-16T13:00:00Z",
            }
        }
    }

    due = build_due_actions(schedule, now=now)

    assert due.forecast_match_ids == ["match"]
    assert due.news_match_ids == []
