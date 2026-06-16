from __future__ import annotations

import asyncio

import httpx
import pytest

from probability_cup_bot.models import Event
from probability_cup_bot.sportspredict import SportsPredictClient, SportsPredictError


def _client_with_events(events: list[Event]) -> SportsPredictClient:
    client = SportsPredictClient.__new__(SportsPredictClient)

    async def list_events(limit: int = 100) -> list[Event]:
        return events

    client.list_events = list_events  # type: ignore[method-assign]
    return client


async def test_find_event_prefers_explicit_event_id() -> None:
    client = _client_with_events([])

    event = await client.find_event("Probability Cup", event_id="event-123")

    assert event.id == "event-123"
    assert event.title == "Probability Cup"


async def test_find_event_matches_title_without_strict_type() -> None:
    client = _client_with_events(
        [
            Event(id="other", title="Other", type="sports"),
            Event(id="jump", title="Jump Trading Probability Cup", type="tournament"),
        ]
    )

    event = await client.find_event("Probability Cup")

    assert event.id == "jump"


async def test_find_event_falls_back_to_single_event() -> None:
    client = _client_with_events([Event(id="only", title="World Cup Forecasting", type="tournament")])

    event = await client.find_event("Probability Cup")

    assert event.id == "only"


async def test_find_event_error_lists_available_events() -> None:
    client = _client_with_events(
        [
            Event(id="one", title="One", type="sports"),
            Event(id="two", title="Two", type="sports"),
        ]
    )

    with pytest.raises(SportsPredictError, match="Available events:"):
        await client.find_event("Probability Cup")


async def test_request_retries_429_with_retry_after(monkeypatch) -> None:
    calls = 0
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, json={"message": "Too Many Requests"}, headers={"Retry-After": "0"})
        return httpx.Response(200, json=[{"id": "event", "title": "Cup", "type": "probability"}])

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    client = SportsPredictClient(
        base_url="https://example.test/api/v1",
        api_key="sportspredict_test_key",
        retry_attempts=3,
        retry_initial_seconds=0,
        retry_max_seconds=1,
    )
    await client.client.aclose()
    client.client = httpx.AsyncClient(base_url=client.base_url, transport=httpx.MockTransport(handler))
    try:
        data = await client._request("GET", "/events")
    finally:
        await client.aclose()

    assert calls == 2
    assert sleeps == [0.0]
    assert data[0]["id"] == "event"
