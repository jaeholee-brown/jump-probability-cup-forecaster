from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from probability_cup_bot.models import Event, Lobby, Market, Match, Prediction


class SportsPredictError(RuntimeError):
    pass


class SportsPredictClient:
    def __init__(self, *, base_url: str, api_key: str, timeout: float = 30.0) -> None:
        if not api_key:
            raise SportsPredictError("SPORTSPREDICT_API_KEY is required")
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "probability-cup-forecaster/0.1",
            },
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError)),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = await self.client.request(method, path, **kwargs)
        if response.status_code == 429:
            await asyncio.sleep(10)
            response = await self.client.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise SportsPredictError(f"{method} {path} failed: {response.status_code} {response.text}")
        if not response.content:
            return None
        return response.json()

    async def list_events(self, limit: int = 100) -> list[Event]:
        data = await self._request("GET", "/events", params={"limit": limit})
        return [Event.model_validate(item) for item in data]

    async def find_event(self, title: str, event_id: str = "") -> Event:
        events = await self.list_events()
        if event_id:
            for event in events:
                if event.id == event_id:
                    return event
            return Event(
                id=event_id,
                title=title or event_id,
                type="probability",
                status=None,
            )

        title_lower = title.lower()
        probability_events = [event for event in events if str(event.type).lower() == "probability"]
        for event in events:
            event_title = event.title.lower()
            if title_lower in event_title or event_title in title_lower:
                return event
        if probability_events:
            return probability_events[0]
        if len(events) == 1:
            return events[0]
        available = ", ".join(
            f"{event.id} title={event.title!r} type={event.type!r} status={event.status!r}"
            for event in events[:10]
        )
        suffix = f" Available events: {available}" if available else " /events returned no events."
        raise SportsPredictError(
            f"No event found matching {title!r}. Set EVENT_ID to the desired event id if needed.{suffix}"
        )

    async def list_lobbies(self, event_id: str) -> list[Lobby]:
        data = await self._request("GET", "/lobbies", params={"event_id": event_id})
        return [Lobby.model_validate(item) for item in data]

    async def ensure_lobby(self, event_id: str) -> Lobby:
        lobbies = await self.list_lobbies(event_id)
        if not lobbies:
            raise SportsPredictError(f"No lobby found for event {event_id}")
        lobby = lobbies[0]
        if not lobby.joined:
            response = await self.client.post(f"/lobbies/{lobby.id}/join")
            if response.status_code not in {200, 201, 204, 409}:
                raise SportsPredictError(
                    f"POST /lobbies/{lobby.id}/join failed: {response.status_code} {response.text}"
                )
            lobby.joined = True
        return lobby

    async def list_matches(self, event_id: str, lobby_id: str | None = None) -> list[Match]:
        params: dict[str, str] = {"event_id": event_id}
        if lobby_id:
            params["lobby_id"] = lobby_id
        data = await self._request("GET", "/matches", params=params)
        return [Match.model_validate(item) for item in data]

    async def list_markets(self, lobby_id: str, match_id: str | None = None) -> list[Market]:
        params: dict[str, str] = {"lobby_id": lobby_id}
        if match_id:
            params["match_id"] = match_id
        data = await self._request("GET", "/markets", params=params)
        return [Market.model_validate(item) for item in data]

    async def list_predictions(self, lobby_id: str | None = None) -> list[Prediction]:
        params = {"lobby_id": lobby_id} if lobby_id else None
        data = await self._request("GET", "/predictions", params=params)
        return [Prediction.model_validate(item) for item in data]

    async def list_results(self, lobby_id: str | None = None) -> list[dict[str, Any]]:
        params = {"lobby_id": lobby_id} if lobby_id else None
        data = await self._request("GET", "/results", params=params)
        return list(data)

    async def submit_batch(self, predictions: list[dict[str, Any]]) -> dict[str, Any]:
        if not predictions:
            return {"total": 0, "succeeded": 0, "failed": 0, "results": []}
        if len(predictions) > 50:
            raise ValueError("SportsPredict batch limit is 50")
        return await self._request("POST", "/predictions/batch", json={"predictions": predictions})

    async def update_prediction(self, prediction_id: str, probability: int) -> Prediction:
        data = await self._request("PATCH", f"/predictions/{prediction_id}", json={"probability": probability})
        return Prediction.model_validate(data)


def chunks(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
