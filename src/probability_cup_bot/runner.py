from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import asdict
from datetime import timezone
from typing import Any

from probability_cup_bot.config import Settings
from probability_cup_bot.evidence import EvidenceCollector
from probability_cup_bot.forecaster import MatchForecaster
from probability_cup_bot.models import AggregatedForecast, Market, Match, Prediction, utcnow
from probability_cup_bot.openai_adapter import OpenAIAdapter
from probability_cup_bot.sportspredict import SportsPredictClient, chunks
from probability_cup_bot.state import ensure_dirs, timestamp_slug, write_json


class ForecastRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def run(self) -> dict[str, Any]:
        ensure_dirs(self.settings.state_dir, self.settings.logs_dir)
        sp = SportsPredictClient(
            base_url=self.settings.sportspredict_base_url,
            api_key=self.settings.sportspredict_api_key,
        )
        try:
            grok = (
                OpenAIAdapter(
                    self.settings.xai_api_key,
                    base_url=self.settings.xai_base_url,
                    provider="xai",
                )
                if self.settings.xai_api_key
                else None
            )
            openai = (
                OpenAIAdapter(self.settings.openai_api_key, provider="openai")
                if self.settings.openai_api_key
                else grok
            )
            if openai is None:
                raise RuntimeError("Set OPENAI_API_KEY or XAI_API_KEY before running forecasts.")
            evidence_collector = EvidenceCollector(self.settings, openai, grok)
            forecaster = MatchForecaster(self.settings, openai, grok)

            event = await sp.find_event(self.settings.event_title)
            lobby = await sp.ensure_lobby(event.id)
            matches = await sp.list_matches(event.id, lobby.id)
            all_markets = await sp.list_markets(lobby.id)
            existing_predictions = await sp.list_predictions(lobby.id)

            selected = self._select_matches(matches, all_markets)
            forecast_results = await self._forecast_selected(
                selected=selected,
                evidence_collector=evidence_collector,
                forecaster=forecaster,
            )
            plan = self._plan_writes(
                forecasts=forecast_results,
                existing_predictions=existing_predictions,
                lobby_id=lobby.id,
            )
            submission_results = await self._write_predictions(sp, plan)
            run_log = {
                "generated_at": utcnow().isoformat(),
                "settings": {
                    key: value
                    for key, value in asdict(self.settings).items()
                    if "key" not in key and key not in {"state_dir", "logs_dir"}
                },
                "event": event.model_dump(),
                "lobby": lobby.model_dump(),
                "matches_seen": len(matches),
                "open_markets_seen": len(all_markets),
                "matches_forecasted": len(selected),
                "forecast_count": len(forecast_results),
                "plan": plan,
                "submission_results": submission_results,
                "forecasts": [forecast.model_dump() for forecast in forecast_results],
            }
            write_json(self.settings.logs_dir / f"run-{timestamp_slug()}.json", run_log)
            write_json(self.settings.state_dir / "latest-run.json", run_log)
            return run_log
        finally:
            await sp.aclose()

    def _select_matches(
        self,
        matches: list[Match],
        markets: list[Market],
    ) -> list[tuple[Match, list[Market]]]:
        markets_by_match: dict[str, list[Market]] = defaultdict(list)
        for market in markets:
            if market.status != "open":
                continue
            markets_by_match[market.match.id].append(market)

        now = utcnow()
        selected: list[tuple[Match, list[Market]]] = []
        match_lookup = {match.id: match for match in matches}
        for match_id, match_markets in markets_by_match.items():
            match = match_lookup.get(match_id) or Match(
                id=match_id,
                name=match_markets[0].match.name,
                opening_time=match_markets[0].match.opening_time,
                closing_time=match_markets[0].match.closing_time,
                open_market_count=len(match_markets),
            )
            closes_at = match.closes_at or match_markets[0].closes_at
            if closes_at is not None:
                hours = (closes_at.astimezone(timezone.utc) - now).total_seconds() / 3600
                if hours < self.settings.min_hours_to_close:
                    continue
                if self.settings.max_hours_to_close and hours > self.settings.max_hours_to_close:
                    continue
            selected.append((match, match_markets))

        selected.sort(key=lambda item: item[0].closes_at or utcnow())
        if self.settings.max_matches_per_run > 0:
            selected = selected[: self.settings.max_matches_per_run]
        return selected

    async def _forecast_selected(
        self,
        *,
        selected: list[tuple[Match, list[Market]]],
        evidence_collector: EvidenceCollector,
        forecaster: MatchForecaster,
    ) -> list[AggregatedForecast]:
        semaphore = asyncio.Semaphore(self.settings.concurrency)
        outputs: list[AggregatedForecast] = []

        async def forecast_one(match: Match, markets: list[Market]) -> list[AggregatedForecast]:
            async with semaphore:
                evidence = await evidence_collector.collect(match, markets)
                return await forecaster.forecast_match(match=match, markets=markets, evidence=evidence)

        tasks = [forecast_one(match, markets) for match, markets in selected]
        for result in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(result, Exception):
                outputs.append(
                    AggregatedForecast(
                        market_id="error",
                        question=str(result),
                        probability=0.5,
                        probability_int=50,
                        component_probabilities=[],
                        confidence="low",
                        evidence_quality="low",
                        notes="Forecasting failed for one match.",
                    )
                )
            else:
                outputs.extend(result)
        return [forecast for forecast in outputs if forecast.market_id != "error"]

    def _plan_writes(
        self,
        *,
        forecasts: list[AggregatedForecast],
        existing_predictions: list[Prediction],
        lobby_id: str,
    ) -> dict[str, Any]:
        existing_by_market = {prediction.market_id: prediction for prediction in existing_predictions}
        creates: list[dict[str, Any]] = []
        updates: list[dict[str, Any]] = []
        skips: list[dict[str, Any]] = []
        for forecast in forecasts:
            existing = existing_by_market.get(forecast.market_id)
            if existing is None:
                creates.append(
                    {
                        "market_id": forecast.market_id,
                        "lobby_id": lobby_id,
                        "probability": forecast.probability_int,
                    }
                )
                continue
            old = existing.probability_int
            new = forecast.probability_int
            if existing.market_status and existing.market_status != "open":
                skips.append(
                    {
                        "market_id": forecast.market_id,
                        "reason": f"existing market status is {existing.market_status}",
                    }
                )
            elif abs(new - old) >= self.settings.update_threshold_points:
                updates.append(
                    {
                        "prediction_id": existing.id,
                        "market_id": forecast.market_id,
                        "old_probability": old,
                        "probability": new,
                    }
                )
            else:
                skips.append(
                    {
                        "market_id": forecast.market_id,
                        "reason": f"change {old}->{new} below threshold",
                    }
                )
        return {"creates": creates, "updates": updates, "skips": skips}

    async def _write_predictions(
        self,
        sp: SportsPredictClient,
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.settings.can_submit:
            return {
                "mode": "dry_run",
                "message": "No writes performed. Set SUBMIT=true to submit or update predictions.",
            }

        create_results: list[dict[str, Any]] = []
        for chunk in chunks(plan["creates"], 50):
            create_results.append(await sp.submit_batch(chunk))

        update_results: list[dict[str, Any]] = []
        for item in plan["updates"]:
            updated = await sp.update_prediction(item["prediction_id"], item["probability"])
            update_results.append(updated.model_dump())

        return {"mode": "submitted", "creates": create_results, "updates": update_results}
