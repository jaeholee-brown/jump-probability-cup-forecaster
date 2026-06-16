from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from probability_cup_bot.anthropic_adapter import AnthropicAdapter
from probability_cup_bot.config import Settings
from probability_cup_bot.evidence import EvidenceCollector
from probability_cup_bot.forecaster import MatchForecaster
from probability_cup_bot.models import AggregatedForecast, Market, Match, Prediction, parse_dt, utcnow
from probability_cup_bot.openai_adapter import OpenAIAdapter
from probability_cup_bot.sportspredict import SportsPredictClient, chunks
from probability_cup_bot.state import ensure_dirs, read_json, timestamp_slug, write_json


VOLATILE_MARKET_TERMS = (
    "assist",
    "booking",
    "card",
    "corner",
    "goal",
    "lineup",
    "penalty",
    "player",
    "red card",
    "shot",
    "start",
    "yellow",
)


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
                else None
            )
            anthropic = (
                AnthropicAdapter(self.settings.anthropic_api_key)
                if self.settings.anthropic_api_key
                else None
            )
            if openai is None and grok is None and anthropic is None:
                raise RuntimeError(
                    "Set OPENAI_API_KEY, XAI_API_KEY, or ANTHROPIC_API_KEY before running forecasts."
                )
            evidence_collector = EvidenceCollector(self.settings, openai, grok)
            forecaster = MatchForecaster(self.settings, openai, grok, anthropic)

            event = await sp.find_event(self.settings.event_title)
            lobby = await sp.ensure_lobby(event.id)
            matches = await sp.list_matches(event.id, lobby.id)
            all_markets = await sp.list_markets(lobby.id)
            existing_predictions = await sp.list_predictions(lobby.id)
            history = read_json(self.settings.state_dir / "forecast-history.json", {})

            selected = self._select_matches(matches, all_markets, existing_predictions, history)
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
            history = self._update_history(history, selected, forecast_results)
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
                "update_gate": {
                    "enabled": self.settings.enable_update_gate,
                    "max_prediction_age_hours": self.settings.max_prediction_age_hours,
                    "force_reforecast_within_hours": self.settings.force_reforecast_within_hours,
                },
                "forecast_count": len(forecast_results),
                "plan": plan,
                "submission_results": submission_results,
                "forecasts": [forecast.model_dump() for forecast in forecast_results],
            }
            write_json(self.settings.state_dir / "forecast-history.json", history)
            write_json(self.settings.logs_dir / f"run-{timestamp_slug()}.json", run_log)
            write_json(self.settings.state_dir / "latest-run.json", run_log)
            return run_log
        finally:
            await sp.aclose()

    def _select_matches(
        self,
        matches: list[Match],
        markets: list[Market],
        existing_predictions: list[Prediction] | None = None,
        history: dict[str, Any] | None = None,
    ) -> list[tuple[Match, list[Market]]]:
        markets_by_match: dict[str, list[Market]] = defaultdict(list)
        for market in markets:
            if market.status != "open":
                continue
            markets_by_match[market.match.id].append(market)

        now = utcnow()
        selected: list[tuple[Match, list[Market]]] = []
        match_lookup = {match.id: match for match in matches}
        existing_by_market = {
            prediction.market_id: prediction
            for prediction in existing_predictions or []
            if not prediction.market_status or prediction.market_status == "open"
        }
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
            if not self._should_forecast_match(match, match_markets, existing_by_market, now, history or {}):
                continue
            selected.append((match, match_markets))

        selected.sort(key=lambda item: item[0].closes_at or utcnow())
        if self.settings.max_matches_per_run > 0:
            selected = selected[: self.settings.max_matches_per_run]
        return selected

    def _should_forecast_match(
        self,
        match: Match,
        markets: list[Market],
        existing_by_market: dict[str, Prediction],
        now: datetime,
        history: dict[str, Any],
    ) -> bool:
        if not self.settings.enable_update_gate:
            return True
        if any(market.id not in existing_by_market for market in markets):
            return True

        closes_at = match.closes_at or markets[0].closes_at
        if closes_at is not None and self.settings.force_reforecast_within_hours >= 0:
            hours_to_close = (closes_at.astimezone(timezone.utc) - now).total_seconds() / 3600
            if hours_to_close <= self.settings.force_reforecast_within_hours:
                return True

        match_history = (history.get("matches") or {}).get(match.id, {})
        history_updated_at = parse_dt(match_history.get("last_forecast_at"))
        if history_updated_at is not None:
            updated_times = [history_updated_at]
        else:
            updated_times = [
                parse_dt(existing_by_market[market.id].updated_date or existing_by_market[market.id].created_date)
                for market in markets
                if market.id in existing_by_market
            ]
        if not updated_times or any(updated_at is None for updated_at in updated_times):
            return True

        oldest_update = min(updated_at for updated_at in updated_times if updated_at is not None)
        age_hours = (now - oldest_update.astimezone(timezone.utc)).total_seconds() / 3600
        cadence_hours = self._cadence_hours(match, markets, history, now)
        return age_hours >= cadence_hours

    def _cadence_hours(
        self,
        match: Match,
        markets: list[Market],
        history: dict[str, Any],
        now: datetime,
    ) -> float:
        closes_at = match.closes_at or markets[0].closes_at
        hours_to_close = 9999.0
        if closes_at is not None:
            hours_to_close = (closes_at.astimezone(timezone.utc) - now).total_seconds() / 3600
        if hours_to_close <= self.settings.force_reforecast_within_hours:
            return 0.0
        if hours_to_close <= 24:
            cadence = 3.0
        elif hours_to_close <= 72:
            cadence = 8.0
        else:
            cadence = self.settings.max_prediction_age_hours

        match_history = (history.get("matches") or {}).get(match.id, {})
        spread = float(match_history.get("max_component_spread_points") or 0)
        low_quality = str(match_history.get("worst_evidence_quality") or "") == "low"
        low_confidence = str(match_history.get("worst_confidence") or "") == "low"
        if spread >= 20 or low_quality or low_confidence:
            cadence *= 0.5
        if any(self._is_volatile_market(market.question) for market in markets):
            cadence *= 0.75

        return max(1.0, min(cadence, self.settings.max_prediction_age_hours))

    @staticmethod
    def _is_volatile_market(question: str) -> bool:
        lowered = question.lower()
        return any(term in lowered for term in VOLATILE_MARKET_TERMS)

    def _update_history(
        self,
        history: dict[str, Any],
        selected: list[tuple[Match, list[Market]]],
        forecasts: list[AggregatedForecast],
    ) -> dict[str, Any]:
        now = utcnow().isoformat()
        updated = dict(history)
        updated.setdefault("matches", {})
        updated.setdefault("markets", {})
        forecasts_by_market = {forecast.market_id: forecast for forecast in forecasts}
        for match, markets in selected:
            match_spreads: list[float] = []
            confidences: list[str] = []
            qualities: list[str] = []
            for market in markets:
                forecast = forecasts_by_market.get(market.id)
                if forecast is None:
                    continue
                spread = self._component_spread_points(forecast.component_probabilities)
                match_spreads.append(spread)
                confidences.append(forecast.confidence)
                qualities.append(forecast.evidence_quality)
                updated["markets"][market.id] = {
                    "last_forecast_at": now,
                    "match_id": match.id,
                    "question": market.question,
                    "probability_int": forecast.probability_int,
                    "component_spread_points": spread,
                    "confidence": forecast.confidence,
                    "evidence_quality": forecast.evidence_quality,
                    "component_count": len(forecast.component_probabilities),
                }
            if match_spreads:
                previous = updated["matches"].get(match.id, {})
                updated["matches"][match.id] = {
                    "last_forecast_at": now,
                    "match_name": match.name,
                    "closing_time": match.closing_time,
                    "market_count": len(markets),
                    "max_component_spread_points": max(match_spreads),
                    "worst_confidence": self._worst_level(confidences),
                    "worst_evidence_quality": self._worst_level(qualities),
                    "forecast_count": int(previous.get("forecast_count") or 0) + 1,
                    "market_ids": [market.id for market in markets],
                }
        return updated

    @staticmethod
    def _component_spread_points(probabilities: list[float]) -> float:
        if len(probabilities) < 2:
            return 0.0
        return round((max(probabilities) - min(probabilities)) * 100, 2)

    @staticmethod
    def _worst_level(values: list[str]) -> str:
        order = {"low": 0, "medium": 1, "high": 2}
        if not values:
            return "medium"
        return min(values, key=lambda value: order.get(value, 1))

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
