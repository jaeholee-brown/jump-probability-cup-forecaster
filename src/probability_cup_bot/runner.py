from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from probability_cup_bot.anthropic_adapter import AnthropicAdapter
from probability_cup_bot.calibration import build_calibration_report
from probability_cup_bot.config import Settings
from probability_cup_bot.evidence import EvidenceCollector
from probability_cup_bot.firecrawl import FirecrawlClient
from probability_cup_bot.forecaster import MatchForecaster
from probability_cup_bot.models import AggregatedForecast, Market, Match, NewsCheck, Prediction, parse_dt, utcnow
from probability_cup_bot.news_monitor import GrokNewsMonitor
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
        firecrawl: FirecrawlClient | None = None
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
            if self.settings.firecrawl_api_key and self.settings.use_firecrawl_retrieval:
                firecrawl = FirecrawlClient(self.settings.firecrawl_api_key)
            previous_calibration = read_json(self.settings.state_dir / "calibration-report.json", {})
            if self.settings.apply_calibration_weights:
                calibration_multipliers = previous_calibration.get("suggested_multipliers") or {}
            else:
                calibration_multipliers = {}
            evidence_collector = EvidenceCollector(self.settings, openai, grok, firecrawl)
            forecaster = MatchForecaster(
                self.settings,
                openai,
                grok,
                anthropic,
                calibration_multipliers=calibration_multipliers,
            )

            event = await sp.find_event(self.settings.event_title, self.settings.event_id)
            lobby = await sp.ensure_lobby(event.id)
            matches = await sp.list_matches(event.id, lobby.id)
            all_markets = await sp.list_markets(lobby.id)
            existing_predictions = await sp.list_predictions(lobby.id)
            history = read_json(self.settings.state_dir / "forecast-history.json", {})
            news_cache = read_json(self.settings.state_dir / "news-cache.json", {"matches": {}})

            selected = self._select_matches(matches, all_markets, existing_predictions, history)
            news_monitor = (
                GrokNewsMonitor(self.settings, grok)
                if self.settings.use_grok_news_monitor and grok is not None
                else None
            )
            selected, news_cache, news_checks = await self._augment_selected_with_news_monitor(
                selected=selected,
                matches=matches,
                markets=all_markets,
                existing_predictions=existing_predictions,
                history=history,
                news_cache=news_cache,
                news_monitor=news_monitor,
                firecrawl=firecrawl,
            )
            forecast_results = await self._forecast_selected(
                selected=selected,
                evidence_collector=evidence_collector,
                forecaster=forecaster,
                history=history,
                news_cache=news_cache,
            )
            plan = self._plan_writes(
                forecasts=forecast_results,
                existing_predictions=existing_predictions,
                lobby_id=lobby.id,
            )
            history = self._update_history(history, selected, forecast_results, plan)
            submission_results = await self._write_predictions(sp, plan)
            calibration_report = await self._build_calibration_report(sp, lobby.id, history, calibration_multipliers)
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
                "news_monitor": {
                    "enabled": bool(news_monitor),
                    "checks": news_checks,
                },
                "plan": plan,
                "submission_results": submission_results,
                "calibration": {
                    "settled_market_count": calibration_report.get("settled_market_count", 0),
                    "aggregate": calibration_report.get("aggregate", {}),
                    "suggested_multipliers": calibration_report.get("suggested_multipliers", {}),
                },
                "forecasts": [forecast.model_dump() for forecast in forecast_results],
            }
            write_json(self.settings.state_dir / "forecast-history.json", history)
            write_json(self.settings.state_dir / "news-cache.json", news_cache)
            write_json(self.settings.state_dir / "calibration-report.json", calibration_report)
            write_json(self.settings.logs_dir / f"calibration-{timestamp_slug()}.json", calibration_report)
            write_json(self.settings.logs_dir / f"run-{timestamp_slug()}.json", run_log)
            write_json(self.settings.state_dir / "latest-run.json", run_log)
            return run_log
        finally:
            if firecrawl is not None:
                await firecrawl.aclose()
            await sp.aclose()

    def _select_matches(
        self,
        matches: list[Match],
        markets: list[Market],
        existing_predictions: list[Prediction] | None = None,
        history: dict[str, Any] | None = None,
    ) -> list[tuple[Match, list[Market]]]:
        now = utcnow()
        selected: list[tuple[Match, list[Market]]] = []
        existing_by_market = {
            prediction.market_id: prediction
            for prediction in existing_predictions or []
            if not prediction.market_status or prediction.market_status == "open"
        }
        for match, match_markets in self._eligible_match_groups(matches, markets, now):
            if not self._should_forecast_match(match, match_markets, existing_by_market, now, history or {}):
                continue
            selected.append((match, match_markets))

        selected.sort(key=lambda item: item[0].closes_at or utcnow())
        if self.settings.max_matches_per_run > 0:
            selected = selected[: self.settings.max_matches_per_run]
        return selected

    def _eligible_match_groups(
        self,
        matches: list[Match],
        markets: list[Market],
        now: datetime,
    ) -> list[tuple[Match, list[Market]]]:
        markets_by_match: dict[str, list[Market]] = defaultdict(list)
        for market in markets:
            if market.status != "open":
                continue
            markets_by_match[market.match.id].append(market)

        groups: list[tuple[Match, list[Market]]] = []
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
            groups.append((match, match_markets))
        groups.sort(key=lambda item: item[0].closes_at or utcnow())
        return groups

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
        hours_to_close = 9999.0
        if closes_at is not None and self.settings.force_reforecast_within_hours >= 0:
            hours_to_close = (closes_at.astimezone(timezone.utc) - now).total_seconds() / 3600

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
        if hours_to_close <= self.settings.force_reforecast_within_hours:
            min_interval_hours = self.settings.final_reforecast_min_interval_minutes / 60
            return age_hours >= min_interval_hours

        if not self.settings.stale_reforecast_without_news:
            return False

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

    async def _augment_selected_with_news_monitor(
        self,
        *,
        selected: list[tuple[Match, list[Market]]],
        matches: list[Match],
        markets: list[Market],
        existing_predictions: list[Prediction],
        history: dict[str, Any],
        news_cache: dict[str, Any],
        news_monitor: GrokNewsMonitor | None,
        firecrawl: FirecrawlClient | None,
    ) -> tuple[list[tuple[Match, list[Market]]], dict[str, Any], list[dict[str, Any]]]:
        if news_monitor is None:
            return selected, news_cache, []
        now = utcnow()
        selected_ids = {match.id for match, _ in selected}
        existing_by_market = {
            prediction.market_id: prediction
            for prediction in existing_predictions
            if not prediction.market_status or prediction.market_status == "open"
        }
        candidates = [
            (match, match_markets)
            for match, match_markets in self._eligible_match_groups(matches, markets, now)
            if match.id not in selected_ids
            and self._should_news_monitor_match(match, match_markets, existing_by_market, history, news_cache, now)
        ]
        if self.settings.max_matches_per_run > 0:
            remaining = max(0, self.settings.max_matches_per_run - len(selected))
            candidates = candidates[:remaining]

        news_cache.setdefault("matches", {})
        checks: list[dict[str, Any]] = []
        promoted: list[tuple[Match, list[Market]]] = []
        semaphore = asyncio.Semaphore(self.settings.concurrency)

        async def check_one(match: Match, match_markets: list[Market]) -> tuple[Match, list[Market], NewsCheck | Exception, str]:
            async with semaphore:
                firecrawl_context = ""
                if self._should_use_firecrawl(match, match_markets, history, news_cache, now, for_monitor=True):
                    firecrawl_context = await self._firecrawl_context_for_monitor(firecrawl, match, match_markets)
                try:
                    news_check = await news_monitor.check_match(
                        match=match,
                        markets=match_markets,
                        match_history=(history.get("matches") or {}).get(match.id, {}),
                        cached_news=(news_cache.get("matches") or {}).get(match.id, {}),
                        firecrawl_context=firecrawl_context,
                    )
                    return match, match_markets, news_check, firecrawl_context
                except Exception as exc:
                    return match, match_markets, exc, firecrawl_context

        for match, match_markets, result, firecrawl_context in await asyncio.gather(
            *(check_one(match, match_markets) for match, match_markets in candidates)
        ):
            if isinstance(result, Exception):
                checks.append({"match_id": match.id, "match_name": match.name, "error": str(result)})
                continue
            cache_entry = self._news_cache_entry(match, result, firecrawl_context)
            news_cache["matches"][match.id] = cache_entry
            row = result.model_dump()
            row["used_firecrawl"] = bool(firecrawl_context)
            checks.append(row)
            if (
                result.should_reforecast
                and result.estimated_delta_points >= self.settings.news_monitor_materiality_threshold_points
            ):
                promoted.append((match, match_markets))

        output = selected + promoted
        output.sort(key=lambda item: item[0].closes_at or utcnow())
        return output, news_cache, checks

    def _should_news_monitor_match(
        self,
        match: Match,
        markets: list[Market],
        existing_by_market: dict[str, Prediction],
        history: dict[str, Any],
        news_cache: dict[str, Any],
        now: datetime,
    ) -> bool:
        if not self.settings.use_grok_news_monitor:
            return False
        if any(market.id not in existing_by_market for market in markets):
            return False
        closes_at = match.closes_at or markets[0].closes_at
        if closes_at is None:
            return True
        hours_to_close = (closes_at.astimezone(timezone.utc) - now).total_seconds() / 3600
        if hours_to_close < self.settings.min_hours_to_close:
            return False
        if hours_to_close > self.settings.news_monitor_max_hours_to_close:
            return False
        cache_entry = (news_cache.get("matches") or {}).get(match.id, {})
        last_checked_at = parse_dt(cache_entry.get("last_checked_at"))
        if last_checked_at is None:
            return True
        age_hours = (now - last_checked_at.astimezone(timezone.utc)).total_seconds() / 3600
        interval_hours = self._news_monitor_interval_hours(match, markets, history, hours_to_close)
        return age_hours >= interval_hours

    def _news_monitor_interval_hours(
        self,
        match: Match,
        markets: list[Market],
        history: dict[str, Any],
        hours_to_close: float,
    ) -> float:
        if hours_to_close <= 2:
            interval = 0.25
        elif hours_to_close <= 6:
            interval = 0.5
        elif hours_to_close <= 24:
            interval = 1.0
        elif hours_to_close <= 72:
            interval = 3.0
        else:
            interval = 6.0
        match_history = (history.get("matches") or {}).get(match.id, {})
        spread = float(match_history.get("max_component_spread_points") or 0)
        low_quality = str(match_history.get("worst_evidence_quality") or "") == "low"
        volatile = any(self._is_volatile_market(market.question) for market in markets)
        if spread >= self.settings.firecrawl_disagreement_threshold_points or low_quality or volatile:
            interval *= 0.5
        return max(0.25, interval)

    def _should_use_firecrawl(
        self,
        match: Match,
        markets: list[Market],
        history: dict[str, Any],
        news_cache: dict[str, Any],
        now: datetime,
        *,
        for_monitor: bool = False,
    ) -> bool:
        if not self.settings.use_firecrawl_retrieval:
            return False
        if self.settings.firecrawl_mode == "always":
            return True
        if self.settings.firecrawl_mode == "off":
            return False
        closes_at = match.closes_at or markets[0].closes_at
        hours_to_close = 9999.0
        if closes_at is not None:
            hours_to_close = (closes_at.astimezone(timezone.utc) - now).total_seconds() / 3600
        match_history = (history.get("matches") or {}).get(match.id, {})
        cache_entry = (news_cache.get("matches") or {}).get(match.id, {})
        spread = float(match_history.get("max_component_spread_points") or 0)
        low_quality = str(match_history.get("worst_evidence_quality") or "") == "low"
        volatile = any(self._is_volatile_market(market.question) for market in markets)
        material_cached_news = int(cache_entry.get("estimated_delta_points") or 0) >= (
            self.settings.news_monitor_materiality_threshold_points
        )
        if hours_to_close <= self.settings.firecrawl_force_within_hours:
            return True
        if volatile and hours_to_close <= self.settings.firecrawl_volatile_within_hours:
            return True
        if spread >= self.settings.firecrawl_disagreement_threshold_points or low_quality or material_cached_news:
            return True
        return for_monitor and hours_to_close <= 6

    async def _firecrawl_context_for_monitor(
        self,
        firecrawl: FirecrawlClient | None,
        match: Match,
        markets: list[Market],
    ) -> str:
        if firecrawl is None:
            return ""
        contexts: list[str] = []
        total_credits = 0
        market_terms = " ".join(market.question for market in markets[:6])
        queries = [
            f"{match.name} confirmed lineup injury suspension team news",
            f"{match.name} late news X lineup weather odds {market_terms}",
        ][: self.settings.firecrawl_search_queries]
        for query in queries:
            try:
                results, credits = await firecrawl.search(
                    query,
                    limit=self.settings.firecrawl_search_limit,
                    sources=("web",),
                    tbs="qdr:d,sbd:1",
                )
            except Exception as exc:
                contexts.append(f"Firecrawl monitor query failed for {query!r}: {exc}")
                continue
            total_credits += credits
            rendered = "\n".join(result.compact() for result in results[: self.settings.firecrawl_search_limit])
            if rendered:
                contexts.append(f"Firecrawl monitor query: {query}\n{rendered}")
        if not contexts:
            return ""
        return f"Firecrawl monitor credits used: {total_credits}\n" + "\n\n".join(contexts)

    @staticmethod
    def _news_cache_entry(match: Match, news_check: NewsCheck, firecrawl_context: str) -> dict[str, Any]:
        return {
            "match_id": match.id,
            "match_name": match.name,
            "closing_time": match.closing_time,
            "last_checked_at": news_check.checked_at,
            "summary": news_check.summary,
            "new_developments": news_check.new_developments,
            "sources": [source.model_dump() for source in news_check.sources],
            "should_reforecast": news_check.should_reforecast,
            "estimated_delta_points": news_check.estimated_delta_points,
            "materiality": news_check.materiality,
            "evidence_quality": news_check.evidence_quality,
            "reason": news_check.reason,
            "firecrawl_context": firecrawl_context,
        }

    def _update_history(
        self,
        history: dict[str, Any],
        selected: list[tuple[Match, list[Market]]],
        forecasts: list[AggregatedForecast],
        plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = utcnow().isoformat()
        updated = dict(history)
        updated.setdefault("matches", {})
        updated.setdefault("markets", {})
        forecasts_by_market = {forecast.market_id: forecast for forecast in forecasts}
        written_market_ids = self._written_market_ids(plan or {}) if self.settings.can_submit else set()
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
                should_record_submission = market.id in written_market_ids
                if should_record_submission:
                    updated["markets"][market.id] = {
                        "last_forecast_at": now,
                        "match_id": match.id,
                        "question": market.question,
                        "probability_int": forecast.probability_int,
                        "probability": forecast.probability,
                        "component_spread_points": spread,
                        "confidence": forecast.confidence,
                        "evidence_quality": forecast.evidence_quality,
                        "component_count": len(forecast.component_probabilities),
                        "components": self._component_records(forecast),
                    }
                else:
                    updated["markets"].setdefault(
                        market.id,
                        {
                            "match_id": match.id,
                            "question": market.question,
                        },
                    )
                    updated["markets"][market.id]["last_model_forecast_at"] = now
                    updated["markets"][market.id]["last_skipped_probability_int"] = forecast.probability_int
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
    def _written_market_ids(plan: dict[str, Any]) -> set[str]:
        return {
            item["market_id"]
            for key in ("creates", "updates")
            for item in plan.get(key, [])
            if "market_id" in item
        }

    @staticmethod
    def _component_records(forecast: AggregatedForecast) -> list[dict[str, Any]]:
        variants = forecast.metadata.get("variants") or []
        models = forecast.metadata.get("models") or []
        providers = forecast.metadata.get("providers") or []
        weights = forecast.metadata.get("weights") or []
        resolution_interpretations = forecast.metadata.get("resolution_interpretations") or []
        reference_classes = forecast.metadata.get("reference_classes") or []
        base_rates = forecast.metadata.get("base_rates") or []
        base_rate_rationales = forecast.metadata.get("base_rate_rationales") or []
        yes_reasons = forecast.metadata.get("yes_reasons") or []
        no_reasons = forecast.metadata.get("no_reasons") or []
        probability_rationales = forecast.metadata.get("probability_rationales") or []
        calibration_notes = forecast.metadata.get("calibration_notes") or []
        consistency_notes = forecast.metadata.get("consistency_notes") or []
        records: list[dict[str, Any]] = []
        for index, probability in enumerate(forecast.component_probabilities):
            records.append(
                {
                    "probability": probability,
                    "variant": variants[index] if index < len(variants) else "",
                    "model": models[index] if index < len(models) else "",
                    "provider": providers[index] if index < len(providers) else "",
                    "weight": weights[index] if index < len(weights) else 1.0,
                    "resolution_interpretation": (
                        resolution_interpretations[index] if index < len(resolution_interpretations) else ""
                    ),
                    "reference_class": reference_classes[index] if index < len(reference_classes) else "",
                    "base_rate": base_rates[index] if index < len(base_rates) else None,
                    "base_rate_rationale": (
                        base_rate_rationales[index] if index < len(base_rate_rationales) else ""
                    ),
                    "yes_reasons": yes_reasons[index] if index < len(yes_reasons) else [],
                    "no_reasons": no_reasons[index] if index < len(no_reasons) else [],
                    "probability_rationale": (
                        probability_rationales[index] if index < len(probability_rationales) else ""
                    ),
                    "calibration_notes": calibration_notes[index] if index < len(calibration_notes) else "",
                    "consistency_notes": consistency_notes[index] if index < len(consistency_notes) else "",
                }
            )
        return records

    async def _build_calibration_report(
        self,
        sp: SportsPredictClient,
        lobby_id: str,
        history: dict[str, Any],
        calibration_multipliers: dict[str, float],
    ) -> dict[str, Any]:
        try:
            results = await sp.list_results(lobby_id)
        except Exception as exc:
            return {
                "generated_at": utcnow().isoformat(),
                "settled_market_count": 0,
                "error": f"Could not fetch settled results: {exc}",
                "current_multipliers": calibration_multipliers,
                "suggested_multipliers": calibration_multipliers,
            }
        return build_calibration_report(
            results=results,
            history=history,
            current_multipliers=calibration_multipliers,
            learning_rate=self.settings.calibration_learning_rate,
            prior_count=self.settings.calibration_prior_count,
        )

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
        history: dict[str, Any],
        news_cache: dict[str, Any],
    ) -> list[AggregatedForecast]:
        semaphore = asyncio.Semaphore(self.settings.concurrency)
        outputs: list[AggregatedForecast] = []

        async def forecast_one(match: Match, markets: list[Market]) -> list[AggregatedForecast]:
            async with semaphore:
                use_firecrawl = self._should_use_firecrawl(match, markets, history, news_cache, utcnow())
                cached_news = (news_cache.get("matches") or {}).get(match.id, {})
                cached_news_context = self._cached_news_context(cached_news)
                firecrawl_context = ""
                if use_firecrawl:
                    firecrawl_context = await evidence_collector.firecrawl_context(match, markets)
                    if firecrawl_context:
                        self._record_forecast_firecrawl_context(news_cache, match, firecrawl_context)
                evidence = await evidence_collector.collect(
                    match,
                    markets,
                    use_firecrawl=False,
                    cached_news_context=cached_news_context,
                    firecrawl_context_override=firecrawl_context,
                )
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

    @staticmethod
    def _cached_news_context(cached_news: dict[str, Any]) -> str:
        if not cached_news:
            return ""
        parts = [
            f"Cached news checked at: {cached_news.get('last_checked_at', '')}",
            f"Cached news materiality: {cached_news.get('materiality', '')}",
            f"Cached news estimated delta points: {cached_news.get('estimated_delta_points', '')}",
            f"Cached news summary: {cached_news.get('summary', '')}",
        ]
        developments = cached_news.get("new_developments") or []
        if developments:
            parts.append("New developments:\n" + "\n".join(f"- {item}" for item in developments))
        sources = cached_news.get("sources") or []
        if sources:
            rendered_sources = []
            for source in sources[:8]:
                rendered_sources.append(
                    f"- {source.get('title', '')} ({source.get('source', '')}) "
                    f"{source.get('url', '')}: {source.get('summary', '')}"
                )
            parts.append("News monitor sources:\n" + "\n".join(rendered_sources))
        firecrawl_context = cached_news.get("firecrawl_context") or ""
        if firecrawl_context:
            parts.append("Cached Firecrawl snippets:\n" + firecrawl_context[:12000])
        forecast_firecrawl_context = cached_news.get("forecast_firecrawl_context") or ""
        if forecast_firecrawl_context:
            parts.append("Cached full-research Firecrawl snippets:\n" + forecast_firecrawl_context[:12000])
        return "\n\n".join(parts)

    @staticmethod
    def _record_forecast_firecrawl_context(
        news_cache: dict[str, Any],
        match: Match,
        firecrawl_context: str,
    ) -> None:
        news_cache.setdefault("matches", {})
        entry = news_cache["matches"].setdefault(
            match.id,
            {
                "match_id": match.id,
                "match_name": match.name,
                "closing_time": match.closing_time,
            },
        )
        checked_at = utcnow().isoformat()
        compact_context = firecrawl_context[:12000]
        credits = ForecastRunner._parse_firecrawl_credits(firecrawl_context)
        entry.update(
            {
                "match_id": match.id,
                "match_name": match.name,
                "closing_time": match.closing_time,
                "last_forecast_firecrawl_at": checked_at,
                "forecast_firecrawl_context": compact_context,
                "forecast_firecrawl_credits": credits,
            }
        )
        history = list(entry.get("forecast_firecrawl_history") or [])
        history.append(
            {
                "checked_at": checked_at,
                "credits": credits,
                "context": compact_context,
            }
        )
        entry["forecast_firecrawl_history"] = history[-3:]

    @staticmethod
    def _parse_firecrawl_credits(firecrawl_context: str) -> int | None:
        match = re.search(r"Firecrawl (?:monitor )?credits used: (\d+)", firecrawl_context)
        return int(match.group(1)) if match else None

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
