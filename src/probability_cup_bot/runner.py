from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from contextlib import suppress
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from probability_cup_bot.anthropic_adapter import AnthropicAdapter
from probability_cup_bot.calibration import build_calibration_report
from probability_cup_bot.config import Settings
from probability_cup_bot.evidence import EvidenceCollector
from probability_cup_bot.firecrawl import FirecrawlClient
from probability_cup_bot.forecaster import MatchForecaster
from probability_cup_bot.market_analysis import profile_market
from probability_cup_bot.models import (
    AggregatedForecast,
    Market,
    Match,
    MatchEvidence,
    NewsCheck,
    Prediction,
    parse_dt,
    utcnow,
)
from probability_cup_bot.news_monitor import GrokNewsMonitor
from probability_cup_bot.openai_adapter import OpenAIAdapter
from probability_cup_bot.sportspredict import SportsPredictClient, chunks
from probability_cup_bot.state import ensure_dirs, read_json, timestamp_slug, write_json
from probability_cup_bot.usage import UsageTracker, reset_current_tracker, set_current_tracker, update_usage_ledger


logger = logging.getLogger(__name__)

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


def _safe_match_name(match: Match) -> str:
    return " ".join(match.name.split())[:120]


class ForecastRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def run(
        self,
        *,
        news_monitor_only: bool = False,
        target_match_ids: set[str] | None = None,
        force_target_matches: bool = False,
        force_news_monitor: bool = False,
    ) -> dict[str, Any]:
        ensure_dirs(self.settings.state_dir, self.settings.logs_dir)
        target_match_ids = target_match_ids or set()
        usage_tracker = UsageTracker()
        usage_token = set_current_tracker(usage_tracker)
        self._write_forecast_checkpoint(
            [],
            completed_matches=0,
            failed_matches=0,
            total_matches=0,
            status="started",
            stage="run_start",
        )
        logger.info(
            "Run start event_title=%r dry_run=%s max_matches=%s concurrency=%d news_monitor_only=%s target_matches=%d force_target=%s force_news=%s",
            self.settings.event_title,
            not self.settings.can_submit,
            self.settings.max_matches_per_run or "unlimited",
            self.settings.concurrency,
            news_monitor_only,
            len(target_match_ids),
            force_target_matches,
            force_news_monitor,
        )
        sp = SportsPredictClient(
            base_url=self.settings.sportspredict_base_url,
            api_key=self.settings.sportspredict_api_key,
            retry_attempts=self.settings.sportspredict_retry_attempts,
            retry_initial_seconds=self.settings.sportspredict_retry_initial_seconds,
            retry_max_seconds=self.settings.sportspredict_retry_max_seconds,
        )
        firecrawl: FirecrawlClient | None = None
        usage_written = False
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
            logger.info(
                "Model adapters configured openai=%s xai=%s anthropic=%s",
                openai is not None,
                grok is not None,
                anthropic is not None,
            )
            if openai is None and grok is None and anthropic is None:
                raise RuntimeError(
                    "Set OPENAI_API_KEY, XAI_API_KEY, or ANTHROPIC_API_KEY before running forecasts."
                )
            if self.settings.firecrawl_api_key and self.settings.use_firecrawl_retrieval:
                firecrawl = FirecrawlClient(self.settings.firecrawl_api_key)
            logger.info("Firecrawl retrieval enabled=%s", firecrawl is not None)
            previous_calibration = read_json(self.settings.state_dir / "calibration-report.json", {})
            if self.settings.apply_calibration_weights:
                calibration_multipliers = previous_calibration.get("suggested_multipliers") or {}
            else:
                calibration_multipliers = {}
            logger.info(
                "Loaded calibration multipliers enabled=%s count=%d",
                self.settings.apply_calibration_weights,
                len(calibration_multipliers),
            )
            evidence_collector = EvidenceCollector(self.settings, openai, grok, firecrawl)
            forecaster = MatchForecaster(
                self.settings,
                openai,
                grok,
                anthropic,
                calibration_multipliers=calibration_multipliers,
            )

            logger.info("Fetching event")
            event = await sp.find_event(self.settings.event_title, self.settings.event_id)
            logger.info("Fetched event id=%s title=%r", event.id, event.title)
            logger.info("Fetching lobby event_id=%s", event.id)
            lobby = await sp.ensure_lobby(event.id)
            logger.info("Fetched lobby id=%s joined=%s", lobby.id, lobby.joined)
            logger.info("Fetching matches event_id=%s lobby_id=%s", event.id, lobby.id)
            matches = await sp.list_matches(event.id, lobby.id)
            logger.info("Fetched matches count=%d", len(matches))
            logger.info("Fetching markets lobby_id=%s", lobby.id)
            all_markets = await sp.list_markets(lobby.id)
            logger.info(
                "Fetched markets count=%d open=%d",
                len(all_markets),
                sum(1 for market in all_markets if market.status == "open"),
            )
            logger.info("Fetching predictions lobby_id=%s", lobby.id)
            existing_predictions = await sp.list_predictions(lobby.id)
            logger.info("Fetched predictions count=%d", len(existing_predictions))
            history = read_json(self.settings.state_dir / "forecast-history.json", {})
            news_cache = read_json(self.settings.state_dir / "news-cache.json", {"matches": {}})
            logger.info(
                "Loaded state history_matches=%d news_cache_matches=%d",
                len(history.get("matches") or {}),
                len(news_cache.get("matches") or {}),
            )

            selected = (
                []
                if news_monitor_only
                else self._select_matches(
                    matches,
                    all_markets,
                    existing_predictions,
                    history,
                    target_match_ids=target_match_ids,
                    force_target_matches=force_target_matches,
                )
            )
            logger.info(
                "Selected matches count=%d markets=%d",
                len(selected),
                sum(len(markets) for _, markets in selected),
            )
            news_monitor = (
                GrokNewsMonitor(self.settings, grok)
                if self.settings.use_grok_news_monitor and grok is not None
                else None
            )
            pre_news_selected_count = len(selected)
            selected, news_cache, news_checks = await self._augment_selected_with_news_monitor(
                selected=selected,
                matches=matches,
                markets=all_markets,
                existing_predictions=existing_predictions,
                history=history,
                news_cache=news_cache,
                news_monitor=news_monitor,
                firecrawl=firecrawl,
                target_match_ids=target_match_ids,
                force_news_monitor=force_news_monitor,
            )
            logger.info(
                "Selection after news monitor matches=%d promoted=%d checks=%d",
                len(selected),
                max(0, len(selected) - pre_news_selected_count),
                len(news_checks),
            )
            forecast_results = await self._forecast_selected(
                selected=selected,
                evidence_collector=evidence_collector,
                forecaster=forecaster,
                history=history,
                news_cache=news_cache,
            )
            logger.info("Forecasting complete forecasts=%d", len(forecast_results))
            plan = self._plan_writes(
                forecasts=forecast_results,
                existing_predictions=existing_predictions,
                lobby_id=lobby.id,
            )
            component_coverage = self._component_coverage(forecast_results)
            if component_coverage["missing_by_model"]:
                logger.warning(
                    "Forecast component coverage incomplete full_coverage=%d/%d missing_by_model=%s",
                    component_coverage["full_coverage_market_count"],
                    component_coverage["forecast_count"],
                    component_coverage["missing_by_model"],
                )
            history = self._update_history(history, selected, forecast_results, plan)
            submission_results = await self._write_predictions(sp, plan)
            calibration_report = await self._build_calibration_report(sp, lobby.id, history, calibration_multipliers)
            generated_at = utcnow().isoformat()
            usage_summary = self._usage_summary(usage_tracker, generated_at=generated_at, status="complete")
            usage_ledger = self._write_usage_artifacts(usage_summary)
            usage_written = True
            run_log = {
                "generated_at": generated_at,
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
                "news_monitor_only": news_monitor_only,
                "target_match_ids": sorted(target_match_ids),
                "force_target_matches": force_target_matches,
                "force_news_monitor": force_news_monitor,
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
                "component_coverage": component_coverage,
                "submission_results": submission_results,
                "calibration": {
                    "settled_market_count": calibration_report.get("settled_market_count", 0),
                    "aggregate": calibration_report.get("aggregate", {}),
                    "suggested_multipliers": calibration_report.get("suggested_multipliers", {}),
                },
                "usage": usage_summary,
                "usage_cumulative": usage_ledger.get("cumulative", {}),
                "forecasts": [forecast.model_dump() for forecast in forecast_results],
            }
            logger.info("Writing run artifacts logs_dir=%s state_dir=%s", self.settings.logs_dir, self.settings.state_dir)
            write_json(self.settings.state_dir / "forecast-history.json", history)
            write_json(self.settings.state_dir / "news-cache.json", news_cache)
            write_json(self.settings.state_dir / "calibration-report.json", calibration_report)
            write_json(self.settings.logs_dir / f"calibration-{timestamp_slug()}.json", calibration_report)
            write_json(self.settings.logs_dir / f"run-{timestamp_slug()}.json", run_log)
            write_json(self.settings.state_dir / "latest-run.json", run_log)
            logger.info("Run artifacts written latest=%s", self.settings.state_dir / "latest-run.json")
            logger.info("Run complete forecasts=%d mode=%s", len(forecast_results), submission_results["mode"])
            return run_log
        except Exception as exc:
            logger.exception("Run failed error_type=%s", type(exc).__name__)
            if not usage_written:
                with suppress(Exception):
                    generated_at = utcnow().isoformat()
                    usage_summary = self._usage_summary(
                        usage_tracker,
                        generated_at=generated_at,
                        status="failed",
                        error={
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    )
                    usage_ledger = self._write_usage_artifacts(usage_summary, failed=True)
                    write_json(
                        self.settings.state_dir / "last-failed-run.json",
                        {
                            "generated_at": generated_at,
                            "error": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                            "usage": usage_summary,
                            "usage_cumulative": usage_ledger.get("cumulative", {}),
                        },
                    )
            raise
        finally:
            reset_current_tracker(usage_token)
            if firecrawl is not None:
                await firecrawl.aclose()
            await sp.aclose()

    def _usage_summary(
        self,
        usage_tracker: UsageTracker,
        *,
        generated_at: str,
        status: str,
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        usage_summary = usage_tracker.summary()
        usage_summary["generated_at"] = generated_at
        usage_summary["status"] = status
        if error:
            usage_summary["error"] = error
        return usage_summary

    def _write_usage_artifacts(self, usage_summary: dict[str, Any], *, failed: bool = False) -> dict[str, Any]:
        usage_ledger = update_usage_ledger(
            read_json(self.settings.state_dir / "usage-ledger.json", {}),
            usage_summary,
        )
        provider_costs = {
            provider: bucket.get("estimated_cost_usd", 0)
            for provider, bucket in (usage_summary.get("by_provider") or {}).items()
        }
        logger.info(
            "Usage summary status=%s calls=%d estimated_cost_usd=%.6f by_provider=%s",
            usage_summary.get("status"),
            usage_summary.get("call_count", 0),
            float(usage_summary.get("estimated_cost_usd") or 0),
            provider_costs,
        )
        write_json(self.settings.state_dir / "usage-ledger.json", usage_ledger)
        write_json(self.settings.state_dir / "latest-usage.json", usage_summary)
        if failed:
            write_json(self.settings.state_dir / "last-failed-usage.json", usage_summary)
            write_json(self.settings.logs_dir / f"usage-failed-{timestamp_slug()}.json", usage_summary)
        else:
            write_json(self.settings.logs_dir / f"usage-{timestamp_slug()}.json", usage_summary)
        return usage_ledger

    def _select_matches(
        self,
        matches: list[Match],
        markets: list[Market],
        existing_predictions: list[Prediction] | None = None,
        history: dict[str, Any] | None = None,
        *,
        target_match_ids: set[str] | None = None,
        force_target_matches: bool = False,
    ) -> list[tuple[Match, list[Market]]]:
        now = utcnow()
        target_match_ids = target_match_ids or set()
        selected: list[tuple[Match, list[Market]]] = []
        existing_by_market = {
            prediction.market_id: prediction
            for prediction in existing_predictions or []
            if not prediction.market_status or prediction.market_status == "open"
        }
        for match, match_markets in self._eligible_match_groups(matches, markets, now):
            if target_match_ids and match.id not in target_match_ids:
                continue
            if force_target_matches and match.id in target_match_ids:
                selected.append((match, match_markets))
                continue
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
        target_match_ids: set[str] | None = None,
        force_news_monitor: bool = False,
    ) -> tuple[list[tuple[Match, list[Market]]], dict[str, Any], list[dict[str, Any]]]:
        if news_monitor is None:
            logger.info("News monitor disabled")
            return selected, news_cache, []
        now = utcnow()
        target_match_ids = target_match_ids or set()
        selected_ids = {match.id for match, _ in selected}
        existing_by_market = {
            prediction.market_id: prediction
            for prediction in existing_predictions
            if not prediction.market_status or prediction.market_status == "open"
        }
        candidates: list[tuple[Match, list[Market]]] = []
        for match, match_markets in self._eligible_match_groups(matches, markets, now):
            if match.id in selected_ids:
                continue
            if target_match_ids and match.id not in target_match_ids:
                continue
            if force_news_monitor and match.id in target_match_ids:
                if all(market.id in existing_by_market for market in match_markets):
                    candidates.append((match, match_markets))
                continue
            if self._should_news_monitor_match(
                match,
                match_markets,
                existing_by_market,
                history,
                news_cache,
                now,
            ):
                candidates.append((match, match_markets))
        if self.settings.max_matches_per_run > 0:
            remaining = max(0, self.settings.max_matches_per_run - len(selected))
            candidates = candidates[:remaining]

        logger.info(
            "News monitor checks start candidates=%d already_selected=%d",
            len(candidates),
            len(selected),
        )
        news_cache.setdefault("matches", {})
        checks: list[dict[str, Any]] = []
        promoted: list[tuple[Match, list[Market]]] = []
        semaphore = asyncio.Semaphore(self.settings.concurrency)

        async def check_one(match: Match, match_markets: list[Market]) -> tuple[Match, list[Market], NewsCheck | Exception, str]:
            async with semaphore:
                logger.info(
                    "News monitor check start match_id=%s match=%r markets=%d",
                    match.id,
                    _safe_match_name(match),
                    len(match_markets),
                )
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
                    logger.warning(
                        "News monitor check failed match_id=%s error_type=%s",
                        match.id,
                        type(exc).__name__,
                    )
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
            logger.info(
                "News monitor check end match_id=%s should_reforecast=%s delta_points=%d materiality=%s affected_markets=%d used_firecrawl=%s",
                match.id,
                result.should_reforecast,
                result.estimated_delta_points,
                result.materiality,
                len(result.affected_market_ids),
                bool(firecrawl_context),
            )
            if (
                result.should_reforecast
                and result.estimated_delta_points >= self.settings.news_monitor_materiality_threshold_points
            ):
                promoted_markets = self._affected_markets(match_markets, result.affected_market_ids)
                promoted.append((match, promoted_markets))

        output = selected + promoted
        output.sort(key=lambda item: item[0].closes_at or utcnow())
        logger.info("News monitor checks complete checks=%d promoted=%d", len(checks), len(promoted))
        return output, news_cache, checks

    @staticmethod
    def _affected_markets(markets: list[Market], affected_market_ids: list[str]) -> list[Market]:
        affected_ids = {market_id for market_id in affected_market_ids if market_id}
        if not affected_ids:
            return markets
        affected_markets = [market for market in markets if market.id in affected_ids]
        return affected_markets or markets

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
        logger.info("Firecrawl monitor start match_id=%s queries=%d", match.id, len(queries))
        for index, query in enumerate(queries, start=1):
            logger.info(
                "Firecrawl monitor query start match_id=%s query=%d/%d",
                match.id,
                index,
                len(queries),
            )
            try:
                results, credits = await firecrawl.search(
                    query,
                    limit=self.settings.firecrawl_search_limit,
                    sources=("web",),
                    tbs="qdr:d,sbd:1",
                )
            except Exception as exc:
                logger.warning(
                    "Firecrawl monitor query failed match_id=%s query=%d/%d error_type=%s",
                    match.id,
                    index,
                    len(queries),
                    type(exc).__name__,
                )
                contexts.append(f"Firecrawl monitor query failed for {query!r}: {exc}")
                continue
            total_credits += credits
            logger.info(
                "Firecrawl monitor query end match_id=%s query=%d/%d results=%d credits=%d",
                match.id,
                index,
                len(queries),
                len(results),
                credits,
            )
            rendered = "\n".join(result.compact() for result in results[: self.settings.firecrawl_search_limit])
            if rendered:
                contexts.append(f"Firecrawl monitor query: {query}\n{rendered}")
        if not contexts:
            logger.info("Firecrawl monitor end match_id=%s contexts=0 credits=%d", match.id, total_credits)
            return ""
        logger.info(
            "Firecrawl monitor end match_id=%s contexts=%d credits=%d",
            match.id,
            len(contexts),
            total_credits,
        )
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
            "affected_market_ids": news_check.affected_market_ids,
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
                market_profile = profile_market(market.id, market.question, match.name)
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
                        "market_family": market_profile.family,
                        "market_profile": market_profile.model_payload(),
                        "probability_int": forecast.probability_int,
                        "probability": forecast.probability,
                        "component_spread_points": spread,
                        "confidence": forecast.confidence,
                        "evidence_quality": forecast.evidence_quality,
                        "component_count": len(forecast.component_probabilities),
                        "coherence_adjustments": forecast.metadata.get("coherence_adjustments", []),
                        "components": self._component_records(forecast),
                    }
                else:
                    updated["markets"].setdefault(
                        market.id,
                        {
                            "match_id": match.id,
                            "question": market.question,
                            "market_family": market_profile.family,
                            "market_profile": market_profile.model_payload(),
                        },
                    )
                    updated["markets"][market.id]["market_family"] = market_profile.family
                    updated["markets"][market.id]["market_profile"] = market_profile.model_payload()
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

    def _component_coverage(self, forecasts: list[AggregatedForecast]) -> dict[str, Any]:
        expected_models = self._expected_forecast_models()
        missing_by_model = {model: 0 for model in expected_models}
        model_component_counts = {model: 0 for model in expected_models}
        market_reports: list[dict[str, Any]] = []
        full_coverage_count = 0

        for forecast in forecasts:
            observed_models = list((forecast.metadata or {}).get("models") or [])
            observed_unique = sorted(set(str(model) for model in observed_models if model))
            observed_counts = {
                model: sum(1 for observed in observed_models if observed == model)
                for model in observed_unique
            }
            for model, count in observed_counts.items():
                model_component_counts[model] = model_component_counts.get(model, 0) + count
            missing = [model for model in expected_models if model not in observed_counts]
            if missing:
                for model in missing:
                    missing_by_model[model] = missing_by_model.get(model, 0) + 1
                market_reports.append(
                    {
                        "market_id": forecast.market_id,
                        "question": forecast.question,
                        "observed_models": observed_unique,
                        "missing_models": missing,
                        "component_count": len(forecast.component_probabilities),
                    }
                )
            else:
                full_coverage_count += 1

        missing_by_model = {model: count for model, count in missing_by_model.items() if count}
        return {
            "forecast_count": len(forecasts),
            "expected_models": expected_models,
            "full_coverage_market_count": full_coverage_count,
            "partial_coverage_market_count": len(forecasts) - full_coverage_count,
            "model_component_counts": model_component_counts,
            "missing_by_model": missing_by_model,
            "markets_missing_components": market_reports,
        }

    def _expected_forecast_models(self) -> list[str]:
        models: list[str] = []
        if self.settings.use_openai_forecast and self.settings.openai_api_key:
            models.append(self.settings.forecast_model)
        if self.settings.use_grok_forecast and self.settings.xai_api_key:
            models.extend(self.settings.grok_forecast_models)
        if self.settings.use_claude_forecast and self.settings.anthropic_api_key:
            models.extend(self.settings.claude_forecast_models)
        seen: set[str] = set()
        output: list[str] = []
        for model in models:
            if model and model not in seen:
                seen.add(model)
                output.append(model)
        return output

    async def _build_calibration_report(
        self,
        sp: SportsPredictClient,
        lobby_id: str,
        history: dict[str, Any],
        calibration_multipliers: dict[str, float],
    ) -> dict[str, Any]:
        logger.info(
            "Calibration start lobby_id=%s current_multipliers=%d",
            lobby_id,
            len(calibration_multipliers),
        )
        try:
            results = await sp.list_results(lobby_id)
        except Exception as exc:
            logger.warning("Calibration results fetch failed error_type=%s", type(exc).__name__)
            return {
                "generated_at": utcnow().isoformat(),
                "settled_market_count": 0,
                "error": f"Could not fetch settled results: {exc}",
                "current_multipliers": calibration_multipliers,
                "suggested_multipliers": calibration_multipliers,
            }
        logger.info("Calibration fetched results=%d", len(results))
        report = build_calibration_report(
            results=results,
            history=history,
            current_multipliers=calibration_multipliers,
            learning_rate=self.settings.calibration_learning_rate,
            prior_count=self.settings.calibration_prior_count,
        )
        logger.info(
            "Calibration complete settled_market_count=%d suggested_multipliers=%d",
            report.get("settled_market_count", 0),
            len(report.get("suggested_multipliers") or {}),
        )
        return report

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
        started_at = utcnow()
        logger.info(
            "Forecast selected start matches=%d markets=%d concurrency=%d",
            len(selected),
            sum(len(markets) for _, markets in selected),
            self.settings.concurrency,
        )

        async def forecast_one(
            match: Match,
            markets: list[Market],
        ) -> tuple[Match, list[Market], list[AggregatedForecast], Exception | None]:
            async with semaphore:
                use_firecrawl = self._should_use_firecrawl(match, markets, history, news_cache, utcnow())
                cached_news = (news_cache.get("matches") or {}).get(match.id, {})
                cached_news_context = self._cached_news_context(cached_news)
                firecrawl_context = ""
                logger.info(
                    "Match pipeline start match_id=%s match=%r markets=%d use_firecrawl=%s cached_news=%s",
                    match.id,
                    _safe_match_name(match),
                    len(markets),
                    use_firecrawl,
                    bool(cached_news),
                )
                try:
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
                    forecasts = await self._forecast_match_with_missing_retry(
                        forecaster=forecaster,
                        match=match,
                        markets=markets,
                        evidence=evidence,
                    )
                    logger.info("Match pipeline end match_id=%s forecasts=%d", match.id, len(forecasts))
                    return match, markets, forecasts, None
                except Exception as exc:
                    logger.warning(
                        "Match pipeline failed match_id=%s error_type=%s",
                        match.id,
                        type(exc).__name__,
                    )
                    return match, markets, [], exc

        tasks = [asyncio.create_task(forecast_one(match, markets)) for match, markets in selected]
        failed_matches = 0
        completed_matches = 0
        latest_match_id: str | None = None
        latest_match_name: str | None = None
        self._write_forecast_checkpoint(
            outputs,
            completed_matches=0,
            failed_matches=0,
            total_matches=len(selected),
            status="started",
            stage="forecasting",
        )

        async def heartbeat() -> None:
            while True:
                await asyncio.sleep(60)
                elapsed_seconds = round((utcnow() - started_at).total_seconds(), 1)
                forecast_count = len([forecast for forecast in outputs if forecast.market_id != "error"])
                logger.info(
                    "Forecast heartbeat completed_matches=%d/%d failed_matches=%d forecasts=%d elapsed=%.1fs",
                    completed_matches,
                    len(selected),
                    failed_matches,
                    forecast_count,
                    elapsed_seconds,
                )
                self._write_forecast_checkpoint(
                    outputs,
                    completed_matches=completed_matches,
                    failed_matches=failed_matches,
                    total_matches=len(selected),
                    status="running",
                    stage="forecasting",
                    latest_match_id=latest_match_id,
                    latest_match_name=latest_match_name,
                    elapsed_seconds=elapsed_seconds,
                )

        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            for task in asyncio.as_completed(tasks):
                match, _markets, forecasts, error = await task
                latest_match_id = match.id
                latest_match_name = match.name
                completed_matches += 1
                if error is not None:
                    failed_matches += 1
                    outputs.append(
                        AggregatedForecast(
                            market_id="error",
                            question=str(error),
                            probability=0.5,
                            probability_int=50,
                            component_probabilities=[],
                            confidence="low",
                            evidence_quality="low",
                            notes="Forecasting failed for one match.",
                        )
                    )
                else:
                    outputs.extend(forecasts)
                self._write_forecast_checkpoint(
                    outputs,
                    completed_matches=completed_matches,
                    failed_matches=failed_matches,
                    total_matches=len(selected),
                    status="running" if completed_matches < len(selected) else "complete",
                    stage="forecasting",
                    latest_match_id=match.id,
                    latest_match_name=match.name,
                    elapsed_seconds=round((utcnow() - started_at).total_seconds(), 1),
                )
        finally:
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task
        forecasts = [forecast for forecast in outputs if forecast.market_id != "error"]
        logger.info(
            "Forecast selected complete forecasts=%d failed_matches=%d",
            len(forecasts),
            failed_matches,
        )
        return forecasts

    async def _forecast_match_with_missing_retry(
        self,
        *,
        forecaster: MatchForecaster,
        match: Match,
        markets: list[Market],
        evidence: MatchEvidence,
    ) -> list[AggregatedForecast]:
        forecasts = await forecaster.forecast_match(match=match, markets=markets, evidence=evidence)
        missing_markets = self._missing_forecast_markets(markets, forecasts)
        if not missing_markets:
            return forecasts

        logger.warning(
            "Match forecast omitted markets; retrying missing subset match_id=%s missing=%d market_ids=%s",
            match.id,
            len(missing_markets),
            [market.id for market in missing_markets],
        )
        try:
            retry_forecasts = await forecaster.forecast_match(
                match=match,
                markets=missing_markets,
                evidence=evidence,
            )
        except Exception as exc:
            logger.warning(
                "Missing-market retry failed match_id=%s missing=%d error_type=%s",
                match.id,
                len(missing_markets),
                type(exc).__name__,
            )
            return forecasts

        by_market = {forecast.market_id: forecast for forecast in forecasts}
        for forecast in retry_forecasts:
            by_market.setdefault(forecast.market_id, forecast)
        combined = [by_market[market.id] for market in markets if market.id in by_market]
        still_missing = self._missing_forecast_markets(markets, combined)
        if still_missing:
            logger.error(
                "Match forecast still missing markets after retry match_id=%s missing=%d market_ids=%s",
                match.id,
                len(still_missing),
                [market.id for market in still_missing],
            )
        else:
            logger.info(
                "Missing-market retry filled all omitted markets match_id=%s added=%d",
                match.id,
                len(retry_forecasts),
            )
        return combined

    @staticmethod
    def _missing_forecast_markets(
        markets: list[Market],
        forecasts: list[AggregatedForecast],
    ) -> list[Market]:
        forecast_market_ids = {forecast.market_id for forecast in forecasts if forecast.market_id}
        return [market for market in markets if market.id not in forecast_market_ids]

    def _write_forecast_checkpoint(
        self,
        forecasts: list[AggregatedForecast],
        *,
        completed_matches: int,
        failed_matches: int,
        total_matches: int,
        status: str,
        stage: str,
        latest_match_id: str | None = None,
        latest_match_name: str | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        valid_forecasts = [forecast for forecast in forecasts if forecast.market_id != "error"]
        errors = [forecast.question for forecast in forecasts if forecast.market_id == "error"]
        checkpoint = {
            "generated_at": utcnow().isoformat(),
            "status": status,
            "stage": stage,
            "completed_matches": completed_matches,
            "failed_matches": failed_matches,
            "total_matches": total_matches,
            "forecast_count": len(valid_forecasts),
            "latest_match_id": latest_match_id,
            "latest_match_name": latest_match_name,
            "errors": errors[-10:],
            "forecasts": [forecast.model_dump() for forecast in valid_forecasts],
        }
        if elapsed_seconds is not None:
            checkpoint["elapsed_seconds"] = elapsed_seconds
        write_json(self.settings.state_dir / "in-progress-run.json", checkpoint)

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
        logger.info(
            "Plan writes start forecasts=%d existing_predictions=%d threshold_points=%d",
            len(forecasts),
            len(existing_predictions),
            self.settings.update_threshold_points,
        )
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
        plan = {"creates": creates, "updates": updates, "skips": skips}
        logger.info(
            "Plan writes ready creates=%d updates=%d skips=%d",
            len(creates),
            len(updates),
            len(skips),
        )
        return plan

    async def _write_predictions(
        self,
        sp: SportsPredictClient,
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        create_count = len(plan["creates"])
        update_count = len(plan["updates"])
        skip_count = len(plan["skips"])
        if not self.settings.can_submit:
            logger.info(
                "Submission dry-run creates=%d updates=%d skips=%d",
                create_count,
                update_count,
                skip_count,
            )
            return {
                "mode": "dry_run",
                "message": "No writes performed. Set SUBMIT=true to submit or update predictions.",
            }

        logger.info("Submission start creates=%d updates=%d skips=%d", create_count, update_count, skip_count)
        create_results: list[dict[str, Any]] = []
        create_errors: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks(plan["creates"], 50), start=1):
            if index > 1 and self.settings.sportspredict_update_interval_seconds > 0:
                await asyncio.sleep(self.settings.sportspredict_update_interval_seconds)
            logger.info("Submission create batch start batch=%d size=%d", index, len(chunk))
            try:
                result = await sp.submit_batch(chunk)
            except Exception as exc:
                logger.warning(
                    "Submission create batch failed batch=%d error_type=%s",
                    index,
                    type(exc).__name__,
                )
                create_errors.append(
                    {
                        "batch": index,
                        "size": len(chunk),
                        "error": self._error_summary(exc),
                    }
                )
                continue
            create_results.append(result)
            logger.info(
                "Submission create batch end batch=%d total=%s succeeded=%s failed=%s",
                index,
                result.get("total"),
                result.get("succeeded"),
                result.get("failed"),
            )

        update_results: list[dict[str, Any]] = []
        update_errors: list[dict[str, Any]] = []
        for index, item in enumerate(plan["updates"], start=1):
            if index > 1 and self.settings.sportspredict_update_interval_seconds > 0:
                await asyncio.sleep(self.settings.sportspredict_update_interval_seconds)
            logger.info(
                "Submission update start index=%d market_id=%s probability=%d",
                index,
                item["market_id"],
                item["probability"],
            )
            try:
                updated = await sp.update_prediction(item["prediction_id"], item["probability"])
            except Exception as exc:
                logger.warning(
                    "Submission update failed index=%d market_id=%s error_type=%s",
                    index,
                    item["market_id"],
                    type(exc).__name__,
                )
                update_errors.append(
                    {
                        "index": index,
                        "prediction_id": item["prediction_id"],
                        "market_id": item["market_id"],
                        "probability": item["probability"],
                        "error": self._error_summary(exc),
                    }
                )
                continue
            update_results.append(updated.model_dump())
            logger.info("Submission update end index=%d market_id=%s", index, item["market_id"])

        mode = "submitted" if not create_errors and not update_errors else "submitted_with_errors"
        logger.info(
            "Submission complete mode=%s create_batches=%d updates=%d failed_create_batches=%d failed_updates=%d",
            mode,
            len(create_results),
            len(update_results),
            len(create_errors),
            len(update_errors),
        )
        return {
            "mode": mode,
            "creates": create_results,
            "updates": update_results,
            "failed_create_batches": create_errors,
            "failed_updates": update_errors,
        }

    @staticmethod
    def _error_summary(exc: Exception) -> dict[str, Any]:
        return {
            "type": type(exc).__name__,
            "status_code": getattr(exc, "status_code", None),
            "message": str(exc)[:2000],
        }
