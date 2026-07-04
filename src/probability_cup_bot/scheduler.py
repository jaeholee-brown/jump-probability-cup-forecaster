from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from probability_cup_bot.config import Settings
from probability_cup_bot.models import Match, parse_dt, utcnow
from probability_cup_bot.runner import ForecastRunner
from probability_cup_bot.sportspredict import SportsPredictClient
from probability_cup_bot.state import ensure_dirs, read_json, timestamp_slug, write_json


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DueActions:
    forecast_match_ids: list[str]
    news_match_ids: list[str]


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse(value: str | None) -> datetime | None:
    parsed = parse_dt(value)
    return parsed.astimezone(timezone.utc) if parsed else None


def build_due_actions(
    schedule: dict[str, Any],
    *,
    now: datetime,
    forecast_offset_minutes: float = 1440.0,
    final_forecast_offset_minutes: float = 55.0,
    news_offset_minutes: float = 40.0,
) -> DueActions:
    now = now.astimezone(timezone.utc)
    forecast_ids: list[str] = []
    news_ids: list[str] = []
    for match_id, entry in sorted((schedule.get("matches") or {}).items()):
        closes_at = _parse(entry.get("opening_time") or entry.get("closing_time"))
        if closes_at is None or closes_at <= now:
            continue
        forecast_due_at = closes_at - timedelta(minutes=forecast_offset_minutes)
        final_due_at = closes_at - timedelta(minutes=final_forecast_offset_minutes)
        news_due_at = closes_at - timedelta(minutes=news_offset_minutes)
        forecast_completed_at = _parse(entry.get("late_forecast_completed_at"))
        final_completed_at = _parse(entry.get("final_forecast_completed_at"))
        news_completed_at = _parse(entry.get("news_check_completed_at"))
        latest_full = max(
            (value for value in (forecast_completed_at, final_completed_at) if value is not None),
            default=None,
        )

        final_satisfied = final_completed_at is not None or (
            forecast_completed_at is not None and forecast_completed_at >= final_due_at
        )

        if now >= forecast_due_at and forecast_completed_at is None:
            forecast_ids.append(match_id)
            continue
        # Unconditional full-ensemble pass once confirmed lineups are typically
        # out (~T-60m). Not gated on news materiality: the 2026-06-29..07-02
        # stale cohort showed a single one-shot news window is too fragile.
        if now >= final_due_at and not final_satisfied:
            forecast_ids.append(match_id)
            continue
        if now >= news_due_at and news_completed_at is None:
            if latest_full is None:
                forecast_ids.append(match_id)
            elif latest_full < news_due_at:
                news_ids.append(match_id)
    return DueActions(forecast_ids, news_ids)


class MatchScheduler:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.schedule_path = settings.state_dir / "match-schedule.json"

    async def refresh(self) -> dict[str, Any]:
        ensure_dirs(self.settings.state_dir, self.settings.logs_dir)
        logger.info("Schedule refresh start")
        state = read_json(self.schedule_path, {"matches": {}})
        sp = SportsPredictClient(
            base_url=self.settings.sportspredict_base_url,
            api_key=self.settings.sportspredict_api_key,
            retry_attempts=self.settings.sportspredict_retry_attempts,
            retry_initial_seconds=self.settings.sportspredict_retry_initial_seconds,
            retry_max_seconds=self.settings.sportspredict_retry_max_seconds,
        )
        try:
            event = await sp.find_event(self.settings.event_title, self.settings.event_id)
            lobby = await sp.ensure_lobby(event.id)
            matches = await sp.list_matches(event.id, lobby.id)
        finally:
            await sp.aclose()

        now = utcnow()
        updated = self._merge_matches(state, matches, now)
        updated["event"] = event.model_dump()
        updated["lobby"] = lobby.model_dump()
        updated["last_refreshed_at"] = _iso(now)
        updated["match_count"] = len(updated.get("matches") or {})
        write_json(self.schedule_path, updated)
        write_json(self.settings.logs_dir / f"schedule-refresh-{timestamp_slug()}.json", updated)
        logger.info("Schedule refresh complete matches=%d", len(matches))
        return {
            "mode": "refresh",
            "generated_at": _iso(now),
            "matches_seen": len(matches),
            "scheduled_matches": len(updated.get("matches") or {}),
            "schedule": str(self.schedule_path),
        }

    async def run_due(self) -> dict[str, Any]:
        ensure_dirs(self.settings.state_dir, self.settings.logs_dir)
        logger.info("Refreshing schedule before due check")
        await self.refresh()
        state = read_json(self.schedule_path, {"matches": {}})

        now = utcnow()
        due = build_due_actions(
            state,
            now=now,
            forecast_offset_minutes=self.settings.scheduler_forecast_offset_minutes,
            final_forecast_offset_minutes=self.settings.scheduler_final_forecast_offset_minutes,
            news_offset_minutes=self.settings.scheduler_news_offset_minutes,
        )
        logger.info(
            "Schedule due check forecast_due=%d news_due=%d",
            len(due.forecast_match_ids),
            len(due.news_match_ids),
        )
        report: dict[str, Any] = {
            "mode": "run_due",
            "generated_at": _iso(now),
            "forecast_match_ids": due.forecast_match_ids,
            "news_match_ids": due.news_match_ids,
            "forecast_result": None,
            "news_result": None,
        }
        runner = ForecastRunner(self.settings)

        just_forecasted: set[str] = set()
        if due.forecast_match_ids:
            forecast_result = await runner.run(
                target_match_ids=set(due.forecast_match_ids),
                force_target_matches=True,
            )
            report["forecast_result"] = self._compact_run_result(forecast_result)
            just_forecasted = set(due.forecast_match_ids)
            self._mark_forecast_completed(state, due.forecast_match_ids, forecast_result["generated_at"])

        news_ids = [match_id for match_id in due.news_match_ids if match_id not in just_forecasted]
        if news_ids:
            news_result = await runner.run(
                news_monitor_only=True,
                target_match_ids=set(news_ids),
                force_news_monitor=True,
            )
            report["news_result"] = self._compact_run_result(news_result)
            self._mark_news_completed(state, news_ids, news_result["generated_at"], news_result)

        if just_forecasted:
            self._mark_late_forecast_as_news_check_if_needed(state, just_forecasted)

        state["last_due_checked_at"] = _iso(utcnow())
        write_json(self.schedule_path, state)
        write_json(self.settings.logs_dir / f"schedule-due-{timestamp_slug()}.json", report)
        return report

    def _merge_matches(
        self,
        state: dict[str, Any],
        matches: list[Match],
        now: datetime,
    ) -> dict[str, Any]:
        updated = dict(state)
        updated.setdefault("matches", {})
        entries = updated["matches"]
        seen_ids = set()
        for match in matches:
            closes_at = match.closes_at
            if closes_at is not None and closes_at.astimezone(timezone.utc) <= now:
                continue
            seen_ids.add(match.id)
            entry = dict(entries.get(match.id) or {})
            entry.setdefault("first_seen_at", _iso(now))
            entry.update(
                {
                    "match_id": match.id,
                    "name": match.name,
                    "closing_time": match.closing_time,
                    "opening_time": match.opening_time,
                    "open_market_count": match.open_market_count,
                    "last_seen_at": _iso(now),
                }
            )
            if closes_at is not None:
                close = closes_at.astimezone(timezone.utc)
                entry["late_forecast_due_at"] = _iso(
                    close - timedelta(minutes=self.settings.scheduler_forecast_offset_minutes)
                )
                entry["final_forecast_due_at"] = _iso(
                    close - timedelta(minutes=self.settings.scheduler_final_forecast_offset_minutes)
                )
                entry["news_check_due_at"] = _iso(
                    close - timedelta(minutes=self.settings.scheduler_news_offset_minutes)
                )
            entries[match.id] = entry
        for match_id, entry in list(entries.items()):
            if match_id not in seen_ids:
                entry["last_missing_at"] = _iso(now)
        return updated

    def _schedule_is_stale(self, state: dict[str, Any]) -> bool:
        refreshed_at = _parse(state.get("last_refreshed_at"))
        if refreshed_at is None:
            return True
        age_hours = (utcnow() - refreshed_at).total_seconds() / 3600
        return age_hours >= 6

    @staticmethod
    def _compact_run_result(run_log: dict[str, Any]) -> dict[str, Any]:
        return {
            "generated_at": run_log.get("generated_at"),
            "matches_forecasted": run_log.get("matches_forecasted"),
            "forecast_count": run_log.get("forecast_count"),
            "creates": len((run_log.get("plan") or {}).get("creates") or []),
            "updates": len((run_log.get("plan") or {}).get("updates") or []),
            "skips": len((run_log.get("plan") or {}).get("skips") or []),
            "submission_mode": (run_log.get("submission_results") or {}).get("mode"),
            "news_checks": len(((run_log.get("news_monitor") or {}).get("checks") or [])),
        }

    @staticmethod
    def _mark_forecast_completed(
        state: dict[str, Any],
        match_ids: list[str],
        completed_at: str,
    ) -> None:
        completed = _parse(completed_at)
        for match_id in match_ids:
            entry = (state.get("matches") or {}).setdefault(match_id, {"match_id": match_id})
            entry.setdefault("late_forecast_completed_at", completed_at)
            final_due_at = _parse(entry.get("final_forecast_due_at"))
            if completed and final_due_at and completed >= final_due_at:
                entry["final_forecast_completed_at"] = completed_at

    def _mark_news_completed(
        self,
        state: dict[str, Any],
        match_ids: list[str],
        completed_at: str,
        run_log: dict[str, Any],
    ) -> None:
        checks = (run_log.get("news_monitor") or {}).get("checks") or []
        checks_by_match = {check.get("match_id"): check for check in checks if isinstance(check, dict)}
        for match_id in match_ids:
            entry = (state.get("matches") or {}).setdefault(match_id, {"match_id": match_id})
            if match_id not in checks_by_match:
                # Leave incomplete so the next due tick retries instead of
                # silently locking with a stale forecast, unless the monitor is
                # disabled and can never produce a check.
                if not self.settings.use_grok_news_monitor:
                    entry["news_check_completed_at"] = completed_at
                    entry["news_check_skipped_reason"] = "news monitor disabled"
                continue
            check = checks_by_match[match_id]
            entry["news_check_completed_at"] = completed_at
            entry["news_check_should_reforecast"] = check.get("should_reforecast")
            entry["news_check_estimated_delta_points"] = check.get("estimated_delta_points")
            entry["news_check_materiality"] = check.get("materiality")
            if check.get("should_reforecast"):
                entry["news_reforecast_completed_at"] = completed_at

    @staticmethod
    def _mark_late_forecast_as_news_check_if_needed(state: dict[str, Any], match_ids: set[str]) -> None:
        for match_id in match_ids:
            entry = (state.get("matches") or {}).get(match_id) or {}
            completions = [
                value
                for value in (
                    _parse(entry.get("late_forecast_completed_at")),
                    _parse(entry.get("final_forecast_completed_at")),
                )
                if value is not None
            ]
            news_due_at = _parse(entry.get("news_check_due_at"))
            if completions and news_due_at and max(completions) >= news_due_at:
                entry["news_check_completed_at"] = _iso(max(completions))
                entry["news_check_skipped_reason"] = "full forecast completed after news-check due time"
