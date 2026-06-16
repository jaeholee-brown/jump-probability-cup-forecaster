from __future__ import annotations

import json
from datetime import timezone
from difflib import SequenceMatcher
from typing import Any

import httpx

from probability_cup_bot.config import Settings
from probability_cup_bot.models import Match, MatchEvidence, Market, utcnow
from probability_cup_bot.openai_adapter import OpenAIAdapter
from probability_cup_bot.prompts import RESEARCH_INSTRUCTIONS


def split_match_name(match_name: str) -> tuple[str, str] | None:
    lowered = match_name.replace(" vs. ", " vs ").replace(" v. ", " v ")
    for delimiter in (" vs ", " v ", " - "):
        if delimiter in lowered:
            left, right = lowered.split(delimiter, 1)
            return left.strip(), right.strip()
    return None


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


class EvidenceCollector:
    def __init__(self, settings: Settings, openai: OpenAIAdapter) -> None:
        self.settings = settings
        self.openai = openai

    async def collect(self, match: Match, markets: list[Market]) -> MatchEvidence:
        odds_context = await self._odds_context(match)
        user_input = json.dumps(
            {
                "today_utc": utcnow().astimezone(timezone.utc).isoformat(),
                "match": match.model_dump(),
                "markets": [market.model_dump() for market in markets],
                "odds_context": odds_context,
                "research_task": (
                    "Gather compact current evidence for forecasting these Jump Probability Cup "
                    "markets. Focus on match result, goals, cards, player/team props, injuries, "
                    "lineups, odds, weather, and tournament incentives when relevant."
                ),
            },
            ensure_ascii=True,
        )
        try:
            evidence = await self.openai.structured_response(
                model=self.settings.research_model,
                instructions=RESEARCH_INSTRUCTIONS,
                user_input=user_input,
                schema_model=MatchEvidence,
                schema_name="match_evidence",
                reasoning_effort="low",
                tools=[{"type": "web_search"}],
            )
        except Exception as exc:
            evidence = MatchEvidence(
                match_id=match.id,
                match_name=match.name,
                generated_at=utcnow().isoformat(),
                query_summary=f"Evidence collection failed; using platform data only: {exc}",
                odds_context=odds_context,
                key_facts=[],
                items=[],
                evidence_quality="low",
            )
        if not evidence.match_id:
            evidence.match_id = match.id
        if not evidence.match_name:
            evidence.match_name = match.name
        if odds_context and not evidence.odds_context:
            evidence.odds_context = odds_context
        return evidence

    async def _odds_context(self, match: Match) -> str:
        if not self.settings.odds_api_key:
            return ""
        teams = split_match_name(match.name)
        if teams is None:
            return ""
        home, away = teams
        sport_keys = await self._sport_keys()
        if not sport_keys:
            return ""
        contexts: list[str] = []
        async with httpx.AsyncClient(timeout=20.0) as client:
            for sport_key in sport_keys[:3]:
                url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
                params = {
                    "apiKey": self.settings.odds_api_key,
                    "regions": "us,uk,eu",
                    "markets": "h2h,totals,btts",
                    "oddsFormat": "decimal",
                }
                try:
                    response = await client.get(url, params=params)
                    if response.status_code >= 400:
                        continue
                    for event in response.json():
                        home_team = event.get("home_team", "")
                        away_team = event.get("away_team", "")
                        score = max(
                            (_similar(home, home_team) + _similar(away, away_team)) / 2,
                            (_similar(home, away_team) + _similar(away, home_team)) / 2,
                        )
                        if score < 0.62:
                            continue
                        contexts.append(self._render_odds_event(event))
                except Exception:
                    continue
        return "\n".join(context for context in contexts if context)[:4000]

    async def _sport_keys(self) -> list[str]:
        configured = [key.strip() for key in self.settings.odds_sport_key.split(",") if key.strip()]
        if configured and configured != ["soccer"]:
            return configured
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(
                    "https://api.the-odds-api.com/v4/sports",
                    params={"apiKey": self.settings.odds_api_key},
                )
                if response.status_code >= 400:
                    return configured
                sports = response.json()
        except Exception:
            return configured
        soccer_keys = [
            item["key"]
            for item in sports
            if item.get("active") and str(item.get("key", "")).startswith("soccer")
        ]
        preferred = [key for key in soccer_keys if "world_cup" in key or "fifa" in key]
        return preferred or soccer_keys[:3] or configured

    @staticmethod
    def _render_odds_event(event: dict[str, Any]) -> str:
        parts = [
            f"Odds event: {event.get('home_team')} vs {event.get('away_team')} "
            f"commence_time={event.get('commence_time')}"
        ]
        for bookmaker in event.get("bookmakers", [])[:5]:
            markets = []
            for market in bookmaker.get("markets", []):
                outcomes = ", ".join(
                    f"{outcome.get('name')} {outcome.get('price')}"
                    for outcome in market.get("outcomes", [])
                )
                markets.append(f"{market.get('key')}: {outcomes}")
            if markets:
                parts.append(f"{bookmaker.get('title')}: " + " | ".join(markets))
        return "\n".join(parts)

