from __future__ import annotations

import asyncio
import json
from datetime import timezone
from difflib import SequenceMatcher
from typing import Any

import httpx

from probability_cup_bot.config import Settings
from probability_cup_bot.models import Match, MatchEvidence, Market, utcnow
from probability_cup_bot.openai_adapter import OpenAIAdapter
from probability_cup_bot.prompts import RESEARCH_INSTRUCTIONS


RESEARCH_PASS_TASKS: dict[str, str] = {
    "overview": (
        "Build the stable outside-view picture for this match: team strength, recent form, "
        "tournament incentives, likely tactical setup, historical/base rates for the listed "
        "market families, and any reliable public odds context."
    ),
    "late_news": (
        "Search specifically for current lineups, injuries, suspensions, player availability, "
        "manager comments, weather, travel/rest issues, and credible late-breaking news. Use "
        "recent web/X evidence and prefer timestamped sources."
    ),
    "market_micro": (
        "Work market by market. For every listed question, find the most decision-relevant "
        "public facts, especially for player props, cards, shots, corners, goals, penalties, "
        "and other volatile soccer markets."
    ),
}


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
    def __init__(
        self,
        settings: Settings,
        openai: OpenAIAdapter | None,
        grok: OpenAIAdapter | None = None,
    ) -> None:
        self.settings = settings
        self.openai = openai
        self.grok = grok

    async def collect(self, match: Match, markets: list[Market]) -> MatchEvidence:
        odds_context = await self._odds_context(match)
        adapter = self.grok if self.settings.use_grok_research and self.grok else self.openai
        if adapter is None:
            return MatchEvidence(
                match_id=match.id,
                match_name=match.name,
                generated_at=utcnow().isoformat(),
                query_summary="No OpenAI-compatible search adapter configured; using platform data only.",
                odds_context=odds_context,
                key_facts=[],
                items=[],
                evidence_quality="low",
            )

        pass_names = self.settings.grok_research_passes if adapter.provider == "xai" else ("overview",)
        if not pass_names:
            pass_names = ("overview",)
        results = await asyncio.gather(
            *[
                self._research_pass(
                    adapter=adapter,
                    match=match,
                    markets=markets,
                    odds_context=odds_context,
                    pass_name=pass_name,
                )
                for pass_name in pass_names
            ],
            return_exceptions=True,
        )
        evidences = [result for result in results if isinstance(result, MatchEvidence)]
        if evidences:
            return self._merge_evidence(match, odds_context, evidences)

        failures = "; ".join(str(result)[:300] for result in results if isinstance(result, Exception))
        return MatchEvidence(
            match_id=match.id,
            match_name=match.name,
            generated_at=utcnow().isoformat(),
            query_summary=f"Evidence collection failed; using platform data only: {failures}",
            odds_context=odds_context,
            key_facts=[],
            items=[],
            evidence_quality="low",
        )

    async def _research_pass(
        self,
        *,
        adapter: OpenAIAdapter,
        match: Match,
        markets: list[Market],
        odds_context: str,
        pass_name: str,
    ) -> MatchEvidence:
        research_task = RESEARCH_PASS_TASKS.get(pass_name, pass_name)
        user_input = json.dumps(
            {
                "today_utc": utcnow().astimezone(timezone.utc).isoformat(),
                "match": match.model_dump(),
                "markets": [market.model_dump() for market in markets],
                "odds_context": odds_context,
                "research_pass": pass_name,
                "research_task": research_task,
            },
            ensure_ascii=True,
        )
        model = self.settings.grok_research_model if adapter.provider == "xai" else self.settings.research_model
        tools = [{"type": "web_search"}]
        if adapter.provider == "xai":
            tools.append({"type": "x_search"})
        reasoning_effort = (
            self.settings.grok_research_reasoning_effort if adapter.provider == "xai" else "low"
        )
        evidence = await adapter.structured_response(
            model=model,
            instructions=f"{RESEARCH_INSTRUCTIONS}\n\nResearch pass focus:\n{research_task}",
            user_input=user_input,
            schema_model=MatchEvidence,
            schema_name="match_evidence",
            reasoning_effort=reasoning_effort,
            tools=tools,
        )
        if not evidence.match_id:
            evidence.match_id = match.id
        if not evidence.match_name:
            evidence.match_name = match.name
        if odds_context and not evidence.odds_context:
            evidence.odds_context = odds_context
        evidence.query_summary = f"[{pass_name}] {evidence.query_summary}"
        return evidence

    def _merge_evidence(
        self,
        match: Match,
        odds_context: str,
        evidences: list[MatchEvidence],
    ) -> MatchEvidence:
        key_facts: list[str] = []
        seen_facts: set[str] = set()
        for evidence in evidences:
            for fact in evidence.key_facts:
                normalized = " ".join(fact.lower().split())
                if normalized and normalized not in seen_facts:
                    seen_facts.add(normalized)
                    key_facts.append(fact)

        items = []
        seen_items: set[str] = set()
        for evidence in evidences:
            for item in evidence.items:
                key = item.url or f"{item.title.lower()}::{item.summary[:80].lower()}"
                if key and key not in seen_items:
                    seen_items.add(key)
                    items.append(item)
        items = sorted(items, key=lambda item: item.relevance, reverse=True)[:18]

        query_summary = "\n".join(evidence.query_summary for evidence in evidences if evidence.query_summary)
        quality = self._best_evidence_quality(evidence.evidence_quality for evidence in evidences)
        return MatchEvidence(
            match_id=match.id,
            match_name=match.name,
            generated_at=utcnow().isoformat(),
            query_summary=f"Merged {len(evidences)} research passes.\n{query_summary}",
            key_facts=key_facts[:30],
            odds_context=odds_context or next((e.odds_context for e in evidences if e.odds_context), ""),
            items=items,
            evidence_quality=quality,
        )

    @staticmethod
    def _best_evidence_quality(values: Any) -> str:
        order = {"low": 0, "medium": 1, "high": 2}
        best = "low"
        for value in values:
            if order.get(value, 0) > order[best]:
                best = value
        return best

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
