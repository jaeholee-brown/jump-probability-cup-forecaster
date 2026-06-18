from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import timezone
from difflib import SequenceMatcher
from typing import Any

import httpx

from probability_cup_bot.config import Settings
from probability_cup_bot.firecrawl import FirecrawlClient
from probability_cup_bot.models import EvidenceAudit, Match, MatchEvidence, Market, utcnow
from probability_cup_bot.openai_adapter import OpenAIAdapter
from probability_cup_bot.prompts import RESEARCH_INSTRUCTIONS


logger = logging.getLogger(__name__)

RESEARCH_PASS_TASKS: dict[str, str] = {
    "overview": (
        "Build the stable outside-view picture for this match: team strength, recent form, "
        "tournament incentives, likely tactical setup, historical/base rates for the listed "
        "market families, and any reliable public odds context."
    ),
    "base_rates": (
        "Spend this pass on base rates and reference classes. Estimate market-family priors "
        "for soccer outcomes, goals, cards, corners, shots, player participation/starts, and "
        "common prop markets. Prefer quantitative ranges and explain when no good base-rate "
        "data is available."
    ),
    "late_news": (
        "Search specifically for current lineups, injuries, suspensions, player availability, "
        "manager comments, weather, travel/rest issues, and credible late-breaking news. Use "
        "recent web/X evidence and prefer timestamped sources. Treat X as a fast discovery "
        "channel for official accounts and credible reporters; label social-only claims as "
        "uncertain unless corroborated."
    ),
    "market_micro": (
        "Work market by market. For every listed question, find the most decision-relevant "
        "public facts, especially for player props, cards, shots, corners, goals, penalties, "
        "and other volatile soccer markets."
    ),
    "lineup_roles": (
        "Focus on confirmed or expected lineups, substitutions risk, player roles, set pieces, "
        "formation, tactical matchup, injuries, suspensions, minutes expectations, and manager "
        "comments. For player markets, separate start probability, expected minutes, role, and "
        "per-minute event rate."
    ),
    "volatile_market_anchors": (
        "Focus only on volatile soccer prop families: cards, fouls, penalties, red cards, corners, "
        "shots on target, offsides, BTTS, team totals, half-specific goals, and joint markets. "
        "Find statistical or odds anchors where possible, quantify broad base rates when narrow "
        "rates are unavailable, and flag markets where narrative evidence is too weak."
    ),
}


EVIDENCE_AUDIT_INSTRUCTIONS = """
You are an evidence QA analyst for a soccer probability forecasting bot.

Review the merged evidence before the expensive forecast ensemble sees it. Your job is not to
make final forecasts. Your job is to detect evidence problems that could cause bad probabilities.

Use web search and X search if needed to verify freshness, but be disciplined:
- Flag missing denominators, invented-looking stats, stale previews, unsupported lineups, and
  social-only claims that are not corroborated.
- For each volatile market, say whether the evidence has a real statistical/odds/source anchor
  or only narrative reasoning.
- For player props, check whether start probability, expected minutes, tactical role, and recent
  per-minute rate were established.
- For joint or union markets, check whether the evidence supports a decomposition instead of
  independent addition.
- For cards/penalties/reds/corners/shots/offsides, ask for external anchors and warn if absent.

Return compact, actionable QA notes. Do not overstate uncertainty: if the evidence is good, say so.
""".strip()


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
        firecrawl: FirecrawlClient | None = None,
    ) -> None:
        self.settings = settings
        self.openai = openai
        self.grok = grok
        self.firecrawl = firecrawl

    async def collect(
        self,
        match: Match,
        markets: list[Market],
        *,
        use_firecrawl: bool = True,
        cached_news_context: str = "",
        firecrawl_context_override: str | None = None,
    ) -> MatchEvidence:
        started_at = time.perf_counter()
        logger.info(
            "Evidence start match_id=%s match=%r markets=%d use_firecrawl=%s cached_news=%s",
            match.id,
            " ".join(match.name.split())[:120],
            len(markets),
            use_firecrawl and firecrawl_context_override is None,
            bool(cached_news_context),
        )
        odds_context = await self._odds_context(match)
        if firecrawl_context_override is not None:
            firecrawl_context = firecrawl_context_override
        else:
            firecrawl_context = await self._firecrawl_context(match, markets) if use_firecrawl else ""
        adapter = self.grok if self.settings.use_grok_research and self.grok else self.openai
        if adapter is None:
            evidence = MatchEvidence(
                match_id=match.id,
                match_name=match.name,
                generated_at=utcnow().isoformat(),
                query_summary="No OpenAI-compatible search adapter configured; using platform data only.",
                odds_context=odds_context,
                key_facts=[],
                items=[],
                evidence_quality="low",
            )
            logger.info(
                "Evidence end match_id=%s quality=low facts=0 items=0 reason=no_adapter elapsed=%.1fs",
                match.id,
                time.perf_counter() - started_at,
            )
            return evidence

        pass_names = self.settings.grok_research_passes if adapter.provider == "xai" else ("overview",)
        if not pass_names:
            pass_names = ("overview",)
        logger.info(
            "Evidence research passes match_id=%s provider=%s passes=%s",
            match.id,
            adapter.provider,
            ",".join(pass_names),
        )
        results = await asyncio.gather(
            *[
                self._research_pass(
                    adapter=adapter,
                    match=match,
                    markets=markets,
                    odds_context=odds_context,
                    firecrawl_context=firecrawl_context,
                    cached_news_context=cached_news_context,
                    pass_name=pass_name,
                )
                for pass_name in pass_names
            ],
            return_exceptions=True,
        )
        failed_passes = sum(isinstance(result, Exception) for result in results)
        evidences = [result for result in results if isinstance(result, MatchEvidence)]
        if evidences:
            evidence = self._merge_evidence(match, odds_context, evidences)
            evidence = await self._maybe_audit_evidence(
                adapter=adapter,
                match=match,
                markets=markets,
                evidence=evidence,
            )
            logger.info(
                "Evidence end match_id=%s quality=%s facts=%d items=%d passes=%d failed_passes=%d elapsed=%.1fs",
                match.id,
                evidence.evidence_quality,
                len(evidence.key_facts),
                len(evidence.items),
                len(evidences),
                failed_passes,
                time.perf_counter() - started_at,
            )
            return evidence

        failures = "; ".join(str(result)[:300] for result in results if isinstance(result, Exception))
        evidence = MatchEvidence(
            match_id=match.id,
            match_name=match.name,
            generated_at=utcnow().isoformat(),
            query_summary=f"Evidence collection failed; using platform data only: {failures}",
            odds_context=odds_context,
            key_facts=[],
            items=[],
            evidence_quality="low",
        )
        logger.warning(
            "Evidence end match_id=%s quality=low facts=0 items=0 passes=0 failed_passes=%d elapsed=%.1fs",
            match.id,
            failed_passes,
            time.perf_counter() - started_at,
        )
        return evidence

    async def firecrawl_context(self, match: Match, markets: list[Market]) -> str:
        return await self._firecrawl_context(match, markets)

    async def _research_pass(
        self,
        *,
        adapter: OpenAIAdapter,
        match: Match,
        markets: list[Market],
        odds_context: str,
        firecrawl_context: str,
        cached_news_context: str,
        pass_name: str,
    ) -> MatchEvidence:
        research_task = RESEARCH_PASS_TASKS.get(pass_name, pass_name)
        user_input = json.dumps(
            {
                "today_utc": utcnow().astimezone(timezone.utc).isoformat(),
                "match": match.model_dump(),
                "markets": [market.model_dump() for market in markets],
                "odds_context": odds_context,
                "firecrawl_context": firecrawl_context,
                "cached_news_context": cached_news_context,
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
        started_at = time.perf_counter()
        logger.info(
            "Evidence research pass start match_id=%s pass=%s provider=%s model=%s tools=%d",
            match.id,
            pass_name,
            adapter.provider,
            model,
            len(tools),
        )
        try:
            evidence = await adapter.structured_response(
                model=model,
                instructions=f"{RESEARCH_INSTRUCTIONS}\n\nResearch pass focus:\n{research_task}",
                user_input=user_input,
                schema_model=MatchEvidence,
                schema_name="match_evidence",
                reasoning_effort=reasoning_effort,
                tools=tools,
            )
        except Exception as exc:
            logger.warning(
                "Evidence research pass failed match_id=%s pass=%s provider=%s model=%s error_type=%s elapsed=%.1fs",
                match.id,
                pass_name,
                adapter.provider,
                model,
                type(exc).__name__,
                time.perf_counter() - started_at,
            )
            raise
        if not evidence.match_id:
            evidence.match_id = match.id
        if not evidence.match_name:
            evidence.match_name = match.name
        if odds_context and not evidence.odds_context:
            evidence.odds_context = odds_context
        evidence.query_summary = f"[{pass_name}] {evidence.query_summary}"
        logger.info(
            "Evidence research pass end match_id=%s pass=%s quality=%s facts=%d items=%d elapsed=%.1fs",
            match.id,
            pass_name,
            evidence.evidence_quality,
            len(evidence.key_facts),
            len(evidence.items),
            time.perf_counter() - started_at,
        )
        return evidence

    async def _maybe_audit_evidence(
        self,
        *,
        adapter: OpenAIAdapter,
        match: Match,
        markets: list[Market],
        evidence: MatchEvidence,
    ) -> MatchEvidence:
        if adapter.provider != "xai" or not self.settings.use_grok_evidence_qa:
            return evidence

        user_input = json.dumps(
            {
                "today_utc": utcnow().astimezone(timezone.utc).isoformat(),
                "match": match.model_dump(),
                "markets": [
                    {
                        "id": market.id,
                        "question": market.question,
                        "status": market.status,
                        "closing_time": market.match.closing_time,
                    }
                    for market in markets
                ],
                "merged_evidence": evidence.model_dump(),
                "decision_context": (
                    "The forecast ensemble will see this evidence after your audit. Add cautions "
                    "that should change probabilistic forecasts, especially for volatile markets."
                ),
            },
            ensure_ascii=True,
        )
        started_at = time.perf_counter()
        logger.info(
            "Evidence QA start match_id=%s provider=%s model=%s markets=%d",
            match.id,
            adapter.provider,
            self.settings.grok_evidence_qa_model,
            len(markets),
        )
        try:
            audit = await adapter.structured_response(
                model=self.settings.grok_evidence_qa_model,
                instructions=EVIDENCE_AUDIT_INSTRUCTIONS,
                user_input=user_input,
                schema_model=EvidenceAudit,
                schema_name="evidence_audit",
                reasoning_effort=self.settings.grok_evidence_qa_reasoning_effort,
                tools=[{"type": "web_search"}, {"type": "x_search"}],
            )
        except Exception as exc:
            logger.warning(
                "Evidence QA failed match_id=%s provider=%s model=%s error_type=%s elapsed=%.1fs",
                match.id,
                adapter.provider,
                self.settings.grok_evidence_qa_model,
                type(exc).__name__,
                time.perf_counter() - started_at,
            )
            return evidence

        audited = self._apply_evidence_audit(evidence, audit)
        logger.info(
            "Evidence QA end match_id=%s quality=%s alerts=%d missing=%d stale=%d elapsed=%.1fs",
            match.id,
            audit.evidence_quality,
            len(audit.volatile_market_alerts),
            len(audit.missing_or_weak_anchors),
            len(audit.stale_or_conflicting_claims),
            time.perf_counter() - started_at,
        )
        return audited

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

    def _apply_evidence_audit(self, evidence: MatchEvidence, audit: EvidenceAudit) -> MatchEvidence:
        audit_lines = self._evidence_audit_lines(audit)
        if not audit_lines:
            return evidence

        return evidence.model_copy(
            update={
                "query_summary": (
                    f"{evidence.query_summary}\n"
                    f"[evidence_qa] {audit.overall_assessment}"
                ),
                "key_facts": [*evidence.key_facts, *audit_lines][:40],
                "evidence_quality": self._audit_adjusted_quality(
                    evidence.evidence_quality,
                    audit.evidence_quality,
                    audit,
                ),
            }
        )

    @staticmethod
    def _evidence_audit_lines(audit: EvidenceAudit) -> list[str]:
        lines = [f"Evidence QA: {audit.overall_assessment}"]
        lines.extend(f"Evidence QA missing/weak anchor: {item}" for item in audit.missing_or_weak_anchors[:6])
        lines.extend(
            f"Evidence QA stale/conflicting claim: {item}" for item in audit.stale_or_conflicting_claims[:4]
        )
        for alert in audit.volatile_market_alerts[:10]:
            question = f" ({alert.question})" if alert.question else ""
            family = f"{alert.market_family}: " if alert.market_family else ""
            recommendation = f" Recommendation: {alert.recommendation}" if alert.recommendation else ""
            lines.append(
                f"Evidence QA {alert.severity} alert{question}: {family}{alert.issue}{recommendation}"
            )
        lines.extend(f"Evidence QA source note: {item}" for item in audit.source_quality_notes[:4])
        lines.extend(
            f"Evidence QA forecaster caution: {item}" for item in audit.recommended_forecaster_cautions[:6]
        )
        return lines[:24]

    @staticmethod
    def _audit_adjusted_quality(
        current_quality: str,
        audit_quality: str,
        audit: EvidenceAudit,
    ) -> str:
        order = {"low": 0, "medium": 1, "high": 2}
        severe_alert = any(alert.severity == "high" for alert in audit.volatile_market_alerts)
        hard_problem = bool(audit.stale_or_conflicting_claims) or severe_alert
        if not hard_problem and audit_quality == "low":
            hard_problem = len(audit.missing_or_weak_anchors) >= 3
        if not hard_problem:
            return current_quality
        current_score = order.get(current_quality, 1)
        audit_score = order.get(audit_quality, 1)
        adjusted_score = min(current_score, audit_score)
        for quality, score in order.items():
            if score == adjusted_score:
                return quality
        return current_quality

    @staticmethod
    def _best_evidence_quality(values: Any) -> str:
        order = {"low": 0, "medium": 1, "high": 2}
        best = "low"
        for value in values:
            if order.get(value, 0) > order[best]:
                best = value
        return best

    async def _firecrawl_context(self, match: Match, markets: list[Market]) -> str:
        if (
            not self.firecrawl
            or not self.settings.use_firecrawl_retrieval
            or self.settings.firecrawl_search_queries <= 0
        ):
            return ""
        queries = self._firecrawl_queries(match, markets)[: self.settings.firecrawl_search_queries]
        contexts: list[str] = []
        total_credits = 0
        logger.info("Firecrawl evidence start match_id=%s queries=%d", match.id, len(queries))
        for index, query in enumerate(queries, start=1):
            logger.info(
                "Firecrawl evidence query start match_id=%s query=%d/%d",
                match.id,
                index,
                len(queries),
            )
            try:
                results, credits = await self.firecrawl.search(
                    query,
                    limit=self.settings.firecrawl_search_limit,
                    sources=("web",),
                    tbs="qdr:w,sbd:1",
                )
            except Exception as exc:
                logger.warning(
                    "Firecrawl evidence query failed match_id=%s query=%d/%d error_type=%s",
                    match.id,
                    index,
                    len(queries),
                    type(exc).__name__,
                )
                contexts.append(f"Firecrawl query failed for {query!r}: {exc}")
                continue
            total_credits += credits
            logger.info(
                "Firecrawl evidence query end match_id=%s query=%d/%d results=%d credits=%d",
                match.id,
                index,
                len(queries),
                len(results),
                credits,
            )
            rendered = "\n".join(result.compact() for result in results[: self.settings.firecrawl_search_limit])
            if rendered:
                contexts.append(f"Firecrawl query: {query}\n{rendered}")
        if not contexts:
            logger.info("Firecrawl evidence end match_id=%s contexts=0 credits=%d", match.id, total_credits)
            return ""
        logger.info(
            "Firecrawl evidence end match_id=%s contexts=%d credits=%d",
            match.id,
            len(contexts),
            total_credits,
        )
        return f"Firecrawl credits used: {total_credits}\n" + "\n\n".join(contexts)

    @staticmethod
    def _firecrawl_queries(match: Match, markets: list[Market]) -> list[str]:
        market_terms = " ".join(market.question for market in markets[:8])
        return [
            f"{match.name} confirmed lineup injuries suspensions team news weather",
            f"{match.name} odds preview goals cards corners shots player props {market_terms}",
            f"{match.name} soccer statistics xG team strength recent form base rates",
        ]

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
