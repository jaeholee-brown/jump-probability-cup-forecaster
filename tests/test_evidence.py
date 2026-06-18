from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from probability_cup_bot.config import Settings
from probability_cup_bot.evidence import EvidenceCollector
from probability_cup_bot.models import EvidenceAudit, EvidenceItem, Market, MarketEvidenceAlert, MarketMatch, Match, MatchEvidence


T = TypeVar("T", bound=BaseModel)


class FakeAdapter:
    provider = "xai"

    async def structured_response(
        self,
        *,
        model: str,
        instructions: str,
        user_input: str,
        schema_model: type[T],
        schema_name: str,
        reasoning_effort: str = "medium",
        tools: list[dict[str, Any]] | None = None,
    ) -> T:
        raise AssertionError("not called")


class FakeFirecrawl:
    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        sources: tuple[str, ...] = ("web", "news"),
        tbs: str | None = None,
        country: str = "US",
    ) -> tuple[list[Any], int]:
        class Result:
            def compact(self, max_chars: int = 1800) -> str:
                return f"- Result for {query}: Team news markdown"

        return [Result()], 7


class FakeResearchAndAuditAdapter:
    provider = "xai"

    def __init__(self) -> None:
        self.schema_names: list[str] = []

    async def structured_response(
        self,
        *,
        model: str,
        instructions: str,
        user_input: str,
        schema_model: type[T],
        schema_name: str,
        reasoning_effort: str = "medium",
        tools: list[dict[str, Any]] | None = None,
    ) -> T:
        self.schema_names.append(schema_name)
        if schema_model is MatchEvidence:
            return MatchEvidence(
                match_id="match",
                match_name="A vs B",
                generated_at="2026-06-16T00:00:00Z",
                query_summary="summary",
                key_facts=["A striker likely starts"],
                evidence_quality="high",
            )  # type: ignore[return-value]
        if schema_model is EvidenceAudit:
            return EvidenceAudit(
                match_id="match",
                match_name="A vs B",
                audited_at="2026-06-16T00:01:00Z",
                evidence_quality="medium",
                overall_assessment="Player prop anchors are incomplete.",
                missing_or_weak_anchors=["No denominator for striker shots-on-target rate."],
                stale_or_conflicting_claims=[],
                volatile_market_alerts=[
                    MarketEvidenceAlert(
                        market_id="market",
                        question="Will A striker have 1+ SOT?",
                        market_family="player shots",
                        issue="Start is supported but conditional SOT rate is missing.",
                        recommendation="Shrink toward broad player SOT base rate.",
                        severity="high",
                    )
                ],
                source_quality_notes=["Lineup source is current, stat source is weak."],
                recommended_forecaster_cautions=["Do not infer SOT probability from team favorite status."],
            )  # type: ignore[return-value]
        raise AssertionError(f"unexpected schema {schema_name}")


def test_merge_evidence_deduplicates_and_keeps_best_quality() -> None:
    collector = EvidenceCollector(
        Settings(
            sportspredict_api_key="sportspredict_test_key",
            openai_api_key="",
            xai_api_key="xai_test_key",
        ),
        openai=None,
        grok=FakeAdapter(),
    )
    match = Match(id="match", name="A vs B", closing_time="2026-06-20T12:00:00Z")
    evidence = collector._merge_evidence(
        match,
        odds_context="odds",
        evidences=[
            MatchEvidence(
                match_id="match",
                match_name="A vs B",
                generated_at="2026-06-16T00:00:00Z",
                query_summary="[overview] summary",
                key_facts=["A striker fit", "A striker fit"],
                evidence_quality="medium",
                items=[
                    EvidenceItem(
                        title="Team news",
                        url="https://example.com/news",
                        summary="A striker fit",
                        relevance=6,
                    )
                ],
            ),
            MatchEvidence(
                match_id="match",
                match_name="A vs B",
                generated_at="2026-06-16T00:01:00Z",
                query_summary="[late_news] summary",
                key_facts=["Weather is clear"],
                evidence_quality="high",
                items=[
                    EvidenceItem(
                        title="Team news duplicate",
                        url="https://example.com/news",
                        summary="Duplicate",
                        relevance=5,
                    )
                ],
            ),
        ],
    )

    assert evidence.evidence_quality == "high"
    assert evidence.key_facts == ["A striker fit", "Weather is clear"]
    assert len(evidence.items) == 1
    assert "Merged 2 research passes" in evidence.query_summary


async def test_firecrawl_context_renders_search_results() -> None:
    collector = EvidenceCollector(
        Settings(
            sportspredict_api_key="sportspredict_test_key",
            openai_api_key="",
            xai_api_key="xai_test_key",
            firecrawl_search_queries=1,
        ),
        openai=None,
        grok=FakeAdapter(),
        firecrawl=FakeFirecrawl(),
    )
    match = Match(id="match", name="A vs B", closing_time="2026-06-20T12:00:00Z")

    context = await collector._firecrawl_context(match, [])

    assert "Firecrawl credits used: 7" in context
    assert "Team news markdown" in context


async def test_collect_appends_grok_evidence_qa_notes() -> None:
    adapter = FakeResearchAndAuditAdapter()
    collector = EvidenceCollector(
        Settings(
            sportspredict_api_key="sportspredict_test_key",
            openai_api_key="",
            xai_api_key="xai_test_key",
            grok_research_passes=("overview",),
            use_grok_evidence_qa=True,
        ),
        openai=None,
        grok=adapter,
    )
    match = Match(id="match", name="A vs B", closing_time="2026-06-20T12:00:00Z")
    market = Market(
        id="market",
        question="Will A striker have 1+ SOT?",
        status="open",
        match=MarketMatch(id="match", name="A vs B", closing_time="2026-06-20T12:00:00Z"),
        lobby_id="lobby",
    )

    evidence = await collector.collect(match, [market], use_firecrawl=False)

    assert adapter.schema_names == ["match_evidence", "evidence_audit"]
    assert evidence.evidence_quality == "medium"
    assert "[evidence_qa] Player prop anchors are incomplete." in evidence.query_summary
    assert any("missing/weak anchor" in fact for fact in evidence.key_facts)
    assert any("player shots" in fact for fact in evidence.key_facts)
