from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from probability_cup_bot.config import Settings
from probability_cup_bot.evidence import EvidenceCollector
from probability_cup_bot.models import EvidenceItem, Match, MatchEvidence


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
