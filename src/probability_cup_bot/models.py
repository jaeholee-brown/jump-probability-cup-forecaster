from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Event(BaseModel):
    id: str
    title: str
    type: str
    status: str | None = None
    start_date: str | None = None
    end_date: str | None = None


class Lobby(BaseModel):
    id: str
    name: str
    type: str | None = None
    is_private: bool | None = None
    joined: bool = False


class Match(BaseModel):
    id: str
    name: str
    event_id: str | None = None
    opening_time: str | None = None
    closing_time: str | None = None
    open_market_count: int = 0

    @property
    def closes_at(self) -> datetime | None:
        return parse_dt(self.closing_time or self.opening_time)


class MarketMatch(BaseModel):
    id: str
    name: str
    opening_time: str | None = None
    closing_time: str | None = None


class Market(BaseModel):
    id: str
    question: str
    event_type: str | None = None
    status: str
    match: MarketMatch
    lobby_id: str

    @property
    def closes_at(self) -> datetime | None:
        return parse_dt(self.match.closing_time or self.match.opening_time)


class Prediction(BaseModel):
    id: str
    market_id: str
    lobby_id: str
    probability: int | float
    question: str | None = None
    market_status: str | None = None
    brier_score: float | None = None
    created_date: str | None = None
    updated_date: str | None = None

    @property
    def probability_int(self) -> int:
        probability = float(self.probability)
        if 0 <= probability <= 1:
            probability *= 100
        return int(round(probability))


class EvidenceItem(BaseModel):
    title: str
    url: str = ""
    source: str = ""
    published_at: str = ""
    summary: str
    relevance: int = Field(default=3, ge=1, le=6)


class MatchEvidence(BaseModel):
    match_id: str
    match_name: str
    generated_at: str
    query_summary: str
    key_facts: list[str] = Field(default_factory=list)
    odds_context: str = ""
    items: list[EvidenceItem] = Field(default_factory=list)
    evidence_quality: str = "medium"

    def compact(self) -> str:
        parts = [
            f"Match: {self.match_name}",
            f"Evidence generated at: {self.generated_at}",
            f"Evidence quality: {self.evidence_quality}",
            f"Search summary: {self.query_summary}",
        ]
        if self.odds_context:
            parts.append(f"Odds context: {self.odds_context}")
        if self.key_facts:
            parts.append("Key facts:\n" + "\n".join(f"- {fact}" for fact in self.key_facts))
        if self.items:
            rendered = []
            for item in self.items[:12]:
                source = f" ({item.source})" if item.source else ""
                url = f" {item.url}" if item.url else ""
                rendered.append(f"- {item.title}{source}: {item.summary}{url}")
            parts.append("Retrieved items:\n" + "\n".join(rendered))
        return "\n\n".join(parts)


class NewsSource(BaseModel):
    title: str = ""
    url: str = ""
    source: str = ""
    published_at: str = ""
    summary: str = ""


class NewsCheck(BaseModel):
    match_id: str
    match_name: str
    checked_at: str
    should_reforecast: bool
    estimated_delta_points: int = Field(default=0, ge=0, le=50)
    materiality: str = Field(pattern="^(none|low|medium|high)$")
    evidence_quality: str = Field(pattern="^(low|medium|high)$")
    reason: str
    summary: str
    new_developments: list[str] = Field(default_factory=list)
    sources: list[NewsSource] = Field(default_factory=list)


class MarketForecast(BaseModel):
    market_id: str
    question: str
    probability: float = Field(ge=0.01, le=0.99)
    confidence: str = Field(pattern="^(low|medium|high)$")
    evidence_quality: str = Field(pattern="^(low|medium|high)$")
    reference_class: str
    base_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    yes_reasons: list[str] = Field(default_factory=list)
    no_reasons: list[str] = Field(default_factory=list)
    calibration_notes: str = ""
    consistency_notes: str = ""


class ForecastBatch(BaseModel):
    match_id: str
    match_name: str
    model: str
    prompt_variant: str
    provider: str = ""
    weight: float = 1.0
    forecasts: list[MarketForecast]


class AggregatedForecast(BaseModel):
    market_id: str
    question: str
    probability: float
    probability_int: int
    component_probabilities: list[float]
    confidence: str
    evidence_quality: str
    notes: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
