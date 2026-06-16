from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from probability_cup_bot.anthropic_adapter import AnthropicAdapter
from probability_cup_bot.config import Settings
from probability_cup_bot.models import (
    AggregatedForecast,
    ForecastBatch,
    Market,
    Match,
    MatchEvidence,
)
from probability_cup_bot.openai_adapter import OpenAIAdapter
from probability_cup_bot.prompts import FORECASTING_INSTRUCTIONS, PROMPT_VARIANTS
from probability_cup_bot.scoring import aggregate_probabilities, probability_to_int


QUALITY_WEIGHT = {"low": 0.7, "medium": 1.0, "high": 1.25}
CONFIDENCE_WEIGHT = {"low": 0.85, "medium": 1.0, "high": 1.15}
T = TypeVar("T", bound=BaseModel)


class StructuredAdapter(Protocol):
    provider: str

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
        ...


@dataclass(frozen=True)
class ForecastModelSpec:
    name: str
    adapter: StructuredAdapter
    model: str
    variants: tuple[str, ...]
    tools: list[dict[str, str]] | None = None


class MatchForecaster:
    def __init__(
        self,
        settings: Settings,
        openai: OpenAIAdapter | None = None,
        grok: OpenAIAdapter | None = None,
        anthropic: AnthropicAdapter | None = None,
    ) -> None:
        self.settings = settings
        self.openai = openai
        self.grok = grok
        self.anthropic = anthropic

    async def forecast_match(
        self,
        *,
        match: Match,
        markets: list[Market],
        evidence: MatchEvidence,
    ) -> list[AggregatedForecast]:
        specs = self._forecast_model_specs()
        if not specs:
            raise RuntimeError("Set at least one forecast model API key before running forecasts.")
        tasks = [
            self._forecast_variant(
                match,
                markets,
                evidence,
                f"{spec.name}_{variant}",
                variant_instruction,
                adapter=spec.adapter,
                model=spec.model,
                tools=spec.tools,
            )
            for spec in specs
            for variant, variant_instruction in self._variant_items(spec)
        ]
        batches = await asyncio.gather(*tasks, return_exceptions=True)
        valid_batches = [batch for batch in batches if isinstance(batch, ForecastBatch)]
        if not valid_batches:
            raise RuntimeError(f"No valid forecasts produced for {match.name}")
        return self._aggregate(markets, valid_batches)

    async def _forecast_variant(
        self,
        match: Match,
        markets: list[Market],
        evidence: MatchEvidence,
        variant: str,
        variant_instruction: str,
        *,
        adapter: StructuredAdapter,
        model: str,
        tools: list[dict[str, str]] | None = None,
    ) -> ForecastBatch:
        instructions = f"{FORECASTING_INSTRUCTIONS}\n\nVariant emphasis:\n{variant_instruction}"
        user_input = json.dumps(
            {
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
                "evidence": evidence.model_dump(),
                "output_requirements": (
                    "Return exactly one forecast for every market id. Decimals must be between "
                    "0.01 and 0.99. Keep notes concise."
                ),
            },
            ensure_ascii=True,
        )
        batch = await adapter.structured_response(
            model=model,
            instructions=instructions,
            user_input=user_input,
            schema_model=ForecastBatch,
            schema_name="forecast_batch",
            reasoning_effort=self.settings.reasoning_effort,
            tools=tools,
        )
        batch.prompt_variant = variant
        batch.model = model
        return batch

    def _forecast_model_specs(self) -> list[ForecastModelSpec]:
        specs: list[ForecastModelSpec] = []
        if self.settings.use_openai_forecast and self.openai and self.openai.provider == "openai":
            specs.append(
                ForecastModelSpec(
                    name="openai",
                    adapter=self.openai,
                    model=self.settings.forecast_model,
                    variants=self.settings.openai_forecast_variants,
                )
            )
        if self.settings.use_grok_forecast and self.grok:
            specs.append(
                ForecastModelSpec(
                    name="grok",
                    adapter=self.grok,
                    model=self.settings.grok_forecast_model,
                    variants=self.settings.grok_forecast_variants,
                )
            )
        if self.settings.use_claude_forecast and self.anthropic:
            specs.append(
                ForecastModelSpec(
                    name="claude",
                    adapter=self.anthropic,
                    model=self.settings.claude_forecast_model,
                    variants=self.settings.claude_forecast_variants,
                )
            )
        return specs

    @staticmethod
    def _variant_items(spec: ForecastModelSpec) -> list[tuple[str, str]]:
        variant_names = tuple(PROMPT_VARIANTS) if spec.variants == ("all",) else spec.variants
        items = [(name, PROMPT_VARIANTS[name]) for name in variant_names if name in PROMPT_VARIANTS]
        if not items:
            available = ", ".join(PROMPT_VARIANTS)
            requested = ", ".join(spec.variants)
            raise ValueError(f"No valid prompt variants for {spec.name}: {requested}. Use one of: {available}")
        return items

    def _aggregate(
        self,
        markets: list[Market],
        batches: list[ForecastBatch],
    ) -> list[AggregatedForecast]:
        by_market: dict[str, list[tuple[ForecastBatch, float, str, str]]] = defaultdict(list)
        for batch in batches:
            for forecast in batch.forecasts:
                by_market[forecast.market_id].append(
                    (batch, forecast.probability, forecast.confidence, forecast.evidence_quality)
                )

        output: list[AggregatedForecast] = []
        for market in markets:
            components = by_market.get(market.id, [])
            if not components:
                continue
            probabilities = [p for _, p, _, _ in components]
            weights = [
                QUALITY_WEIGHT.get(quality, 1.0) * CONFIDENCE_WEIGHT.get(confidence, 1.0)
                for _, _, confidence, quality in components
            ]
            evidence_quality = self._mode([quality for _, _, _, quality in components])
            confidence = self._mode([confidence for _, _, confidence, _ in components])
            shrinkage = (
                self.settings.low_evidence_shrinkage
                if evidence_quality == "low"
                else self.settings.base_shrinkage
            )
            p = aggregate_probabilities(
                probabilities,
                alpha=self.settings.extremize_alpha,
                shrinkage=shrinkage,
            )
            # Recompute with explicit weights if there are only a few components. This keeps strong
            # evidence/confidence from being ignored while still using robust trimming for larger sets.
            if len(probabilities) < 5:
                from probability_cup_bot.scoring import extremize, log_odds_mean, shrink_toward_half

                p = shrink_toward_half(
                    extremize(log_odds_mean(probabilities, weights), self.settings.extremize_alpha),
                    shrinkage,
                )
            output.append(
                AggregatedForecast(
                    market_id=market.id,
                    question=market.question,
                    probability=p,
                    probability_int=probability_to_int(p),
                    component_probabilities=probabilities,
                    confidence=confidence,
                    evidence_quality=evidence_quality,
                    notes=f"Aggregated {len(components)} variants by log-odds mean.",
                    metadata={
                        "variants": [batch.prompt_variant for batch, _, _, _ in components],
                        "models": [batch.model for batch, _, _, _ in components],
                        "weights": weights,
                    },
                )
            )
        return output

    @staticmethod
    def _mode(values: list[str]) -> str:
        if not values:
            return "medium"
        order = {"low": 0, "medium": 1, "high": 2}
        counts = {value: values.count(value) for value in set(values)}
        return max(counts, key=lambda value: (counts[value], order.get(value, 1)))
