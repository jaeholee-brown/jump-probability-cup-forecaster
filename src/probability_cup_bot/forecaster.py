from __future__ import annotations

import asyncio
import json
from collections import defaultdict

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


class MatchForecaster:
    def __init__(self, settings: Settings, openai: OpenAIAdapter) -> None:
        self.settings = settings
        self.openai = openai

    async def forecast_match(
        self,
        *,
        match: Match,
        markets: list[Market],
        evidence: MatchEvidence,
    ) -> list[AggregatedForecast]:
        batches = await asyncio.gather(
            *[
                self._forecast_variant(match, markets, evidence, variant, variant_instruction)
                for variant, variant_instruction in PROMPT_VARIANTS.items()
            ],
            return_exceptions=True,
        )
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
        batch = await self.openai.structured_response(
            model=self.settings.forecast_model,
            instructions=instructions,
            user_input=user_input,
            schema_model=ForecastBatch,
            schema_name="forecast_batch",
            reasoning_effort=self.settings.reasoning_effort,
        )
        batch.prompt_variant = variant
        batch.model = self.settings.forecast_model
        return batch

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

