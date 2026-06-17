from __future__ import annotations

import asyncio
import json
import logging
import re
import time
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
    MarketForecast,
    Match,
    MatchEvidence,
)
from probability_cup_bot.openai_adapter import OpenAIAdapter
from probability_cup_bot.prompts import FORECASTING_INSTRUCTIONS, PROMPT_VARIANTS
from probability_cup_bot.scoring import probability_to_int


logger = logging.getLogger(__name__)

QUALITY_WEIGHT = {"low": 0.7, "medium": 1.0, "high": 1.25}
CONFIDENCE_WEIGHT = {"low": 0.85, "medium": 1.0, "high": 1.15}
T = TypeVar("T", bound=BaseModel)
BOUNDARY_REPAIR_MODELS = {"grok"}


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
    provider: str
    adapter: StructuredAdapter
    model: str
    variants: tuple[str, ...]
    weight: float
    tools: list[dict[str, str]] | None = None


class MatchForecaster:
    def __init__(
        self,
        settings: Settings,
        openai: OpenAIAdapter | None = None,
        grok: OpenAIAdapter | None = None,
        anthropic: AnthropicAdapter | None = None,
        calibration_multipliers: dict[str, float] | None = None,
    ) -> None:
        self.settings = settings
        self.openai = openai
        self.grok = grok
        self.anthropic = anthropic
        self.calibration_multipliers = calibration_multipliers or {}

    async def forecast_match(
        self,
        *,
        match: Match,
        markets: list[Market],
        evidence: MatchEvidence,
    ) -> list[AggregatedForecast]:
        started_at = time.perf_counter()
        specs = self._forecast_model_specs()
        if not specs:
            raise RuntimeError("Set at least one forecast model API key before running forecasts.")

        call_specs = [
            (spec, variant, variant_instruction)
            for spec in specs
            for variant, variant_instruction in self._variant_items(spec)
        ]
        logger.info(
            "Forecast start match_id=%s match=%r markets=%d models=%d variant_calls=%d",
            match.id,
            " ".join(match.name.split())[:120],
            len(markets),
            len(specs),
            len(call_specs),
        )
        tasks = [
            self._forecast_variant(
                match,
                markets,
                evidence,
                f"{spec.name}_{variant}",
                variant_instruction,
                adapter=spec.adapter,
                model=spec.model,
                provider=spec.provider,
                weight=spec.weight,
                tools=spec.tools,
            )
            for spec, variant, variant_instruction in call_specs
        ]
        batches = await asyncio.gather(*tasks, return_exceptions=True)
        valid_batches = [batch for batch in batches if isinstance(batch, ForecastBatch)]
        failed_calls = sum(isinstance(batch, Exception) for batch in batches)
        if not valid_batches:
            logger.warning(
                "Forecast failed match_id=%s valid_batches=0 failed_calls=%d elapsed=%.1fs",
                match.id,
                failed_calls,
                time.perf_counter() - started_at,
            )
            raise RuntimeError(f"No valid forecasts produced for {match.name}")
        forecasts = self._aggregate(markets, valid_batches)
        logger.info(
            "Forecast end match_id=%s forecasts=%d valid_calls=%d failed_calls=%d elapsed=%.1fs",
            match.id,
            len(forecasts),
            len(valid_batches),
            failed_calls,
            time.perf_counter() - started_at,
        )
        return forecasts

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
        provider: str,
        weight: float,
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
                    "0.01 and 0.99. Keep notes concise: at most 2 short yes reasons, 2 short no "
                    "reasons, and 1-2 compact sentences for each rationale field."
                ),
            },
            ensure_ascii=True,
        )
        started_at = time.perf_counter()
        logger.info(
            "Forecast variant start match_id=%s provider=%s model=%s variant=%s markets=%d",
            match.id,
            provider,
            model,
            variant,
            len(markets),
        )
        try:
            batch = await adapter.structured_response(
                model=model,
                instructions=instructions,
                user_input=user_input,
                schema_model=ForecastBatch,
                schema_name="forecast_batch",
                reasoning_effort=self._reasoning_effort(provider),
                tools=tools,
            )
        except Exception as exc:
            logger.warning(
                "Forecast variant failed match_id=%s provider=%s model=%s variant=%s error_type=%s elapsed=%.1fs",
                match.id,
                provider,
                model,
                variant,
                type(exc).__name__,
                time.perf_counter() - started_at,
            )
            raise
        batch.prompt_variant = variant
        batch.model = model
        batch.provider = provider
        batch.weight = weight
        logger.info(
            "Forecast variant end match_id=%s provider=%s model=%s variant=%s forecasts=%d elapsed=%.1fs",
            match.id,
            provider,
            model,
            variant,
            len(batch.forecasts),
            time.perf_counter() - started_at,
        )
        return batch

    def _forecast_model_specs(self) -> list[ForecastModelSpec]:
        specs: list[ForecastModelSpec] = []
        if self.settings.use_openai_forecast and self.openai and self.openai.provider == "openai":
            specs.append(
                ForecastModelSpec(
                    name=self._spec_name("openai", self.settings.forecast_model),
                    provider="openai",
                    adapter=self.openai,
                    model=self.settings.forecast_model,
                    variants=self.settings.openai_forecast_variants,
                    weight=self._model_weight(self.settings.forecast_model, self.settings.openai_forecast_weight),
                )
            )
        if self.settings.use_grok_forecast and self.grok:
            for model in self._unique_models(self.settings.grok_forecast_models):
                specs.append(
                    ForecastModelSpec(
                        name=self._spec_name("grok", model),
                        provider="grok",
                        adapter=self.grok,
                        model=model,
                        variants=self.settings.grok_forecast_variants,
                        weight=self._model_weight(model, self.settings.grok_forecast_weight),
                    )
                )
        if self.settings.use_claude_forecast and self.anthropic:
            for model in self._unique_models(self.settings.claude_forecast_models):
                specs.append(
                    ForecastModelSpec(
                        name=self._spec_name("claude", model),
                        provider="claude",
                        adapter=self.anthropic,
                        model=model,
                        variants=self.settings.claude_forecast_variants,
                        weight=self._model_weight(model, self.settings.claude_forecast_weight),
                    )
                )
        return specs

    def _model_weight(self, model: str, provider_default: float) -> float:
        configured = (self.settings.forecast_model_weights or {}).get(model, provider_default)
        return configured * self.calibration_multipliers.get(model, 1.0)

    def _reasoning_effort(self, provider: str) -> str:
        if provider == "claude":
            return "none"
        return self.settings.reasoning_effort

    @staticmethod
    def _unique_models(models: tuple[str, ...]) -> tuple[str, ...]:
        seen: set[str] = set()
        unique: list[str] = []
        for model in models:
            if model and model not in seen:
                seen.add(model)
                unique.append(model)
        return tuple(unique)

    @staticmethod
    def _spec_name(provider: str, model: str) -> str:
        safe_model = model.replace(".", "_").replace("-", "_")
        return f"{provider}_{safe_model}"

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
        by_market: dict[str, list[tuple[ForecastBatch, MarketForecast]]] = defaultdict(list)
        for batch in batches:
            for forecast in batch.forecasts:
                by_market[forecast.market_id].append((batch, forecast))

        output: list[AggregatedForecast] = []
        for market in markets:
            components = by_market.get(market.id, [])
            if not components:
                continue
            probability_repairs = [
                self._repair_boundary_probability(batch, forecast) for batch, forecast in components
            ]
            probabilities = [repair["probability"] for repair in probability_repairs]
            weights = [
                batch.weight
                * QUALITY_WEIGHT.get(forecast.evidence_quality, 1.0)
                * CONFIDENCE_WEIGHT.get(forecast.confidence, 1.0)
                for batch, forecast in components
            ]
            evidence_quality = self._mode([forecast.evidence_quality for _, forecast in components])
            confidence = self._mode([forecast.confidence for _, forecast in components])
            shrinkage = (
                self.settings.low_evidence_shrinkage
                if evidence_quality == "low"
                else self.settings.base_shrinkage
            )
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
                        "variants": [batch.prompt_variant for batch, _ in components],
                        "models": [batch.model for batch, _ in components],
                        "providers": [batch.provider for batch, _ in components],
                        "weights": weights,
                        "resolution_interpretations": [
                            forecast.resolution_interpretation for _, forecast in components
                        ],
                        "reference_classes": [forecast.reference_class for _, forecast in components],
                        "base_rates": [forecast.base_rate for _, forecast in components],
                        "base_rate_rationales": [forecast.base_rate_rationale for _, forecast in components],
                        "yes_reasons": [forecast.yes_reasons for _, forecast in components],
                        "no_reasons": [forecast.no_reasons for _, forecast in components],
                        "probability_rationales": [
                            forecast.probability_rationale for _, forecast in components
                        ],
                        "calibration_notes": [forecast.calibration_notes for _, forecast in components],
                        "consistency_notes": [forecast.consistency_notes for _, forecast in components],
                        "raw_component_probabilities": [
                            repair["raw_probability"] for repair in probability_repairs
                        ],
                        "probability_repairs": [
                            repair for repair in probability_repairs if repair["repaired"]
                        ],
                    },
                )
            )
        return output

    @staticmethod
    def _repair_boundary_probability(batch: ForecastBatch, forecast: MarketForecast) -> dict[str, Any]:
        probability = float(forecast.probability)
        repair = {
            "provider": batch.provider,
            "model": batch.model,
            "variant": batch.prompt_variant,
            "raw_probability": probability,
            "probability": probability,
            "repaired": False,
            "source": "",
        }
        if batch.provider not in BOUNDARY_REPAIR_MODELS:
            return repair
        if 0.02 < probability < 0.98:
            return repair

        recovered = MatchForecaster._extract_final_probability(forecast.probability_rationale)
        if recovered is None:
            return repair
        if abs(recovered - probability) < 0.025:
            return repair

        repair.update(
            {
                "probability": recovered,
                "repaired": True,
                "source": "probability_rationale",
            }
        )
        logger.warning(
            "Repaired boundary probability provider=%s model=%s market_id=%s old=%.3f new=%.3f",
            batch.provider,
            batch.model,
            forecast.market_id,
            probability,
            recovered,
        )
        return repair

    @staticmethod
    def _extract_final_probability(text: str) -> float | None:
        candidates: list[tuple[int, float]] = []
        for match in re.finditer(r"(?<!\d)(?:0?\.\d{1,3}|1\.0+)(?!\d)", text or ""):
            probability = float(match.group(0))
            if 0.01 <= probability <= 0.99:
                candidates.append((match.start(), probability))
        for match in re.finditer(r"(?<!\d)([1-9]\d?)(?:\.\d+)?\s*%", text or ""):
            probability = float(match.group(0).replace("%", "")) / 100.0
            if 0.01 <= probability <= 0.99:
                candidates.append((match.start(), probability))
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: item[0])[-1][1]

    @staticmethod
    def _mode(values: list[str]) -> str:
        if not values:
            return "medium"
        order = {"low": 0, "medium": 1, "high": 2}
        counts = {value: values.count(value) for value in set(values)}
        return max(counts, key=lambda value: (counts[value], order.get(value, 1)))
