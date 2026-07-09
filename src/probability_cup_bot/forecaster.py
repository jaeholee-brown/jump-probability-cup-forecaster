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
from probability_cup_bot.market_analysis import profile_markets
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
    # Independent specs research the match themselves via server-side search
    # and are deliberately not given the shared evidence pack, so their errors
    # decorrelate from the pack-fed models (which sit at 0.99 correlation).
    independent: bool = False


class MatchForecaster:
    def __init__(
        self,
        settings: Settings,
        openai: OpenAIAdapter | None = None,
        grok: OpenAIAdapter | None = None,
        anthropic: AnthropicAdapter | None = None,
        calibration_multipliers: dict[str, float] | None = None,
        family_corrections: dict[str, Any] | None = None,
    ) -> None:
        self.settings = settings
        self.openai = openai
        self.grok = grok
        self.anthropic = anthropic
        self.calibration_multipliers = calibration_multipliers or {}
        self.family_corrections = family_corrections or {}

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
                independent=spec.independent,
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
        independent: bool = False,
    ) -> ForecastBatch:
        instructions = f"{FORECASTING_INSTRUCTIONS}\n\nVariant emphasis:\n{variant_instruction}"
        if independent:
            instructions += (
                "\n\nIndependent research mode: you have deliberately NOT been given the shared "
                "research pack that other ensemble members see. Use your web and X search tools to "
                "research this match yourself before forecasting: confirmed or probable lineups, "
                "injuries, suspensions, team form, referee tendencies, weather, and market-relevant "
                "statistical rates. The odds context and tournament-to-date rates provided are "
                "objective anchors you should still use. Corroborate social-media claims; discount "
                "fan speculation. You are forecasting BEFORE the match kicks off: if search results "
                "appear to show this match already played or resolved, they are mismatched fixtures, "
                "wrong dates, or unreliable pages - ignore them and forecast the upcoming match."
            )
        profiles = profile_markets(markets)
        if independent:
            evidence_payload: dict[str, Any] = {
                "odds_context": evidence.odds_context,
                "note": "Shared research pack withheld; research independently with your search tools.",
            }
        else:
            evidence_payload = evidence.model_dump()
        user_input = json.dumps(
            {
                "match": match.model_dump(),
                "markets": [
                    {
                        "id": market.id,
                        "question": market.question,
                        "status": market.status,
                        "closing_time": market.match.closing_time,
                        "profile": profiles[market.id].model_payload(),
                        "tournament_to_date": self._tournament_context(
                            profiles[market.id].family, market.question
                        ),
                    }
                    for market in markets
                ],
                "market_profiles": [profiles[market.id].model_payload() for market in markets],
                "evidence": evidence_payload,
                "output_requirements": (
                    "Return exactly one forecast for every market id. Decimals must be between "
                    "0.01 and 0.99. Use enough structured audit detail to justify the number. "
                    "For decomposable markets, show the participation/conditional/joint or union "
                    "calculation in probability_rationale. For favorite, noisy, or correlated prop "
                    "markets, include the overconfidence/correlation check. End each "
                    "probability_rationale with 'Final probability: 0.xx' and make the probability "
                    "field exactly match that value."
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
        if (
            self.settings.use_grok_independent_forecast
            and self.grok
            and self.settings.grok_independent_forecast_model
        ):
            model = self.settings.grok_independent_forecast_model
            specs.append(
                ForecastModelSpec(
                    name=f"{self._spec_name('grok', model)}_independent",
                    provider="grok",
                    adapter=self.grok,
                    model=model,
                    variants=self.settings.grok_forecast_variants,
                    weight=self._model_weight(model, self.settings.grok_forecast_weight),
                    tools=[{"type": "web_search"}, {"type": "x_search"}],
                    independent=True,
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
        profiles = profile_markets(markets)
        by_market: dict[str, list[tuple[ForecastBatch, MarketForecast]]] = defaultdict(list)
        for batch in batches:
            for forecast in batch.forecasts:
                by_market[forecast.market_id].append((batch, forecast))

        output: list[AggregatedForecast] = []
        for market in markets:
            profile = profiles[market.id]
            components = by_market.get(market.id, [])
            if not components:
                continue
            components, dropped_independent = self._drop_divergent_independent(market.id, components)
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

            p_raw = log_odds_mean(probabilities, weights)
            correction_note = None
            if self.family_corrections.get("enabled"):
                p, correction_note = self._apply_family_correction(
                    p_raw, profile.family, market.question
                )
                if evidence_quality == "low":
                    p = shrink_toward_half(p, self.settings.low_evidence_shrinkage)
            else:
                p = shrink_toward_half(
                    extremize(p_raw, self.settings.extremize_alpha),
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
                        "market_family": profile.family,
                        "market_profile": profile.model_payload(),
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
                        "family_correction": correction_note,
                        "dropped_independent_components": dropped_independent,
                    },
                )
            )
        if self.settings.enable_coherence_adjustments:
            self._apply_coherence_adjustments(output)
        return output

    _COMPARISON_PATTERN = re.compile(r"\bmore\b.+\bthan\b|\bfewer\b.+\bthan\b", re.IGNORECASE)

    def _tournament_context(self, family: str, question: str) -> dict[str, Any] | None:
        """Realized this-tournament rates for the market family, plus structural notes."""
        from probability_cup_bot.market_analysis import market_subtype

        all_stats = self.family_corrections.get("family_stats") or {}
        subtype = market_subtype(question)
        group = family
        stats = all_stats.get(f"{family}|{subtype}")
        if stats and int(stats.get("count", 0)) >= 8:
            group = f"{family} {subtype.replace('_', ' ')}"
        else:
            stats = all_stats.get(family)
        context: dict[str, Any] = {}
        if stats and int(stats.get("count", 0)) >= 8:
            context["family_settled_count"] = int(stats["count"])
            context["family_yes_rate"] = stats["yes_rate"]
            context["note"] = (
                f"Across this tournament so far, {group} markets on this platform resolved YES "
                f"{stats['yes_rate']:.0%} of the time (n={int(stats['count'])}). Platform thresholds "
                "are similar across matches, so treat this as a strong reference class and deviate "
                "only with concrete match-specific evidence."
            )
        if self._COMPARISON_PATTERN.search(question):
            context["comparison_note"] = (
                "Strictly-greater comparison: a tie resolves NO. For low-count stats (cards, "
                "goals, offsides, corners in a half), P(tie) is often 20-35%, so P(A beats B) "
                "for evenly matched sides is usually 33-40%, not 50%."
            )
        return context or None

    def _apply_family_correction(
        self, p_raw: float, family: str, question: str
    ) -> tuple[float, dict[str, Any]]:
        from probability_cup_bot.calibration import lookup_shift
        from probability_cup_bot.market_analysis import market_subtype
        from probability_cup_bot.scoring import clamp_probability, inv_logit, logit

        corrections = self.family_corrections
        subtype = market_subtype(question)
        shift = lookup_shift(corrections.get("shifts") or {}, family, subtype)
        intercept = float(corrections.get("intercept") or 0.0)
        slope = float(corrections.get("slope") or 1.0)
        corrected = clamp_probability(inv_logit(intercept + slope * (logit(p_raw) + shift)))
        note = {
            "family": family,
            "subtype": subtype,
            "shift": shift,
            "intercept": intercept,
            "slope": slope,
            "raw_probability": round(p_raw, 4),
            "corrected_probability": round(corrected, 4),
        }
        return corrected, note

    def _apply_coherence_adjustments(self, forecasts: list[AggregatedForecast]) -> None:
        if not forecasts:
            return
        min_delta = self.settings.coherence_min_adjustment_points / 100.0
        penalty_probability = max(
            (
                forecast.probability
                for forecast in forecasts
                if self._market_family(forecast) == "penalty"
            ),
            default=None,
        )
        if penalty_probability is not None:
            for forecast in forecasts:
                family = self._market_family(forecast)
                if family not in {"player_shot_on_target", "player_goal"}:
                    continue
                if not self._has_penalty_taker_signal(forecast):
                    continue
                fraction = (
                    self.settings.penalty_taker_sot_floor_fraction
                    if family == "player_shot_on_target"
                    else self.settings.penalty_taker_goal_floor_fraction
                )
                floor = max(0.01, min(0.99, penalty_probability * fraction))
                if floor - forecast.probability >= min_delta:
                    self._adjust_probability(
                        forecast,
                        floor,
                        reason=(
                            f"Raised to preserve explicit penalty-taker channel: penalty market "
                            f"{penalty_probability:.2f} x floor fraction {fraction:.2f}."
                        ),
                    )

        best_goal_by_subject: dict[str, AggregatedForecast] = {}
        best_sot_by_subject: dict[str, AggregatedForecast] = {}
        for forecast in forecasts:
            profile = self._market_profile(forecast)
            subject = str(profile.get("subject_key") or "")
            if not subject:
                continue
            family = self._market_family(forecast)
            if family == "player_goal":
                if subject not in best_goal_by_subject or forecast.probability > best_goal_by_subject[subject].probability:
                    best_goal_by_subject[subject] = forecast
            elif family == "player_shot_on_target":
                if subject not in best_sot_by_subject or forecast.probability > best_sot_by_subject[subject].probability:
                    best_sot_by_subject[subject] = forecast
        for subject, goal_forecast in best_goal_by_subject.items():
            sot_forecast = best_sot_by_subject.get(subject)
            if sot_forecast is None:
                continue
            if goal_forecast.probability - sot_forecast.probability >= min_delta:
                self._adjust_probability(
                    sot_forecast,
                    goal_forecast.probability,
                    reason=(
                        "Raised player shot-on-target probability because the same player's goal "
                        "probability was higher; a credited goal normally implies a shot on target."
                    ),
                )

    @staticmethod
    def _market_family(forecast: AggregatedForecast) -> str:
        return str((forecast.metadata or {}).get("market_family") or "")

    @staticmethod
    def _market_profile(forecast: AggregatedForecast) -> dict[str, Any]:
        profile = (forecast.metadata or {}).get("market_profile")
        return profile if isinstance(profile, dict) else {}

    @staticmethod
    def _has_penalty_taker_signal(forecast: AggregatedForecast) -> bool:
        fields: list[str] = [forecast.question]
        for key in (
            "reference_classes",
            "base_rate_rationales",
            "yes_reasons",
            "probability_rationales",
            "consistency_notes",
        ):
            value = (forecast.metadata or {}).get(key)
            fields.append(json.dumps(value, ensure_ascii=True) if value is not None else "")
        text = " ".join(fields).lower()
        penalty_terms = (
            "penalty taker",
            "takes penalties",
            "on penalties",
            "primary penalty",
            "first-choice penalty",
            "spot kick",
            "spot-kick",
            "penalty duty",
            "penalty duties",
        )
        plausible_terms = (
            "plausible penalty",
            "possible penalty taker",
            "likely penalty taker",
            "set-piece",
            "dead-ball",
        )
        return any(term in text for term in penalty_terms) or (
            "penalty" in text and any(term in text for term in plausible_terms)
        )

    @staticmethod
    def _adjust_probability(forecast: AggregatedForecast, probability: float, *, reason: str) -> None:
        old_probability = forecast.probability
        new_probability = max(0.01, min(0.99, probability))
        forecast.probability = new_probability
        forecast.probability_int = probability_to_int(new_probability)
        forecast.notes = f"{forecast.notes} Coherence adjusted {old_probability:.3f}->{new_probability:.3f}."
        adjustments = list((forecast.metadata or {}).get("coherence_adjustments") or [])
        adjustments.append(
            {
                "old_probability": old_probability,
                "new_probability": new_probability,
                "reason": reason,
            }
        )
        forecast.metadata["coherence_adjustments"] = adjustments
        logger.warning(
            "Coherence adjusted market_id=%s old=%.3f new=%.3f reason=%s",
            forecast.market_id,
            old_probability,
            new_probability,
            reason,
        )

    INDEPENDENT_MAX_DIVERGENCE_LOGITS = 3.0

    def _drop_divergent_independent(
        self,
        market_id: str,
        components: list[tuple[ForecastBatch, MarketForecast]],
    ) -> tuple[list[tuple[ForecastBatch, MarketForecast]], list[dict[str, Any]]]:
        """Drop independent-research components that wildly diverge from the rest.

        An independent searcher that latches onto mismatched fixtures or
        "already resolved" pages can emit a confident boundary probability
        that carries no forecast information but moves a log-odds ensemble by
        double-digit points. 3 logits still allows ~25x odds-ratio
        disagreement, so genuine contrarian reads survive.
        """
        from probability_cup_bot.scoring import logit

        independent = [
            (batch, forecast)
            for batch, forecast in components
            if "_independent_" in (batch.prompt_variant or "")
        ]
        others = [
            (batch, forecast)
            for batch, forecast in components
            if "_independent_" not in (batch.prompt_variant or "")
        ]
        if not independent or not others:
            return components, []
        reference = sum(logit(float(f.probability)) for _, f in others) / len(others)
        dropped: list[dict[str, Any]] = []
        kept = list(others)
        for batch, forecast in independent:
            divergence = logit(float(forecast.probability)) - reference
            if abs(divergence) > self.INDEPENDENT_MAX_DIVERGENCE_LOGITS:
                dropped.append(
                    {
                        "model": batch.model,
                        "probability": float(forecast.probability),
                        "divergence_logits": round(divergence, 3),
                    }
                )
                logger.warning(
                    "Dropped divergent independent component market_id=%s model=%s p=%.3f divergence=%.2f logits",
                    market_id,
                    batch.model,
                    float(forecast.probability),
                    divergence,
                )
            else:
                kept.append((batch, forecast))
        return (kept, dropped) if dropped else (components, [])

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
