from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from probability_cup_bot.models import utcnow


def infer_outcome(result: dict[str, Any]) -> int | None:
    probability = result.get("probability_submitted")
    brier = result.get("brier_score")
    if probability is None or brier is None:
        return None
    p = float(probability)
    if p > 1:
        p /= 100
    error_if_yes = (p - 1.0) ** 2
    error_if_no = p**2
    return 1 if abs(error_if_yes - float(brier)) <= abs(error_if_no - float(brier)) else 0


def brier(probability: float, outcome: int) -> float:
    p = max(0.01, min(0.99, probability))
    return (p - outcome) ** 2


def build_calibration_report(
    *,
    results: list[dict[str, Any]],
    history: dict[str, Any],
    current_multipliers: dict[str, float] | None = None,
    learning_rate: float = 1.8,
    prior_count: int = 20,
) -> dict[str, Any]:
    current_multipliers = current_multipliers or {}
    market_history = history.get("markets") or {}
    settled_records: list[dict[str, Any]] = []
    provider_scores: dict[str, list[float]] = defaultdict(list)
    model_scores: dict[str, list[float]] = defaultdict(list)
    aggregate_scores: list[float] = []

    for result in results:
        market_id = result.get("market_id")
        if not market_id or market_id not in market_history:
            continue
        outcome = infer_outcome(result)
        if outcome is None:
            continue
        hist = market_history[market_id]
        aggregate_brier = float(result.get("brier_score"))
        aggregate_scores.append(aggregate_brier)
        components = hist.get("components") or []
        component_rows: list[dict[str, Any]] = []
        for component in components:
            probability = float(component.get("probability", 0.5))
            score = brier(probability, outcome)
            model = str(component.get("model") or "unknown")
            provider = str(component.get("provider") or "unknown")
            model_scores[model].append(score)
            provider_scores[provider].append(score)
            component_rows.append(
                {
                    "model": model,
                    "provider": provider,
                    "variant": component.get("variant"),
                    "probability": probability,
                    "weight": component.get("weight"),
                    "brier": score,
                }
            )
        settled_records.append(
            {
                "market_id": market_id,
                "question": result.get("question") or hist.get("question"),
                "outcome": outcome,
                "probability_submitted": result.get("probability_submitted"),
                "aggregate_brier": aggregate_brier,
                "components": component_rows,
            }
        )

    model_summary = _summary_table(model_scores)
    provider_summary = _summary_table(provider_scores)
    model_mean = _weighted_reference_mean(model_summary)
    suggested_multipliers = _suggest_multipliers(
        model_summary=model_summary,
        reference_mean=model_mean,
        current_multipliers=current_multipliers,
        learning_rate=learning_rate,
        prior_count=prior_count,
    )
    return {
        "generated_at": utcnow().isoformat(),
        "settled_market_count": len(settled_records),
        "aggregate": {
            "count": len(aggregate_scores),
            "mean_brier": _mean(aggregate_scores),
        },
        "providers": provider_summary,
        "models": model_summary,
        "current_multipliers": current_multipliers,
        "suggested_multipliers": suggested_multipliers,
        "settled_records": settled_records,
        "method": (
            "Model suggestions use exponentially weighted Brier regret versus the settled model "
            "mean, shrunk by n/(n+prior_count). This is conservative until each model has enough "
            "settled markets."
        ),
    }


def _summary_table(scores_by_key: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "count": len(scores),
            "mean_brier": _mean(scores),
        }
        for key, scores in sorted(scores_by_key.items())
        if scores
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _weighted_reference_mean(summary: dict[str, dict[str, float]]) -> float | None:
    total_count = sum(int(row["count"]) for row in summary.values())
    if total_count <= 0:
        return None
    return sum(float(row["mean_brier"]) * int(row["count"]) for row in summary.values()) / total_count


def _suggest_multipliers(
    *,
    model_summary: dict[str, dict[str, float]],
    reference_mean: float | None,
    current_multipliers: dict[str, float],
    learning_rate: float,
    prior_count: int,
) -> dict[str, float]:
    if reference_mean is None:
        return current_multipliers
    suggestions: dict[str, float] = {}
    for model, row in model_summary.items():
        count = int(row["count"])
        mean_brier = float(row["mean_brier"])
        shrink = count / (count + max(1, prior_count))
        regret = mean_brier - reference_mean
        multiplier = current_multipliers.get(model, 1.0) * math.exp(-learning_rate * shrink * regret)
        suggestions[model] = round(max(0.5, min(1.5, multiplier)), 4)
    return suggestions
