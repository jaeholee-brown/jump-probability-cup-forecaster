from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from probability_cup_bot.market_analysis import classify_market_family
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
    family_correction_prior_n: float = 12.0,
    family_correction_damp: float = 0.9,
    family_correction_min_settled: int = 150,
    family_correction_max_shift: float = 0.6,
) -> dict[str, Any]:
    current_multipliers = current_multipliers or {}
    market_history = history.get("markets") or {}
    settled_records: list[dict[str, Any]] = []
    provider_scores: dict[str, list[float]] = defaultdict(list)
    model_scores: dict[str, list[float]] = defaultdict(list)
    family_scores: dict[str, list[float]] = defaultdict(list)
    model_family_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    provider_family_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    aggregate_scores: list[float] = []

    for result in results:
        market_id = result.get("market_id")
        if not market_id or market_id not in market_history:
            continue
        outcome = infer_outcome(result)
        if outcome is None:
            continue
        hist = market_history[market_id]
        question = result.get("question") or hist.get("question") or ""
        market_family = str(hist.get("market_family") or classify_market_family(question))
        aggregate_brier = float(result.get("brier_score"))
        aggregate_scores.append(aggregate_brier)
        family_scores[market_family].append(aggregate_brier)
        components = hist.get("components") or []
        component_rows: list[dict[str, Any]] = []
        for component in components:
            probability = float(component.get("probability", 0.5))
            score = brier(probability, outcome)
            model = str(component.get("model") or "unknown")
            provider = str(component.get("provider") or "unknown")
            model_scores[model].append(score)
            provider_scores[provider].append(score)
            model_family_scores[market_family][model].append(score)
            provider_family_scores[market_family][provider].append(score)
            component_rows.append(
                {
                    "model": model,
                    "provider": provider,
                    "market_family": market_family,
                    "variant": component.get("variant"),
                    "probability": probability,
                    "weight": component.get("weight"),
                    "brier": score,
                }
            )
        settled_records.append(
            {
                "market_id": market_id,
                "question": question,
                "market_family": market_family,
                "outcome": outcome,
                "probability_submitted": result.get("probability_submitted"),
                "aggregate_brier": aggregate_brier,
                "components": component_rows,
            }
        )

    model_summary = _summary_table(model_scores)
    provider_summary = _summary_table(provider_scores)
    family_summary = _summary_table(family_scores)
    model_by_family_summary = {
        family: _summary_table(scores_by_model)
        for family, scores_by_model in sorted(model_family_scores.items())
    }
    provider_by_family_summary = {
        family: _summary_table(scores_by_provider)
        for family, scores_by_provider in sorted(provider_family_scores.items())
    }
    model_mean = _weighted_reference_mean(model_summary)
    suggested_multipliers = _suggest_multipliers(
        model_summary=model_summary,
        reference_mean=model_mean,
        current_multipliers=current_multipliers,
        learning_rate=learning_rate,
        prior_count=prior_count,
    )
    family_corrections = build_family_corrections(
        settled_records,
        prior_n=family_correction_prior_n,
        damp=family_correction_damp,
        min_settled=family_correction_min_settled,
        max_shift=family_correction_max_shift,
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
        "market_families": family_summary,
        "models_by_family": model_by_family_summary,
        "providers_by_family": provider_by_family_summary,
        "current_multipliers": current_multipliers,
        "suggested_multipliers": suggested_multipliers,
        "family_corrections": family_corrections,
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
        # Stateless in the current multiplier: regret is already cumulative, so
        # chaining exp-updates onto the previous multiplier re-counts the same
        # settled markets every run and saturates at the clamps.
        multiplier = math.exp(-learning_rate * shrink * regret)
        suggestions[model] = round(max(0.5, min(1.5, multiplier)), 4)
    return suggestions


def _logit(p: float) -> float:
    p = max(0.01, min(0.99, p))
    return math.log(p / (1.0 - p))


def _inv_logit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _record_raw_probability(record: dict[str, Any]) -> float | None:
    """Equal-weight log-odds mean of saved component probabilities.

    This is the correction basis rather than the submitted probability so the
    fit stays consistent once corrected forecasts start settling.
    """
    components = record.get("components") or []
    probabilities = [float(c.get("probability", 0.5)) for c in components]
    if not probabilities:
        submitted = record.get("probability_submitted")
        if submitted is None:
            return None
        p = float(submitted)
        return p / 100.0 if p > 1 else p
    return _inv_logit(sum(_logit(p) for p in probabilities) / len(probabilities))


def build_family_corrections(
    settled_records: list[dict[str, Any]],
    *,
    prior_n: float = 12.0,
    damp: float = 0.9,
    min_settled: int = 150,
    max_shift: float = 0.6,
    min_family_n: int = 8,
) -> dict[str, Any]:
    """Fit per-family logit shifts plus a global slope on settled outcomes.

    Validated 2026-07-03 on 50/50, 60/40, 70/30, and 80/20 time-ordered folds:
    -0.006 to -0.013 Brier on every held-out split (t -1.5 to -3.4), with
    full-history fits beating recency-weighted ones. Replaces the old
    extremize/shrink pair, whose defaults cancelled each other out.
    """
    from probability_cup_bot.market_analysis import market_subtype

    rows: list[tuple[str, float, int]] = []
    for record in settled_records:
        praw = _record_raw_probability(record)
        outcome = record.get("outcome")
        family = record.get("market_family")
        if praw is None or outcome is None or not family:
            continue
        subtype = market_subtype(str(record.get("question") or ""))
        rows.append((f"{family}|{subtype}", praw, int(outcome)))
    if len(rows) < min_settled:
        return {
            "enabled": False,
            "reason": f"only {len(rows)} settled records; need {min_settled}",
            "settled_count": len(rows),
        }

    by_family: dict[str, list[tuple[float, int]]] = defaultdict(list)
    by_subtype: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for key, praw, outcome in rows:
        by_family[key.split("|", 1)[0]].append((praw, outcome))
        by_subtype[key].append((praw, outcome))

    def _group_stats(data: list[tuple[float, int]]) -> tuple[int, float, float]:
        n = len(data)
        return n, sum(p for p, _ in data) / n, sum(y for _, y in data) / n

    shifts: dict[str, float] = {}
    family_stats: dict[str, dict[str, float]] = {}
    for family, data in sorted(by_family.items()):
        n, mean_p, yes_rate = _group_stats(data)
        family_stats[family] = {
            "count": n,
            "yes_rate": round(yes_rate, 4),
            "mean_forecast": round(mean_p, 4),
        }
        if n < min_family_n:
            continue
        clamped_rate = max(0.03, min(0.97, yes_rate))
        raw_shift = _logit(clamped_rate) - _logit(mean_p)
        shrink = n / (n + max(1.0, prior_n))
        shift = raw_shift * shrink * damp
        shifts[family] = round(max(-max_shift, min(max_shift, shift)), 4)

    # Sub-type shifts shrink toward their family shift, which itself shrinks
    # toward zero — families pool structurally different questions (threshold
    # totals vs strictly-greater comparisons) whose biases can cancel.
    for key, data in sorted(by_subtype.items()):
        n, mean_p, yes_rate = _group_stats(data)
        family_stats[key] = {
            "count": n,
            "yes_rate": round(yes_rate, 4),
            "mean_forecast": round(mean_p, 4),
        }
        if n < min_family_n:
            continue
        parent = shifts.get(key.split("|", 1)[0], 0.0)
        clamped_rate = max(0.03, min(0.97, yes_rate))
        raw_shift = _logit(clamped_rate) - _logit(mean_p)
        shrink = n / (n + max(1.0, prior_n))
        shift = parent + (raw_shift * damp - parent) * shrink
        shifts[key] = round(max(-max_shift, min(max_shift, shift)), 4)

    intercept, slope = _fit_recalibration_slope(rows, shifts)
    return {
        "enabled": True,
        "basis": "equal_weight_component_log_odds_mean; shifts keyed family|subtype with family fallback",
        "settled_count": len(rows),
        "shifts": shifts,
        "intercept": round(max(-0.2, min(0.2, intercept)), 4),
        "slope": round(max(0.9, min(1.4, slope)), 4),
        "family_stats": family_stats,
    }


def lookup_shift(shifts: dict[str, float], family: str, subtype: str | None = None) -> float:
    """Most specific available shift: family|subtype, then family, then 0."""
    if subtype is not None:
        specific = shifts.get(f"{family}|{subtype}")
        if specific is not None:
            return specific
    if "|" in family:
        specific = shifts.get(family)
        if specific is not None:
            return specific
        family = family.split("|", 1)[0]
    return shifts.get(family, 0.0)


def _fit_recalibration_slope(
    rows: list[tuple[str, float, int]],
    shifts: dict[str, float],
) -> tuple[float, float]:
    """One-covariate logistic fit of outcome on shifted logit(p)."""
    intercept, slope = 0.0, 1.0
    for _ in range(60):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for family, praw, outcome in rows:
            x = _logit(praw) + lookup_shift(shifts, family)
            mu = _inv_logit(intercept + slope * x)
            g0 += outcome - mu
            g1 += (outcome - mu) * x
            w = mu * (1.0 - mu)
            h00 += w
            h01 += w * x
            h11 += w * x * x
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        d_intercept = (h11 * g0 - h01 * g1) / det
        d_slope = (-h01 * g0 + h00 * g1) / det
        intercept += d_intercept
        slope += d_slope
        if abs(d_intercept) + abs(d_slope) < 1e-10:
            break
    if not (math.isfinite(intercept) and math.isfinite(slope)):
        return 0.0, 1.0
    return intercept, slope
