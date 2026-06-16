from __future__ import annotations

import math
from statistics import median


def clamp_probability(p: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, p))


def logit(p: float) -> float:
    p = clamp_probability(p)
    return math.log(p / (1.0 - p))


def inv_logit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def log_odds_mean(probabilities: list[float], weights: list[float] | None = None) -> float:
    if not probabilities:
        raise ValueError("probabilities cannot be empty")
    ps = [clamp_probability(p) for p in probabilities]
    if weights is None:
        weights = [1.0] * len(ps)
    if len(weights) != len(ps):
        raise ValueError("weights length must equal probabilities length")
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must have positive sum")
    return inv_logit(sum(w * logit(p) for p, w in zip(ps, weights, strict=True)) / total)


def extremize(p: float, alpha: float = 1.0) -> float:
    if alpha == 1.0:
        return clamp_probability(p)
    return clamp_probability(inv_logit(alpha * logit(p)))


def shrink_toward_half(p: float, shrinkage: float) -> float:
    shrinkage = max(0.0, min(1.0, shrinkage))
    return clamp_probability((1.0 - shrinkage) * p + shrinkage * 0.5)


def probability_to_int(p: float) -> int:
    return int(max(1, min(99, round(clamp_probability(p) * 100))))


def aggregate_probabilities(
    probabilities: list[float],
    *,
    alpha: float = 1.0,
    shrinkage: float = 0.0,
    robust: bool = True,
) -> float:
    if not probabilities:
        raise ValueError("probabilities cannot be empty")
    cleaned = [clamp_probability(p) for p in probabilities]
    if robust and len(cleaned) >= 5:
        med = median(cleaned)
        distances = sorted((abs(p - med), p) for p in cleaned)
        keep = [p for _, p in distances[: max(3, math.ceil(len(cleaned) * 0.8))]]
    else:
        keep = cleaned
    p = log_odds_mean(keep)
    p = extremize(p, alpha)
    return shrink_toward_half(p, shrinkage)

