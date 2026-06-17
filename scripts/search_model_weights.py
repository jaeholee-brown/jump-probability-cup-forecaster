from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from probability_cup_bot.calibration import brier, build_calibration_report, infer_outcome
from probability_cup_bot.config import DEFAULT_FORECAST_MODEL_WEIGHTS, load_settings
from probability_cup_bot.scoring import extremize, log_odds_mean, shrink_toward_half
from probability_cup_bot.sportspredict import SportsPredictClient


DEFAULT_MODELS = tuple(DEFAULT_FORECAST_MODEL_WEIGHTS)
RAW_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
CONSTRAINED_GRID = (0.5, 0.75, 1.0, 1.25, 1.5)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _market_family(question: str) -> str:
    text = question.lower()
    if "offside" in text:
        return "offsides"
    if "foul" in text:
        return "fouls"
    if "shot on target" in text or "shots on target" in text:
        return "shots-on-target"
    if "corner" in text:
        return "corners"
    if "card" in text:
        return "cards"
    if "penalty" in text or "red card" in text:
        return "penalty-red"
    if "score a goal" in text or "assist" in text:
        return "player-goal-assist"
    if "win the match" in text or "half-time" in text or "halftime" in text:
        return "result"
    if "goal" in text or "both teams score" in text:
        return "goals"
    return "other"


async def _fetch_results(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.results_json:
        return json.loads(Path(args.results_json).read_text())
    settings = load_settings(args.dotenv, force_dry_run=True)
    sp = SportsPredictClient(
        base_url=settings.sportspredict_base_url,
        api_key=settings.sportspredict_api_key,
        retry_attempts=settings.sportspredict_retry_attempts,
        retry_initial_seconds=settings.sportspredict_retry_initial_seconds,
        retry_max_seconds=settings.sportspredict_retry_max_seconds,
    )
    try:
        event = await sp.find_event(settings.event_title, settings.event_id)
        lobby = await sp.ensure_lobby(event.id)
        return await sp.list_results(lobby.id)
    finally:
        await sp.aclose()


def _records(
    *,
    history: dict[str, Any],
    results: list[dict[str, Any]],
    models: tuple[str, ...],
) -> list[dict[str, Any]]:
    result_by_market = {result.get("market_id"): result for result in results}
    records: list[dict[str, Any]] = []
    for market_id, row in (history.get("markets") or {}).items():
        result = result_by_market.get(market_id)
        if not result:
            continue
        outcome = infer_outcome(result)
        if outcome is None:
            continue
        components = [
            component
            for component in row.get("components") or []
            if component.get("model") in models and component.get("probability") is not None
        ]
        if not components:
            continue
        question = result.get("question") or row.get("question") or ""
        records.append(
            {
                "market_id": market_id,
                "match_id": row.get("match_id") or "",
                "question": question,
                "family": _market_family(question),
                "outcome": outcome,
                "components": components,
            }
        )
    return records


def _aggregate_probability(
    components: list[dict[str, Any]],
    weights: dict[str, float],
    *,
    extremize_alpha: float,
    base_shrinkage: float,
) -> float | None:
    probabilities: list[float] = []
    component_weights: list[float] = []
    for component in components:
        model = str(component.get("model") or "")
        weight = weights.get(model, 0.0)
        if weight <= 0:
            continue
        probabilities.append(float(component["probability"]))
        component_weights.append(weight)
    if not probabilities:
        return None
    return shrink_toward_half(
        extremize(log_odds_mean(probabilities, component_weights), extremize_alpha),
        base_shrinkage,
    )


def _score(
    records: list[dict[str, Any]],
    weights: dict[str, float],
    *,
    extremize_alpha: float,
    base_shrinkage: float,
) -> dict[str, Any]:
    scores: list[float] = []
    by_family: dict[str, list[float]] = defaultdict(list)
    by_match: dict[str, list[float]] = defaultdict(list)
    for record in records:
        probability = _aggregate_probability(
            record["components"],
            weights,
            extremize_alpha=extremize_alpha,
            base_shrinkage=base_shrinkage,
        )
        if probability is None:
            continue
        score = brier(probability, record["outcome"])
        scores.append(score)
        by_family[record["family"]].append(score)
        by_match[record["match_id"]].append(score)
    return {
        "count": len(scores),
        "mean_brier": _mean(scores),
        "by_family": {
            key: {"count": len(values), "mean_brier": _mean(values)}
            for key, values in sorted(by_family.items())
        },
        "by_match": {
            key: {"count": len(values), "mean_brier": _mean(values)}
            for key, values in sorted(by_match.items())
        },
    }


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    mean_weight = sum(weights.values()) / max(len(weights), 1)
    if mean_weight <= 0:
        return dict(weights)
    return {model: round(weight / mean_weight, 4) for model, weight in weights.items()}


def _top_weight_sets(
    *,
    records: list[dict[str, Any]],
    models: tuple[str, ...],
    grid: tuple[float, ...],
    constrained: bool,
    extremize_alpha: float,
    base_shrinkage: float,
) -> list[dict[str, Any]]:
    rows: list[tuple[float, float, dict[str, float], tuple[float, ...], dict[str, Any]]] = []
    for multipliers in itertools.product(grid, repeat=len(models)):
        if not any(multipliers):
            continue
        weights = {
            model: DEFAULT_FORECAST_MODEL_WEIGHTS.get(model, 1.0) * multiplier
            for model, multiplier in zip(models, multipliers, strict=True)
        }
        score = _score(
            records,
            weights,
            extremize_alpha=extremize_alpha,
            base_shrinkage=base_shrinkage,
        )
        if score["count"] != len(records):
            continue
        penalty = 0.0
        if constrained:
            penalty = 0.002 * sum(math.log(multiplier) ** 2 for multiplier in multipliers) / len(multipliers)
        rows.append((float(score["mean_brier"]) + penalty, float(score["mean_brier"]), weights, multipliers, score))
    rows.sort(key=lambda row: row[0])
    return [
        {
            "rank": index + 1,
            "objective": objective,
            "mean_brier": mean_brier,
            "weights": weights,
            "normalized_weights": _normalize(weights),
            "multipliers": dict(zip(models, multipliers, strict=True)),
        }
        for index, (objective, mean_brier, weights, multipliers, _score_row) in enumerate(rows[:10])
    ]


def _component_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_model: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for component in record["components"]:
            by_model[str(component.get("model") or "unknown")].append(
                brier(float(component["probability"]), record["outcome"])
            )
    return {
        model: {"count": len(scores), "mean_brier": _mean(scores)}
        for model, scores in sorted(by_model.items())
    }


def build_report(args: argparse.Namespace, results: list[dict[str, Any]]) -> dict[str, Any]:
    history = json.loads(Path(args.history).read_text())
    models = tuple(args.models.split(",")) if args.models else DEFAULT_MODELS
    records = _records(history=history, results=results, models=models)
    calibration = build_calibration_report(
        results=results,
        history=history,
        learning_rate=args.calibration_learning_rate,
        prior_count=args.calibration_prior_count,
    )
    default_weights = {model: DEFAULT_FORECAST_MODEL_WEIGHTS.get(model, 1.0) for model in models}
    family_counts: dict[str, int] = defaultdict(int)
    for record in records:
        family_counts[record["family"]] += 1
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "history": str(Path(args.history).resolve()),
        "record_count": len(records),
        "match_count": len({record["match_id"] for record in records}),
        "family_counts": dict(sorted(family_counts.items())),
        "component_summary": _component_summary(records),
        "calibration_report": {
            key: calibration[key]
            for key in ("settled_market_count", "aggregate", "models", "providers", "suggested_multipliers")
        },
        "default_weights": default_weights,
        "default_score": _score(
            records,
            default_weights,
            extremize_alpha=args.extremize_alpha,
            base_shrinkage=args.base_shrinkage,
        ),
        "raw_best_top10": _top_weight_sets(
            records=records,
            models=models,
            grid=RAW_GRID,
            constrained=False,
            extremize_alpha=args.extremize_alpha,
            base_shrinkage=args.base_shrinkage,
        ),
        "constrained_best_top10": _top_weight_sets(
            records=records,
            models=models,
            grid=CONSTRAINED_GRID,
            constrained=True,
            extremize_alpha=args.extremize_alpha,
            base_shrinkage=args.base_shrinkage,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search model weights from settled component forecasts.")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--history", default="state/forecast-history.json")
    parser.add_argument("--results-json", default="", help="Optional saved /results JSON instead of live API fetch.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--output", default=f"logs/weight-search-{_timestamp()}.json")
    parser.add_argument("--extremize-alpha", type=float, default=1.05)
    parser.add_argument("--base-shrinkage", type=float, default=0.04)
    parser.add_argument("--calibration-learning-rate", type=float, default=1.8)
    parser.add_argument("--calibration-prior-count", type=int, default=20)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results = asyncio.run(_fetch_results(args))
    report = build_report(args, results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "record_count": report["record_count"],
                "component_summary": report["component_summary"],
                "default_score": report["default_score"],
                "best_constrained": report["constrained_best_top10"][0]
                if report["constrained_best_top10"]
                else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
