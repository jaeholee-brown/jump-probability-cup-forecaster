from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from probability_cup_bot.config import load_settings  # noqa: E402
from probability_cup_bot.forecaster import MatchForecaster  # noqa: E402
from probability_cup_bot.scoring import extremize, log_odds_mean, probability_to_int, shrink_toward_half  # noqa: E402
from probability_cup_bot.sportspredict import SportsPredictClient  # noqa: E402


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _repair_components(forecast: dict[str, Any]) -> tuple[list[float], list[dict[str, Any]]]:
    metadata = forecast.get("metadata") or {}
    providers = metadata.get("providers") or []
    models = metadata.get("models") or []
    variants = metadata.get("variants") or []
    rationales = metadata.get("probability_rationales") or []
    components = [float(value) for value in forecast.get("component_probabilities") or []]
    repaired = list(components)
    repairs: list[dict[str, Any]] = []

    for index, probability in enumerate(components):
        provider = providers[index] if index < len(providers) else ""
        if provider != "grok" or 0.02 < probability < 0.98:
            continue
        rationale = rationales[index] if index < len(rationales) else ""
        recovered = MatchForecaster._extract_final_probability(rationale)
        if recovered is None or abs(recovered - probability) < 0.025:
            continue
        repaired[index] = recovered
        repairs.append(
            {
                "index": index,
                "provider": provider,
                "model": models[index] if index < len(models) else "",
                "variant": variants[index] if index < len(variants) else "",
                "old_probability": probability,
                "new_probability": recovered,
                "source": "probability_rationale",
                "rationale": rationale,
            }
        )
    return repaired, repairs


def _correct_forecast(forecast: dict[str, Any], run_settings: dict[str, Any]) -> dict[str, Any]:
    repaired_components, repairs = _repair_components(forecast)
    corrected = dict(forecast)
    metadata = dict(forecast.get("metadata") or {})
    old_probability = float(forecast.get("probability"))
    old_probability_int = int(forecast.get("probability_int"))

    if repaired_components:
        weights = metadata.get("weights") or [1.0] * len(repaired_components)
        if len(weights) != len(repaired_components):
            weights = [1.0] * len(repaired_components)
        alpha = float(run_settings.get("extremize_alpha", 1.0))
        base_shrinkage = float(run_settings.get("base_shrinkage", 0.0))
        low_evidence_shrinkage = float(run_settings.get("low_evidence_shrinkage", base_shrinkage))
        shrinkage = low_evidence_shrinkage if forecast.get("evidence_quality") == "low" else base_shrinkage
        probability = shrink_toward_half(
            extremize(log_odds_mean(repaired_components, [float(weight) for weight in weights]), alpha),
            shrinkage,
        )
    else:
        probability = old_probability

    metadata["raw_component_probabilities"] = forecast.get("component_probabilities") or []
    metadata["component_probability_repairs"] = repairs
    corrected["metadata"] = metadata
    corrected["component_probabilities"] = repaired_components
    corrected["probability"] = probability
    corrected["probability_int"] = probability_to_int(probability)
    corrected["repair_summary"] = {
        "old_probability": old_probability,
        "old_probability_int": old_probability_int,
        "new_probability": probability,
        "new_probability_int": corrected["probability_int"],
        "repair_count": len(repairs),
        "delta_points": corrected["probability_int"] - old_probability_int,
    }
    return corrected


def build_repair_report(artifact: Path, threshold: int) -> dict[str, Any]:
    run_log = json.loads(artifact.read_text(encoding="utf-8"))
    run_settings = run_log.get("settings") or {}
    corrected_forecasts = [
        _correct_forecast(forecast, run_settings) for forecast in run_log.get("forecasts") or []
    ]
    updates = []
    for forecast in corrected_forecasts:
        summary = forecast["repair_summary"]
        if abs(int(summary["delta_points"])) < threshold:
            continue
        updates.append(
            {
                "market_id": forecast["market_id"],
                "question": forecast["question"],
                "old_probability": summary["old_probability_int"],
                "probability": forecast["probability_int"],
                "delta_points": summary["delta_points"],
                "repair_count": summary["repair_count"],
                "component_probabilities": forecast["component_probabilities"],
                "raw_component_probabilities": forecast["metadata"].get("raw_component_probabilities"),
                "models": forecast["metadata"].get("models"),
            }
        )

    corrected_run = dict(run_log)
    corrected_run["forecasts"] = corrected_forecasts
    corrected_run["repair_generated_at"] = datetime.now(timezone.utc).isoformat()
    corrected_run["repair_source_artifact"] = str(artifact)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_artifact": str(artifact),
        "forecast_count": len(corrected_forecasts),
        "repair_count": sum(f["repair_summary"]["repair_count"] for f in corrected_forecasts),
        "updates_planned": len(updates),
        "threshold_points": threshold,
        "lobby_id": (run_log.get("lobby") or {}).get("id"),
        "updates": updates,
        "corrected_run": corrected_run,
    }


async def submit_updates(report: dict[str, Any]) -> dict[str, Any]:
    settings = load_settings()
    lobby_id = report.get("lobby_id")
    sp = SportsPredictClient(
        base_url=settings.sportspredict_base_url,
        api_key=settings.sportspredict_api_key,
        retry_attempts=settings.sportspredict_retry_attempts,
        retry_initial_seconds=settings.sportspredict_retry_initial_seconds,
        retry_max_seconds=settings.sportspredict_retry_max_seconds,
    )
    try:
        predictions = await sp.list_predictions(lobby_id)
        existing = {prediction.market_id: prediction for prediction in predictions}
        submitted = []
        skipped = []
        errors = []
        for index, update in enumerate(report["updates"], start=1):
            prediction = existing.get(update["market_id"])
            if prediction is None:
                skipped.append({**update, "reason": "no existing prediction found"})
                continue
            current = prediction.probability_int
            new = int(update["probability"])
            if prediction.market_status and prediction.market_status != "open":
                skipped.append({**update, "reason": f"market status is {prediction.market_status}"})
                continue
            if abs(new - current) < int(report["threshold_points"]):
                skipped.append({**update, "reason": f"current change {current}->{new} below threshold"})
                continue
            if index > 1 and settings.sportspredict_update_interval_seconds > 0:
                await asyncio.sleep(settings.sportspredict_update_interval_seconds)
            try:
                result = await sp.update_prediction(prediction.id, new)
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        **update,
                        "prediction_id": prediction.id,
                        "current_probability": current,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            submitted.append(
                {
                    **update,
                    "prediction_id": prediction.id,
                    "current_probability": current,
                    "submitted_probability": result.probability_int,
                }
            )
        return {
            "mode": "submitted" if not errors else "submitted_with_errors",
            "existing_prediction_count": len(predictions),
            "submitted_count": len(submitted),
            "skipped_count": len(skipped),
            "error_count": len(errors),
            "submitted": submitted,
            "skipped": skipped,
            "errors": errors,
        }
    finally:
        await sp.aclose()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair Grok boundary probabilities from an existing latest-run artifact."
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("logs") / f"repair-{_timestamp()}.json")
    parser.add_argument("--corrected-run-output", type=Path, default=None)
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_repair_report(args.artifact, args.threshold)
    if args.submit:
        report["submission_results"] = asyncio.run(submit_updates(report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.corrected_run_output:
        args.corrected_run_output.parent.mkdir(parents=True, exist_ok=True)
        args.corrected_run_output.write_text(
            json.dumps(report["corrected_run"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "forecast_count": report["forecast_count"],
                "repair_count": report["repair_count"],
                "updates_planned": report["updates_planned"],
                "submission": (
                    {
                        "mode": report["submission_results"]["mode"],
                        "submitted_count": report["submission_results"]["submitted_count"],
                        "skipped_count": report["submission_results"]["skipped_count"],
                        "error_count": report["submission_results"]["error_count"],
                    }
                    if report.get("submission_results")
                    else None
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
