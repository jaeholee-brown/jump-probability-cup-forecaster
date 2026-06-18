from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from probability_cup_bot.calibration import brier, infer_outcome
from probability_cup_bot.config import DEFAULT_FORECAST_MODEL_WEIGHTS, load_settings
from probability_cup_bot.models import Market, Match, utcnow
from probability_cup_bot.scoring import extremize, log_odds_mean, shrink_toward_half
from probability_cup_bot.sportspredict import SportsPredictClient


DEFAULT_POST_CHANGE_AT = "2026-06-18T05:06:00+00:00"
ANCHOR_PATTERN = re.compile(
    r"\b(odds?|bookmaker|book|market[- ]?implied|implied|de-?vig|polymarket|line|price)\b",
    re.IGNORECASE,
)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _market_family(question: str) -> str:
    text = question.lower()
    if "penalty" in text or "red card" in text:
        return "penalty-red"
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
    if "score or assist" in text or "goal or assist" in text:
        return "player-goal-assist"
    if "score a goal" in text:
        return "player-goal"
    if "win the match" in text or "halftime" in text or "half-time" in text:
        return "result"
    if "goal" in text or "both teams score" in text or "total goals" in text:
        return "goals"
    return "other"


def _probability_bucket(probability_int: int | float) -> str:
    p = int(round(float(probability_int)))
    if p <= 20:
        return "01-20"
    if p <= 40:
        return "21-40"
    if p <= 50:
        return "41-50"
    if p <= 60:
        return "51-60"
    if p <= 80:
        return "61-80"
    return "81-99"


def _forecast_stage(row: dict[str, Any], post_change_at: datetime) -> str:
    updated = _parse_dt(row.get("last_model_forecast_at") or row.get("last_forecast_at"))
    if updated is None:
        return "platform-only"
    return "post-evidence-qa" if updated >= post_change_at else "pre-evidence-qa"


def _aggregate_probability(components: list[dict[str, Any]]) -> float | None:
    probabilities: list[float] = []
    weights: list[float] = []
    for component in components:
        model = str(component.get("model") or "")
        probability = component.get("probability")
        if model not in DEFAULT_FORECAST_MODEL_WEIGHTS or probability is None:
            continue
        probabilities.append(float(probability))
        weights.append(float(DEFAULT_FORECAST_MODEL_WEIGHTS[model]))
    if not probabilities:
        return None
    return shrink_toward_half(extremize(log_odds_mean(probabilities, weights), 1.05), 0.04)


def _anchor_text(component: dict[str, Any]) -> str:
    fields = [
        component.get("reference_class"),
        component.get("base_rate_rationale"),
        component.get("probability_rationale"),
        " ".join(component.get("yes_reasons") or []),
        " ".join(component.get("no_reasons") or []),
        component.get("calibration_notes"),
    ]
    return "\n".join(str(field or "") for field in fields)


def _has_market_anchor(row: dict[str, Any]) -> bool:
    return any(ANCHOR_PATTERN.search(_anchor_text(component)) for component in row.get("components") or [])


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "count": 0,
            "mean_brier": None,
            "mean_probability": None,
            "outcome_rate": None,
            "bias_probability_minus_outcome": None,
            "mean_abs_error": None,
        }
    probabilities = [float(record["probability"]) for record in records]
    outcomes = [int(record["outcome"]) for record in records]
    scores = [float(record["brier"]) for record in records]
    errors = [probability - outcome for probability, outcome in zip(probabilities, outcomes)]
    return {
        "count": len(records),
        "mean_brier": _round(_mean(scores)),
        "mean_probability": _round(_mean(probabilities)),
        "outcome_rate": _round(_mean([float(outcome) for outcome in outcomes])),
        "bias_probability_minus_outcome": _round(_mean(errors)),
        "mean_abs_error": _round(_mean([abs(error) for error in errors])),
    }


def _group_summary(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        value = record.get(key)
        bucket = "unknown" if value is None or value == "" else str(value)
        buckets[bucket].append(record)
    return {
        bucket: _summary(rows)
        for bucket, rows in sorted(
            buckets.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
    }


def _component_summary(component_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in component_records:
        buckets[str(record["model"])].append(record)
    return {
        model: _summary(rows)
        for model, rows in sorted(
            buckets.items(),
            key=lambda item: (_mean([float(row["brier"]) for row in item[1]]) or 999, item[0]),
        )
    }


def _worst(records: list[dict[str, Any]], *, reverse: bool = True, limit: int = 15) -> list[dict[str, Any]]:
    selected = sorted(records, key=lambda record: float(record["brier"]), reverse=reverse)[:limit]
    keys = [
        "match_name",
        "question",
        "family",
        "probability_int",
        "outcome",
        "brier",
        "stage",
        "component_count",
        "component_spread_points",
    ]
    return [{key: record.get(key) for key in keys} for record in selected]


async def _fetch_platform(args: argparse.Namespace) -> dict[str, Any]:
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
        matches = await sp.list_matches(event.id, lobby.id)
        markets = await sp.list_markets(lobby.id)
        predictions = await sp.list_predictions(lobby.id)
        results = await sp.list_results(lobby.id)
    finally:
        await sp.aclose()
    return {
        "event": event.model_dump(),
        "lobby": lobby.model_dump(),
        "matches": matches,
        "markets": markets,
        "predictions": predictions,
        "results": results,
    }


def _market_match_name(market: Market | None, matches_by_id: dict[str, Match]) -> str:
    if market is None:
        return ""
    match = matches_by_id.get(market.match.id)
    return match.name if match else market.match.name


def _history_match_name(history: dict[str, Any], match_id: str) -> str:
    return str(((history.get("matches") or {}).get(match_id) or {}).get("match_name") or "")


def _build_records(
    *,
    platform: dict[str, Any],
    history: dict[str, Any],
    post_change_at: datetime,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matches_by_id = {match.id: match for match in platform["matches"]}
    markets_by_id = {market.id: market for market in platform["markets"]}
    predictions_by_market = {prediction.market_id: prediction for prediction in platform["predictions"]}
    history_markets = history.get("markets") or {}

    records: list[dict[str, Any]] = []
    component_records: list[dict[str, Any]] = []
    for result in platform["results"]:
        market_id = result.get("market_id")
        if not market_id:
            continue
        outcome = infer_outcome(result)
        brier_score = result.get("brier_score")
        probability_submitted = result.get("probability_submitted")
        if outcome is None or brier_score is None or probability_submitted is None:
            continue
        market = markets_by_id.get(market_id)
        prediction = predictions_by_market.get(market_id)
        history_row = history_markets.get(market_id) or {}
        match_id = (market.match.id if market else history_row.get("match_id") or "")
        match_name = _market_match_name(market, matches_by_id) or _history_match_name(history, match_id)
        probability_int = int(round(float(probability_submitted)))
        probability = probability_int / 100.0
        components = history_row.get("components") or []
        aggregate_probability = _aggregate_probability(components)
        family = _market_family(result.get("question") or history_row.get("question") or "")
        row = {
            "market_id": market_id,
            "match_id": match_id,
            "match_name": match_name,
            "question": result.get("question") or (market.question if market else history_row.get("question") or ""),
            "family": family,
            "probability": probability,
            "probability_int": probability_int,
            "outcome": outcome,
            "brier": float(brier_score),
            "probability_bucket": _probability_bucket(probability_int),
            "market_status": result.get("market_status") or (prediction.market_status if prediction else ""),
            "prediction_created_date": result.get("created_date") or (prediction.created_date if prediction else ""),
            "prediction_updated_date": prediction.updated_date if prediction else None,
            "component_count": len(components),
            "component_spread_points": history_row.get("component_spread_points"),
            "component_aggregate_probability": aggregate_probability,
            "component_aggregate_brier": (
                brier(aggregate_probability, outcome) if aggregate_probability is not None else None
            ),
            "stage": _forecast_stage(history_row, post_change_at),
            "has_component_history": bool(components),
            "has_market_anchor_in_rationale": _has_market_anchor(history_row),
            "last_forecast_at": history_row.get("last_forecast_at"),
            "last_model_forecast_at": history_row.get("last_model_forecast_at"),
        }
        records.append(row)
        for component in components:
            probability_value = component.get("probability")
            if probability_value is None:
                continue
            component_probability = float(probability_value)
            component_records.append(
                {
                    "market_id": market_id,
                    "match_id": row["match_id"],
                    "match_name": row["match_name"],
                    "question": row["question"],
                    "family": family,
                    "model": component.get("model") or "unknown",
                    "provider": component.get("provider") or "unknown",
                    "variant": component.get("variant") or "",
                    "probability": component_probability,
                    "probability_int": round(component_probability * 100),
                    "outcome": outcome,
                    "brier": brier(component_probability, outcome),
                    "stage": row["stage"],
                    "has_market_anchor_in_rationale": bool(ANCHOR_PATTERN.search(_anchor_text(component))),
                    "base_rate": component.get("base_rate"),
                    "reference_class": component.get("reference_class"),
                    "probability_rationale": component.get("probability_rationale"),
                }
            )
    return records, component_records


def _calibration_bins(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return _group_summary(records, "probability_bucket")


def _disagreement_bins(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for record in records:
        spread = record.get("component_spread_points")
        if spread is None:
            bucket = "no-components"
        elif float(spread) < 8:
            bucket = "00-07"
        elif float(spread) < 15:
            bucket = "08-14"
        elif float(spread) < 25:
            bucket = "15-24"
        else:
            bucket = "25+"
        enriched.append({**record, "spread_bucket": bucket})
    return _group_summary(enriched, "spread_bucket")


def _candidate_findings(records: list[dict[str, Any]], component_records: list[dict[str, Any]]) -> list[str]:
    findings: list[str] = []
    overall = _summary(records)
    findings.append(
        f"Settled platform predictions: {overall['count']} markets, mean Brier {overall['mean_brier']}."
    )

    family_rows = _group_summary(records, "family")
    weak_families = [
        (family, row)
        for family, row in family_rows.items()
        if row["count"] >= 5 and row["mean_brier"] is not None
    ]
    for family, row in sorted(weak_families, key=lambda item: item[1]["mean_brier"], reverse=True)[:5]:
        findings.append(
            f"Family {family}: n={row['count']}, mean Brier {row['mean_brier']}, "
            f"mean p {row['mean_probability']} vs outcome rate {row['outcome_rate']} "
            f"(bias {row['bias_probability_minus_outcome']})."
        )

    stage_rows = _group_summary(records, "stage")
    for stage, row in stage_rows.items():
        findings.append(
            f"Stage {stage}: n={row['count']}, mean Brier {row['mean_brier']}, "
            f"bias {row['bias_probability_minus_outcome']}."
        )

    component_rows = _component_summary(component_records)
    if component_rows:
        best = min(component_rows.items(), key=lambda item: item[1]["mean_brier"] or 999)
        worst = max(component_rows.items(), key=lambda item: item[1]["mean_brier"] or -1)
        findings.append(
            f"Best component so far: {best[0]} n={best[1]['count']} mean Brier {best[1]['mean_brier']}; "
            f"worst: {worst[0]} n={worst[1]['count']} mean Brier {worst[1]['mean_brier']}."
        )

    anchor_rows = _group_summary(records, "has_market_anchor_in_rationale")
    if "True" in anchor_rows and "False" in anchor_rows:
        findings.append(
            "Markets with explicit odds/market-anchor language in component rationales: "
            f"n={anchor_rows['True']['count']} mean Brier {anchor_rows['True']['mean_brier']}; "
            f"without: n={anchor_rows['False']['count']} mean Brier {anchor_rows['False']['mean_brier']}."
        )
    return findings


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Forecast Audit",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Executive Findings",
        "",
    ]
    lines.extend(f"- {finding}" for finding in report["findings"])
    lines.extend(["", "## Platform Summary", ""])
    lines.append(_markdown_table(report["platform_summary"]))
    lines.extend(["", "## By Family", ""])
    lines.append(_markdown_summary_rows(report["by_family"]))
    lines.extend(["", "## Calibration Buckets", ""])
    lines.append(_markdown_summary_rows(report["calibration_bins"]))
    lines.extend(["", "## Component Scores", ""])
    lines.append(_markdown_summary_rows(report["component_summary"]))
    lines.extend(["", "## Worst Settled Markets", ""])
    lines.append(_markdown_record_rows(report["worst_markets"]))
    lines.extend(["", "## Best Settled Markets", ""])
    lines.append(_markdown_record_rows(report["best_markets"]))
    lines.extend(["", "## Data Notes", ""])
    lines.extend(f"- {note}" for note in report["data_notes"])
    lines.append("")
    return "\n".join(lines)


def _markdown_table(row: dict[str, Any]) -> str:
    return _markdown_rows([{"metric": key, "value": value} for key, value in row.items()])


def _markdown_summary_rows(rows: dict[str, dict[str, Any]]) -> str:
    output = [{"group": key, **value} for key, value in rows.items()]
    return _markdown_rows(output)


def _markdown_record_rows(rows: list[dict[str, Any]]) -> str:
    return _markdown_rows(rows)


def _markdown_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    keys = list(rows[0].keys())
    header = "| " + " | ".join(keys) + " |"
    separator = "| " + " | ".join("---" for _ in keys) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_cell(row.get(key)) for key in keys) + " |")
    return "\n".join([header, separator, *body])


def _cell(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("|", "\\|")
    return text[:260]


def build_report(args: argparse.Namespace, platform: dict[str, Any]) -> dict[str, Any]:
    history = json.loads(Path(args.history).read_text()) if args.history else {"markets": {}, "matches": {}}
    post_change_at = _parse_dt(args.post_change_at) or _parse_dt(DEFAULT_POST_CHANGE_AT)
    assert post_change_at is not None
    records, component_records = _build_records(
        platform=platform,
        history=history,
        post_change_at=post_change_at,
    )

    open_predictions = [
        prediction
        for prediction in platform["predictions"]
        if not prediction.market_status or prediction.market_status == "open"
    ]
    report = {
        "generated_at": utcnow().isoformat(),
        "event": platform["event"],
        "lobby": platform["lobby"],
        "post_change_at": post_change_at.isoformat(),
        "platform_summary": {
            "matches_seen": len(platform["matches"]),
            "markets_seen": len(platform["markets"]),
            "predictions_seen": len(platform["predictions"]),
            "open_predictions": len(open_predictions),
            "results_seen": len(platform["results"]),
            "settled_scored_predictions": len(records),
            "history_markets": len((history.get("markets") or {})),
            "history_matches": len((history.get("matches") or {})),
            "component_scored_predictions": len([record for record in records if record["has_component_history"]]),
            "component_records": len(component_records),
        },
        "overall": _summary(records),
        "by_family": _group_summary(records, "family"),
        "by_match": _group_summary(records, "match_name"),
        "by_stage": _group_summary(records, "stage"),
        "calibration_bins": _calibration_bins(records),
        "disagreement_bins": _disagreement_bins(records),
        "market_anchor_bins": _group_summary(records, "has_market_anchor_in_rationale"),
        "component_summary": _component_summary(component_records),
        "component_by_family": {
            family: _component_summary([row for row in component_records if row["family"] == family])
            for family in sorted({row["family"] for row in component_records})
        },
        "worst_markets": _worst(records, reverse=True, limit=args.limit),
        "best_markets": _worst(records, reverse=False, limit=args.limit),
        "records": records,
        "component_records": component_records,
        "data_notes": [
            "Platform-level scoring uses SportsPredict /results and covers every settled submitted prediction returned by the API.",
            "Component/model scoring is available only for markets present in saved forecast-history.json.",
            "External odds are not available through the Jump API; this report flags explicit odds/market-anchor language in saved rationales as a proxy until a historical odds feed is attached.",
            "Post-change split uses forecast-history timestamps, not Git metadata inside artifacts.",
        ],
    }
    report["findings"] = _candidate_findings(records, component_records)
    return report


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"forecast-audit-{stamp}.json"
    md_path = output_dir / f"forecast-audit-{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_report_markdown(report), encoding="utf-8")
    latest_json = output_dir / "forecast-audit-latest.json"
    latest_md = output_dir / "forecast-audit-latest.md"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Probability Cup forecasts against settled results.")
    parser.add_argument("--history", default="state/forecast-history.json", help="forecast-history.json path.")
    parser.add_argument("--dotenv", default=None, help="Optional .env path.")
    parser.add_argument("--output-dir", default="reports", help="Directory for JSON and Markdown reports.")
    parser.add_argument("--post-change-at", default=DEFAULT_POST_CHANGE_AT, help="ISO timestamp for post-change split.")
    parser.add_argument("--limit", type=int, default=15, help="Worst/best markets to include in Markdown.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    platform = asyncio.run(_fetch_platform(args))
    report = build_report(args, platform)
    json_path, md_path = write_report(report, Path(args.output_dir))
    print(
        json.dumps(
            {
                "json": str(json_path.resolve()),
                "markdown": str(md_path.resolve()),
                "settled_scored_predictions": report["platform_summary"]["settled_scored_predictions"],
                "component_scored_predictions": report["platform_summary"]["component_scored_predictions"],
                "overall": report["overall"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
