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
EVIDENCE_LEVEL_ORDER = {
    "direct_market_line": 0,
    "direct_or_adjacent_stat": 1,
    "adjacent_match_market": 2,
    "match_context_only": 3,
    "no_external_source": 4,
}
LEGACY_MARKET_MATCHES = {
    "a42ddd0a-f256-4673-b105-14f678bba629": "FRA vs SEN",
    "02f1082e-cb12-43c6-aadb-0e8b89e66f49": "FRA vs SEN",
    "b86d03ef-3e9b-4029-bf1a-2a7706a53a86": "FRA vs SEN",
    "49c796d9-068f-4889-bfb8-ff99a884c0d0": "FRA vs SEN",
    "44edc25a-6d65-4602-a0f0-7319d4731fbd": "FRA vs SEN",
    "d09a87e1-f556-40d1-97c8-ecbe8bbb83ca": "FRA vs SEN",
    "6f0a2ca7-d1e8-4f11-ab6b-cce2e91448a2": "FRA vs SEN",
    "9c5459a1-4edc-429b-bdaf-1223d3764b5a": "FRA vs SEN",
    "b8fd5895-4f4d-4706-8bf8-8dbf72487c89": "FRA vs SEN",
    "fc032969-f65c-4e9f-913b-c21f5474b020": "FRA vs SEN",
}


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
        "external_evidence_level",
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


def _infer_legacy_match_name(market_id: str, question: str) -> str:
    if market_id in LEGACY_MARKET_MATCHES:
        return LEGACY_MARKET_MATCHES[market_id]
    text = question.lower()
    if "france" in text or "senegal" in text or "sadio" in text or "mane" in text:
        return "FRA vs SEN"
    return ""


def _match_key(match_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", match_name.lower())


def _load_external_sources(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    source_path = Path(path)
    if not source_path.exists():
        return {}
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    matches = raw.get("matches") if isinstance(raw, dict) and "matches" in raw else raw
    if not isinstance(matches, dict):
        raise ValueError(f"External source file must contain a match-name object: {source_path}")
    indexed: dict[str, dict[str, Any]] = {}
    for match_name, payload in matches.items():
        if not isinstance(payload, dict):
            continue
        entry = {**payload, "match_name": str(payload.get("match_name") or match_name)}
        names = {str(match_name), str(entry["match_name"])}
        names.update(str(alias) for alias in entry.get("aliases") or [])
        for name in names:
            indexed[_match_key(name)] = entry
    return indexed


def _external_for_match(match_name: str, external_sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not match_name:
        return {}
    return external_sources.get(_match_key(match_name)) or {}


def _external_counts(entry: dict[str, Any]) -> dict[str, int]:
    sources = entry.get("sources") or []
    type_counts: dict[str, int] = defaultdict(int)
    for source in sources:
        if isinstance(source, dict):
            type_counts[str(source.get("type") or "unknown")] += 1
    return {
        "external_source_count": len(sources),
        "external_odds_source_count": type_counts.get("odds", 0),
        "external_stats_source_count": type_counts.get("stats", 0),
        "external_recap_source_count": type_counts.get("recap", 0),
    }


def _external_fields(entry: dict[str, Any]) -> dict[str, Any]:
    counts = _external_counts(entry)
    urls = [
        str(source.get("url"))
        for source in entry.get("sources") or []
        if isinstance(source, dict) and source.get("url")
    ]
    return {
        **counts,
        "external_odds_summary": entry.get("odds_summary") or "",
        "external_result_summary": entry.get("result_summary") or "",
        "external_audit_notes": entry.get("audit_notes") or [],
        "external_source_urls": urls,
    }


def _external_source_facts(entry: dict[str, Any], *, source_types: set[str] | None = None) -> list[str]:
    facts: list[str] = []
    for source in entry.get("sources") or []:
        if not isinstance(source, dict):
            continue
        source_type = str(source.get("type") or "unknown")
        if source_types is not None and source_type not in source_types:
            continue
        for fact in source.get("facts") or []:
            facts.append(str(fact))
    return facts


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _matched_facts(facts: list[str], needles: list[str]) -> list[str]:
    matches = []
    for fact in facts:
        fact_text = fact.lower()
        if _contains_any(fact_text, needles):
            matches.append(fact)
    return matches[:5]


def _player_terms(question: str) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'’-]+", question):
        lower = token.lower().strip("'’")
        if lower not in {
            "will",
            "in",
            "at",
            "and",
            "or",
            "the",
            "first",
            "second",
            "half",
        }:
            terms.append(lower)
    return terms


def _external_evidence_for_record(question: str, family: str, entry: dict[str, Any]) -> dict[str, Any]:
    if not entry:
        return {
            "external_evidence_level": "no_external_source",
            "external_evidence_reason": "No configured external source matched this forecast's match.",
            "external_evidence_basis": [],
        }

    odds_or_stats_facts = _external_source_facts(entry, source_types={"odds", "stats"})
    context = "\n".join([entry.get("odds_summary") or "", *odds_or_stats_facts]).lower()
    question_text = question.lower()
    has_moneyline = _contains_any(context, ["moneyline", "favorite", "favourite", "favored", "favour"])
    has_total = _contains_any(context, ["total 2.5", "over 2.5", "under 2.5", "match total", "total went over"])
    has_btts = _contains_any(context, ["btts", "both teams score"])
    has_sot = _contains_any(context, ["shot-on-target", "shots-on-target", "shots on target", "sot"])
    has_shots = has_sot or _contains_any(context, [" shot ", " shots ", "shot and shot-on-target"])
    has_fouls = "foul" in context
    has_corners = "corner" in context
    has_cards = "card" in context
    has_offsides = "offside" in context
    has_penalty_red = "penalty" in context or "red card" in context
    has_anytime_scorer = _contains_any(context, ["anytime", "goalscorer", "score a goal"])

    direct_facts: list[str] = []
    adjacent_facts: list[str] = []
    reason = ""
    level = "match_context_only"

    if family == "result":
        if "win the match" in question_text and has_moneyline:
            level = "direct_market_line"
            reason = "Moneyline/favorite odds directly describe match-winner likelihood."
            direct_facts = _matched_facts(odds_or_stats_facts, ["moneyline", "favorite", "favored"])
        elif has_moneyline:
            level = "adjacent_match_market"
            reason = "Match moneyline is relevant but does not directly price halftime or partial-result framing."
            adjacent_facts = _matched_facts(odds_or_stats_facts, ["moneyline", "favorite", "favored"])
    elif family == "goals":
        needs_total = _contains_any(question_text, ["total goals", "3 or more", "2 or fewer", "second half have"])
        needs_btts = "both teams score" in question_text
        if (needs_total and has_total) or (needs_btts and has_btts):
            level = "direct_market_line"
            reason = "Totals and/or BTTS odds directly price the core goal-market leg."
            direct_facts = _matched_facts(odds_or_stats_facts, ["total", "over 2.5", "under 2.5", "btts"])
        elif has_total or has_btts or has_moneyline:
            level = "adjacent_match_market"
            reason = "Moneyline, totals, or BTTS odds are adjacent to team-scoring probability but do not directly price this exact market."
            adjacent_facts = _matched_facts(odds_or_stats_facts, ["moneyline", "total", "btts"])
    elif family in {"player-goal", "player-goal-assist"}:
        player_terms = _player_terms(question)
        mentions_player = any(term in context for term in player_terms)
        if mentions_player and has_anytime_scorer:
            level = "direct_market_line"
            reason = "A player goalscorer market was found for the named player."
            direct_facts = _matched_facts(odds_or_stats_facts, [*player_terms, "anytime", "goalscorer"])
        elif has_total or has_btts or has_moneyline:
            level = "adjacent_match_market"
            reason = "Team/match scoring odds are relevant but do not directly price the named player's goal/assist market."
            adjacent_facts = _matched_facts(odds_or_stats_facts, ["moneyline", "total", "btts"])
    elif family == "shots-on-target":
        if has_sot:
            level = "direct_or_adjacent_stat"
            reason = "External source includes shots-on-target facts or a shots-on-target market, but may not match the exact team/player/half threshold."
            direct_facts = _matched_facts(odds_or_stats_facts, ["shot-on-target", "shots-on-target", "shots on target", "sot"])
        elif has_shots:
            level = "direct_or_adjacent_stat"
            reason = "External source includes shot-volume facts adjacent to the shots-on-target market."
            direct_facts = _matched_facts(odds_or_stats_facts, [" shot ", " shots "])
        elif has_moneyline or has_total:
            level = "adjacent_match_market"
            reason = "Match winner/total odds are only loose proxies for shot-on-target props."
            adjacent_facts = _matched_facts(odds_or_stats_facts, ["moneyline", "total"])
    elif family == "fouls":
        if has_fouls:
            level = "direct_or_adjacent_stat"
            reason = "External source includes foul-rate/foul-count evidence."
            direct_facts = _matched_facts(odds_or_stats_facts, ["foul"])
        elif has_moneyline:
            level = "adjacent_match_market"
            reason = "Moneyline context is a weak proxy for foul-market game script."
            adjacent_facts = _matched_facts(odds_or_stats_facts, ["moneyline"])
    elif family == "corners":
        if has_corners:
            level = "direct_or_adjacent_stat"
            reason = "External source includes corner evidence."
            direct_facts = _matched_facts(odds_or_stats_facts, ["corner"])
        elif has_moneyline or has_total:
            level = "adjacent_match_market"
            reason = "Match odds are only loose proxies for corner props."
            adjacent_facts = _matched_facts(odds_or_stats_facts, ["moneyline", "total"])
    elif family == "cards":
        if has_cards:
            level = "direct_or_adjacent_stat"
            reason = "External source includes card evidence."
            direct_facts = _matched_facts(odds_or_stats_facts, ["card"])
        elif has_moneyline:
            level = "adjacent_match_market"
            reason = "Moneyline context is a weak proxy for card props."
            adjacent_facts = _matched_facts(odds_or_stats_facts, ["moneyline"])
    elif family == "offsides":
        if has_offsides:
            level = "direct_or_adjacent_stat"
            reason = "External source includes offside evidence."
            direct_facts = _matched_facts(odds_or_stats_facts, ["offside"])
        elif has_moneyline:
            level = "adjacent_match_market"
            reason = "Moneyline context is a weak proxy for offside props."
            adjacent_facts = _matched_facts(odds_or_stats_facts, ["moneyline"])
    elif family == "penalty-red":
        if has_penalty_red and has_cards:
            level = "direct_or_adjacent_stat"
            reason = "External source includes penalty/red-card related evidence."
            direct_facts = _matched_facts(odds_or_stats_facts, ["penalty", "red card", "card"])
        elif has_moneyline or has_total:
            level = "adjacent_match_market"
            reason = "Match odds are only loose proxies for penalty/red-card likelihood."
            adjacent_facts = _matched_facts(odds_or_stats_facts, ["moneyline", "total"])
    else:
        if _contains_any(question_text, ["score in the first half", "score in the second half", "score in the second"]):
            if has_total or has_btts:
                level = "adjacent_match_market"
                reason = "Totals/BTTS odds are adjacent to team half-scoring markets."
                adjacent_facts = _matched_facts(odds_or_stats_facts, ["total", "btts"])
        elif has_moneyline or has_total or has_btts:
            level = "adjacent_match_market"
            reason = "Match-level odds are relevant context but do not directly price this market."
            adjacent_facts = _matched_facts(odds_or_stats_facts, ["moneyline", "total", "btts"])

    if level == "match_context_only":
        reason = "External sources cover the match but not this market family with a direct or adjacent market/stat line."
    return {
        "external_evidence_level": level,
        "external_evidence_reason": reason,
        "external_evidence_basis": direct_facts or adjacent_facts,
    }


def _build_records(
    *,
    platform: dict[str, Any],
    history: dict[str, Any],
    post_change_at: datetime,
    external_sources: dict[str, dict[str, Any]],
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
        question = result.get("question") or (market.question if market else history_row.get("question") or "")
        match_name = (
            _market_match_name(market, matches_by_id)
            or _history_match_name(history, match_id)
            or _infer_legacy_match_name(market_id, question)
        )
        external_entry = _external_for_match(match_name, external_sources)
        probability_int = int(round(float(probability_submitted)))
        probability = probability_int / 100.0
        components = history_row.get("components") or []
        aggregate_probability = _aggregate_probability(components)
        family = _market_family(question)
        row = {
            "market_id": market_id,
            "match_id": match_id,
            "match_name": match_name,
            "question": question,
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
            **_external_fields(external_entry),
            **_external_evidence_for_record(question, family, external_entry),
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


def _external_coverage(records: list[dict[str, Any]], external_sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_match: dict[str, dict[str, Any]] = {}
    for match_name, rows in _records_by_match(records).items():
        source_counts = [int(row.get("external_source_count") or 0) for row in rows]
        odds_counts = [int(row.get("external_odds_source_count") or 0) for row in rows]
        stats_counts = [int(row.get("external_stats_source_count") or 0) for row in rows]
        by_match[match_name] = {
            "settled_records": len(rows),
            "records_with_sources": sum(1 for count in source_counts if count > 0),
            "source_count": max(source_counts) if source_counts else 0,
            "odds_source_count": max(odds_counts) if odds_counts else 0,
            "stats_source_count": max(stats_counts) if stats_counts else 0,
            "odds_summary": rows[0].get("external_odds_summary") or "",
            "result_summary": rows[0].get("external_result_summary") or "",
        }
    missing = [
        match_name
        for match_name, row in sorted(by_match.items())
        if int(row["settled_records"]) > int(row["records_with_sources"])
    ]
    return {
        "configured_matches": len({_match_key(entry.get("match_name") or "") for entry in external_sources.values()}),
        "settled_matches": len(by_match),
        "settled_records": len(records),
        "settled_records_with_sources": sum(1 for row in records if int(row.get("external_source_count") or 0) > 0),
        "settled_records_with_direct_or_adjacent_evidence": sum(
            1
            for row in records
            if row.get("external_evidence_level") in {"direct_market_line", "direct_or_adjacent_stat"}
        ),
        "matches_missing_sources": missing,
        "by_match": by_match,
    }


def _records_by_match(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[str(record.get("match_name") or "unknown")].append(record)
    return dict(buckets)


def _candidate_findings(
    records: list[dict[str, Any]],
    component_records: list[dict[str, Any]],
    external_coverage: dict[str, Any],
) -> list[str]:
    findings: list[str] = []
    overall = _summary(records)
    findings.append(
        f"Settled platform predictions: {overall['count']} markets, mean Brier {overall['mean_brier']}."
    )
    findings.append(
        "External source coverage: "
        f"{external_coverage['settled_records_with_sources']}/{external_coverage['settled_records']} "
        f"settled forecasts across {external_coverage['settled_matches']} match groups."
    )
    findings.append(
        "Direct/specific evidence coverage: "
        f"{external_coverage['settled_records_with_direct_or_adjacent_evidence']}/"
        f"{external_coverage['settled_records']} settled forecasts."
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
    lines.extend(["", "## External Source Coverage", ""])
    lines.append(_markdown_external_coverage(report["external_source_coverage"]))
    lines.extend(["", "## External Evidence Strength", ""])
    lines.append(_markdown_summary_rows(report["external_evidence_bins"]))
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
    output = [{"group": key, **value} for key, value in _sort_summary_rows(rows).items()]
    return _markdown_rows(output)


def _sort_summary_rows(rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if set(rows).issubset(EVIDENCE_LEVEL_ORDER):
        return dict(sorted(rows.items(), key=lambda item: EVIDENCE_LEVEL_ORDER.get(item[0], 99)))
    return rows


def _markdown_record_rows(rows: list[dict[str, Any]]) -> str:
    return _markdown_rows(rows)


def _markdown_external_coverage(coverage: dict[str, Any]) -> str:
    rows = []
    for match_name, row in sorted((coverage.get("by_match") or {}).items()):
        rows.append(
            {
                "match_name": match_name,
                "settled_records": row.get("settled_records"),
                "records_with_sources": row.get("records_with_sources"),
                "source_count": row.get("source_count"),
                "odds_source_count": row.get("odds_source_count"),
                "stats_source_count": row.get("stats_source_count"),
                "odds_summary": row.get("odds_summary"),
                "result_summary": row.get("result_summary"),
            }
        )
    if not rows:
        return "_No external source rows._"
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
    external_sources = _load_external_sources(args.external_sources)
    post_change_at = _parse_dt(args.post_change_at) or _parse_dt(DEFAULT_POST_CHANGE_AT)
    assert post_change_at is not None
    records, component_records = _build_records(
        platform=platform,
        history=history,
        post_change_at=post_change_at,
        external_sources=external_sources,
    )
    external_coverage = _external_coverage(records, external_sources)

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
            "external_source_matches_configured": external_coverage["configured_matches"],
            "settled_records_with_external_sources": external_coverage["settled_records_with_sources"],
            "settled_records_with_direct_or_adjacent_evidence": external_coverage[
                "settled_records_with_direct_or_adjacent_evidence"
            ],
        },
        "overall": _summary(records),
        "by_family": _group_summary(records, "family"),
        "by_match": _group_summary(records, "match_name"),
        "by_stage": _group_summary(records, "stage"),
        "calibration_bins": _calibration_bins(records),
        "disagreement_bins": _disagreement_bins(records),
        "market_anchor_bins": _group_summary(records, "has_market_anchor_in_rationale"),
        "external_source_coverage": external_coverage,
        "external_evidence_bins": _group_summary(records, "external_evidence_level"),
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
            "External source enrichment is match-level unless a direct market/prop line is shown in the source facts.",
            "External evidence strength is conservative: generic moneyline/total context is not treated as direct evidence for SOT, cards, fouls, offsides, corners, or half-specific props.",
            "External odds are not available through the Jump API; source coverage comes from public odds, stats, and recap pages captured in reports/external-match-sources.json.",
            "Post-change split uses forecast-history timestamps, not Git metadata inside artifacts.",
        ],
    }
    report["findings"] = _candidate_findings(records, component_records, external_coverage)
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
    parser.add_argument(
        "--external-sources",
        default="reports/external-match-sources.json",
        help="Optional JSON file of public odds/stat/recap sources keyed by match.",
    )
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
