from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
import logging
import json
import os
import sys
import time
from datetime import timezone

from probability_cup_bot.config import load_settings
from probability_cup_bot.models import Market, utcnow
from probability_cup_bot.runner import ForecastRunner
from probability_cup_bot.sportspredict import SportsPredictClient


class _UtcFormatter(logging.Formatter):
    converter = time.gmtime


def _configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _UtcFormatter("%(asctime)sZ %(levelname)s %(name)s: %(message)s", "%Y-%m-%dT%H:%M:%S")
    )
    logging.basicConfig(level=level, handlers=[handler], force=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="probability-cup-bot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Discover open markets, forecast, and optionally submit.")
    run.add_argument("--dry-run", action="store_true", help="Force dry-run mode even if SUBMIT=true.")
    run.add_argument(
        "--news-monitor-only",
        action="store_true",
        help="Run cheap news checks and forecast only markets promoted by material new information.",
    )
    run.add_argument("--dotenv", default=None, help="Optional path to an env file.")
    inspect = subparsers.add_parser(
        "inspect-docket",
        help="List current matches/markets and summarize odds-feed usefulness without model calls.",
    )
    inspect.add_argument("--dotenv", default=None, help="Optional path to an env file.")
    inspect.add_argument("--max-questions", type=int, default=120, help="Maximum questions to print.")
    return parser


async def _run(args: argparse.Namespace) -> int:
    _configure_logging()
    settings = load_settings(dotenv_path=args.dotenv, force_dry_run=args.dry_run)
    runner = ForecastRunner(settings)
    result = await runner.run(news_monitor_only=args.news_monitor_only)
    print(
        json.dumps(
            {
                "mode": result["submission_results"]["mode"],
                "news_monitor_only": result.get("news_monitor_only", False),
                "matches_forecasted": result["matches_forecasted"],
                "forecast_count": result["forecast_count"],
                "creates": len(result["plan"]["creates"]),
                "updates": len(result["plan"]["updates"]),
                "skips": len(result["plan"]["skips"]),
                "log": str(settings.state_dir / "latest-run.json"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _market_category(question: str) -> str:
    q = question.lower()
    if "both teams" in q or "btts" in q:
        return "btts"
    if any(term in q for term in ("over ", "under ", "total goals", "combined goals")):
        return "totals"
    if any(term in q for term in ("win the match", "win in regulation", "to win", "draw")):
        return "match_winner"
    if any(term in q for term in ("corner", "corners")):
        return "corners"
    if any(term in q for term in ("card", "booking", "yellow", "red card")):
        return "cards"
    if any(term in q for term in ("shot", "shots", "shot on target")):
        return "shots"
    if any(term in q for term in ("score a goal", "goalscorer", "assist", "player", "start")):
        return "player_prop"
    if any(term in q for term in ("penalty", "own goal", "clean sheet")):
        return "special_prop"
    return "other"


def _odds_api_verdict(category_counts: Counter[str]) -> str:
    direct = sum(category_counts[key] for key in ("match_winner", "totals", "btts"))
    total = sum(category_counts.values())
    if total == 0:
        return "No open markets found."
    share = direct / total
    if share >= 0.6:
        return "The current basic Odds API integration is highly relevant."
    if share >= 0.3:
        return "The basic Odds API integration is useful as an anchor, but incomplete."
    return "The basic Odds API integration is only marginally useful; web/Grok research matters more."


async def _inspect_docket(args: argparse.Namespace) -> int:
    _configure_logging()
    settings = load_settings(dotenv_path=args.dotenv, force_dry_run=True)
    sp = SportsPredictClient(
        base_url=settings.sportspredict_base_url,
        api_key=settings.sportspredict_api_key,
    )
    try:
        event = await sp.find_event(settings.event_title, settings.event_id)
        lobby = await sp.ensure_lobby(event.id)
        matches = await sp.list_matches(event.id, lobby.id)
        markets = await sp.list_markets(lobby.id)
    finally:
        await sp.aclose()

    markets_by_match: dict[str, list[Market]] = defaultdict(list)
    category_counts: Counter[str] = Counter()
    for market in markets:
        if market.status != "open":
            continue
        category = _market_category(market.question)
        category_counts[category] += 1
        markets_by_match[market.match.id].append(market)

    close_times = []
    now = utcnow()
    for market in markets:
        closes_at = market.closes_at
        if market.status == "open" and closes_at is not None:
            close_times.append((closes_at.astimezone(timezone.utc) - now).total_seconds() / 3600)

    summary = {
        "generated_at": now.isoformat(),
        "event": event.model_dump(),
        "lobby": lobby.model_dump(),
        "matches_seen": len(matches),
        "open_markets_seen": sum(1 for market in markets if market.status == "open"),
        "category_counts": dict(category_counts),
        "basic_odds_api_direct_categories": ["match_winner", "totals", "btts"],
        "basic_odds_api_direct_count": sum(
            category_counts[key] for key in ("match_winner", "totals", "btts")
        ),
        "odds_api_verdict": _odds_api_verdict(category_counts),
        "min_hours_to_close": round(min(close_times), 2) if close_times else None,
        "max_hours_to_close": round(max(close_times), 2) if close_times else None,
    }
    print(json.dumps(summary, indent=2), flush=True)

    printed = 0
    for match in sorted(matches, key=lambda item: item.closes_at or utcnow()):
        match_markets = markets_by_match.get(match.id, [])
        if not match_markets:
            continue
        print(f"\n## {match.name} | close={match.closing_time} | open_markets={len(match_markets)}", flush=True)
        for market in match_markets:
            if printed >= args.max_questions:
                print(f"\n... truncated after {args.max_questions} questions", flush=True)
                return 0
            print(f"- [{_market_category(market.question)}] {market.question}", flush=True)
            printed += 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return asyncio.run(_run(args))
    if args.command == "inspect-docket":
        return asyncio.run(_inspect_docket(args))
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
