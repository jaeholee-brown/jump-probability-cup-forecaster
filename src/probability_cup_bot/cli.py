from __future__ import annotations

import argparse
import asyncio
import json
import sys

from probability_cup_bot.config import load_settings
from probability_cup_bot.runner import ForecastRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="probability-cup-bot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Discover open markets, forecast, and optionally submit.")
    run.add_argument("--dry-run", action="store_true", help="Force dry-run mode even if SUBMIT=true.")
    run.add_argument("--dotenv", default=None, help="Optional path to an env file.")
    return parser


async def _run(args: argparse.Namespace) -> int:
    settings = load_settings(dotenv_path=args.dotenv, force_dry_run=args.dry_run)
    runner = ForecastRunner(settings)
    result = await runner.run()
    print(
        json.dumps(
            {
                "mode": result["submission_results"]["mode"],
                "matches_forecasted": result["matches_forecasted"],
                "forecast_count": result["forecast_count"],
                "creates": len(result["plan"]["creates"]),
                "updates": len(result["plan"]["updates"]),
                "skips": len(result["plan"]["skips"]),
                "log": str(settings.state_dir / "latest-run.json"),
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return asyncio.run(_run(args))
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())

