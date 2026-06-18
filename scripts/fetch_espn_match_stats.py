from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


ESPN_EVENT_IDS = {
    "FRA vs SEN": "760432",
    "IRQ vs NOR": "760430",
    "ARG vs ALG": "760433",
    "AUT vs JOR": "760431",
    "POR vs COD": "760435",
    "ENG vs CRO": "760437",
    "GHA vs PAN": "760434",
    "UZB vs COL": "760436",
    "CZE vs RSA": "760438",
}
ESPN_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary"


def _team_stats(summary: dict[str, Any]) -> dict[str, Any]:
    teams: dict[str, Any] = {}
    for row in (summary.get("boxscore") or {}).get("teams") or []:
        team = row.get("team") or {}
        abbreviation = str(team.get("abbreviation") or "")
        if not abbreviation:
            continue
        teams[abbreviation] = {
            "display_name": team.get("displayName") or "",
            "home_away": row.get("homeAway") or "",
            "statistics": {
                stat.get("name"): _numeric_value(stat.get("displayValue") or stat.get("value"))
                for stat in row.get("statistics") or []
                if stat.get("name")
            },
        }
    return teams


def _numeric_value(value: Any) -> int | float | str:
    if isinstance(value, int | float):
        return value
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() else number


def _events(summary: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in summary.get("commentary") or []:
        play = row.get("play") or {}
        play_type = play.get("type") or {}
        team = play.get("team") or {}
        clock = play.get("clock") or {}
        period = play.get("period") or {}
        participants = []
        for participant in play.get("participants") or []:
            athlete = participant.get("athlete") or {}
            name = athlete.get("displayName")
            if name:
                participants.append(str(name))
        events.append(
            {
                "sequence": row.get("sequence"),
                "type": play_type.get("text") or "",
                "type_slug": play_type.get("type") or "",
                "period": period.get("number"),
                "clock_value": clock.get("value"),
                "clock_display": clock.get("displayValue") or "",
                "team": team.get("displayName") or "",
                "participants": participants,
                "text": play.get("text") or row.get("text") or "",
            }
        )
    return events


def fetch_event(client: httpx.Client, match_name: str, event_id: str) -> dict[str, Any]:
    response = client.get(ESPN_SUMMARY_URL, params={"event": event_id})
    response.raise_for_status()
    summary = response.json()
    header = summary.get("header") or {}
    competitions = header.get("competitions") or []
    competitors = (competitions[0].get("competitors") if competitions else []) or []
    return {
        "match_name": match_name,
        "espn_event_id": event_id,
        "summary_url": f"{ESPN_SUMMARY_URL}?event={event_id}",
        "espn_match_url": f"https://www.espn.com/soccer/match/_/gameId/{event_id}",
        "event_name": header.get("name") or "",
        "date": header.get("timeValid") or header.get("competitions", [{}])[0].get("date", ""),
        "competitors": [
            {
                "home_away": row.get("homeAway") or "",
                "abbreviation": (row.get("team") or {}).get("abbreviation") or "",
                "display_name": (row.get("team") or {}).get("displayName") or "",
                "score": row.get("score"),
            }
            for row in competitors
        ],
        "teams": _team_stats(summary),
        "events": _events(summary),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch structured ESPN match stats for settled matches.")
    parser.add_argument("--output", default="reports/espn-match-stats.json")
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=args.timeout, follow_redirects=True) as client:
        matches = {
            match_name: fetch_event(client, match_name, event_id)
            for match_name, event_id in ESPN_EVENT_IDS.items()
        }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "ESPN public site API",
        "matches": matches,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "matches": len(matches)}, indent=2))


if __name__ == "__main__":
    main()
