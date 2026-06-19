from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from datetime import timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from probability_cup_bot.config import Settings
from probability_cup_bot.models import Market, Match, Prediction, utcnow
from probability_cup_bot.sportspredict import SportsPredictClient
from probability_cup_bot.state import ensure_dirs, read_json, write_json


logger = logging.getLogger(__name__)


def _hours_to_close(match: Match | None, market: Market | None) -> float | None:
    closes_at = (match.closes_at if match else None) or (market.closes_at if market else None)
    if closes_at is None:
        return None
    return round((closes_at.astimezone(timezone.utc) - utcnow()).total_seconds() / 3600, 2)


def _forecast_lookup(state_dir: Path) -> dict[str, dict[str, Any]]:
    latest_repaired = read_json(state_dir / "latest-run-repaired.json", {})
    latest = latest_repaired or read_json(state_dir / "latest-run.json", {})
    return {
        forecast.get("market_id"): forecast
        for forecast in latest.get("forecasts") or []
        if forecast.get("market_id")
    }


def build_dashboard_data(
    *,
    matches: list[Match],
    markets: list[Market],
    predictions: list[Prediction],
    state_dir: Path,
) -> dict[str, Any]:
    now = utcnow()
    matches_by_id = {match.id: match for match in matches}
    markets_by_id = {market.id: market for market in markets}
    forecast_by_market = _forecast_lookup(state_dir)
    history = read_json(state_dir / "forecast-history.json", {})
    schedule = read_json(state_dir / "match-schedule.json", {"matches": {}})
    usage = read_json(state_dir / "usage-ledger.json", {})

    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        market = markets_by_id.get(prediction.market_id)
        match = matches_by_id.get(market.match.id) if market else None
        forecast = forecast_by_market.get(prediction.market_id) or {}
        match_history = (history.get("matches") or {}).get(match.id if match else "", {})
        market_history = (history.get("markets") or {}).get(prediction.market_id, {})
        schedule_entry = (schedule.get("matches") or {}).get(match.id if match else "", {})
        rows.append(
            {
                "prediction_id": prediction.id,
                "market_id": prediction.market_id,
                "match_id": match.id if match else (market.match.id if market else ""),
                "match_name": match.name if match else (market.match.name if market else ""),
                "question": market.question if market else prediction.question or "",
                "market_family": market_history.get("market_family"),
                "probability": prediction.probability_int,
                "platform_market_status": prediction.market_status or (market.status if market else ""),
                "created_date": prediction.created_date,
                "updated_date": prediction.updated_date,
                "closing_time": (match.closing_time if match else None)
                or (market.match.closing_time if market else None),
                "hours_to_close": _hours_to_close(match, market),
                "latest_bot_probability": forecast.get("probability_int"),
                "latest_bot_confidence": forecast.get("confidence"),
                "latest_bot_evidence_quality": forecast.get("evidence_quality"),
                "component_count": len(forecast.get("component_probabilities") or []),
                "component_spread_points": market_history.get("component_spread_points"),
                "last_forecast_at": match_history.get("last_forecast_at"),
                "late_forecast_due_at": schedule_entry.get("late_forecast_due_at"),
                "late_forecast_completed_at": schedule_entry.get("late_forecast_completed_at"),
                "news_check_due_at": schedule_entry.get("news_check_due_at"),
                "news_check_completed_at": schedule_entry.get("news_check_completed_at"),
                "news_check_should_reforecast": schedule_entry.get("news_check_should_reforecast"),
                "brier_score": prediction.brier_score,
            }
        )

    rows.sort(
        key=lambda row: (
            row["hours_to_close"] is None,
            row["hours_to_close"] if row["hours_to_close"] is not None else 999999,
            row["match_name"],
            row["question"],
        )
    )
    open_rows = [row for row in rows if str(row.get("platform_market_status") or "open") == "open"]
    mismatched = [
        row
        for row in rows
        if row.get("latest_bot_probability") is not None
        and int(row["probability"]) != int(row["latest_bot_probability"])
    ]
    return {
        "generated_at": now.isoformat(),
        "summary": {
            "prediction_count": len(rows),
            "open_prediction_count": len(open_rows),
            "market_count": len(markets),
            "match_count": len(matches),
            "bot_probability_mismatch_count": len(mismatched),
            "next_close_time": next((row["closing_time"] for row in rows if row.get("closing_time")), None),
            "usage_cumulative": usage.get("cumulative", {}),
        },
        "rows": rows,
    }


class DashboardBuilder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def write(self, output_dir: Path) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        sp = SportsPredictClient(
            base_url=self.settings.sportspredict_base_url,
            api_key=self.settings.sportspredict_api_key,
            retry_attempts=self.settings.sportspredict_retry_attempts,
            retry_initial_seconds=self.settings.sportspredict_retry_initial_seconds,
            retry_max_seconds=self.settings.sportspredict_retry_max_seconds,
        )
        try:
            event = await sp.find_event(self.settings.event_title, self.settings.event_id)
            lobby = await sp.ensure_lobby(event.id)
            matches = await sp.list_matches(event.id, lobby.id)
            markets = await sp.list_markets(lobby.id)
            predictions = await sp.list_predictions(lobby.id)
        finally:
            await sp.aclose()

        data = build_dashboard_data(
            matches=matches,
            markets=markets,
            predictions=predictions,
            state_dir=self.settings.state_dir,
        )
        data["event"] = event.model_dump()
        data["lobby"] = lobby.model_dump()
        write_json(output_dir / "dashboard-data.json", data)
        (output_dir / "index.html").write_text(_render_index_html(data), encoding="utf-8")
        logger.info(
            "Dashboard written output=%s predictions=%d",
            output_dir,
            data["summary"]["prediction_count"],
        )
        return data


def serve_dashboard(
    *,
    builder: DashboardBuilder,
    output_dir: Path,
    host: str,
    port: int,
    refresh_seconds: int,
) -> None:
    ensure_dirs(output_dir)

    def refresh_loop() -> None:
        while True:
            try:
                asyncio.run(builder.write(output_dir))
            except Exception:  # noqa: BLE001
                logger.exception("Dashboard refresh failed")
            time.sleep(max(10, refresh_seconds))

    thread = threading.Thread(target=refresh_loop, daemon=True)
    thread.start()
    handler = partial(SimpleHTTPRequestHandler, directory=str(output_dir))
    server = ThreadingHTTPServer((host, port), handler)
    logger.info("Dashboard serving at http://%s:%d", host, port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _render_index_html(data: dict[str, Any]) -> str:
    payload = _json_script_payload(data)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Probability Cup Forecast Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #1f2528;
      --muted: #637076;
      --line: #d9dddc;
      --accent: #136f63;
      --warn: #a24b2a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      padding: 18px 24px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 16px;
      color: var(--muted);
      font-size: 13px;
    }}
    .summary b {{ color: var(--ink); }}
    main {{ padding: 16px 24px 28px; }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 12px;
    }}
    input, select {{
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-size: 13px;
    }}
    input {{ min-width: 280px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      vertical-align: top;
      text-align: left;
    }}
    th {{
      position: sticky;
      top: 83px;
      background: #eef1ef;
      z-index: 1;
      font-weight: 650;
    }}
    td.num, th.num {{ text-align: right; white-space: nowrap; }}
    td.question {{ min-width: 320px; max-width: 620px; }}
    .muted {{ color: var(--muted); }}
    .pill {{
      display: inline-block;
      padding: 2px 6px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #f8faf9;
      font-size: 12px;
      white-space: nowrap;
    }}
    .mismatch {{ color: var(--warn); font-weight: 650; }}
    .ok {{ color: var(--accent); font-weight: 650; }}
  </style>
</head>
<body>
  <header>
    <h1>Probability Cup Forecast Dashboard</h1>
    <div class="summary" id="summary"></div>
  </header>
  <main>
    <div class="controls">
      <input id="search" placeholder="Filter by match, question, or market id">
      <select id="status">
        <option value="all">All statuses</option>
        <option value="open">Open only</option>
        <option value="mismatch">Bot mismatch</option>
        <option value="due">Scheduled/due tracked</option>
      </select>
      <span class="muted" id="refreshStatus"></span>
    </div>
    <table>
      <thead>
        <tr>
          <th>Match</th>
          <th class="question">Question</th>
          <th class="num">Platform API</th>
          <th class="num">Bot State</th>
          <th>Status</th>
          <th class="num">Hrs</th>
          <th>Schedule</th>
          <th>Updated</th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </main>
  <script id="initial-data" type="application/json">{payload}</script>
  <script>
    let data = JSON.parse(document.getElementById('initial-data').textContent);
    const search = document.getElementById('search');
    const status = document.getElementById('status');
    const rows = document.getElementById('rows');
    const summary = document.getElementById('summary');
    const refreshStatus = document.getElementById('refreshStatus');

    function fmt(value) {{
      return value === null || value === undefined || value === '' ? '-' : value;
    }}
    function esc(value) {{
      return String(fmt(value)).replace(/[&<>"']/g, c => ({{
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }}[c]));
    }}
    function shortTime(value) {{
      if (!value) return '-';
      const d = new Date(value);
      return isNaN(d.getTime()) ? value : d.toLocaleString();
    }}
    function renderSummary() {{
      const s = data.summary || {{}};
      summary.innerHTML = [
        `<span>Generated <b>${{shortTime(data.generated_at)}}</b></span>`,
        `<span>Platform source <b>SportsPredict /predictions</b></span>`,
        `<span>Predictions <b>${{s.prediction_count || 0}}</b></span>`,
        `<span>Open <b>${{s.open_prediction_count || 0}}</b></span>`,
        `<span>Matches <b>${{s.match_count || 0}}</b></span>`,
        `<span>Mismatches <b>${{s.bot_probability_mismatch_count || 0}}</b></span>`
      ].join('');
    }}
    function rowMatches(row, query, mode) {{
      const haystack = `${{row.match_name}} ${{row.question}} ${{row.market_id}}`.toLowerCase();
      if (query && !haystack.includes(query)) return false;
      if (mode === 'open' && row.platform_market_status && row.platform_market_status !== 'open') return false;
      if (mode === 'mismatch' && Number(row.probability) === Number(row.latest_bot_probability)) return false;
      if (mode === 'due' && !row.late_forecast_due_at && !row.news_check_due_at) return false;
      return true;
    }}
    function renderRows() {{
      const query = search.value.trim().toLowerCase();
      const mode = status.value;
      const filtered = (data.rows || []).filter(row => rowMatches(row, query, mode));
      rows.textContent = '';
      for (const row of filtered) {{
        const tr = document.createElement('tr');
        const mismatch = row.latest_bot_probability !== null && row.latest_bot_probability !== undefined &&
          Number(row.probability) !== Number(row.latest_bot_probability);
        const schedule = [
          row.late_forecast_completed_at ? 'forecast done' : (row.late_forecast_due_at ? 'forecast scheduled' : ''),
          row.news_check_completed_at ? 'news done' : (row.news_check_due_at ? 'news scheduled' : '')
        ].filter(Boolean).join(' / ');
        const cells = [
          `<b>${{esc(row.match_name)}}</b><br><span class="muted">${{esc(shortTime(row.closing_time))}}</span>`,
          `${{esc(row.question)}}<br><span class="muted">${{esc(row.market_id)}}</span>`,
          `<span class="${{mismatch ? 'mismatch' : 'ok'}}">${{esc(row.probability)}}%</span>`,
          `${{esc(row.latest_bot_probability)}}%<br><span class="muted">${{esc(row.latest_bot_confidence)}}/${{esc(row.latest_bot_evidence_quality)}}</span>`,
          `<span class="pill">${{esc(row.platform_market_status || 'open')}}</span>`,
          esc(row.hours_to_close),
          `${{esc(schedule)}}<br><span class="muted">${{esc(shortTime(row.late_forecast_due_at))}} / ${{esc(shortTime(row.news_check_due_at))}}</span>`,
          esc(shortTime(row.updated_date || row.created_date))
        ];
        cells.forEach((html, index) => {{
          const td = document.createElement('td');
          if (index === 1) td.className = 'question';
          if (index === 2 || index === 3 || index === 5) td.classList.add('num');
          td.innerHTML = html;
          tr.appendChild(td);
        }});
        rows.appendChild(tr);
      }}
    }}
    async function refreshData() {{
      try {{
        const res = await fetch(`dashboard-data.json?ts=${{Date.now()}}`, {{ cache: 'no-store' }});
        if (!res.ok) throw new Error(String(res.status));
        data = await res.json();
        refreshStatus.textContent = `auto-refreshed ${{new Date().toLocaleTimeString()}}`;
        renderSummary();
        renderRows();
      }} catch (err) {{
        refreshStatus.textContent = 'snapshot loaded; serve over HTTP for auto-refresh';
      }}
    }}
    search.addEventListener('input', renderRows);
    status.addEventListener('change', renderRows);
    renderSummary();
    renderRows();
    setInterval(refreshData, 30000);
    refreshData();
  </script>
</body>
</html>
"""


def _json_script_payload(data: dict[str, Any]) -> str:
    return (
        json.dumps(data, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
