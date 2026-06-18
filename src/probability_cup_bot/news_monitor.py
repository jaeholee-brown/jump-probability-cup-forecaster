from __future__ import annotations

import json
from typing import Any

from probability_cup_bot.config import Settings
from probability_cup_bot.models import Market, Match, NewsCheck, utcnow
from probability_cup_bot.openai_adapter import OpenAIAdapter


NEWS_MONITOR_INSTRUCTIONS = """
You are a low-cost change detector for a soccer forecasting bot.

Goal: decide whether new public information is important enough to rerun the expensive full
OpenAI/Grok/Claude forecast ensemble before kickoff.

Use web search and X search aggressively for freshness, but be conservative about rerunning.
On X, look for official club/tournament/player accounts, credible journalists, team reporters,
and timestamped first-hand posts. Treat X as a discovery and corroboration channel, not as a
raw sentiment signal. Discount fan speculation, anonymous aggregator claims, and repeated old
posts unless corroborated by official, reputable media, bookmaker/odds, or weather sources.

Trigger rerun only for information likely to move at least one listed market by the configured
threshold. Important examples: confirmed or strongly credible lineup changes, injury/suspension
news, weather that materially affects play, major odds movement if found, tactical/motivation
news, and player availability for player props.

Do not trigger a rerun for generic previews, repeated old news, fan speculation without credible
source support, or broad sentiment that does not affect the listed markets.

Return a compact current news summary that can be cached and reused by later forecast calls.
If a rerun is needed, set affected_market_ids to only the listed market ids whose probabilities
should be recomputed. Leave it empty only when all listed markets are affected. Confirmed starting
lineups, tactical shape, major weather changes, or broad referee/disciplinary news often affect
the whole match; when that is true or when you are unsure which props move, leave affected_market_ids
empty so the runner recomputes every listed market.
"""


class GrokNewsMonitor:
    def __init__(self, settings: Settings, grok: OpenAIAdapter) -> None:
        self.settings = settings
        self.grok = grok

    async def check_match(
        self,
        *,
        match: Match,
        markets: list[Market],
        match_history: dict[str, Any],
        cached_news: dict[str, Any],
        firecrawl_context: str = "",
    ) -> NewsCheck:
        user_input = json.dumps(
            {
                "today_utc": utcnow().isoformat(),
                "match": match.model_dump(),
                "markets": [
                    {
                        "id": market.id,
                        "question": market.question,
                        "status": market.status,
                        "closing_time": market.match.closing_time,
                    }
                    for market in markets
                ],
                "previous_forecast_state": match_history,
                "cached_news": cached_news,
                "firecrawl_context": firecrawl_context,
                "source_policy": (
                    "Mark developments as new only if they are newer than cached_news or materially "
                    "clarify prior uncertainty. Prefer official/team/journalist/market/weather "
                    "sources; X-only claims need corroboration or explicit uncertainty."
                ),
                "rerun_threshold_points": self.settings.news_monitor_materiality_threshold_points,
                "decision_rule": (
                    "Set should_reforecast=true only if new information is credible and likely "
                    "to change at least one market by the threshold. Populate affected_market_ids "
                    "with only the changed market ids. If the information is match-wide or you "
                    "cannot confidently enumerate the affected subset, set should_reforecast=true "
                    "and affected_market_ids=[] to mean all listed markets. Otherwise update "
                    "summary, set affected_market_ids=[], and set should_reforecast=false."
                ),
            },
            ensure_ascii=True,
        )
        news_check = await self.grok.structured_response(
            model=self.settings.grok_news_model,
            instructions=NEWS_MONITOR_INSTRUCTIONS,
            user_input=user_input,
            schema_model=NewsCheck,
            schema_name="news_check",
            reasoning_effort=self.settings.grok_news_reasoning_effort,
            tools=[{"type": "web_search"}, {"type": "x_search"}],
        )
        if not news_check.match_id:
            news_check.match_id = match.id
        if not news_check.match_name:
            news_check.match_name = match.name
        return news_check
