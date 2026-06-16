# Autonomous System Design

## Goal

Run an LLM forecasting bot for the Jump Trading Probability Cup from GitHub Actions every hour or half-hour. The bot should discover recently open or updated questions, gather fresh context, forecast with the optimized prompt, aggregate multiple LLM calls, and submit or update predictions through the SportsPredict REST API.

## SportsPredict API Surface

Available through the attached docs:

- `GET /events`: find the Probability Cup event.
- `GET /lobbies?event_id=...`: find the shared lobby and joined status.
- `POST /lobbies/{id}/join`: join if needed.
- `GET /matches?event_id=...&lobby_id=...`: discover open matches and `open_market_count`.
- `GET /markets?lobby_id=...`: list open binary markets. REST can fetch all open markets in one call; LLM/MCP clients should filter by match.
- `GET /predictions?lobby_id=...`: recover existing prediction ids and current probabilities.
- `POST /predictions/batch`: submit up to 50 new predictions.
- `PATCH /predictions/{id}`: update an existing prediction before close.
- `GET /results?lobby_id=...`: retrieve settled Brier scores for calibration.

Important limitations:

- No `current_price`, crowd forecast, bookmaker line, result probability, or market history is exposed.
- No webhook for settlement; polling is required.
- The leaderboard aggregate/RBP is not exposed by API.
- Probabilities must be 1-99 integers.
- Batch update is not exposed; updates are one PATCH each.
- Rate limit is 60 requests/minute/IP.

## Data Flow

1. Discover event and lobby.
2. Fetch matches and all open markets.
3. Fetch existing predictions.
4. Group markets by match.
5. Filter by close window:
   - default: markets closing within 168 hours;
   - set `MAX_HOURS_TO_CLOSE=0` to remove this filter.
6. For each selected match:
   - optional The Odds API lookup;
   - OpenAI web-search evidence summary;
   - four independent forecasting variants;
   - log-odds aggregation;
   - mild extremization and evidence-quality shrinkage;
   - integer 1-99 conversion.
7. Compare with existing predictions.
8. Submit creates in batches of 50 and PATCH material updates.
9. Write `logs/run-*.json` and `state/latest-run.json`.

## Model Strategy

Default:

- Research/evidence model with OpenAI key: `gpt-5.4-mini`.
- Forecast model with OpenAI key: `gpt-5.5`.
- Optional high-volume research/ensemble model: `grok-4.20-multi-agent-0309` via xAI when `XAI_API_KEY` is set.
- Grok-only mode is supported if `OPENAI_API_KEY` is absent.
- Not used: `gpt-5.5-pro`.

OpenAI docs checked for current API behavior:

- The [models docs](https://developers.openai.com/api/docs/models) list `gpt-5.5` as the flagship model and `gpt-5.4-mini` as the cheaper/lower-latency option.
- The [Responses API docs](https://developers.openai.com/api/reference/responses/overview/) describe direct text/structured/tool-using requests.
- The [structured outputs docs](https://developers.openai.com/api/docs/guides/structured-outputs) recommend Structured Outputs over JSON mode when schema adherence matters.
- The [web search tool docs](https://developers.openai.com/api/docs/guides/tools-web-search) show `tools: [{"type": "web_search"}]` for current Responses integrations.

xAI docs checked for current API behavior:

- The [xAI quickstart](https://docs.x.ai/developers/quickstart) documents OpenAI-compatible SDK usage with `base_url="https://api.x.ai/v1"`.
- The [Grok 4.20 Multi-Agent model page](https://docs.x.ai/developers/models/grok-4.20-multi-agent-beta-0309) lists the multi-agent model as a deep-research model with a 1M context window and structured outputs.
- The [multi-agent guide](https://docs.x.ai/developers/model-capabilities/text/multi-agent) recommends `grok-4.20-multi-agent` for coordinated research.
- The [xAI web search docs](https://docs.x.ai/developers/tools/web-search) and [X search docs](https://docs.x.ai/developers/tools/x-search) list `web_search` and `x_search` tools for OpenAI-compatible Responses API calls.

## Why This Architecture Matches the Literature

- Halawi et al. show that retrieval plus reasoning scaffolding beats raw model prompting by a large Brier margin.
- ForecastBench and Silicon Crowd show that aggregation and external crowd/market context are strong.
- Prompt-engineering studies show base-rate/frequency/step-back prompts are the only prompt-only components worth keeping, and even those are modest.
- Baron/Satopaa-style extremization motivates a mild log-odds extremization after aggregation.
- Pitfalls papers warn against overfitting backtests and correlated bets, so live logs and per-market calibration matter.

## Similar Competition Signals

ForecastBench's current tournament leaderboard, checked June 16, 2026, is dominated by superforecaster/crowd baselines, major frontier-model teams, and systems whose names imply ensembles, crowd adjustment, and decision-flow calibration. The ForecastBench tournament rules explicitly allow external tools, extra context, fine-tuning, and ensembling. The practical lesson for Jump is to treat the LLM as one component in a forecasting system, not the whole system.

## GitHub Action Operation

The workflow runs hourly at minute 7 UTC and also supports manual `workflow_dispatch`.

Repository secrets:

- `SPORTSPREDICT_API_KEY`
- `OPENAI_API_KEY` or `XAI_API_KEY`
- optional `ODDS_API_KEY`

Key environment controls:

- `SUBMIT=true`: actually writes predictions.
- `MAX_HOURS_TO_CLOSE=168`: forecast next seven days by default.
- `UPDATE_THRESHOLD_POINTS=2`: avoid noisy one-point updates.
- `CONCURRENCY=4`: bound concurrent match forecasts.
- `USE_GROK_RESEARCH=true`: use xAI multi-agent search for evidence when `XAI_API_KEY` exists.
- `USE_GROK_FORECAST=true`: add one Grok forecast variant to the ensemble when `XAI_API_KEY` exists.
- `EXTREMIZE_ALPHA=1.05`: mild log-odds extremization.
- `BASE_SHRINKAGE=0.04`: mild shrinkage toward 50.
- `LOW_EVIDENCE_SHRINKAGE=0.12`: stronger shrinkage when evidence is weak.

## Future Improvements

1. Add a true sports model layer: Elo/SPI-style team ratings, Poisson goal model, and de-vigged bookmaker consensus.
2. Add hard coherence repair for mutually exclusive markets and monotone totals.
3. Use settled `GET /results` to learn calibration by market family.
4. Persist historical odds snapshots and detect meaningful odds moves before updating.
5. Add a second bot key with an intentionally different strategy if platform rules allow two bots per user.
6. Add a nightly calibration report that decomposes Brier score by confidence bin and market type.
