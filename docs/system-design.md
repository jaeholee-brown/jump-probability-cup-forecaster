# Autonomous System Design

## Goal

Run an LLM forecasting bot for the Jump Trading Probability Cup from GitHub Actions every hour or half-hour. The bot should discover new, stale, or closing-soon questions, gather fresh context with Grok multi-agent research as the primary path, forecast with the optimized prompt, aggregate multiple LLM calls, and submit or update predictions through the SportsPredict REST API.

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
   - three xAI/Grok multi-agent web/X-search evidence passes when `XAI_API_KEY` exists;
   - OpenAI web-search evidence summary as fallback when Grok is unavailable;
   - configurable forecasting variants per configured forecast provider;
   - log-odds aggregation;
   - mild extremization and evidence-quality shrinkage;
   - integer 1-99 conversion.
7. Compare with existing predictions.
8. Submit creates in batches of 50 and PATCH material updates.
9. Write `logs/run-*.json` and `state/latest-run.json`.

## Model Strategy

Default:

- Primary high-volume research model: `grok-4.20-multi-agent-0309` via xAI when `XAI_API_KEY` is set.
- Research passes: `overview`, `late_news`, and `market_micro`.
- Grok forecast models: `grok-4.3` and `grok-4.20-0309-reasoning`.
- OpenAI forecast model: `gpt-5`.
- Claude forecast models: `claude-opus-4-8` and `claude-opus-4-6`.
- Fallback research/evidence model with OpenAI key: `gpt-5.4-mini`.
- Default forecast variants: one `base_rate_frequency` call per configured forecast model.
- Default forecast weights: OpenAI 1.0, Claude 1.0 per model, Grok 0.35 per model.
- Full prompt-ensemble mode: set `OPENAI_FORECAST_VARIANTS=all`, `GROK_FORECAST_VARIANTS=all`, and/or `CLAUDE_FORECAST_VARIANTS=all`.
- Grok-only mode is supported if `OPENAI_API_KEY` is absent.
- Not used: `gpt-5.5-pro`.

OpenAI docs checked for current API behavior:

- The [GPT-5 model docs](https://developers.openai.com/api/docs/models/gpt-5) list `gpt-5` as a reasoning model with configurable reasoning effort at $1.25/$10 per million input/output tokens.
- The [pricing docs](https://developers.openai.com/api/docs/pricing) list `gpt-5.5` as substantially more expensive, which is why the workflow now uses `gpt-5` by default.
- The [Responses API docs](https://developers.openai.com/api/reference/responses/overview/) describe direct text/structured/tool-using requests.
- The [structured outputs docs](https://developers.openai.com/api/docs/guides/structured-outputs) recommend Structured Outputs over JSON mode when schema adherence matters.
- The [web search tool docs](https://developers.openai.com/api/docs/guides/tools-web-search) show `tools: [{"type": "web_search"}]` for current Responses integrations.

xAI docs checked for current API behavior:

- The [xAI quickstart](https://docs.x.ai/developers/quickstart) documents OpenAI-compatible SDK usage with `base_url="https://api.x.ai/v1"`.
- The [xAI models page](https://docs.x.ai/developers/models) recommends `grok-4.3` for general text work and lists Grok 4.20 variants for multi-agent/reasoning use.
- The [Grok 4.20 Multi-Agent model page](https://docs.x.ai/developers/models/grok-4.20-multi-agent-beta-0309) lists the multi-agent model as a deep-research model with a 1M context window and structured outputs.
- The [multi-agent guide](https://docs.x.ai/developers/model-capabilities/text/multi-agent) recommends `grok-4.20-multi-agent` for coordinated research.
- The [xAI web search docs](https://docs.x.ai/developers/tools/web-search) and [X search docs](https://docs.x.ai/developers/tools/x-search) list `web_search` and `x_search` tools for OpenAI-compatible Responses API calls.

Anthropic docs checked for current API behavior:

- The [Claude Opus 4.8 docs](https://www.anthropic.com/claude/opus) list `claude-opus-4-8` as the API model name, with standard pricing of $5/$25 per million input/output tokens.
- The [Claude models overview](https://platform.claude.com/docs/en/about-claude/models/overview) lists `claude-opus-4-8` as Anthropic's strongest Opus-tier model and keeps earlier Opus IDs available.
- Anthropic's extended thinking docs state that thinking tokens are billed as output tokens, so cost estimates treat hidden reasoning as billed output.

## Why This Architecture Matches the Literature

- Halawi et al. show that retrieval plus reasoning scaffolding beats raw model prompting by a large Brier margin.
- ForecastBench and Silicon Crowd show that aggregation and external crowd/market context are strong.
- Prompt-engineering studies show base-rate/frequency/step-back prompts are the only prompt-only components worth keeping, and even those are modest.
- ForecastBench and Silicon Crowd support cross-model aggregation more directly than prompt-variant-only ensembling, so the bot defaults to model diversity, explicit component weights, and better retrieval instead of many same-family forecast votes.
- Baron/Satopaa-style extremization motivates a mild log-odds extremization after aggregation.
- Pitfalls papers warn against overfitting backtests and correlated bets, so live logs and per-market calibration matter.

## Similar Competition Signals

ForecastBench's current tournament leaderboard, checked June 16, 2026, is dominated by superforecaster/crowd baselines, major frontier-model teams, and systems whose names imply ensembles, crowd adjustment, and decision-flow calibration. The ForecastBench tournament rules explicitly allow external tools, extra context, fine-tuning, and ensembling. The practical lesson for Jump is to treat the LLM as one component in a forecasting system, not the whole system.

## GitHub Action Operation

The workflow runs hourly at minute 7 UTC and also supports manual `workflow_dispatch`.

Repository secrets:

- `SPORTSPREDICT_API_KEY`
- `OPENAI_API_KEY` or `XAI_API_KEY`
- optional `ANTHROPIC_API_KEY`
- optional `ODDS_API_KEY`

Key environment controls:

- `SUBMIT=true`: actually writes predictions.
- `MAX_HOURS_TO_CLOSE=168`: forecast next seven days by default.
- `ENABLE_UPDATE_GATE=true`: skip full reforecasting of already-fresh matches.
- `MAX_PREDICTION_AGE_HOURS=12`: outside the force window, refresh a fully predicted match after this many hours.
- `FORCE_REFORECAST_WITHIN_HOURS=6`: refresh every selected run inside the final pre-kickoff window.
- `UPDATE_THRESHOLD_POINTS=2`: avoid noisy one-point updates.
- `CONCURRENCY=4`: bound concurrent match forecasts.
- `USE_GROK_RESEARCH=true`: use xAI multi-agent search for evidence when `XAI_API_KEY` exists.
- `GROK_RESEARCH_PASSES=overview,late_news,market_micro`: specialized xAI research passes to run and merge.
- `GROK_RESEARCH_REASONING_EFFORT=medium`: xAI research effort.
- `USE_OPENAI_FORECAST=true`: run configured variants with `gpt-5` when `OPENAI_API_KEY` exists.
- `USE_GROK_FORECAST=true`: run configured variants with `GROK_FORECAST_MODELS` when `XAI_API_KEY` exists.
- `USE_CLAUDE_FORECAST=true`: run configured variants with `CLAUDE_FORECAST_MODELS` when `ANTHROPIC_API_KEY` exists.
- `GROK_FORECAST_MODELS=grok-4.3,grok-4.20-0309-reasoning`: comma-separated xAI forecast models.
- `CLAUDE_FORECAST_MODELS=claude-opus-4-8,claude-opus-4-6`: comma-separated Claude forecast models.
- `OPENAI_FORECAST_VARIANTS=base_rate_frequency`: comma-separated OpenAI variants, or `all`.
- `GROK_FORECAST_VARIANTS=base_rate_frequency`: comma-separated Grok variants, or `all`.
- `CLAUDE_FORECAST_VARIANTS=base_rate_frequency`: comma-separated Claude variants, or `all`.
- `OPENAI_FORECAST_WEIGHT=1.0`, `GROK_FORECAST_WEIGHT=0.35`, `CLAUDE_FORECAST_WEIGHT=1.0`: component weights before confidence/evidence adjustments.
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
