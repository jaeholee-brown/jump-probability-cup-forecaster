# Autonomous System Design

## Goal

Run an LLM forecasting bot for the Jump Trading Probability Cup from GitHub Actions. SportsPredict scores the latest prediction submitted before market close, not a time-weighted forecast path, so the bot should submit a baseline for coverage, use cheap Grok news monitoring between full updates, and concentrate the paid OpenAI/Grok/Claude ensemble on daily refreshes, near kickoff, or on material news.

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
   - optional cached Grok news-monitor summary;
   - targeted Firecrawl search+scrape snippets when `FIRECRAWL_API_KEY` exists and the match is close, volatile, low-evidence, high-disagreement, or already has material cached news;
   - four xAI/Grok multi-agent web/X-search evidence passes when `XAI_API_KEY` exists;
   - OpenAI web-search evidence summary as fallback when Grok is unavailable;
   - configurable forecasting variants per configured forecast provider;
   - log-odds aggregation;
   - mild extremization and evidence-quality shrinkage;
   - conservative coherence adjustments for explicit penalty-taker channels and player goal/SOT consistency;
   - integer 1-99 conversion.
7. Compare with existing predictions.
8. Submit creates in batches of 50 and PATCH material updates.
9. For skipped-but-open matches, run one Grok-only news monitor call per match group with web/X search. Replace cached news with the current summary even when nothing changed. Promote only affected market ids to the full ensemble if credible news is likely to move those markets by at least the threshold.
10. Fetch settled results and update calibration telemetry.
11. Write `state/in-progress-run.json` at run start, before forecasting starts, every heartbeat during long forecast batches, and after each completed match. Completed runs also write `logs/run-*.json`, `logs/calibration-*.json`, `logs/usage-*.json`, `state/latest-run.json`, `state/news-cache.json`, `state/calibration-report.json`, and `state/usage-ledger.json`.

## Model Strategy

Default:

- Primary high-volume research model: `grok-4.20-multi-agent-0309` via xAI when `XAI_API_KEY` is set.
- Research passes: `overview`, `base_rates`, `late_news`, `market_micro`, `lineup_roles`, and `volatile_market_anchors`.
- Evidence QA: after the research passes are merged, a low-reasoning Grok audit checks whether the evidence has stale claims, missing denominators, weak player-role assumptions, or unsupported volatile-market anchors. The audit is appended to the evidence package before OpenAI/Grok/Claude forecasts run.
- Base-rate pass policy: bucket markets by family, search for the narrowest reliable reference class, and report explicit frequencies/rates when available. For the current Jump docket this matters more than generic match odds because shots, cards, corners, fouls, offsides, halves, and player props outnumber vanilla match-winner/goal-total markets. The pass may use StatMuse FC, FBref/Stathead-style tables, StatBunker, API-Football/Sportmonks/Sportradar-style pages, official competition pages, and bookmaker lines as evidence sources, but natural-language stats answers should be corroborated or downweighted.
- Grok news monitor: `grok-4.20-multi-agent-0309` with low reasoning, `web_search`, and `x_search`, used as a change detector before spending on the full ensemble. It returns `affected_market_ids`, so narrow player/prop news can target only those markets. If the list is empty while `should_reforecast=true`, the runner treats the news as match-wide and reruns every listed market.
- Optional Firecrawl retrieval: targeted web-only searches, five scraped results per search, fed into Grok monitor/research as source context.
- Grok forecast models: `grok-4.3` (pack-fed) plus `grok-4.5` in independent-research mode. `grok-4.20-0309-reasoning` was dropped 2026-07-04: it was the most redundant ensemble member (0.974 probability correlation with grok-4.3) and the best leave-one-out counterfactual.
- Independent forecaster: `grok-4.5` runs its own `web_search`/`x_search` and is deliberately not given the shared research pack (it keeps the odds context and tournament-to-date anchors). With all pack-fed models at 0.99 error correlation, an independent information path is the only ensemble addition with support in the forecast-combination literature. A divergence guard drops the independent component when it sits more than 3 logits from the rest of the ensemble (retrieval failures such as "match already resolved" pages), logged in `dropped_independent_components`.
- Judgment slots run `grok-4.5` (evidence QA audit and news-monitor materiality); the parallel-search research passes stay on `grok-4.20-multi-agent-0309`, whose fan-out architecture is the point of that slot.
- OpenAI forecast model: `gpt-5`.
- Claude forecast models: `claude-opus-4-8` and `claude-opus-4-6`.
- Fallback research/evidence model with OpenAI key: `gpt-5.4-mini`.
- Default forecast variants: one `base_rate_frequency` call per configured forecast model.
- Default model-specific forecast weights: uniform `1.0` for every model. The 2026-07-03 audit of 517 common settled markets found no statistically distinguishable model differences (all pairwise |t| < 1.8) and 0.99 error correlation, because all models share the same evidence pack. Earlier non-uniform weights were fit to noise at n=48-74 and their ordering did not replicate.
- Claude forecast calls use Anthropic tool-choice structured output and do not enable extended thinking. The forecaster passes `reasoning_effort=none` for Claude so logs and future adapter behavior stay explicit; Opus still uses normal inference at standard token pricing.
- Calibration multipliers are report-only by default (`APPLY_CALIBRATION_WEIGHTS=false`). The pre-July formula compounded cumulative regret every run until weights hit the clamps; the formula is now stateless, but model reweighting stays off because the settled data cannot rank these models.
- Family correction layer (`ENABLE_FAMILY_CORRECTION=true`): each run fits per-family logit shifts plus one global recalibration slope from settled outcomes against the raw equal-weight component ensemble, shrunk by n/(n+12), damped 0.9, shift-clamped to ±0.6 and slope-clamped to [0.9, 1.4]. It replaces the old `EXTREMIZE_ALPHA=1.05` + `BASE_SHRINKAGE=0.04` pair, which nearly cancelled to identity. Validated out-of-sample on 50/50 through 80/20 time-ordered folds at -0.004 to -0.013 Brier per market (t -1.5 to -3.4). The same family stats feed `tournament_to_date` prompt anchors (realized family YES rates plus a ties-resolve-NO note for strictly-greater comparison markets).
- Every market is deterministically tagged with a market family before forecasting. The forecast payload includes the family, a broad prior range, and a decomposition hint, and `forecast-history.json` persists the same tag for calibration.
- A narrow post-aggregation coherence layer can raise player SOT/goal probabilities when the model explicitly identified a penalty-taker path and the match also has a penalty market. It also enforces the basic relation that a player's goal probability should not exceed that player's SOT probability when both markets exist. These repairs are logged in `coherence_adjustments` metadata.
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
- The base-rate instructions are deliberately operational: find a source, extract a rate, state the reference class, then update. The bot avoids asking models to perform formal Bayesian-reasoning theater because forecasting-specific prompt studies found that kind of instruction can be neutral or harmful.
- ForecastBench and Silicon Crowd support cross-model aggregation more directly than prompt-variant-only ensembling, so the bot defaults to model diversity, explicit component weights, and better retrieval instead of many same-family forecast votes.
- Baron/Satopaa-style extremization motivates a mild log-odds extremization after aggregation.
- Brier score is a strictly proper scoring rule, so settled results can support conservative online reweighting without inventing an untested sports model layer.
- Pitfalls papers warn against overfitting backtests and correlated bets, so live logs and per-market calibration matter.
- X/Twitter search is not treated as a proven forecasting edge by itself; it is a late-breaking discovery channel for lineups, injuries, suspensions, weather disruption, and beat-reporting signals that should be corroborated or discounted by the research prompt.

## Grok, X Search, and Firecrawl Policy

Because xAI credits are abundant here, Grok is used freely for monitoring and research but not allowed to dominate the final ensemble. The full paid ensemble still uses one GPT-5 forecast, two Claude Opus forecasts, and two downweighted Grok forecasts. The xAI surplus goes into evidence:

- scheduled Grok monitor checks on covered matches;
- six specialized Grok research passes plus one evidence-QA audit for selected matches;
- `web_search` plus `x_search` in both the monitor and research stages;
- cached summaries and source lists in `state/news-cache.json`.

The X-search rule is deliberately narrow. Grok should use X for official team/player/tournament posts, credible journalists, team reporters, lineup leaks, injury reports, suspension news, weather disruption, and fast-moving availability claims. It should not use raw fan sentiment as a forecast input. Brown et al. found Twitter tone added information to Betfair prices for EPL matches, especially immediately after goals and red cards, which is stronger evidence for time-sensitive discovery than for pre-kickoff sentiment modeling. The Jump markets close at kickoff, so the bot uses X mainly for pre-close lineup and availability discovery.

I did not find public evidence that Firecrawl itself has been used by winning forecasting bots. The analogous component in published and open-source systems is the retrieval/content layer: Halawi et al. use query generation, news retrieval, relevance ranking, and summarization; the Panshul42/Q2 bot fork describes Google Search, Google News, AskNews, Perplexity, and custom extraction; No-Stream uses AskNews, Gemini grounded search, optional Grok native search, Perplexity, Exa, and persistent research. Firecrawl fills the content-extraction slot: it turns targeted web results into clean markdown snippets with URLs and timestamps.

Firecrawl is therefore not used as a broad search replacement. It is used when clean source text is most likely to change the final pre-kickoff forecast:

- always inside `FIRECRAWL_FORCE_WITHIN_HOURS` before close, default 2h;
- for volatile player/card/shot/corner/lineup-sensitive markets inside `FIRECRAWL_VOLATILE_WITHIN_HOURS`, default 24h;
- when prior component spread is at least `FIRECRAWL_DISAGREEMENT_THRESHOLD_POINTS`, default 20 points;
- when prior evidence quality was low;
- when cached Grok news is material enough to justify a full ensemble rerun;
- for monitor checks inside 6h before close.

Full-research Firecrawl snippets and monitor Firecrawl snippets are cached in `state/news-cache.json` for auditability. The bot keeps the latest full-research snippet block plus the last three full-research Firecrawl blocks per match, capped before storage so the state file remains manageable.

## Similar Competition Signals

ForecastBench's current tournament leaderboard, checked June 16, 2026, is dominated by superforecaster/crowd baselines, major frontier-model teams, and systems whose names imply ensembles, crowd adjustment, and decision-flow calibration. The ForecastBench tournament rules explicitly allow external tools, extra context, fine-tuning, and ensembling. The practical lesson for Jump is to treat the LLM as one component in a forecasting system, not the whole system.

## GitHub Action Operation

The workflow has two scheduled modes and also supports manual `workflow_dispatch`:

- `17 */6 * * *`: schedule refresh mode. It polls SportsPredict for newly posted matches and updates `state/match-schedule.json`.
- `7,22,37,52 * * * *`: watchdog due-check mode. GitHub scheduled events are best-effort and can be delayed or dropped, so any received watchdog tick runs one cheap due check and restarts the self-dispatching loop when no scheduler run is already active or queued.
- `workflow_dispatch mode=loop`: self-dispatching due-check loop. Each loop covers about 40 minutes with 10-minute checks, then dispatches the next loop if no duplicate scheduler run is already active. This is the primary reliability layer; cron is now a watchdog/backstop, not the only timer.

Repository secrets:

- `SPORTSPREDICT_API_KEY`
- `OPENAI_API_KEY` or `XAI_API_KEY`
- optional `ANTHROPIC_API_KEY`
- optional `ODDS_API_KEY`
- optional `FIRECRAWL_API_KEY`

Key environment controls:

- `SUBMIT=true`: actually writes predictions.
- `MAX_HOURS_TO_CLOSE=168`: forecast next seven days by default.
- `ENABLE_UPDATE_GATE=true`: skip full reforecasting of already-fresh matches.
- `STALE_REFORECAST_WITHOUT_NEWS=true`: daily scheduled runs can refresh stale forecasts.
- `SCHEDULER_FORECAST_OFFSET_MINUTES=1440`: run the first full paid forecast about 24 hours before market lock for coverage. This is deliberately much earlier than the original 30-minute lead because GitHub scheduled runs can be delayed by minutes or hours.
- `SCHEDULER_FINAL_FORECAST_OFFSET_MINUTES=55`: run an unconditional full-ensemble pass about 55 minutes before lock, once confirmed lineups are typically published (~T-60). Added after the 2026-06-29..07-02 cohort locked 16-27 hours stale when the single news-check window was missed; a full forecast completed after this point also satisfies the pass.
- `SCHEDULER_NEWS_OFFSET_MINUTES=40`: run the cheap late-news check about 40 minutes before market lock as a late-shock safety net after the final pass; it retries on later ticks if the monitor call fails instead of being marked complete.
- `MAX_PREDICTION_AGE_HOURS=24`: default stale cadence for daily full refreshes.
- `FORCE_REFORECAST_WITHIN_HOURS=1.5`: full ensemble enters mandatory final-window cadence 90 minutes before kickoff.
- `FINAL_REFORECAST_MIN_INTERVAL_MINUTES=30`: avoid paid-model spam inside the final window while still catching confirmed lineups.
- `UPDATE_THRESHOLD_POINTS=2`: avoid noisy one-point updates.
- `CONCURRENCY=10`: bound concurrent match forecasts. The bot is already async across matches, Grok research passes, Firecrawl requests, and forecast models; this value controls how many match pipelines run at once. Forecasting also emits one-minute heartbeat logs and refreshes `state/in-progress-run.json` so long provider calls remain visible in GitHub Actions artifacts.
- `USE_GROK_NEWS_MONITOR=true`: run cheap Grok web/X checks on otherwise-skipped matches.
- `NEWS_MONITOR_MAX_HOURS_TO_CLOSE=168`: news-monitor eligible matches within the same close window.
- `NEWS_MONITOR_MATERIALITY_THRESHOLD_POINTS=2`: promote to full ensemble only when expected movement clears the update threshold.
- `USE_GROK_RESEARCH=true`: use xAI multi-agent search for evidence when `XAI_API_KEY` exists.
- `GROK_RESEARCH_PASSES=overview,base_rates,late_news,market_micro,lineup_roles,volatile_market_anchors`: specialized xAI research passes to run and merge.
- `GROK_RESEARCH_REASONING_EFFORT=medium`: xAI research effort.
- `USE_GROK_EVIDENCE_QA=true`: run a post-merge Grok audit before paid model forecasts.
- `GROK_EVIDENCE_QA_MODEL=grok-4.20-multi-agent-0309`, `GROK_EVIDENCE_QA_REASONING_EFFORT=low`: default evidence-audit model and effort.
- `GROK_NEWS_MODEL=grok-4.20-multi-agent-0309`, `GROK_NEWS_REASONING_EFFORT=low`: default news monitor model/effort.
- `USE_FIRECRAWL_RETRIEVAL=true`: allow Firecrawl snippets when targeted gate says they are useful.
- `FIRECRAWL_MODE=targeted`: use Firecrawl for close, volatile, low-evidence, high-disagreement, or material-news cases, not every run.
- `FIRECRAWL_SEARCH_QUERIES=2`, `FIRECRAWL_SEARCH_LIMIT=5`: default Firecrawl budget controls.
- `FIRECRAWL_FORCE_WITHIN_HOURS=2`, `FIRECRAWL_VOLATILE_WITHIN_HOURS=24`, `FIRECRAWL_DISAGREEMENT_THRESHOLD_POINTS=20`: Firecrawl gate controls.
- `USE_OPENAI_FORECAST=true`: run configured variants with `gpt-5` when `OPENAI_API_KEY` exists.
- `USE_GROK_FORECAST=true`: run configured variants with `GROK_FORECAST_MODELS` when `XAI_API_KEY` exists.
- `USE_CLAUDE_FORECAST=true`: run configured variants with `CLAUDE_FORECAST_MODELS` when `ANTHROPIC_API_KEY` exists.
- `GROK_FORECAST_MODELS=grok-4.3,grok-4.20-0309-reasoning`: comma-separated xAI forecast models.
- `CLAUDE_FORECAST_MODELS=claude-opus-4-8,claude-opus-4-6`: comma-separated Claude forecast models.
- `OPENAI_FORECAST_VARIANTS=base_rate_frequency`: comma-separated OpenAI variants, or `all`.
- `GROK_FORECAST_VARIANTS=base_rate_frequency`: comma-separated Grok variants, or `all`.
- `CLAUDE_FORECAST_VARIANTS=base_rate_frequency`: comma-separated Claude variants, or `all`.
- `FORECAST_MODEL_WEIGHTS=gpt-5=1.0,...`: uniform model weights (see Model Strategy for why).
- `APPLY_CALIBRATION_WEIGHTS=false`: model multipliers are report-only telemetry.
- `ENABLE_FAMILY_CORRECTION=true`: post-aggregation family logit shifts + global slope fit from settled results (see Model Strategy). `FAMILY_CORRECTION_PRIOR_N=12`, `FAMILY_CORRECTION_DAMP=0.9`, `FAMILY_CORRECTION_MIN_SETTLED=150`, `FAMILY_CORRECTION_MAX_SHIFT=0.6`.
- `EXTREMIZE_ALPHA=1.0`, `BASE_SHRINKAGE=0.0`: legacy pair now neutral; the fitted slope in the family correction supplies extremization instead.
- `LOW_EVIDENCE_SHRINKAGE=0.12`: stronger shrinkage when evidence is weak.
- `ENABLE_COHERENCE_ADJUSTMENTS=false`: post-aggregation coherence repairs are off; all three settled adjustments hurt (including one team-market misfire), so simplicity wins until there is evidence they help.
- `ODDS_SPORT_KEY=soccer_fifa_world_cup`, `ODDS_REGIONS=eu`, `ODDS_MARKETS=h2h,totals`: The Odds API consensus anchors on the free tier (500 credits/month; a call costs regions x markets). Responses are cached per match in `state/odds-cache.json` for `ODDS_CACHE_HOURS=3` (or `ODDS_CACHE_FINAL_MINUTES=45` inside the last `ODDS_CACHE_FINAL_WINDOW_HOURS=2`).

Cost telemetry:

- Every completed run includes `usage` and `usage_cumulative` in `state/latest-run.json`.
- `state/usage-ledger.json` keeps cumulative estimated cost by provider across recent runs.
- `logs/usage-*.json` stores per-call usage events for audit.
- The estimate uses configured token prices plus xAI server-side search tool counts when the SDK exposes them. Provider dashboards remain authoritative.

## Future Improvements

1. Add bookmaker consensus and de-vigged odds anchors if a reliable sports odds feed is available.
2. Add broader hard coherence repair for mutually exclusive markets and monotone totals.
3. Turn market-family telemetry into regularized family-specific model weights once enough settled results exist.
4. Persist historical odds snapshots and detect meaningful odds moves before updating.
5. Add domain-targeted Firecrawl allowlists for official team, tournament, weather, and lineup sources.
6. Add a second bot key with an intentionally different strategy if platform rules allow two bots per user.
