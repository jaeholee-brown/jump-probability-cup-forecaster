# Follow-up Recommendations: Grok Research, Ensembles, Updates, and Cost

Last checked: 2026-06-16.

## 1. Replace OpenAI Research With Grok Multi-Agent?

Yes for the primary research path, with one caveat: keep direct structured odds when available.

The bot should treat `grok-4.20-multi-agent-0309` as the default research engine when `XAI_API_KEY` exists. The reason is practical rather than mystical: the strongest LLM-forecasting evidence supports retrieval and evidence quality more than prompt phrasing. Halawi et al., "Approaching Human-Level Forecasting with Language Models" ([arXiv:2402.18563](https://arxiv.org/abs/2402.18563)), improved Brier from 0.208 for a raw GPT-4 baseline to 0.179 for a retrieval, summarization, reasoning, fine-tuning, and ensembling system over 914 post-cutoff test questions. That is much larger than prompt-only effects in later work.

xAI's docs describe Grok 4.20 Multi-Agent as parallel collaborating agents for deep research, with `web_search` and `x_search` tools available in the Responses API ([model page](https://docs.x.ai/developers/models/grok-4.20-multi-agent-beta-0309), [multi-agent guide](https://docs.x.ai/developers/model-capabilities/text/multi-agent)). Given the user's $2,500 Grok credit and 450 RPM / 2.5M TPM limit, Grok should do all routine match research, late news, lineup, injury, weather, social-source checking, and base-rate gathering.

If forced to spend more xAI budget, spend it on evidence, not extra same-family forecast votes. The best marginal use is a decomposed `base_rates` pass that explicitly searches for historical team, league, player, and market-family frequencies. That is more literature-aligned than asking Grok to cast many additional ballots, because retrieval/system effects are larger and better-supported than prompt-only variance. The current research stack is therefore `overview`, `base_rates`, `late_news`, and `market_micro`.

Firecrawl is useful, but as a retrieval plumbing layer rather than as a replacement for Grok's synthesis. Firecrawl should do search+scrape to provide clean, recent public snippets to the Grok research passes. Firecrawl search is priced in credits per search-result batch and scrape is priced per page, so the default uses two web-only searches per match-cycle and five scraped results per query. That is roughly `2 * (2 search credits + 5 scrape credits) = 14` Firecrawl credits per match-cycle. With 50,000 credits, this supports about 3,571 full match-cycles. The planning upper bound below uses about 2,704 match-cycles, or about 37,856 Firecrawl credits. Enabling both `web` and `news` sources would roughly double result volume and can burn through credits; web-only is the higher-confidence default.

Do not remove the optional odds feed. Bookmaker odds are not just another text source. They are a compact real-money market signal. ForecastBench and "Wisdom of the Silicon Crowd" both point toward crowd/market anchors being high-value. Jump's API exposes no platform current price or crowd forecast, so public odds are the closest legal proxy.

Repo consequence: docs and defaults now describe Grok multi-agent as primary. OpenAI web search is fallback/secondary.

## 2. Prompt Variants and Ensemble Ranking

Current forecast ensemble:

- `base_rate_frequency`: force outside-view base rates and frequency reasoning first.
- `balanced_scratchpad`: rephrase, consider yes/no cases, and do a calibration check.
- `late_information`: overweight late-breaking lineups, injuries, odds shifts, tactical news, and weather.
- `coherence_checker`: check related markets on the same match for obvious probability inconsistency.

Default configured forecast providers do not all run all variants. The evidence says model diversity is more valuable than prompt-only diversity, and xAI forecast calls are more correlated with each other than OpenAI/Anthropic calls. The workflow now defaults to:

- OpenAI `gpt-5` x 1 variant: `base_rate_frequency`, weight 1.0.
- xAI `grok-4.3` x 1 variant: `base_rate_frequency`, weight 0.4.
- xAI `grok-4.20-0309-reasoning` x 1 variant: `base_rate_frequency`, weight 0.6.
- Anthropic `claude-opus-4-8` x 1 variant: `base_rate_frequency`, weight 0.7.
- Anthropic `claude-opus-4-6` x 1 variant: `base_rate_frequency`, weight 0.8.

Set `OPENAI_FORECAST_VARIANTS=all`, `GROK_FORECAST_VARIANTS=all`, or `CLAUDE_FORECAST_VARIANTS=all` for full prompt ensembling when desired. With all keys present, the default is 5 forecast batches per match-cycle.

Stack ranking:

1. **Ensemble both models and prompts**. Best default when budget permits.
2. **Ensemble models**. More important than prompt variants.
3. **Ensemble prompt variants within one model**. Useful, but the weakest of the three.

Why: prompt-only evidence is weak. Schoenegger et al., "Prompt Engineering Large Language Models' Forecasting Capabilities" ([arXiv:2506.01578](https://arxiv.org/abs/2506.01578)), tested 38 prompts across four models and 100 ForecastBench questions. In stricter mixed-effects results, no positive prompt effect survived correction; several apparently sophisticated prompts hurt. The only prompt ingredients worth keeping are base-rate/frequency/step-back style components, and even those should be viewed as modest.

Model/crowd diversity has better support. ForecastBench reports that geometric/log-odds aggregation slightly beat median/trimmed mean for top model aggregates (0.194 vs 0.197 Brier), and its top systems lean on tools, extra context, ensembling, and crowd context ([ForecastBench](https://www.forecastbench.org/)). Schoenegger et al., "Wisdom of the Silicon Crowd" ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11800985/)), found a 12-LLM ensemble on live Metaculus questions statistically indistinguishable from a 925-human crowd in one study, and showed human median context improved GPT-4 and Claude forecasts 17-28 percent in another.

Public competition code points the same way. The official Metaculus starter bot runs `research_reports_per_question=1`, `predictions_per_research_report=5`, and skips previously forecasted questions by default ([Metaculus template](https://github.com/Metaculus/metac-bot-template)). The more aggressive No-Stream Metaculus bot uses a six-model ensemble, conditional stacking, targeted disagreement research, multiple research providers, gap-fill, and median fallback when models agree ([No-Stream bot](https://github.com/No-Stream/metaculus-bot)).

Recommended Jump stack:

- Grok multi-agent primary research for every selected match.
- Grok-only fallback mode if no OpenAI key exists.
- Default: use GPT-5, two Claude Opus generations, and two lightly weighted Grok forecast models. This keeps model diversity while preventing xAI from overruling OpenAI/Anthropic through volume.
- Spend the xAI surplus on four specialized Grok research passes: stable overview, base rates, late-news/lineups, and market-specific micro evidence.
- If paid budget gets tight, keep Grok research and one Grok forecast on every selected match, then reserve GPT-5/Claude for close-to-kickoff, high-disagreement, or high-value matches.

## 3. Blind Reruns vs Cheap Update Gate

Use a gate. Do not blindly full-reforecast every open match every run with the full OpenAI/Grok/Claude ensemble.

SportsPredict is latest-only, not Metaculus time-weighted. The attached API docs say `closing_time` matches match start and the latest value at market close is scored. Early forecasts are therefore operational insurance and context-building, not scoring alpha by themselves. The scoring target is the best reliable pre-kickoff update.

The literature and public bot practice favor selective updating:

- Halawi et al. gain from fresh retrieval, but their system is a research-and-ensemble pipeline, not a constant blind rerun loop.
- Superforecasting practice rewards updating when evidence changes, but not random churn.
- Prompt-only retesting is noisy, and LLM outputs are correlated. Blind reruns can create false update confidence from model variance.
- The official Metaculus template defaults to skipping previously forecasted questions, and the No-Stream bot spends extra research/stacker budget mainly when models disagree or factual gaps are found.

Best execution for Jump:

1. Forecast new markets immediately to avoid missing coverage and to create fallback predictions if providers fail near kickoff.
2. Run the workflow every 15 minutes, but do not run the full paid ensemble every time.
3. Use Grok web/X monitoring for already-covered matches. Promote to the full ensemble only if new evidence is credible and likely to move at least one market by 2+ points.
4. Enter mandatory full-ensemble cadence in the final 90 minutes, with a 30-minute minimum interval. This catches confirmed lineups without paying for every 15-minute tick.
5. Shorten Grok monitor cadence for matches with high model disagreement, low confidence, low evidence quality, or volatile market families such as player goals, cards, shots, and lineup-sensitive props.
6. Preserve `state/forecast-history.json` and `state/news-cache.json` across GitHub Action runs so the gates can use prior component spread, evidence quality, cached news, and raw Firecrawl snippets.
7. Reuse the change detector's evidence summary in the full forecast to avoid doing two totally independent research passes.

Implemented now:

- `ENABLE_UPDATE_GATE=true`
- `MAX_PREDICTION_AGE_HOURS=12`
- `STALE_REFORECAST_WITHOUT_NEWS=false`
- `FORCE_REFORECAST_WITHIN_HOURS=1.5`
- `FINAL_REFORECAST_MIN_INTERVAL_MINUTES=30`
- `USE_GROK_NEWS_MONITOR=true`
- Five-attempt exponential retry around OpenAI, Anthropic, xAI, and Firecrawl calls.
- If a forecast component still fails, aggregate over the surviving components with the configured weights renormalized implicitly by weighted log-odds averaging.
- Settled-result calibration through `GET /results`, written to `state/calibration-report.json` and timestamped `logs/calibration-*.json`.

This gate is now stateful and latest-only. It avoids suppressing new markets, avoids paid stale refreshes without news, refreshes on a controlled final-window cadence, and uses Grok news monitoring to promote material changes. It records per-market probability, component count, component spread, evidence quality, confidence, and per-match worst-case summaries.

Calibration is deliberately conservative. The SportsPredict results endpoint returns settled Brier scores and submitted probabilities, but not explicit outcomes. The outcome is inferable because Brier is either `p^2` for a no-resolution or `(1-p)^2` for a yes-resolution. The bot scores each saved model component against the inferred outcome, compares each model's mean Brier to the ensemble mean, and proposes multiplicative weight updates using an exponentially weighted regret rule with prior-count shrinkage. This is rooted in proper scoring and online aggregation theory, but it avoids overfitting a tiny early sample. The report is designed for a daily coding-agent review: model/provider Brier, sample counts, current multipliers, suggested multipliers, and per-market component records are all visible.

Firecrawl is now targeted. Use it for close matches, volatile markets, low-evidence or high-disagreement forecasts, and material Grok-news cases. Its raw snippets are cached in `state/news-cache.json` for auditability when used by the monitor.

The implementation now also caches Firecrawl snippets used by the full research pass, not only the monitor pass. This matters operationally because a daily review should be able to inspect the actual source text that influenced a submitted update. The cache keeps the latest full-research snippet block and the three most recent full-research Firecrawl blocks per match.

Grok news polling is the default cheap gate. Already-covered matches are not rerun with GPT-5 and Claude just because time has passed. Instead, Grok 4.20 Multi-Agent checks web and X on a cadence that tightens near kickoff: 6h far out, 3h inside 72h, 1h inside 24h, 30m inside 6h, and 15m inside 2h, halved for volatile, low-evidence, or high-disagreement matches. It promotes to the full ensemble only when credible new information is likely to move at least one market by the update threshold.

X/Twitter is useful, but the evidence is narrower than "social sentiment wins." Brown et al. studied 13.8m EPL tweets and found Twitter tone had information beyond Betfair prices especially after goals and red cards. Since Jump closes markets at kickoff, the bot should not spend effort on in-play sentiment. It should use X for pre-close discovery: official team posts, confirmed lineups, credible reporters, injury/suspension news, travel/weather disruption, and fast player-availability updates. The prompt now tells Grok to discount fan speculation and social-only claims unless corroborated.

## 4. Platform Facts That Shape the System

Public Jump/SportsPredict pages and the attached API docs say:

- Probability Cup event: `2026-06-11T15:00:00Z` to `2026-07-19T22:00:00Z`, about 38.3 days or 919 hours.
- Jump's public announcement says the competition begins June 11, 2026 and runs through July 19, 2026.
- Public SportsPredict/LinkedIn campaign copy says 104 matches and 1,000+ probability questions.
- The API docs' older examples mention about 72 matches and about 720 group-stage-scale markets; the live public tournament framing is broader, so cost estimates use 104 matches and 1,000 markets.
- Markets close at match start.
- The API exposes event, lobby, matches, markets, predictions, and results.
- It does not expose current market price, crowd forecast, bookmaker line, leaderboard aggregate/RBP, market history, or webhooks.
- Predictions can be updated before market close with `PATCH /predictions/{id}`.
- Batch creates support up to 50 predictions, but updates are one PATCH each.

Those constraints mean the bot should forecast per match, submit in batches, patch only material changes, and build its own market/odds context from public sources.

## 5. Cost Model

Prices checked 2026-06-16:

- OpenAI `gpt-5`: $1.25 / 1M input tokens, $10.00 / 1M output tokens.
- OpenAI `gpt-5.4-mini`: $0.75 / 1M input tokens, $4.50 / 1M output tokens.
- OpenAI web search: $10 / 1k calls, search content tokens free.
- xAI `grok-4.3`, `grok-4.20-multi-agent-0309`, and `grok-4.20-0309-reasoning`: $1.25 / 1M input tokens, $2.50 / 1M output tokens.
- xAI `web_search` and `x_search`: $5 / 1k calls each. xAI explicitly bills reasoning tokens, completion tokens, and tool invocations.
- Anthropic `claude-opus-4-8` and `claude-opus-4-6`: $5 / 1M input tokens, $25 / 1M output tokens. Anthropic bills full thinking tokens, not just visible summaries.

Assumptions:

- One full match-cycle forecasts all markets for one match, about 9-10 markets.
- Four Grok research passes, each about 9K billed input tokens, 6K billed reasoning/completion tokens, and 2 web/X tool invocations.
- Each forecast call: 12K billed input tokens. OpenAI and Grok forecast calls assume 3.5K billed reasoning/completion tokens at `REASONING_EFFORT=medium`; Claude defaults to no explicit extended-thinking parameter, so the base estimate uses 1.5K visible output tokens, with a sensitivity case of 3.5K if hidden/adaptive thinking is billed similarly.
- xAI research cost: `4 * (9K * $1.25/M + 6K * $2.50/M + 2 * $5/1K) = $0.145`.
- OpenAI GPT-5 forecast call: `12K * $1.25/M + 3.5K * $10/M = $0.050`.
- Grok forecast call: `12K * $1.25/M + 3.5K * $2.50/M = $0.0238`.
- Claude forecast call: `12K * $5/M + 1.5K * $25/M = $0.0975`; sensitivity with 3.5K billed output is `$0.1475`.
- Firecrawl retrieval cost: about 14 credits per match-cycle under the default two-query, five-result-per-query web-only configuration.
- Default cycle: four Grok research passes, two lightly weighted Grok forecasts, one GPT-5 forecast, and two Claude Opus forecasts, about `$0.438`, plus about 14 Firecrawl credits if Firecrawl is enabled.
- Provider split per cycle: about `$0.193` xAI, `$0.050` OpenAI, and `$0.195` Anthropic.
- Bot considers matches within the default 168-hour close window, not the full 919-hour event window.

Approximate costs:

| Scenario | Match cycles | xAI share | OpenAI share | Claude share | Total | Firecrawl credits |
|---|---:|---:|---:|---:|---:|---:|
| Forecast once per match | 104 | $20 | $5 | $20 | $46 | 1,456 |
| Latest-only expected: initial + 2 final + 1 material-news average | 416 | $80 | $21 | $81 | $182 | 5,824 |
| Latest-only high-activity: initial + 4 final + 3 material-news average | 832 | $160 | $42 | $162 | $364 | 11,648 |
| Legacy selective: 10 refreshes per match | 1,040 | $200 | $52 | $203 | $455 | 14,560 |
| Legacy stateful planning upper bound: about 26 refreshes per match | 2,704 | $521 | $135 | $527 | $1,183 | 37,856 |
| Blind hourly within 168h window | 17,472 | $3,363 | $874 | $3,407 | $7,644 | 244,608 |

The user's $2,500 xAI credit covers about 13,000 default xAI match-cycles under these assumptions. The user's $500 Claude credit covers about 2,560 default cycles with two Claude calls, or about 1,690 cycles under the higher hidden-output sensitivity case. The hypothetical 50,000 Firecrawl credits comfortably covers the latest-only plan and the old planning upper bound, but not blind hourly refreshes for every eligible match.

Interpretation:

- Replacing GPT-5.5 with GPT-5 cuts the OpenAI forecast call from about `$0.165` to about `$0.050`, making Claude the main paid marginal cost.
- The two Grok forecasts have a combined raw weight of 1.0, equal to GPT-5 alone and below the two-Claude total of 1.5, so xAI can contribute useful disagreement without dominating the full ensemble.
- The extra xAI budget is better spent on research decomposition, base-rate gathering, and source cleanup than forecast over-voting because retrieval/evidence quality has the strongest measured effect in the forecasting literature.

## Recommendation

Use Grok multi-agent as the research engine, keep direct odds as an anchor, ensemble both prompts and models when paid budget is acceptable, and gate reforecasts. For this tournament, the highest expected-value architecture is:

1. 15-minute GitHub Action.
2. Stateful latest-only gate on by default.
3. Grok news monitor on already-covered matches.
4. Default hybrid OpenAI/Grok/Claude ensemble on new markets, material-news promotions, and the final 90-minute pre-kickoff cadence.
5. Aggregate in log-odds space, mildly extremize, and patch only changes of at least 2 points.
