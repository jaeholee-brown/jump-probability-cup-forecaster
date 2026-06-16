# Follow-up Recommendations: Grok Research, Ensembles, Updates, and Cost

Last checked: 2026-06-16.

## 1. Replace OpenAI Research With Grok Multi-Agent?

Yes for the primary research path, with one caveat: keep direct structured odds when available.

The bot should treat `grok-4.20-multi-agent-0309` as the default research engine when `XAI_API_KEY` exists. The reason is practical rather than mystical: the strongest LLM-forecasting evidence supports retrieval and evidence quality more than prompt phrasing. Halawi et al., "Approaching Human-Level Forecasting with Language Models" ([arXiv:2402.18563](https://arxiv.org/abs/2402.18563)), improved Brier from 0.208 for a raw GPT-4 baseline to 0.179 for a retrieval, summarization, reasoning, fine-tuning, and ensembling system over 914 post-cutoff test questions. That is much larger than prompt-only effects in later work.

xAI's docs describe Grok 4.20 Multi-Agent as parallel collaborating agents for deep research, with `web_search` and `x_search` tools available in the Responses API ([model page](https://docs.x.ai/developers/models/grok-4.20-multi-agent-beta-0309), [multi-agent guide](https://docs.x.ai/developers/model-capabilities/text/multi-agent)). Given the user's $2,500 Grok credit and 450 RPM / 2.5M TPM limit, Grok should do all routine match research, late news, lineup, injury, weather, and social-source checking.

Do not remove the optional odds feed. Bookmaker odds are not just another text source. They are a compact real-money market signal. ForecastBench and "Wisdom of the Silicon Crowd" both point toward crowd/market anchors being high-value. Jump's API exposes no platform current price or crowd forecast, so public odds are the closest legal proxy.

Repo consequence: docs and defaults now describe Grok multi-agent as primary. OpenAI web search is fallback/secondary.

## 2. Prompt Variants and Ensemble Ranking

Current forecast ensemble:

- `base_rate_frequency`: force outside-view base rates and frequency reasoning first.
- `balanced_scratchpad`: rephrase, consider yes/no cases, and do a calibration check.
- `late_information`: overweight late-breaking lineups, injuries, odds shifts, tactical news, and weather.
- `coherence_checker`: check related markets on the same match for obvious probability inconsistency.

Default configured forecast providers do not all run all variants. The evidence says model diversity is more valuable than prompt-only diversity, so the workflow defaults to:

- OpenAI `gpt-5.5` x 1 variant: `base_rate_frequency`.
- xAI `grok-4.20-multi-agent-0309` x 4 variants.
- Anthropic `claude-opus-4-6` x 1 variant: `base_rate_frequency`.

Set `OPENAI_FORECAST_VARIANTS=all` or `CLAUDE_FORECAST_VARIANTS=all` for expensive full prompt ensembling when desired. With all keys present, the default is 6 forecast batches per match-cycle; full strong mode is 12.

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
- Default: run one high-quality `base_rate_frequency` forecast on GPT-5.5 and Claude Opus 4.6, plus all four Grok variants. This keeps model diversity while avoiding 4x paid-provider prompt costs.
- Full strong mode: run all four variants on GPT-5.5, Grok 4.20 Multi-Agent, and Claude Opus 4.6 only for close-to-kickoff, high-disagreement, low-evidence, or manual high-value refreshes.
- If paid budget gets tight, keep Grok on every selected match and reserve GPT-5.5/Claude for close-to-kickoff, high-disagreement, or high-value matches.

## 3. Blind Reruns vs Cheap Update Gate

Use a gate. Do not blindly full-reforecast every open match every run with the full OpenAI/Grok/Claude ensemble.

The literature and public bot practice favor selective updating:

- Halawi et al. gain from fresh retrieval, but their system is a research-and-ensemble pipeline, not a constant blind rerun loop.
- Superforecasting practice rewards updating when evidence changes, but not random churn.
- Prompt-only retesting is noisy, and LLM outputs are correlated. Blind reruns can create false update confidence from model variance.
- The official Metaculus template defaults to skipping previously forecasted questions, and the No-Stream bot spends extra research/stacker budget mainly when models disagree or factual gaps are found.

Best execution for Jump:

1. Always forecast new markets immediately.
2. Always refresh inside the final pre-kickoff window because lineups, injuries, weather, and odds are most valuable then.
3. Outside that window, refresh stale forecasts on a cadence that depends on time-to-close.
4. Shorten cadence for matches with high model disagreement, low confidence, low evidence quality, or volatile market families such as player goals, cards, shots, and lineup-sensitive props.
5. Preserve `state/forecast-history.json` across GitHub Action runs so the gate can use its own prior component spread and evidence quality.
6. Add a cheap Grok change detector later for all open matches if we want maximum quality with controlled cost:
   - Inputs: match, markets, existing probabilities, last update time, time to close, last evidence summary, last component spread.
   - Tools: `web_search` plus `x_search`, date-restricted to recent days where possible.
   - Output: `should_reforecast`, `estimated_delta_points`, `new_evidence_summary`, `sources`, `reason`.
   - Trigger full forecast if new material evidence appears, estimated move is at least 2 points, kickoff is close, or current prediction is stale.
7. Reuse the change detector's evidence summary in the full forecast to avoid doing two totally independent research passes.

Implemented now:

- `ENABLE_UPDATE_GATE=true`
- `MAX_PREDICTION_AGE_HOURS=12`
- `FORCE_REFORECAST_WITHIN_HOURS=6`

This gate is now stateful and slightly smarter than a fixed timer. It avoids suppressing new markets, refreshes every selected run in the final six hours, refreshes stable far-future matches slowly, and refreshes uncertain or volatile matches faster. It records per-market probability, component count, component spread, evidence quality, confidence, and per-match worst-case summaries.

Recommended next implementation:

- Add `GROK_CHANGE_DETECTOR=true`.
- Run a low-effort Grok multi-agent check for open matches that already have predictions and are outside the force window.
- Save the change-detector evidence into `logs/run-*.json`.
- Pass that evidence into the full forecaster when a reforecast is triggered.

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

- OpenAI `gpt-5.5`: $5.00 / 1M input tokens, $30.00 / 1M output tokens.
- OpenAI `gpt-5.4-mini`: $0.75 / 1M input tokens, $4.50 / 1M output tokens.
- OpenAI web search: $10 / 1k calls, search content tokens free.
- xAI `grok-4.20-multi-agent-0309`: $1.25 / 1M input tokens, $2.50 / 1M output tokens.
- xAI `web_search` and `x_search`: $5 / 1k calls each. xAI explicitly bills reasoning tokens, completion tokens, and tool invocations.
- Anthropic `claude-opus-4-6`: $5 / 1M input tokens, $25 / 1M output tokens. Anthropic bills full thinking tokens, not just visible summaries.

Assumptions:

- One full match-cycle forecasts all markets for one match, about 9-10 markets.
- Grok research call: 12K billed input tokens, 8K billed reasoning/completion tokens, and 3 web/X tool invocations.
- Each forecast call: 12K billed input tokens. OpenAI and Grok forecast calls assume 3.5K billed reasoning/completion tokens at `REASONING_EFFORT=medium`; Claude defaults to no explicit extended-thinking parameter, so the base estimate uses 1.5K visible output tokens, with a sensitivity case of 3.5K if hidden/adaptive thinking is billed similarly.
- xAI research cost: `12K * $1.25/M + 8K * $2.50/M + 3 * $5/1K = $0.050`.
- OpenAI forecast call: `12K * $5/M + 3.5K * $30/M = $0.165`.
- Grok forecast call: `12K * $1.25/M + 3.5K * $2.50/M = $0.0238`.
- Claude forecast call: `12K * $5/M + 1.5K * $25/M = $0.0975`; sensitivity with 3.5K billed output is `$0.1475`.
- Cross-model-only cycle: one Grok research call plus one forecast from each provider, about `$0.336`.
- Default hybrid cycle: one Grok research call, four Grok variants, one OpenAI forecast, and one Claude forecast, about `$0.4075`.
- Full strong cycle: one Grok research call plus four variants for all three providers, about `$1.195`.
- Bot considers matches within the default 168-hour close window, not the full 919-hour event window.

Approximate costs:

| Scenario | Match cycles | Cross-model only | Default hybrid | Full strong mode |
|---|---:|---:|---:|---:|
| Forecast once per match | 104 | $35 | $42 | $124 |
| Selective: 10 refreshes per match | 1,040 | $350 | $424 | $1,243 |
| Expected stateful gate: 20 refreshes per match | 2,080 | $699 | $848 | $2,486 |
| Stateful gate planning upper bound: about 26 refreshes per match | 2,704 | $909 | $1,102 | $3,231 |
| Blind hourly within 168h window | 17,472 | $5,875 | $7,120 | $20,875 |

Default hybrid per-cycle spend is approximately `$0.145` xAI, `$0.165` OpenAI, and `$0.098` Claude. The user's $2,500 xAI credit covers about 17,200 default-hybrid xAI match-cycles under these assumptions. The user's $500 Claude credit covers about 5,100 default Claude forecast calls, or about 3,400 calls under the higher hidden-output sensitivity case.

Interpretation:

- Prompt ensembling across OpenAI and Claude is the main marginal cost. Moving from cross-model-only to default hybrid costs only about `$0.071` per match-cycle because the extra variants are cheap Grok calls. Moving from default hybrid to full strong mode costs about `$0.788` more per match-cycle, mostly OpenAI and Claude.
- Full strong mode needs to improve Brier enough to justify about 3x the default hybrid cost. The forecasting literature does not support that as a blanket default.
- OpenAI remains the limiting paid marginal cost. The realistic default-hybrid stateful gate is likely a few hundred dollars of OpenAI spend over the tournament, not thousands, unless the gate is effectively disabled.

## Recommendation

Use Grok multi-agent as the research engine, keep direct odds as an anchor, ensemble both prompts and models when paid budget is acceptable, and gate reforecasts. For this tournament, the highest expected-value architecture is:

1. Hourly GitHub Action.
2. Stateful gate on by default.
3. Default hybrid OpenAI/Grok/Claude ensemble on selected matches.
4. Full forecast on new markets, stale markets, material evidence changes, and all matches inside six hours to close.
5. Aggregate in log-odds space, mildly extremize, and patch only changes of at least 2 points.
