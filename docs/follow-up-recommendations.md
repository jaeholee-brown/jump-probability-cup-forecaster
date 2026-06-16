# Follow-up Recommendations: Grok Research, Ensembles, Updates, and Cost

Last checked: 2026-06-16.

## 1. Replace OpenAI Research With Grok Multi-Agent?

Yes for the primary research path, with one caveat: keep direct structured odds when available.

The bot should treat `grok-4.20-multi-agent-0309` as the default research engine when `XAI_API_KEY` exists. The reason is practical rather than mystical: the strongest LLM-forecasting evidence supports retrieval and evidence quality more than prompt phrasing. Halawi et al., "Approaching Human-Level Forecasting with Language Models" ([arXiv:2402.18563](https://arxiv.org/abs/2402.18563)), improved Brier from 0.208 for a raw GPT-4 baseline to 0.179 for a retrieval, summarization, reasoning, fine-tuning, and ensembling system over 914 post-cutoff test questions. That is much larger than prompt-only effects in later work.

xAI's docs describe Grok 4.20 Multi-Agent as parallel collaborating agents for deep research, with `web_search` and `x_search` tools available in the Responses API ([model page](https://docs.x.ai/developers/models/grok-4.20-multi-agent-beta-0309), [multi-agent guide](https://docs.x.ai/developers/model-capabilities/text/multi-agent)). Given the user's $2,500 Grok credit and 450 RPM / 2.5M TPM limit, Grok should do all routine match research, late news, lineup, injury, weather, and social-source checking.

Do not remove the optional odds feed. Bookmaker odds are not just another text source. They are a compact real-money market signal. ForecastBench and "Wisdom of the Silicon Crowd" both point toward crowd/market anchors being high-value. Jump's API exposes no platform current price or crowd forecast, so public odds are the closest legal proxy.

Repo consequence: docs and defaults now describe Grok multi-agent as primary. OpenAI web search is fallback/secondary.

## 2. Prompt Variants and Ensemble Ranking

Current prompt variants:

- `base_rate_frequency`: force outside-view base rates and frequency reasoning first.
- `balanced_scratchpad`: rephrase, consider yes/no cases, and do a calibration check.
- `late_information`: overweight late-breaking lineups, injuries, odds shifts, tactical news, and weather.
- `coherence_checker`: check related markets on the same match for obvious probability inconsistency.
- `grok_multi_agent_check`: added only when both OpenAI and Grok are available, giving Grok an independent challenge/search role.

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
- If OpenAI budget is acceptable, use GPT-5.5 for four prompt variants and Grok as an independent check.
- If OpenAI budget is constrained, use Grok-only with the four prompt variants, then add one small OpenAI/Gemini/Claude check only on high-disagreement or high-value matches.

## 3. Blind Reruns vs Cheap Update Gate

Use a gate. Do not blindly full-reforecast every open match every run unless using Grok-only and deliberately spending the free credit aggressively.

The literature and public bot practice favor selective updating:

- Halawi et al. gain from fresh retrieval, but their system is a research-and-ensemble pipeline, not a constant blind rerun loop.
- Superforecasting practice rewards updating when evidence changes, but not random churn.
- Prompt-only retesting is noisy, and LLM outputs are correlated. Blind reruns can create false update confidence from model variance.
- The official Metaculus template defaults to skipping previously forecasted questions, and the No-Stream bot spends extra research/stacker budget mainly when models disagree or factual gaps are found.

Best execution for Jump:

1. Always forecast new markets immediately.
2. Always refresh inside the final pre-kickoff window because lineups, injuries, weather, and odds are most valuable then.
3. Outside that window, refresh stale forecasts on a fixed cadence.
4. Add a cheap Grok change detector for all open matches if we want maximum quality with controlled cost:
   - Inputs: match, markets, existing probabilities, last update time, time to close.
   - Tools: `web_search` plus `x_search`, date-restricted to recent days where possible.
   - Output: `should_reforecast`, `estimated_delta_points`, `new_evidence_summary`, `sources`, `reason`.
   - Trigger full forecast if new material evidence appears, estimated move is at least 2 points, kickoff is close, or current prediction is stale.
5. Reuse the change detector's evidence summary in the full forecast to avoid doing two totally independent research passes.

Implemented now:

- `ENABLE_UPDATE_GATE=true`
- `MAX_PREDICTION_AGE_HOURS=12`
- `FORCE_REFORECAST_WITHIN_HOURS=6`

This deterministic gate is deliberately conservative. It avoids suppressing new markets, refreshes stale matches, and refreshes every selected run in the final six hours.

Recommended next implementation:

- Add `GROK_CHANGE_DETECTOR=true`.
- Run a low-effort Grok multi-agent check for open matches that already have predictions and are outside the force window.
- Save the change-detector evidence into `logs/run-*.json`.
- Pass that evidence into the full forecaster when a reforecast is triggered.

## 4. Platform Facts That Shape the System

The attached SportsPredict API docs say:

- Probability Cup event: `2026-06-11T15:00:00Z` to `2026-07-19T22:00:00Z`, about 38.3 days or 919 hours.
- About 72 matches and about 10 binary markets per match.
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
- xAI `web_search` and `x_search`: $5 / 1k calls each.

Assumptions:

- One full match-cycle forecasts all markets for one match.
- Grok-only full cycle: one Grok research call plus four Grok forecast variants.
- OpenAI-only full cycle: one GPT-5.4-mini research call plus four GPT-5.5 forecast variants.
- Both-model cycle: Grok research, four GPT-5.5 variants, one Grok check.
- Average full cycle: about 52K input tokens, 7.8K output tokens, and 2-4 search/X tool invocations for Grok-only; OpenAI costs are dominated by GPT-5.5 forecast outputs.
- Bot considers matches within the default 168-hour close window, not the full 919-hour event window.

Approximate costs:

| Scenario | Match full cycles | xAI-only | OpenAI-only | OpenAI + Grok |
|---|---:|---:|---:|---:|
| Forecast once per match | 72 | $8 | $25 | $27 |
| Blind hourly full reforecast, 168h window | 12,096 | $1,270 | $4,149 | $4,476 |
| Blind half-hour full reforecast, 168h window | 24,192 | $2,540 | $8,298 | $8,951 |
| Deterministic gate: 12h cadence plus final 6h hourly | 1,512 | $159 | $519 | $559 |
| Grok change detector hourly plus about 10 full refreshes per match | 720 full + 12,096 checks | $269 | $440 if full forecasts use OpenAI | about $460-$500 |

Interpretation:

- With the user's xAI free credit, blind hourly Grok-only is financially tolerable and likely under the $2,500 credit under these assumptions.
- Blind half-hour Grok-only is right on the edge and can exceed the credit if Grok uses more search/tool calls than assumed.
- OpenAI blind reruns are not cost-rational.
- The best quality/cost compromise is Grok change detection plus forced late-window full refreshes.
- If using OpenAI GPT-5.5 forecasts, gate aggressively.

## Recommendation

Use Grok multi-agent as the research engine, keep direct odds as an anchor, ensemble both prompts and models when OpenAI budget is acceptable, and gate reforecasts. For this tournament, the highest expected-value architecture is:

1. Hourly GitHub Action.
2. Deterministic gate on by default.
3. Add Grok change detector for non-stale, non-closing matches.
4. Full forecast on new markets, stale markets, material evidence changes, and all matches inside six hours to close.
5. Aggregate in log-odds space, mildly extremize, and patch only changes of at least 2 points.
