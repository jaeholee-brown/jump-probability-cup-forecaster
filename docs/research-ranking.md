# Ranked Research Synthesis: Prompting and Scaffolding for LLM Forecasting

This ranking weights sample size more than apparent effect size. The main conclusion is that prompt wording alone has weak and unstable effects on LLM forecasting. The strongest interventions are system-level: fresh evidence retrieval, external forecast/odds anchors, aggregation, calibration, and disciplined updating.

## Priority Ranking

| Rank | Technique | Evidence weight | Direction and effect | How it enters this bot |
|---:|---|---|---|---|
| 1 | Fresh retrieval plus relevance filtering and summarization | Very high | Halawi et al. built a retrieval-augmented forecasting system over 914 test questions from a 5,516-question curated set. Best raw GPT-4 baseline Brier was 0.208; full system reached 0.179. Removing both retrieval and fine-tuning fell to 0.206. | Six Grok multi-agent web/X-search evidence passes per match when available: stable overview, explicit base rates, late-news/lineups, market-specific micro evidence, lineup/role evidence, and volatile-market statistical anchors; then a Grok evidence-QA audit flags stale claims, weak denominators, and unsupported prop anchors before the forecast ensemble sees the evidence. OpenAI web search remains fallback; optional odds data and Firecrawl snippets provide source context. |
| 2 | External crowd/market/odds anchor | Very high | ForecastBench top systems often use crowd forecast context for market questions; superforecasters scored 0.096 vs top LLM 0.122 on the 200-item human subset. Silicon-crowd Study 2 found LLM forecasts improved 17-28 percent when shown human medians, but simple averaging with the human median was better. | Jump API exposes no crowd price, so public bookmaker odds are the closest legal proxy when available. Prompt treats odds as a strong but not decisive anchor. |
| 3 | Multi-model plus selective multi-prompt aggregation in log-odds space | High | ForecastBench's top-three-model aggregate compared aggregation methods; geometric mean/log-odds methods scored 0.194 vs 0.197 for median/trimmed mean. Silicon-crowd Study 1 used 12 LLMs on 31 live Metaculus questions and was statistically indistinguishable from 925-human crowd. Prompt-only studies are weaker, so model diversity gets priority over variant-only diversity. | OpenAI `gpt-5`, xAI `grok-4.3` and `grok-4.20-0309-reasoning`, plus Claude `claude-opus-4-8` and `claude-opus-4-6` are combined by weighted log-odds mean. Current live-data base weights are 0.5, 0.225, 0.2, 1.35, and 0.6 respectively, then adjusted by conservative settled-result calibration. |
| 4 | Base-rate and frequency-first prompting | Medium-high | Schoenegger et al. tested 38 prompts across four models and 100 ForecastBench questions. In one-sample tests, Frequency-Based Reasoning improved Brier by -0.019, Base Rate First by -0.016, and Step-Back by -0.015 after BH correction; the stricter mixed-effects model found no positive prompt survived correction. | A dedicated base-rate/frequency variant and mandatory reference-class field in every forecast. |
| 5 | Balanced scratchpad: rephrase, yes/no reasons, calibration check | Medium | Halawi and ForecastBench use scratchpad prompts in strong baselines, and ForecastBench's top baseline models often use scratchpad. But the prompt-engineering paper finds prompt-only effects are mostly negligible. | One structured audit-note variant; final output is JSON, not free-form hidden reasoning. |
| 6 | Cross-market coherence checking | Medium | Not directly isolated in LLM forecasting papers, but probability coherence is basic forecasting hygiene and is valuable for related sports markets. | A coherence-checking variant and aggregation metadata; future work can add hard constraint repair. |
| 7 | Prompt-generated prompt optimization | Low-medium | Prompt-engineering Study 2 tested OpenAI/Anthropic generated prompts and compound prompts. Some unadjusted effects were positive, but none survived multiple-comparison correction. | The optimized prompt borrows the best recurring components but does not assume prompt optimization alone is enough. |
| 8 | Self-consistency inside a single call | Low / mixed | Generic reasoning literature supports self-consistency for some tasks, but forecasting-specific tests did not find robust gains; in Study 1, the self-consistency prompt had worse mean difference (+0.005) and was not significant. | Use independent calls plus aggregation instead of asking one call to self-consistently choose its own answer. |
| 9 | Explicit "Bayesian reasoning" prompt, propose-evaluate-select, conditional odds-ratio | Negative | Schoenegger et al. found Bayesian Reasoning (+0.030 mixed-effects; +0.025 one-sample) and Propose-Evaluate-Select (+0.033 mixed-effects; +0.028 one-sample) significantly worsened forecasts. Conditional odds-ratio worsened Study 2 (+0.023, adjusted p = 0.01). | Avoid those instructions. Use base rates and updates without forcing formal Bayesian theater. |
| 10 | Autonomous self-reflection/self-improvement loops | Negative / unproven | Kacholia's 2025 thesis reports nine LLMs did not successfully use self-reflection to improve forecasting prompts. | Store results for later calibration, but do not trust free-form self-reflection as an optimizer. |

## Key Findings

### 1. Retrieval and evidence quality dominate prompt wording

Halawi et al., "Approaching Human-Level Forecasting with Language Models" ([arXiv:2402.18563](https://arxiv.org/abs/2402.18563)), is the strongest system-building paper for this task. They start from a large raw corpus of 48,754 questions and 7.17 million user forecasts, curate 5,516 binary questions, and evaluate on 914 post-cutoff test questions. The full system uses search-query generation, retrieval, relevance filtering, summarization, multiple reasoning prompts, fine-tuning, and ensembling. The Brier improvement from raw GPT-4-1106 baseline 0.208 to system 0.179 is far larger than prompt-only effects in later work.

For Jump, this means a GitHub Action should not merely call a frontier model on the market text. It should first create the missing context: current odds, team strength, injuries, lineups, weather, tournament motivation, and recent form.

### 2. Market/crowd information is extremely valuable, but Jump does not expose it

ForecastBench ([arXiv:2409.19839](https://arxiv.org/abs/2409.19839), [leaderboard](https://www.forecastbench.org/leaderboards/)) reports that superforecasters achieved 0.096 Brier on the 200-item human subset, the public median 0.121, and the top LLM 0.122. The top LLM performers often had "freeze values" or crowd forecasts on market questions. ForecastBench's current tournament page explicitly allows tools, added context, fine-tuning, and ensembling.

Schoenegger et al., "Wisdom of the silicon crowd" ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11800985/)), tested 12 LLMs on 31 live Metaculus questions against 925 human forecasters. The LLM ensemble beat the no-information benchmark and was statistically indistinguishable from the human crowd. Their second study found GPT-4 and Claude 2 improved when shown human medians, but a simple human-machine average beat the models' own updates.

Jump's API does not return current prices or crowd forecasts. That is a hard limitation. The bot substitutes public bookmaker odds when available and treats them as a strong anchor because they are a real-money crowd/market signal.

### 3. Aggregation should be outside the model

ForecastBench Appendix E aggregates top model forecasts and finds geometric/log-odds aggregation slightly better than median/trimmed mean (0.194 vs 0.197 Brier). Baron et al., "Two Reasons to Make Aggregated Probability Forecasts More Extreme" ([Decision Analysis / PDF](https://faculty.wharton.upenn.edu/wp-content/uploads/2015/07/2015---two-reasons-to-make-aggregated-probability-forecasts_1.pdf)), gives the classic reason to extremize aggregated forecasts: averaging independent uncertain estimates pulls probabilities toward 0.5.

The bot therefore runs separate model-family forecasts and a selective set of prompt variants, then combines probabilities in log-odds space with explicit model-family weights. It applies configurable mild extremization (`EXTREMIZE_ALPHA=1.05`) and shrinkage when evidence is weak. Prompt ensembling is available, but paid-provider prompt diversity is not the default because prompt-only gains are not strong enough to justify a blanket 3x-4x cost increase.

Given the available Grok API budget and high rate limits, the implementation uses Grok 4.20 Multi-Agent as the primary evidence path when `XAI_API_KEY` exists, not merely as an optional add-on. This is not because there is direct evidence that that exact model wins forecasting contests, but because the literature rewards independent information gathering and model diversity, and xAI's multi-agent model is explicitly designed for parallel deep research. The xAI surplus is therefore spent on decomposed research passes, especially explicit base-rate gathering, lineup/role checks, volatile-market anchors, and evidence QA, rather than many same-provider forecast votes. Firecrawl, when enabled, is used to feed cleaner retrieved snippets into those passes; it does not replace the model's synthesis role.

X/Twitter search is treated as a freshness and discovery channel, not as an authority layer. I did not find strong forecasting-tournament evidence that social posts independently improve probabilistic accuracy; the better-supported pattern is multi-source retrieval, relevance filtering, and aggregation. The bot therefore uses xAI `x_search` for late lineup, injury, suspension, and availability discovery, then asks the model to discount weak/social-only claims unless they are corroborated by official, bookmaker, weather, or reputable reporting sources.

### 4. Base rates are the best prompt-only ingredient

Schoenegger, Jones, Tetlock, and Mellers, "Prompt Engineering Large Language Models' Forecasting Capabilities" ([arXiv:2506.01578](https://arxiv.org/abs/2506.01578)), is the most directly relevant prompt-only study. It tested 38 prompts in Study 1 across Claude 3.5 Sonnet, Claude 3.5 Haiku, GPT-4o, and Llama 3.1 405B on 100 ForecastBench questions. The strict mixed-effects model found only negative significant prompt effects; the simpler one-sample test found Frequency-Based Reasoning, Base Rate First, and Step-Back improved Brier after correction.

The practical lesson is to include base-rate and reference-class reasoning, but not to expect a prompt alone to carry the tournament.

### 5. Some "smart" prompts are actively harmful

The same prompt-engineering paper found robust negative effects for explicit Bayesian Reasoning and Propose-Evaluate-Select. Their interpretation is that models may mimic formal reasoning without doing reliable probabilistic updating. The ForecastBench superforecaster conditional-odds prompt also underperformed in Study 2.

The final prompt says "start from a base rate and update," but it avoids asking for formal Bayesian mechanics, odds-ratio trees, or a one-call propose/evaluate/select loop.

### 6. One-shot LLM forecasting can exploit data imbalance rather than become skilled

Phan et al., "Can Language Models Use Forecasting Strategies?" ([arXiv:2406.04446](https://arxiv.org/abs/2406.04446)), found a basic prompt outperformed a human baseline on raw validation Brier (0.1221 vs 0.1334), but a weighted Brier analysis showed humans were better (0.1746 vs 0.1955). The apparent LLM win came from a low-probability bias on a dataset where most events resolved "No." More complex strategies did not beat the basic prompt.

This matters for Jump because blindly biasing low probabilities could win some skewed prop markets and fail badly on balanced match markets. The bot logs component probabilities and should be calibrated from settled results.

### 7. Evaluation pitfalls should inform live strategy

Paleka et al., "Pitfalls in Evaluating Language Model Forecasters" ([arXiv:2506.00723](https://arxiv.org/abs/2506.00723)), warns about temporal leakage, base-rate confounds, scoring quirks, and correlated risks. Since Jump is a live contest, temporal leakage is less of an evaluation issue, but correlated risk remains: overconfident wrong assumptions across related markets can hurt many Brier scores at once.

That is why the system forecasts per match, checks related markets, and should avoid taking the same unsupported stance across all markets.

## Prompt Components Kept

- Base-rate/reference-class first.
- Frequency framing.
- Concise yes/no evidence.
- Step-back calibration.
- Current evidence and odds anchoring.
- Related-market coherence.
- Strict structured output.

## Prompt Components Rejected

- "Do Bayesian reasoning" as a formal instruction.
- Propose-evaluate-select inside one answer.
- Long conditional odds-ratio trees.
- Self-consistency inside a single prompt.
- Pure superforecaster persona without data.
- Tipping, emotional stakes, deep-breath-only, or other weak prompt hacks.
