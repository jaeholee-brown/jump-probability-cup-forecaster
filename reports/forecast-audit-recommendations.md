# Forecast Audit Recommendations

Generated: 2026-06-18

These recommendations are based on `forecast-audit-latest.md` and `forecast-audit-external-notes.md`.

## High-Confidence Findings

1. Do not punish odds anchors globally.

The largest single miss, Portugal to beat DR Congo at 77%, was broadly consistent with public betting odds. That was a market-consensus miss, not a unique bot failure. The bot should still use odds/market anchors when available.

2. Stat props are the clearest systematic weakness.

Shots-on-target, fouls, offsides, cards, and penalty/red markets have the worst early calibration profile. The common failure mode is converting team quality, possession expectation, or favorite status into medium-high probabilities for noisy stat events.

3. Fouls and discipline markets need team-rate dominance over generic priors.

The Czechia vs South Africa foul market is the clean example. Generic "underdog fouls more" reasoning was weaker than the concrete team-rate evidence that Czechia commit a lot of fouls. Future forecasts should make the team-rate anchor explicit and cap narrative updates when it conflicts with a direct rate.

4. The 41-60% range is over-optimistic.

The 41-50 and 51-60 buckets are both producing too many false positives so far. This should not trigger a blanket shrinkage change yet, but it should inform prop-market prompt wording and calibration review.

5. Claude 4.8 remains the best component, but the sample is still small.

Current component ranking: Claude 4.8 best, Grok 4.20 worst. Current weights already reflect this. Do not chase the latest constrained optimum until there are more settled markets after the evidence-QA change.

## Recommended Next Changes

1. Add a family-level calibration layer once each family has at least 20-30 settled markets.

For now, log it. Do not apply family calibration from seven foul markets or eight offside markets.

2. Tighten the prompt for noisy stat props.

For shots-on-target, fouls, offsides, cards, and half-specific markets, require:

- explicit team/player rate;
- opponent allowance rate or tournament base rate;
- threshold conversion method;
- game-state adjustment;
- final shrinkage toward 50 if no direct odds/stat line exists.

3. Add a "favorite-control cap" for stat props in the prompt, not hard code.

If the only positive evidence is "team is better / favorite / expected possession," forecasts should usually stay close to the base rate. The model must cite a direct stat or market line before going above roughly 58-60% on noisy team-control props.

4. Track external odds/stat availability per market.

The current report only flags whether rationales mention odds/market anchors. We need a real external-data field: moneyline, total, BTTS, team shots/SOT lines, cards/corners lines when found, and whether the forecast was above/below that anchor.

5. Keep the current model weights for now.

The current live score still supports overweighting Claude 4.8 and downweighting Grok 4.20. But the new evidence-QA stack has only one settled match. Re-run the weight search after at least 30 post-change settled markets.

## What Not To Do Yet

- Do not globally lower all favorites because Portugal drew DR Congo.
- Do not remove odds anchors because odds-anchored markets happened to underperform in this tiny sample.
- Do not overfit CZE vs RSA's penalty/second-half-goal outcomes. Those were plausible 40-45% events that happened.
- Do not use hard-coded sport heuristics without a source. Prefer prompted source requirements and telemetry first.
