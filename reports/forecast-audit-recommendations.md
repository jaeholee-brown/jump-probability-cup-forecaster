# Forecast Audit Recommendations

Generated: 2026-06-18

These recommendations are based on `forecast-audit-latest.md` and
`external-match-sources.json`.

## High-Confidence Findings

1. Keep odds anchors, but label their scope.

Portugal drawing DR Congo was the largest single miss, but Portugal-win at `77%` was close
to public market context. This is a bookmaker-consensus miss, not evidence that odds
anchors are bad.

2. Treat noisy stat props as the primary weakness.

Shots on target, fouls, offsides, cards, and half-specific props are the most consistent
source of avoidable error. The bot is too willing to map favorite status, possession, or
team quality into medium-high probabilities.

The evidence-strength split now makes that concrete: only 25 of 90 settled forecasts have
direct market-line evidence or direct/adjacent stat evidence, and the direct/adjacent stat
bucket is currently badly over-optimistic. Keep the sample-size warning, but prioritize
this failure mode.

3. Require direct rates before moving far from 50 on control props.

For SOT, fouls, offsides, cards, corners, and half markets, the prompt should require:

- team/player event rate;
- opponent allowance rate;
- threshold conversion method;
- game-state adjustment;
- shrinkage toward 50 when no direct rate or market line exists.

4. Do not overreact to plausible low-probability hits.

Iraq scoring at `39%`, England/Croatia BTTS plus 3+ goals at `32%`, and CZE/RSA penalty at
`43%` were damaging but not implausible. The lesson is not "raise all tails"; it is
"decompose correlated/volatile events better."

5. Use family-level calibration only after enough samples.

The current family counts are still small: fouls `n=7`, offsides `n=8`, cards `n=3`.
Keep logging family calibration, but do not hard-code corrections from these samples yet.

## Recommended Next Changes

1. Extend external-source capture from match-level to prop-level where possible.

The audit now verifies match-level source coverage for every settled forecast and labels
each market as direct line, direct/adjacent stat, adjacent match market, or context-only.
The next increment is direct lines for cards, SOT, corners, offsides, and player props
whenever public pages preserve them.

2. Add a "source strength" field to component forecasts.

Store whether each component used a direct prop line, adjacent team stat, generic
moneyline/total, or only narrative context. This lets future audits distinguish good odds
usage from loose market language.

3. Keep the prompt focused on sourced base rates.

The strongest prompt change is not more verbosity. It is forcing the model to name the
base rate, explain the threshold conversion, and admit when it only has weak proxy
evidence.

4. Re-run model-weight search after at least 30 post-evidence-QA settled markets.

Current component scoring still favors Claude 4.8, with Grok 4.20 worst, but most settled
markets predate the latest evidence-QA change. Do not overfit weights to the first day.

## What Not To Do Yet

- Do not globally lower favorites because Portugal drew DR Congo.
- Do not remove market anchors because a few odds-aligned forecasts missed.
- Do not hard-code sport heuristics without source evidence.
- Do not treat every low-probability hit as a calibration failure; compare bucket hit rate,
  not just worst individual Brier scores.
