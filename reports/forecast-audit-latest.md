# Forecast Audit

Generated: `2026-06-18T18:30:08.776602+00:00`

## Executive Findings

- Settled platform predictions: 90 markets, mean Brier 0.2154.
- External source coverage: 90/90 settled forecasts across 9 match groups.
- Direct/specific evidence coverage: 25/90 settled forecasts.
- Family goals: n=14, mean Brier 0.2451, mean p 0.425 vs outcome rate 0.5 (bias -0.075).
- Family fouls: n=7, mean Brier 0.2403, mean p 0.5314 vs outcome rate 0.2857 (bias 0.2457).
- Family offsides: n=8, mean Brier 0.2387, mean p 0.5212 vs outcome rate 0.375 (bias 0.1463).
- Family penalty-red: n=6, mean Brier 0.2339, mean p 0.4117 vs outcome rate 0.3333 (bias 0.0783).
- Family result: n=9, mean Brier 0.2222, mean p 0.5878 vs outcome rate 0.5556 (bias 0.0322).
- Stage pre-evidence-qa: n=70, mean Brier 0.2108, bias 0.1051.
- Stage platform-only: n=10, mean Brier 0.2356, bias 0.11.
- Stage post-evidence-qa: n=10, mean Brier 0.2273, bias -0.111.
- Best component so far: claude-opus-4-8 n=61 mean Brier 0.2129; worst: grok-4.20-0309-reasoning n=66 mean Brier 0.2282.
- Markets with explicit odds/market-anchor language in component rationales: n=27 mean Brier 0.2254; without: n=63 mean Brier 0.2111.

## Platform Summary

| metric | value |
| --- | --- |
| matches_seen | 47 |
| markets_seen | 465 |
| predictions_seen | 277 |
| open_predictions | 187 |
| results_seen | 90 |
| settled_scored_predictions | 90 |
| history_markets | 80 |
| history_matches | 8 |
| component_scored_predictions | 66 |
| component_records | 299 |
| external_source_matches_configured | 9 |
| settled_records_with_external_sources | 90 |
| settled_records_with_direct_or_adjacent_evidence | 25 |

## By Family

| group | count | mean_brier | mean_probability | outcome_rate | bias_probability_minus_outcome | mean_abs_error |
| --- | --- | --- | --- | --- | --- | --- |
| shots-on-target | 27 | 0.2193 | 0.4993 | 0.3333 | 0.1659 | 0.4541 |
| goals | 14 | 0.2451 | 0.425 | 0.5 | -0.075 | 0.4793 |
| result | 9 | 0.2222 | 0.5878 | 0.5556 | 0.0322 | 0.4478 |
| offsides | 8 | 0.2387 | 0.5212 | 0.375 | 0.1463 | 0.4863 |
| fouls | 7 | 0.2403 | 0.5314 | 0.2857 | 0.2457 | 0.48 |
| other | 6 | 0.194 | 0.6 | 0.8333 | -0.2333 | 0.4267 |
| penalty-red | 6 | 0.2339 | 0.4117 | 0.3333 | 0.0783 | 0.475 |
| corners | 5 | 0.1444 | 0.394 | 0.4 | -0.006 | 0.37 |
| cards | 3 | 0.2998 | 0.5333 | 0.3333 | 0.2 | 0.5467 |
| player-goal-assist | 3 | 0.0508 | 0.2233 | 0.0 | 0.2233 | 0.2233 |
| player-goal | 2 | 0.0482 | 0.215 | 0.0 | 0.215 | 0.215 |

## Calibration Buckets

| group | count | mean_brier | mean_probability | outcome_rate | bias_probability_minus_outcome | mean_abs_error |
| --- | --- | --- | --- | --- | --- | --- |
| 51-60 | 27 | 0.2548 | 0.5504 | 0.4444 | 0.1059 | 0.5015 |
| 41-50 | 25 | 0.2431 | 0.4572 | 0.36 | 0.0972 | 0.4908 |
| 21-40 | 21 | 0.1551 | 0.3143 | 0.1905 | 0.1238 | 0.3676 |
| 61-80 | 15 | 0.2071 | 0.674 | 0.7333 | -0.0593 | 0.4247 |
| 01-20 | 2 | 0.0306 | 0.175 | 0.0 | 0.175 | 0.175 |

## Component Scores

| group | count | mean_brier | mean_probability | outcome_rate | bias_probability_minus_outcome | mean_abs_error |
| --- | --- | --- | --- | --- | --- | --- |
| claude-opus-4-8 | 61 | 0.2129 | 0.4634 | 0.3607 | 0.1028 | 0.4438 |
| grok-4.3 | 66 | 0.2214 | 0.4836 | 0.3636 | 0.12 | 0.4564 |
| claude-opus-4-6 | 41 | 0.2219 | 0.4583 | 0.3659 | 0.0924 | 0.4524 |
| gpt-5 | 65 | 0.2236 | 0.4889 | 0.3692 | 0.1197 | 0.4538 |
| grok-4.20-0309-reasoning | 66 | 0.2282 | 0.4874 | 0.3636 | 0.1238 | 0.4653 |

## External Source Coverage

| match_name | settled_records | records_with_sources | source_count | odds_source_count | stats_source_count | odds_summary | result_summary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARG vs ALG | 10 | 10 | 2 | 1 | 0 | Argentina was a heavy favorite around -260, Algeria around +800, draw around +360; Over 2.5 was near even and BTTS Yes was a plus-money underdog. | Argentina beat Algeria 3-0. |
| AUT vs JOR | 10 | 10 | 2 | 1 | 0 | Austria was a heavy favorite around -270, Jordan around +750, draw around +425; Over 2.5 was favored around -135 and BTTS Yes was close to even. | Austria beat Jordan 3-1. |
| CZE vs RSA | 10 | 10 | 4 | 2 | 1 | Czechia was only a mild favorite, roughly +100 to -122 depending on book/source; South Africa was around +300 to +376, and Under 2.5 was favored. | Czechia and South Africa drew 1-1, with South Africa equalizing late by penalty. |
| ENG vs CRO | 10 | 10 | 3 | 1 | 0 | England was favored around -140, Croatia around +420, draw around +270; Under 2.5 was favored around -142 but BTTS Yes was only slightly plus-money. | England beat Croatia 4-2. |
| FRA vs SEN | 10 | 10 | 3 | 2 | 0 | France was a major tournament favorite and group favorite; direct match moneyline was not captured in the saved public-source set. | France beat Senegal 3-1. |
| GHA vs PAN | 10 | 10 | 3 | 2 | 1 | Ghana was only a slight favorite around +127 to +130, Panama around +220 to +245, draw around +220 to +245; Under 2.5 was materially favored. | Ghana beat Panama 1-0 in a low-event match. |
| IRQ vs NOR | 10 | 10 | 3 | 1 | 0 | Norway was a very heavy favorite around -600, Iraq around +1400, draw around +600; Over 2.5 was favored and BTTS Yes was plus-money but not remote. | Norway beat Iraq 4-1. |
| POR vs COD | 10 | 10 | 3 | 1 | 1 | Portugal was a heavy favorite around -350, DR Congo around +1000, draw around +420; Over 2.5 was favored and BTTS No was favored. | Portugal and DR Congo drew 1-1. |
| UZB vs COL | 10 | 10 | 3 | 1 | 1 | Colombia was a heavy favorite around -280, Uzbekistan around +850, draw around +370; Over 2.5 was close to even and BTTS Yes was plus-money. | Colombia beat Uzbekistan 3-1. |

## External Evidence Strength

| group | count | mean_brier | mean_probability | outcome_rate | bias_probability_minus_outcome | mean_abs_error |
| --- | --- | --- | --- | --- | --- | --- |
| direct_market_line | 17 | 0.2239 | 0.5218 | 0.5294 | -0.0076 | 0.4524 |
| direct_or_adjacent_stat | 8 | 0.248 | 0.515 | 0.125 | 0.39 | 0.48 |
| adjacent_match_market | 64 | 0.2071 | 0.4653 | 0.4062 | 0.0591 | 0.4397 |
| match_context_only | 1 | 0.3364 | 0.58 | 0.0 | 0.58 | 0.58 |

## External Evidence Gaps By Family

| group | count | mean_brier | mean_probability | outcome_rate | bias_probability_minus_outcome | mean_abs_error |
| --- | --- | --- | --- | --- | --- | --- |
| shots-on-target | 20 | 0.2089 | 0.4925 | 0.4 | 0.0925 | 0.4455 |
| offsides | 8 | 0.2387 | 0.5212 | 0.375 | 0.1463 | 0.4863 |
| fouls | 6 | 0.2403 | 0.5383 | 0.3333 | 0.205 | 0.4783 |
| other | 6 | 0.194 | 0.6 | 0.8333 | -0.2333 | 0.4267 |
| penalty-red | 6 | 0.2339 | 0.4117 | 0.3333 | 0.0783 | 0.475 |
| corners | 5 | 0.1444 | 0.394 | 0.4 | -0.006 | 0.37 |
| goals | 5 | 0.2599 | 0.348 | 0.6 | -0.252 | 0.484 |
| cards | 3 | 0.2998 | 0.5333 | 0.3333 | 0.2 | 0.5467 |
| player-goal | 2 | 0.0482 | 0.215 | 0.0 | 0.215 | 0.215 |
| player-goal-assist | 2 | 0.045 | 0.21 | 0.0 | 0.21 | 0.21 |
| result | 2 | 0.194 | 0.44 | 0.0 | 0.44 | 0.44 |

## Worst Settled Markets

| match_name | question | family | probability_int | outcome | brier | stage | external_evidence_level | component_count | component_spread_points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| POR vs COD | Will Portugal win the match? | result | 77 | 0 | 0.5929 | pre-evidence-qa | direct_market_line | 5 | 1.0 |
| ENG vs CRO | Will both teams score AND the match have 3 or more total goals? | goals | 32 | 1 | 0.4624 | pre-evidence-qa | direct_market_line | 0 |  |
| POR vs COD | In the second half, will Portugal have more shots on target than DR Congo? | shots-on-target | 68 | 0 | 0.4624 | pre-evidence-qa | direct_or_adjacent_stat | 0 |  |
| GHA vs PAN | Will Ghana have 3 or more shots on target? | shots-on-target | 67 | 0 | 0.4489 | pre-evidence-qa | direct_or_adjacent_stat | 5 | 6.0 |
| UZB vs COL | Will Uzbekistan score at least 1 goal? | goals | 35 | 1 | 0.4225 | pre-evidence-qa | adjacent_match_market | 5 | 3.0 |
| IRQ vs NOR | Will there be 4 or more total shots on target in the second half? | shots-on-target | 62 | 0 | 0.3844 | pre-evidence-qa | adjacent_match_market | 4 | 13.0 |
| ENG vs CRO | Will a penalty kick be awarded OR a red card be shown? | penalty-red | 38 | 1 | 0.3844 | pre-evidence-qa | adjacent_match_market | 5 | 8.0 |
| IRQ vs NOR | Will Iraq score at least 1 goal? | goals | 39 | 1 | 0.3721 | pre-evidence-qa | adjacent_match_market | 4 | 5.0 |
| FRA vs SEN | Will Senegal receive at least 1 card in the second half? | cards | 59 | 0 | 0.3481 | platform-only | adjacent_match_market | 0 |  |
| ARG vs ALG | Will Algeria commit more fouls than Argentina? | fouls | 58 | 0 | 0.3364 | pre-evidence-qa | adjacent_match_market | 4 | 1.0 |
| FRA vs SEN | Will both teams score AND the match have 3 or more total goals? | goals | 42 | 1 | 0.3364 | platform-only | adjacent_match_market | 0 |  |
| FRA vs SEN | Will France score in the first half? | other | 58 | 0 | 0.3364 | platform-only | match_context_only | 0 |  |
| UZB vs COL | Will Colombia have more shots on target than Uzbekistan in the second half? | shots-on-target | 58 | 0 | 0.3364 | pre-evidence-qa | adjacent_match_market | 5 | 8.0 |
| CZE vs RSA | Will South Africa score in the second half? | other | 43 | 1 | 0.3249 | post-evidence-qa | adjacent_match_market | 5 | 4.0 |
| CZE vs RSA | Will a penalty kick be awarded OR a red card be shown in the match? | penalty-red | 43 | 1 | 0.3249 | post-evidence-qa | adjacent_match_market | 5 | 5.0 |

## Best Settled Markets

| match_name | question | family | probability_int | outcome | brier | stage | external_evidence_level | component_count | component_spread_points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| POR vs COD | Will Gonçalo Ramos score a goal (excluding own goals)? | player-goal | 17 | 0 | 0.0289 | pre-evidence-qa | adjacent_match_market | 5 | 7.0 |
| IRQ vs NOR | Will Mohanad Ali score or assist a goal (excluding own goals)? | player-goal-assist | 18 | 0 | 0.0324 | pre-evidence-qa | adjacent_match_market | 4 | 13.0 |
| AUT vs JOR | Will Jordan have more shots on target than Austria in the second half? | shots-on-target | 22 | 0 | 0.0484 | pre-evidence-qa | adjacent_match_market | 4 | 6.0 |
| CZE vs RSA | Will Oswin Appollis score or assist a goal (excluding own goals)? | player-goal-assist | 24 | 0 | 0.0576 | post-evidence-qa | adjacent_match_market | 0 |  |
| IRQ vs NOR | Will Iraq have more shots on target than Norway in the second half? | shots-on-target | 24 | 0 | 0.0576 | pre-evidence-qa | adjacent_match_market | 3 | 5.0 |
| AUT vs JOR | Will Austria win the match? | result | 75 | 1 | 0.0625 | pre-evidence-qa | direct_market_line | 4 | 1.0 |
| UZB vs COL | Will Eldor Shomurodov score or assist a goal (excluding own goals)? | player-goal-assist | 25 | 0 | 0.0625 | pre-evidence-qa | direct_market_line | 5 | 12.0 |
| AUT vs JOR | Will Jordan finish with more corner kicks than Austria? | corners | 26 | 0 | 0.0676 | pre-evidence-qa | adjacent_match_market | 0 |  |
| FRA vs SEN | Will Sadio Mané score a goal (excluding own goals)? | player-goal | 26 | 0 | 0.0676 | platform-only | adjacent_match_market | 0 |  |
| POR vs COD | Will DR Congo commit more fouls than Portugal? | fouls | 73 | 1 | 0.0729 | pre-evidence-qa | adjacent_match_market | 5 | 8.0 |
| ARG vs ALG | Will Argentina score the first goal of the game and Algeria score in the second half? | goals | 28 | 0 | 0.0784 | pre-evidence-qa | adjacent_match_market | 4 | 16.0 |
| ARG vs ALG | Will Argentina win the match? | result | 71 | 1 | 0.0841 | pre-evidence-qa | direct_market_line | 0 |  |
| ENG vs CRO | At halftime, will Croatia have more corner kicks than England? | corners | 30 | 0 | 0.09 | pre-evidence-qa | adjacent_match_market | 0 |  |
| POR vs COD | Will both teams score AND the match have 3 or more total goals? | goals | 30 | 0 | 0.09 | pre-evidence-qa | direct_market_line | 5 | 3.0 |
| GHA vs PAN | Will Panama score the first goal of the second half? | goals | 30 | 0 | 0.09 | pre-evidence-qa | adjacent_match_market | 5 | 8.0 |

## Data Notes

- Platform-level scoring uses SportsPredict /results and covers every settled submitted prediction returned by the API.
- Component/model scoring is available only for markets present in saved forecast-history.json.
- External source enrichment is match-level unless a direct market/prop line is shown in the source facts.
- External evidence strength is conservative: generic moneyline/total context is not treated as direct evidence for SOT, cards, fouls, offsides, corners, or half-specific props.
- External odds are not available through the Jump API; source coverage comes from public odds, stats, and recap pages captured in reports/external-match-sources.json.
- Post-change split uses forecast-history timestamps, not Git metadata inside artifacts.
