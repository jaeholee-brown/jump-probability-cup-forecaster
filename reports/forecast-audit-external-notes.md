# External Audit Notes

Generated: 2026-06-18

These notes supplement `forecast-audit-latest.md`. They are based on public odds/stat pages found after the platform-level audit identified the largest errors.

## Portugal vs DR Congo

- Public odds broadly agreed with our Portugal favorite anchor. FOX/FanDuel listed Portugal `-350`, DR Congo `+1000`, and draw `+420`; our submitted Portugal-win forecast was `77%`, so the miss was market-consistent rather than uniquely model-bad.
- The more actionable miss is the control/volume layer. Post-match reporting described Portugal as struggling to create real threat despite possession, and AS/SofaScore noted Portugal had a historically low shots/SOT output. That fits our broader pattern of overpricing favorite/team-quality control props.
- Source: https://www.foxsports.com/stories/soccer/2026-world-cup-portugal-dr-congo-odds-prediction-picks
- Source: https://en.as.com/soccer/world-cup/portugal-vs-dr-congo-live-online-score-stats-goals-and-updates-fifa-world-cup-2026-f202606-d/

## Czechia vs South Africa

- Public odds described Czechia as only a mild favorite, not a dominant side. Oddschecker listed Czech Republic `+100`, draw `+230`, South Africa `+300`, and Under 2.5 `-150`; Covers/Kalshi listed Czechia around `-122`, South Africa `+376`, draw `+285`, and Under 2.5 `-122`.
- Our Czechia-win `51%` and Under/low-event lean were not obviously unreasonable. The penalty and South Africa second-half goal were not rare events at our probabilities; the score damage came from near-coinflip markets resolving against us.
- The stronger signal is the fouls market. FootyStats-style pre-match rates showed Czechia committing more fouls than South Africa, which supported Claude 4.6's lower `42%` on South Africa more fouls. The aggregate near `49%` was not disastrous, but other models overrode a concrete team-rate anchor too much.
- Source: https://www.oddschecker.com/us/soccer/world-cup/czech-republic-v-south-africa
- Source: https://www.covers.com/world-cup/czechia-vs-south-africa-prediction-top-picks-odds-today-6-18-2026
- Source: https://footystats.org/international/czech-republic-national-team-vs-south-africa-national-team-h2h-stats

## Ghana vs Panama

- Public odds made Ghana only a slight favorite: FOX/FanDuel listed Ghana `+130`, Panama `+220`, draw `+220`; Covers/Kalshi listed Ghana `+127`, Panama `+245`, draw `+245`. Totals leaned Under 2.5 around `-150` to `-170`.
- The match result and external summaries support the low-event read: Ghana won 1-0 with a late goal. Our biggest Ghana/Panama damage came from Ghana/team shot-on-target props, not from the match result direction.
- Public post-match stats also show a low total SOT environment. Fansided listed total shots on target market `9+ (-110)` resolving at `6`, which aligns with the audit's broader finding that shots-on-target markets are overforecast.
- Source: https://www.foxsports.com/stories/soccer/2026-world-cup-ghana-panama-odds-prediction-picks
- Source: https://www.covers.com/world-cup/ghana-vs-panama-prediction-top-picks-odds-today-6-17-2026
- Source: https://fansided.com/soccer/ghana-vs-panama-live-score-team-player-stats-world-cup-group-stage

## Current Systematic Takeaways

- Several large match-result misses were also bookmaker/prediction-market misses. Do not overfit away from odds anchors because Portugal drew DR Congo.
- The clearest bot-specific miss pattern is noisy stat-prop overconfidence, especially shots-on-target, fouls, offsides, and cards.
- Concrete team-rate evidence should beat generic role priors more often. The Czechia/South Africa foul market is the best example so far.
- For favorite control props, team quality should not automatically imply second-half SOT/corners/fouls dominance. Late game state, substitutions, and low-event tournament incentives should pull these closer to 50 unless there is a direct stat or market line.
