# External Audit Notes

Generated: 2026-06-18

These notes supplement `forecast-audit-latest.md`. The reusable source data lives in
`external-match-sources.json`; the regenerated audit now attaches match-level source
coverage to all settled platform forecasts.

## Coverage

- Settled platform forecasts scored: 90.
- Settled match groups covered by external sources: 9 of 9.
- Settled forecasts with at least one public odds/stat/recap source attached: 90 of 90.
- Remaining limitation: most sources are match-level. Direct prop-level historical lines
  are only available when the public page explicitly preserved that market.

## Argentina vs Algeria

- Public odds made Argentina a heavy favorite: Argentina about `-260`, Algeria about
  `+800`, draw about `+360`.
- The 3-0 Argentina result validated the favorite and team-scoring anchors.
- Bot lesson: this was not a miss pattern; strong favorite/result priors were useful here.

## Austria vs Jordan

- Public odds made Austria a heavy favorite: Austria about `-270`, Jordan about `+750`,
  draw about `+425`, with Over 2.5 favored.
- Austria won 3-1, so the result and over-ish total context were market-consistent.
- Bot lesson: Jordan scoring was not shocking because BTTS Yes was near even.

## Czechia vs South Africa

- Public odds made Czechia only a mild favorite, roughly `+100` to `-122` depending on
  source; South Africa was roughly `+300` to `+376`; Under 2.5 was favored.
- The 1-1 draw and late South Africa penalty were damaging but not extreme-tail outcomes.
- Bot lesson: direct team-rate evidence mattered. Pre-match foul-rate sources pointed
  toward Czechia committing more fouls, which was stronger than a generic
  underdog-fouls-more prior.

## England vs Croatia

- Public odds made England a favorite around `-140`; Under 2.5 was favored, but BTTS Yes
  was only slightly plus-money.
- England won 4-2. That was high-event relative to the total, but not a black swan.
- Bot lesson: our `32%` on BTTS plus 3+ goals was probably low, but not wildly
  unreasonable. The better fix is to model correlated legs explicitly, not globally raise
  all totals.

## France vs Senegal

- Public sources captured France as a major tournament and group favorite, but we do not
  have a preserved direct moneyline in the source set.
- France won 3-1.
- Bot lesson: match/result priors were directionally fine, but the early France/Senegal
  markets remain weaker evidence because they are platform-only and lack component
  rationales.

## Ghana vs Panama

- Public odds made Ghana only a slight favorite: about `+127` to `+130`, Panama about
  `+220` to `+245`, with Under 2.5 materially favored.
- Ghana won 1-0, and public stat summaries indicate a low total shots-on-target game.
- Bot lesson: Ghana 3+ shots on target at `67%` was too high for a slight favorite in an
  under-leaning match without a direct SOT line.

## Iraq vs Norway

- Public odds made Norway a very heavy favorite around `-600`; Over 2.5 was favored and
  BTTS Yes was plus-money but plausible.
- Norway won 4-1 and Iraq scored.
- Bot lesson: Iraq scoring at `39%` was not obviously bad. The sharper issue was
  half-specific shot-volume instability.

## Portugal vs DR Congo

- Public odds made Portugal a heavy favorite around `-350`, DR Congo around `+1000`, draw
  around `+420`.
- Portugal drew 1-1.
- Bot lesson: Portugal-win at `77%` was close to market consensus, so that single miss
  should not cause us to abandon odds anchors. The bot-specific weakness was translating
  favorite status into overconfident second-half SOT/control props.

## Uzbekistan vs Colombia

- Public context made Colombia a heavy favorite; the preserved FOX boxscore odds tab
  showed Colombia winning as a `-316` favorite.
- Colombia won 3-1.
- Bot lesson: Uzbekistan scoring at `35%` was not extreme, but the low-total lean and
  second-half SOT-control props were too conservative/fragile.

## Current Systematic Takeaways

- Several big match-result misses were also market-consensus misses. Do not globally
  punish odds anchors.
- The clearest bot-specific weakness remains noisy stat-prop overconfidence: shots on
  target, fouls, offsides, cards, and half-specific variants.
- Team quality and possession should be weak evidence for stat dominance unless backed by
  direct team/player rates or a public prop line.
- Correlated combo markets need explicit leg decomposition and recombination; treating
  them as generic low-probability undershoots events like BTTS plus 3+ goals.
