# Probability Cup Forecaster

Autonomous forecasting system for the Jump Trading Probability Cup / SportsPredict Model API.

The bot:

- discovers the Probability Cup event, lobby, open matches, and open markets;
- gathers fresh public evidence with xAI/Grok multi-agent web/X search when `XAI_API_KEY` is available, with OpenAI web search as fallback;
- can add optional structured bookmaker odds context before LLM research;
- forecasts each match's markets with OpenAI, xAI, and Claude prompt-variant ensembles when keys are available;
- aggregates forecasts in log-odds space, applies configurable calibration, and outputs 1-99 integer probabilities;
- submits new predictions in `/predictions/batch` chunks and updates existing predictions before close;
- skips already-fresh predictions unless they are stale, new, or close to kickoff;
- runs locally or on a scheduled GitHub Action.

## Quick Start

1. Create a SportsPredict bot key in the Probability Cup UI.
2. Create an xAI API key, an OpenAI API key, an Anthropic API key, or any subset. The default quality path uses Grok multi-agent research plus OpenAI/Grok/Claude forecast ensembling when keys are available.
3. Copy `.env.example` to `.env` for local runs, or add GitHub repository secrets:
   - `SPORTSPREDICT_API_KEY`
   - `XAI_API_KEY`, `OPENAI_API_KEY`, and/or `ANTHROPIC_API_KEY`
   - optional `ODDS_API_KEY`
4. Install and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
probability-cup-bot run --dry-run
```

To submit for real:

```bash
SUBMIT=true probability-cup-bot run
```

## GitHub Action

The workflow in `.github/workflows/forecast.yml` runs hourly and can also be started manually from the Actions tab. It dry-runs unless `SUBMIT=true` is set in the workflow environment and the required secrets exist.

The SportsPredict key should be stored only as a repository secret. Do not put it in the repo or logs.

## Main Documents

- [Ranked research synthesis](docs/research-ranking.md)
- [Optimized forecasting prompt](docs/optimized-prompt.md)
- [System design](docs/system-design.md)
- [Follow-up recommendations and cost model](docs/follow-up-recommendations.md)

## Platform Constraints Encoded

- SportsPredict accepts only integer probabilities from 1 to 99.
- There is one prediction per market per bot; existing open predictions are updated with `PATCH`.
- Batch submissions are capped at 50 predictions.
- The API exposes no crowd forecast or current market price, so the bot creates external context from public sources.
- The latest value before market close is scored under Brier methodology.
