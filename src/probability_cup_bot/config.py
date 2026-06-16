from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


@dataclass(frozen=True)
class Settings:
    sportspredict_api_key: str
    openai_api_key: str
    xai_api_key: str = ""
    anthropic_api_key: str = ""
    xai_base_url: str = "https://api.x.ai/v1"
    sportspredict_base_url: str = "https://api.sportspredict.com/api/v1"
    event_title: str = "Probability Cup"
    forecast_model: str = "gpt-5.5"
    research_model: str = "gpt-5.4-mini"
    grok_research_model: str = "grok-4.20-multi-agent-0309"
    grok_forecast_model: str = "grok-4.20-multi-agent-0309"
    claude_forecast_model: str = "claude-opus-4-6"
    use_openai_forecast: bool = True
    use_grok_research: bool = True
    use_grok_forecast: bool = True
    use_claude_forecast: bool = True
    reasoning_effort: str = "medium"
    submit: bool = False
    max_matches_per_run: int = 0
    min_hours_to_close: float = 0.0
    max_hours_to_close: float = 168.0
    enable_update_gate: bool = True
    max_prediction_age_hours: float = 12.0
    force_reforecast_within_hours: float = 6.0
    update_threshold_points: int = 2
    extremize_alpha: float = 1.05
    base_shrinkage: float = 0.04
    low_evidence_shrinkage: float = 0.12
    concurrency: int = 4
    odds_api_key: str = ""
    odds_sport_key: str = "soccer"
    state_dir: Path = Path("state")
    logs_dir: Path = Path("logs")

    @property
    def can_submit(self) -> bool:
        return self.submit and bool(self.sportspredict_api_key)


def load_settings(dotenv_path: str | None = None, *, force_dry_run: bool = False) -> Settings:
    load_dotenv(dotenv_path=dotenv_path)
    submit = _bool_env("SUBMIT", False) and not force_dry_run
    return Settings(
        sportspredict_api_key=os.getenv("SPORTSPREDICT_API_KEY", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        xai_api_key=os.getenv("XAI_API_KEY", ""),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        xai_base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1").rstrip("/"),
        sportspredict_base_url=os.getenv(
            "SPORTSPREDICT_BASE_URL", "https://api.sportspredict.com/api/v1"
        ).rstrip("/"),
        event_title=os.getenv("EVENT_TITLE", "Probability Cup"),
        forecast_model=os.getenv("FORECAST_MODEL", "gpt-5.5"),
        research_model=os.getenv("RESEARCH_MODEL", "gpt-5.4-mini"),
        grok_research_model=os.getenv("GROK_RESEARCH_MODEL", "grok-4.20-multi-agent-0309"),
        grok_forecast_model=os.getenv("GROK_FORECAST_MODEL", "grok-4.20-multi-agent-0309"),
        claude_forecast_model=os.getenv("CLAUDE_FORECAST_MODEL", "claude-opus-4-6"),
        use_openai_forecast=_bool_env("USE_OPENAI_FORECAST", True),
        use_grok_research=_bool_env("USE_GROK_RESEARCH", True),
        use_grok_forecast=_bool_env("USE_GROK_FORECAST", True),
        use_claude_forecast=_bool_env("USE_CLAUDE_FORECAST", True),
        reasoning_effort=os.getenv("REASONING_EFFORT", "medium"),
        submit=submit,
        max_matches_per_run=_int_env("MAX_MATCHES_PER_RUN", 0),
        min_hours_to_close=_float_env("MIN_HOURS_TO_CLOSE", 0.0),
        max_hours_to_close=_float_env("MAX_HOURS_TO_CLOSE", 168.0),
        enable_update_gate=_bool_env("ENABLE_UPDATE_GATE", True),
        max_prediction_age_hours=_float_env("MAX_PREDICTION_AGE_HOURS", 12.0),
        force_reforecast_within_hours=_float_env("FORCE_REFORECAST_WITHIN_HOURS", 6.0),
        update_threshold_points=_int_env("UPDATE_THRESHOLD_POINTS", 2),
        extremize_alpha=_float_env("EXTREMIZE_ALPHA", 1.05),
        base_shrinkage=_float_env("BASE_SHRINKAGE", 0.04),
        low_evidence_shrinkage=_float_env("LOW_EVIDENCE_SHRINKAGE", 0.12),
        concurrency=max(1, _int_env("CONCURRENCY", 4)),
        odds_api_key=os.getenv("ODDS_API_KEY", ""),
        odds_sport_key=os.getenv("ODDS_SPORT_KEY", "soccer"),
        state_dir=Path(os.getenv("STATE_DIR", "state")),
        logs_dir=Path(os.getenv("LOGS_DIR", "logs")),
    )
