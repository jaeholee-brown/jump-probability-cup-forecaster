from __future__ import annotations

import os
from dataclasses import dataclass, field
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


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _mapping_env(name: str, default: dict[str, float]) -> dict[str, float]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    output: dict[str, float] = {}
    for item in value.split(","):
        if "=" not in item:
            continue
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if key:
            output[key] = float(raw_value.strip())
    return output or default


DEFAULT_FORECAST_MODEL_WEIGHTS = {
    "gpt-5": 0.5,
    "grok-4.3": 0.225,
    "grok-4.20-0309-reasoning": 0.2,
    "claude-opus-4-8": 1.35,
    "claude-opus-4-6": 0.6,
}


@dataclass(frozen=True)
class Settings:
    sportspredict_api_key: str
    openai_api_key: str
    xai_api_key: str = ""
    anthropic_api_key: str = ""
    xai_base_url: str = "https://api.x.ai/v1"
    sportspredict_base_url: str = "https://api.sportspredict.com/api/v1"
    event_id: str = ""
    event_title: str = "Probability Cup"
    forecast_model: str = "gpt-5"
    research_model: str = "gpt-5.4-mini"
    grok_research_model: str = "grok-4.20-multi-agent-0309"
    grok_research_passes: tuple[str, ...] = ("overview", "base_rates", "late_news", "market_micro")
    grok_research_reasoning_effort: str = "medium"
    grok_news_model: str = "grok-4.20-multi-agent-0309"
    grok_news_reasoning_effort: str = "low"
    grok_forecast_model: str = "grok-4.3"
    grok_forecast_models: tuple[str, ...] = ("grok-4.3", "grok-4.20-0309-reasoning")
    claude_forecast_model: str = "claude-opus-4-8"
    claude_forecast_models: tuple[str, ...] = ("claude-opus-4-8", "claude-opus-4-6")
    use_openai_forecast: bool = True
    use_grok_research: bool = True
    use_grok_forecast: bool = True
    use_claude_forecast: bool = True
    openai_forecast_variants: tuple[str, ...] = ("base_rate_frequency",)
    grok_forecast_variants: tuple[str, ...] = ("base_rate_frequency",)
    claude_forecast_variants: tuple[str, ...] = ("base_rate_frequency",)
    openai_forecast_weight: float = 1.0
    grok_forecast_weight: float = 0.5
    claude_forecast_weight: float = 0.75
    forecast_model_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FORECAST_MODEL_WEIGHTS)
    )
    apply_calibration_weights: bool = True
    calibration_learning_rate: float = 1.8
    calibration_prior_count: int = 20
    use_grok_news_monitor: bool = True
    news_monitor_max_hours_to_close: float = 168.0
    news_monitor_materiality_threshold_points: int = 2
    stale_reforecast_without_news: bool = False
    firecrawl_api_key: str = ""
    use_firecrawl_retrieval: bool = True
    firecrawl_mode: str = "targeted"
    firecrawl_search_limit: int = 5
    firecrawl_search_queries: int = 2
    firecrawl_force_within_hours: float = 2.0
    firecrawl_volatile_within_hours: float = 24.0
    firecrawl_disagreement_threshold_points: float = 20.0
    reasoning_effort: str = "medium"
    submit: bool = False
    max_matches_per_run: int = 0
    min_hours_to_close: float = 0.0
    max_hours_to_close: float = 168.0
    enable_update_gate: bool = True
    max_prediction_age_hours: float = 12.0
    force_reforecast_within_hours: float = 1.5
    final_reforecast_min_interval_minutes: float = 30.0
    update_threshold_points: int = 2
    sportspredict_retry_attempts: int = 6
    sportspredict_retry_initial_seconds: float = 2.0
    sportspredict_retry_max_seconds: float = 60.0
    sportspredict_update_interval_seconds: float = 1.1
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
        event_id=os.getenv("EVENT_ID", ""),
        event_title=os.getenv("EVENT_TITLE", "Probability Cup"),
        forecast_model=os.getenv("FORECAST_MODEL", "gpt-5"),
        research_model=os.getenv("RESEARCH_MODEL", "gpt-5.4-mini"),
        grok_research_model=os.getenv("GROK_RESEARCH_MODEL", "grok-4.20-multi-agent-0309"),
        grok_research_passes=_csv_env(
            "GROK_RESEARCH_PASSES",
            ("overview", "base_rates", "late_news", "market_micro"),
        ),
        grok_research_reasoning_effort=os.getenv("GROK_RESEARCH_REASONING_EFFORT", "medium"),
        grok_news_model=os.getenv("GROK_NEWS_MODEL", "grok-4.20-multi-agent-0309"),
        grok_news_reasoning_effort=os.getenv("GROK_NEWS_REASONING_EFFORT", "low"),
        grok_forecast_model=os.getenv("GROK_FORECAST_MODEL", "grok-4.3"),
        grok_forecast_models=_csv_env(
            "GROK_FORECAST_MODELS",
            (os.getenv("GROK_FORECAST_MODEL") or "grok-4.3", "grok-4.20-0309-reasoning"),
        ),
        claude_forecast_model=os.getenv("CLAUDE_FORECAST_MODEL", "claude-opus-4-8"),
        claude_forecast_models=_csv_env(
            "CLAUDE_FORECAST_MODELS",
            (os.getenv("CLAUDE_FORECAST_MODEL") or "claude-opus-4-8", "claude-opus-4-6"),
        ),
        use_openai_forecast=_bool_env("USE_OPENAI_FORECAST", True),
        use_grok_research=_bool_env("USE_GROK_RESEARCH", True),
        use_grok_forecast=_bool_env("USE_GROK_FORECAST", True),
        use_claude_forecast=_bool_env("USE_CLAUDE_FORECAST", True),
        openai_forecast_variants=_csv_env("OPENAI_FORECAST_VARIANTS", ("base_rate_frequency",)),
        grok_forecast_variants=_csv_env("GROK_FORECAST_VARIANTS", ("base_rate_frequency",)),
        claude_forecast_variants=_csv_env("CLAUDE_FORECAST_VARIANTS", ("base_rate_frequency",)),
        openai_forecast_weight=_float_env("OPENAI_FORECAST_WEIGHT", 1.0),
        grok_forecast_weight=_float_env("GROK_FORECAST_WEIGHT", 0.5),
        claude_forecast_weight=_float_env("CLAUDE_FORECAST_WEIGHT", 0.75),
        forecast_model_weights=_mapping_env(
            "FORECAST_MODEL_WEIGHTS",
            dict(DEFAULT_FORECAST_MODEL_WEIGHTS),
        ),
        apply_calibration_weights=_bool_env("APPLY_CALIBRATION_WEIGHTS", True),
        calibration_learning_rate=_float_env("CALIBRATION_LEARNING_RATE", 1.8),
        calibration_prior_count=_int_env("CALIBRATION_PRIOR_COUNT", 20),
        use_grok_news_monitor=_bool_env("USE_GROK_NEWS_MONITOR", True),
        news_monitor_max_hours_to_close=_float_env("NEWS_MONITOR_MAX_HOURS_TO_CLOSE", 168.0),
        news_monitor_materiality_threshold_points=_int_env("NEWS_MONITOR_MATERIALITY_THRESHOLD_POINTS", 2),
        stale_reforecast_without_news=_bool_env("STALE_REFORECAST_WITHOUT_NEWS", False),
        firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY", ""),
        use_firecrawl_retrieval=_bool_env("USE_FIRECRAWL_RETRIEVAL", True),
        firecrawl_mode=os.getenv("FIRECRAWL_MODE", "targeted"),
        firecrawl_search_limit=_int_env("FIRECRAWL_SEARCH_LIMIT", 5),
        firecrawl_search_queries=max(0, _int_env("FIRECRAWL_SEARCH_QUERIES", 2)),
        firecrawl_force_within_hours=_float_env("FIRECRAWL_FORCE_WITHIN_HOURS", 2.0),
        firecrawl_volatile_within_hours=_float_env("FIRECRAWL_VOLATILE_WITHIN_HOURS", 24.0),
        firecrawl_disagreement_threshold_points=_float_env("FIRECRAWL_DISAGREEMENT_THRESHOLD_POINTS", 20.0),
        reasoning_effort=os.getenv("REASONING_EFFORT", "medium"),
        submit=submit,
        max_matches_per_run=_int_env("MAX_MATCHES_PER_RUN", 0),
        min_hours_to_close=_float_env("MIN_HOURS_TO_CLOSE", 0.0),
        max_hours_to_close=_float_env("MAX_HOURS_TO_CLOSE", 168.0),
        enable_update_gate=_bool_env("ENABLE_UPDATE_GATE", True),
        max_prediction_age_hours=_float_env("MAX_PREDICTION_AGE_HOURS", 12.0),
        force_reforecast_within_hours=_float_env("FORCE_REFORECAST_WITHIN_HOURS", 1.5),
        final_reforecast_min_interval_minutes=_float_env("FINAL_REFORECAST_MIN_INTERVAL_MINUTES", 30.0),
        update_threshold_points=_int_env("UPDATE_THRESHOLD_POINTS", 2),
        sportspredict_retry_attempts=max(1, _int_env("SPORTSPREDICT_RETRY_ATTEMPTS", 6)),
        sportspredict_retry_initial_seconds=_float_env("SPORTSPREDICT_RETRY_INITIAL_SECONDS", 2.0),
        sportspredict_retry_max_seconds=_float_env("SPORTSPREDICT_RETRY_MAX_SECONDS", 60.0),
        sportspredict_update_interval_seconds=max(
            0.0,
            _float_env("SPORTSPREDICT_UPDATE_INTERVAL_SECONDS", 1.1),
        ),
        extremize_alpha=_float_env("EXTREMIZE_ALPHA", 1.05),
        base_shrinkage=_float_env("BASE_SHRINKAGE", 0.04),
        low_evidence_shrinkage=_float_env("LOW_EVIDENCE_SHRINKAGE", 0.12),
        concurrency=max(1, _int_env("CONCURRENCY", 4)),
        odds_api_key=os.getenv("ODDS_API_KEY", ""),
        odds_sport_key=os.getenv("ODDS_SPORT_KEY", "soccer"),
        state_dir=Path(os.getenv("STATE_DIR", "state")),
        logs_dir=Path(os.getenv("LOGS_DIR", "logs")),
    )
