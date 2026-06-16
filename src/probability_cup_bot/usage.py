from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator


_CURRENT_TRACKER: ContextVar[UsageTracker | None] = ContextVar("usage_tracker", default=None)

XAI_SEARCH_TOOL_COST_USD = 0.005
XAI_COST_TICKS_PER_USD = 10_000_000_000

MODEL_PRICING_PER_MILLION = {
    ("openai", "gpt-5"): {"input": 1.25, "output": 10.0},
    ("xai", "grok-4.3"): {"input": 1.25, "output": 2.5},
    ("xai", "grok-4.20"): {"input": 1.25, "output": 2.5},
    ("anthropic", "claude-opus"): {"input": 5.0, "output": 25.0},
}


@dataclass
class UsageTracker:
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        *,
        provider: str,
        model: str,
        schema_name: str,
        usage: Any = None,
        server_side_tool_usage: Any = None,
    ) -> dict[str, Any]:
        usage_data = _jsonable(usage)
        tool_usage_data = _jsonable(server_side_tool_usage)
        tokens = _token_counts(usage_data)
        tool_calls = _tool_call_count({"usage": usage_data, "server_side_tool_usage": tool_usage_data})
        estimated_cost_usd, cost_source = _cost_usd(provider, model, usage_data, tokens, tool_calls)
        event = {
            "provider": provider,
            "model": model,
            "schema": schema_name,
            "input_tokens": tokens["input_tokens"],
            "output_tokens": tokens["output_tokens"],
            "reasoning_tokens": tokens["reasoning_tokens"],
            "total_tokens": tokens["total_tokens"],
            "tool_calls": tool_calls,
            "estimated_cost_usd": round(estimated_cost_usd, 6),
            "cost_source": cost_source,
            "usage": usage_data,
            "server_side_tool_usage": tool_usage_data,
        }
        self.events.append(event)
        return event

    def summary(self) -> dict[str, Any]:
        by_provider: dict[str, dict[str, Any]] = {}
        by_model: dict[str, dict[str, Any]] = {}
        for event in self.events:
            provider_bucket = by_provider.setdefault(event["provider"], _empty_bucket())
            model_bucket = by_model.setdefault(f"{event['provider']}:{event['model']}", _empty_bucket())
            for bucket in (provider_bucket, model_bucket):
                bucket["calls"] += 1
                bucket["input_tokens"] += int(event["input_tokens"] or 0)
                bucket["output_tokens"] += int(event["output_tokens"] or 0)
                bucket["reasoning_tokens"] += int(event["reasoning_tokens"] or 0)
                bucket["total_tokens"] += int(event["total_tokens"] or 0)
                bucket["tool_calls"] += int(event["tool_calls"] or 0)
                bucket["estimated_cost_usd"] += float(event["estimated_cost_usd"] or 0)
        for bucket in list(by_provider.values()) + list(by_model.values()):
            bucket["estimated_cost_usd"] = round(bucket["estimated_cost_usd"], 6)
        return {
            "call_count": len(self.events),
            "estimated_cost_usd": round(sum(event["estimated_cost_usd"] for event in self.events), 6),
            "by_provider": by_provider,
            "by_model": by_model,
            "events": self.events,
            "pricing_note": (
                "xAI costs use provider-returned cost_in_usd_ticks when present. Other costs are "
                "estimated from configured per-token prices plus exposed server-side tool counts. "
                "Provider dashboards remain authoritative."
            ),
        }


def record_usage(
    *,
    provider: str,
    model: str,
    schema_name: str,
    usage: Any = None,
    server_side_tool_usage: Any = None,
) -> dict[str, Any] | None:
    tracker = _CURRENT_TRACKER.get()
    if tracker is None:
        return None
    return tracker.record(
        provider=provider,
        model=model,
        schema_name=schema_name,
        usage=usage,
        server_side_tool_usage=server_side_tool_usage,
    )


def set_current_tracker(tracker: UsageTracker) -> Any:
    return _CURRENT_TRACKER.set(tracker)


def reset_current_tracker(token: Any) -> None:
    _CURRENT_TRACKER.reset(token)


@contextmanager
def track_usage(tracker: UsageTracker) -> Iterator[UsageTracker]:
    token = _CURRENT_TRACKER.set(tracker)
    try:
        yield tracker
    finally:
        _CURRENT_TRACKER.reset(token)


def update_usage_ledger(previous: dict[str, Any], run_summary: dict[str, Any]) -> dict[str, Any]:
    cumulative = previous.get("cumulative") if isinstance(previous.get("cumulative"), dict) else {}
    by_provider = dict(cumulative.get("by_provider") or {})
    for provider, bucket in (run_summary.get("by_provider") or {}).items():
        existing = dict(by_provider.get(provider) or _empty_bucket())
        for key in ("calls", "input_tokens", "output_tokens", "reasoning_tokens", "total_tokens", "tool_calls"):
            existing[key] = int(existing.get(key) or 0) + int(bucket.get(key) or 0)
        existing["estimated_cost_usd"] = round(
            float(existing.get("estimated_cost_usd") or 0) + float(bucket.get("estimated_cost_usd") or 0),
            6,
        )
        by_provider[provider] = existing
    runs = list(previous.get("runs") or [])
    runs.append(
        {
            "generated_at": run_summary.get("generated_at"),
            "call_count": run_summary.get("call_count", 0),
            "estimated_cost_usd": run_summary.get("estimated_cost_usd", 0),
            "by_provider": run_summary.get("by_provider", {}),
        }
    )
    total_cost = round(sum(float(bucket.get("estimated_cost_usd") or 0) for bucket in by_provider.values()), 6)
    return {
        "generated_at": run_summary.get("generated_at"),
        "cumulative": {
            "estimated_cost_usd": total_cost,
            "by_provider": by_provider,
        },
        "runs": runs[-200:],
    }


def _empty_bucket() -> dict[str, Any]:
    return {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "tool_calls": 0,
        "estimated_cost_usd": 0.0,
    }


def _estimated_cost_usd(
    provider: str,
    model: str,
    tokens: dict[str, int],
    tool_calls: int,
) -> float:
    pricing = _pricing_for(provider, model)
    input_cost = tokens["input_tokens"] / 1_000_000 * pricing.get("input", 0.0)
    output_cost = tokens["output_tokens"] / 1_000_000 * pricing.get("output", 0.0)
    tool_cost = tool_calls * XAI_SEARCH_TOOL_COST_USD if provider == "xai" else 0.0
    return input_cost + output_cost + tool_cost


def _cost_usd(
    provider: str,
    model: str,
    usage_data: Any,
    tokens: dict[str, int],
    tool_calls: int,
) -> tuple[float, str]:
    if provider == "xai" and isinstance(usage_data, dict):
        ticks = usage_data.get("cost_in_usd_ticks")
        if isinstance(ticks, int | float):
            return float(ticks) / XAI_COST_TICKS_PER_USD, "xai_cost_in_usd_ticks"
    return _estimated_cost_usd(provider, model, tokens, tool_calls), "configured_estimate"


def _pricing_for(provider: str, model: str) -> dict[str, float]:
    for (pricing_provider, prefix), pricing in MODEL_PRICING_PER_MILLION.items():
        if provider == pricing_provider and model.startswith(prefix):
            return pricing
    return {"input": 0.0, "output": 0.0}


def _token_counts(usage: Any) -> dict[str, int]:
    usage_data = usage if isinstance(usage, dict) else {}
    input_tokens = _first_int(usage_data, ("input_tokens", "prompt_tokens"))
    output_tokens = _first_int(usage_data, ("output_tokens", "completion_tokens"))
    total_tokens = _first_int(usage_data, ("total_tokens",))
    reasoning_tokens = _find_int_by_key(usage_data, "reasoning_tokens")
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
    }


def _first_int(data: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        value = data.get(key)
        if isinstance(value, int | float):
            return int(value)
    return 0


def _find_int_by_key(value: Any, target_key: str) -> int:
    if isinstance(value, dict):
        total = 0
        for key, item in value.items():
            if key == target_key and isinstance(item, int | float):
                total += int(item)
            else:
                total += _find_int_by_key(item, target_key)
        return total
    if isinstance(value, list):
        return sum(_find_int_by_key(item, target_key) for item in value)
    return 0


def _tool_call_count(value: Any) -> int:
    if isinstance(value, dict):
        total = 0
        for key, item in value.items():
            lowered = key.lower()
            if isinstance(item, int | float) and ("search" in lowered or lowered.endswith("_calls")):
                total += int(item)
            else:
                total += _tool_call_count(item)
        return total
    if isinstance(value, list):
        return sum(_tool_call_count(item) for item in value)
    return 0


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict | list | str | int | float | bool):
        return value
    return str(value)
