from probability_cup_bot.usage import UsageTracker, update_usage_ledger


def test_usage_tracker_summarizes_cost_by_provider() -> None:
    tracker = UsageTracker()

    tracker.record(
        provider="openai",
        model="gpt-5",
        schema_name="forecast_batch",
        usage={"input_tokens": 1_000_000, "output_tokens": 100_000, "output_tokens_details": {"reasoning_tokens": 50_000}},
    )
    tracker.record(
        provider="xai",
        model="grok-4.3",
        schema_name="news_check",
        usage={"input_tokens": 1_000, "output_tokens": 2_000, "web_search_count": 2},
    )

    summary = tracker.summary()

    assert summary["call_count"] == 2
    assert summary["by_provider"]["openai"]["estimated_cost_usd"] == 2.25
    assert summary["by_provider"]["openai"]["reasoning_tokens"] == 50_000
    assert summary["by_provider"]["xai"]["tool_calls"] == 2
    assert summary["by_provider"]["xai"]["estimated_cost_usd"] > 0.01


def test_update_usage_ledger_accumulates_by_provider() -> None:
    previous = {
        "cumulative": {
            "by_provider": {
                "xai": {
                    "calls": 1,
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "reasoning_tokens": 0,
                    "total_tokens": 30,
                    "tool_calls": 1,
                    "estimated_cost_usd": 0.1,
                }
            }
        },
        "runs": [],
    }
    run_summary = {
        "generated_at": "2026-06-16T00:00:00Z",
        "call_count": 1,
        "estimated_cost_usd": 0.2,
        "by_provider": {
            "xai": {
                "calls": 2,
                "input_tokens": 100,
                "output_tokens": 200,
                "reasoning_tokens": 50,
                "total_tokens": 300,
                "tool_calls": 2,
                "estimated_cost_usd": 0.2,
            }
        },
    }

    ledger = update_usage_ledger(previous, run_summary)

    assert ledger["cumulative"]["by_provider"]["xai"]["calls"] == 3
    assert ledger["cumulative"]["by_provider"]["xai"]["estimated_cost_usd"] == 0.3
    assert ledger["runs"][0]["call_count"] == 1
