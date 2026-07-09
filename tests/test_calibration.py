from probability_cup_bot.calibration import build_calibration_report, infer_outcome


def test_infer_outcome_from_brier_score() -> None:
    assert infer_outcome({"probability_submitted": 75, "brier_score": 0.0625}) == 1
    assert infer_outcome({"probability_submitted": 25, "brier_score": 0.0625}) == 0


def test_calibration_report_scores_components_and_suggests_weights() -> None:
    report = build_calibration_report(
        results=[
            {
                "market_id": "market",
                "question": "Will A win?",
                "probability_submitted": 70,
                "brier_score": 0.09,
            }
        ],
        history={
            "markets": {
                "market": {
                    "question": "Will A win?",
                    "market_family": "match_result",
                    "components": [
                        {
                            "model": "good",
                            "provider": "openai",
                            "variant": "base_rate_frequency",
                            "probability": 0.8,
                            "weight": 1.0,
                        },
                        {
                            "model": "bad",
                            "provider": "grok",
                            "variant": "base_rate_frequency",
                            "probability": 0.2,
                            "weight": 0.4,
                        },
                    ],
                }
            }
        },
        learning_rate=2.0,
        prior_count=1,
    )

    assert report["settled_market_count"] == 1
    assert report["market_families"]["match_result"]["count"] == 1
    assert (
        report["models_by_family"]["match_result"]["good"]["mean_brier"]
        < report["models_by_family"]["match_result"]["bad"]["mean_brier"]
    )
    assert report["models"]["good"]["mean_brier"] < report["models"]["bad"]["mean_brier"]
    assert report["suggested_multipliers"]["good"] > report["suggested_multipliers"]["bad"]


def _settled_record(family: str, probability: float, outcome: int) -> dict:
    return {
        "market_family": family,
        "outcome": outcome,
        "probability_submitted": int(round(probability * 100)),
        "components": [
            {"model": "a", "provider": "openai", "probability": probability, "weight": 1.0},
            {"model": "b", "provider": "claude", "probability": probability, "weight": 1.0},
        ],
    }


def test_family_corrections_disabled_below_min_settled() -> None:
    from probability_cup_bot.calibration import build_family_corrections

    records = [_settled_record("cards", 0.5, 0) for _ in range(10)]
    corrections = build_family_corrections(records, min_settled=150)

    assert corrections["enabled"] is False


def test_family_corrections_shift_direction_matches_bias() -> None:
    from probability_cup_bot.calibration import build_family_corrections

    # cards over-forecast (says 55%, resolves 25%), corners under-forecast.
    records = []
    for i in range(100):
        records.append(_settled_record("cards", 0.55, 1 if i % 4 == 0 else 0))
        records.append(_settled_record("corners", 0.45, 0 if i % 4 == 0 else 1))
    corrections = build_family_corrections(records, min_settled=150)

    assert corrections["enabled"] is True
    assert corrections["shifts"]["cards"] < 0
    assert corrections["shifts"]["corners"] > 0
    assert 0.9 <= corrections["slope"] <= 1.4
    assert corrections["family_stats"]["cards"]["count"] == 100


def test_family_corrections_shift_is_clamped() -> None:
    from probability_cup_bot.calibration import build_family_corrections

    records = [_settled_record("player_assist", 0.6, 0) for _ in range(200)]
    corrections = build_family_corrections(records, min_settled=150, max_shift=0.6)

    assert corrections["shifts"]["player_assist"] == -0.6


def test_suggested_multipliers_are_stateless_in_current_multiplier() -> None:
    report_kwargs = dict(
        results=[
            {
                "market_id": "market",
                "question": "Will A win?",
                "probability_submitted": 70,
                "brier_score": 0.09,
            }
        ],
        history={
            "markets": {
                "market": {
                    "question": "Will A win?",
                    "market_family": "match_result",
                    "components": [
                        {"model": "good", "provider": "openai", "probability": 0.8, "weight": 1.0},
                        {"model": "bad", "provider": "grok", "probability": 0.2, "weight": 1.0},
                    ],
                }
            }
        },
        learning_rate=2.0,
        prior_count=1,
    )
    first = build_calibration_report(current_multipliers={}, **report_kwargs)
    # Feeding the suggestions back in must NOT compound them further.
    second = build_calibration_report(
        current_multipliers=first["suggested_multipliers"], **report_kwargs
    )

    assert first["suggested_multipliers"] == second["suggested_multipliers"]


def test_family_corrections_split_subtypes_within_family() -> None:
    from probability_cup_bot.calibration import build_family_corrections, lookup_shift

    records = []
    for i in range(80):
        # comparisons well-calibrated: says 50%, resolves 50%
        records.append(
            {
                "market_family": "cards",
                "question": "Will Team A receive more cards than Team B?",
                "outcome": 1 if i % 2 == 0 else 0,
                "probability_submitted": 50,
                "components": [
                    {"model": "a", "provider": "openai", "probability": 0.5, "weight": 1.0}
                ],
            }
        )
        # totals badly over-forecast: says 55%, resolves 20%
        records.append(
            {
                "market_family": "cards",
                "question": "Will there be 4 or more total cards shown?",
                "outcome": 1 if i % 5 == 0 else 0,
                "probability_submitted": 55,
                "components": [
                    {"model": "a", "provider": "openai", "probability": 0.55, "weight": 1.0}
                ],
            }
        )
    corrections = build_family_corrections(records, min_settled=150)
    shifts = corrections["shifts"]

    total_shift = lookup_shift(shifts, "cards", "total_threshold")
    comparison_shift = lookup_shift(shifts, "cards", "comparison")
    assert total_shift < -0.3
    assert abs(comparison_shift) < abs(total_shift) / 2
    assert corrections["family_stats"]["cards|total_threshold"]["count"] == 80


def test_family_corrections_since_filters_by_closing_time() -> None:
    from probability_cup_bot.calibration import build_family_corrections

    old = [
        dict(_settled_record("cards", 0.5, 1), closing_time="2026-06-20T19:00:00Z")
        for _ in range(100)
    ]
    new = [
        dict(_settled_record("cards", 0.5, 0), closing_time="2026-07-01T19:00:00Z")
        for _ in range(160)
    ]
    all_fit = build_family_corrections(old + new, min_settled=150)
    windowed = build_family_corrections(old + new, min_settled=150, since="2026-06-28")

    assert all_fit["fit_window"] == "all"
    assert windowed["fit_window"] == "since 2026-06-28"
    assert windowed["settled_count"] == 160
    # windowed sees only the all-NO era, so its cards shift is more negative
    assert windowed["shifts"]["cards"] < all_fit["shifts"]["cards"]

    # falls back to full history when the window is too thin
    fallback = build_family_corrections(old + old + new[:20], min_settled=150, since="2026-06-28")
    assert fallback["fit_window"] == "all"
    assert fallback["enabled"] is True
