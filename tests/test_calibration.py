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
    assert report["models"]["good"]["mean_brier"] < report["models"]["bad"]["mean_brier"]
    assert report["suggested_multipliers"]["good"] > report["suggested_multipliers"]["bad"]
