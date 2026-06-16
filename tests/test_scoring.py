from probability_cup_bot.scoring import (
    aggregate_probabilities,
    clamp_probability,
    extremize,
    log_odds_mean,
    probability_to_int,
)


def test_probability_to_int_clamps_to_platform_bounds() -> None:
    assert probability_to_int(0.0) == 1
    assert probability_to_int(1.0) == 99
    assert probability_to_int(0.754) == 75


def test_log_odds_mean_is_symmetric() -> None:
    assert round(log_odds_mean([0.2, 0.8]), 6) == 0.5


def test_extremize_moves_away_from_half() -> None:
    assert extremize(0.7, 1.2) > 0.7
    assert extremize(0.3, 1.2) < 0.3


def test_aggregate_shrinks_low_evidence() -> None:
    p = aggregate_probabilities([0.8, 0.82, 0.78], alpha=1.0, shrinkage=0.25)
    assert 0.7 < p < 0.82


def test_clamp_probability() -> None:
    assert clamp_probability(-2) == 0.01
    assert clamp_probability(2) == 0.99

