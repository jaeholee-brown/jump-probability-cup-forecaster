from probability_cup_bot.market_analysis import classify_market_family, profile_market


def test_market_classifier_uses_match_teams_to_avoid_false_player_props() -> None:
    assert (
        classify_market_family(
            "Will France score 2+ goals?",
            match_name="France vs Senegal",
        )
        == "team_total"
    )
    assert (
        classify_market_family(
            "Will Granit Xhaka have 1+ shot on target?",
            match_name="Switzerland vs Bosnia",
        )
        == "player_shot_on_target"
    )
    assert (
        classify_market_family(
            "Will Switzerland have 4+ shots on target?",
            match_name="Switzerland vs Bosnia",
        )
        == "shots_on_target"
    )
    assert (
        classify_market_family(
            "Will Argentina have 6 or more shots on target?",
            match_name="ARG vs AUT",
        )
        == "shots_on_target"
    )


def test_market_profile_exposes_prior_and_decomposition_hint() -> None:
    profile = profile_market(
        "market",
        "Will Granit Xhaka have 1+ shot on target?",
        "Switzerland vs Bosnia",
    )

    assert profile.family == "player_shot_on_target"
    assert profile.volatile is True
    assert profile.broad_prior is not None
    assert "penalty" in profile.decomposition_hint.lower()
