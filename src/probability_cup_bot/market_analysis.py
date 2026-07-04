from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


VOLATILE_FAMILIES = {
    "cards",
    "corners",
    "fouls",
    "offsides",
    "penalty",
    "player_assist",
    "player_goal",
    "player_shot",
    "player_shot_on_target",
    "red_card",
    "shots_on_target",
}


@dataclass(frozen=True)
class MarketProfile:
    market_id: str
    question: str
    family: str
    volatile: bool
    broad_prior: tuple[float, float] | None
    prior_note: str
    decomposition_hint: str
    subject_key: str = ""
    threshold: float | None = None

    def model_payload(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "family": self.family,
            "volatile": self.volatile,
            "broad_prior_range": list(self.broad_prior) if self.broad_prior else None,
            "prior_note": self.prior_note,
            "decomposition_hint": self.decomposition_hint,
            "subject_key": self.subject_key,
            "threshold": self.threshold,
        }


def profile_market(market_id: str, question: str, match_name: str = "") -> MarketProfile:
    family = classify_market_family(question, match_name=match_name)
    broad_prior, prior_note = broad_prior_for_family(family)
    return MarketProfile(
        market_id=market_id,
        question=question,
        family=family,
        volatile=family in VOLATILE_FAMILIES,
        broad_prior=broad_prior,
        prior_note=prior_note,
        decomposition_hint=decomposition_hint_for_family(family),
        subject_key=subject_key(question),
        threshold=threshold_from_question(question),
    )


def profile_markets(markets: list[Any]) -> dict[str, MarketProfile]:
    return {
        market.id: profile_market(
            market.id,
            market.question,
            match_name=getattr(getattr(market, "match", None), "name", ""),
        )
        for market in markets
    }


def classify_market_family(question: str, *, match_name: str = "") -> str:
    q = _normalize(question)
    player_like = _looks_like_player_prop(question, match_name)
    if "penalty" in q and any(term in q for term in ("awarded", "given", "taken", "conceded")):
        return "penalty"
    if "red card" in q or "sent off" in q:
        return "red_card"
    if "yellow card" in q or "booking" in q or re.search(r"\bcards?\b", q):
        return "cards"
    if "corner" in q:
        return "corners"
    if "offside" in q:
        return "offsides"
    if "foul" in q:
        return "fouls"
    if "both teams" in q and ("score" in q or "to score" in q):
        return "btts"
    if "shot on target" in q or "shots on target" in q:
        return "player_shot_on_target" if player_like else "shots_on_target"
    if re.search(r"\bshots?\b", q):
        return "player_shot" if player_like else "shots"
    if "assist" in q:
        return "player_assist" if player_like else "other"
    if "score" in q or "goal" in q:
        if player_like:
            return "player_goal"
        if "first half" in q or "second half" in q or "1st half" in q or "2nd half" in q:
            return "half_goal"
        if "total" in q or "over" in q or "under" in q or "+" in q:
            return "team_total"
        return "goal_total"
    if "win" in q or "draw" in q or "advance" in q:
        return "match_result"
    return "other"


def broad_prior_for_family(family: str) -> tuple[tuple[float, float] | None, str]:
    priors: dict[str, tuple[tuple[float, float], str]] = {
        "penalty": ((0.18, 0.34), "Broad soccer match prior for at least one penalty; referee, VAR, and box entries matter."),
        "red_card": ((0.10, 0.24), "Broad soccer match prior for at least one red card; referee and game state dominate."),
        "cards": ((0.40, 0.75), "Card thresholds are referee/team dependent; use broad prior only until a line or referee rate is found."),
        "corners": ((0.35, 0.70), "Corner thresholds need team/opponent rates or market odds; broad prior is weak."),
        "offsides": ((0.25, 0.60), "Offside thresholds depend on attacking style and defensive line; broad prior is weak."),
        "fouls": ((0.35, 0.70), "Foul thresholds depend on referee and matchup intensity; broad prior is weak."),
        "btts": ((0.38, 0.58), "Broad BTTS prior; adjust with team strength, total-goals market, and lineup news."),
        "goal_total": ((0.35, 0.65), "Goal-total markets should anchor on bookmaker total or Poisson-style goal expectations when available."),
        "team_total": ((0.25, 0.65), "Team-total threshold needs team implied goals or odds; broad prior is weak."),
        "half_goal": ((0.25, 0.55), "Half-specific goal markets are noisy and should usually be closer to base rates."),
        "shots_on_target": ((0.35, 0.70), "Team SOT thresholds need recent team and opponent SOT rates or a sportsbook line."),
        "player_shot_on_target": ((0.08, 0.45), "Player SOT props need start/minutes, role, per-90 SOT, opponent concession, and set-piece/penalty paths."),
        "player_shot": ((0.15, 0.60), "Player shot props need start/minutes, role, per-90 shots, and opponent concession."),
        "player_goal": ((0.05, 0.35), "Player goal props need start/minutes, anytime-goalscorer odds or xG/90, and penalty role."),
        "player_assist": ((0.04, 0.25), "Player assist props need start/minutes, creative role, set pieces, and teammate finishing."),
        "match_result": ((0.20, 0.70), "Match-result markets need odds/ratings first, then lineup and motivation adjustments."),
    }
    if family not in priors:
        return None, "No broad prior configured; require a source-backed reference class."
    return priors[family]


def decomposition_hint_for_family(family: str) -> str:
    hints = {
        "penalty": "Estimate union of penalty-award paths: box entries, dribbles, referee/VAR, handball/contact, and overlap.",
        "red_card": "Estimate union of straight red and second-yellow paths; account for referee and match intensity.",
        "cards": "Use referee/team card rates and threshold distribution, not only team strength.",
        "corners": "Use team for/against corner rates, game script, and attacking width.",
        "offsides": "Use attacking line style, opponent high line, and player run profile.",
        "fouls": "Use referee foul rate, pressing/directness, and mismatch stress.",
        "btts": "Estimate P(team A scores) and P(team B scores | A scores), keeping total-goals correlation explicit.",
        "goal_total": "Anchor on implied total goals when available; otherwise use team-strength/Poisson-style expectations.",
        "team_total": "Estimate team expected goals, then threshold probability; do not treat favorite status as enough.",
        "half_goal": "Start with half-specific base rate and adjust for tempo/lineups; avoid overconfidence.",
        "shots_on_target": "Use team SOT for/against rates and game script; translate means to thresholds conservatively.",
        "player_shot_on_target": "Estimate P(start/plays enough) x open-play SOT rate plus penalty/free-kick/set-piece paths.",
        "player_shot": "Estimate P(start/plays enough) x per-90 shot rate adjusted for role/opponent.",
        "player_goal": "Estimate P(start/plays enough) x non-penalty goal rate plus penalty-taker path if applicable.",
        "player_assist": "Estimate P(start/plays enough) x assist/creative rate plus set-piece role.",
        "match_result": "Anchor on odds/ratings if available, then adjust for lineup, rotation, rest, and incentives.",
    }
    return hints.get(family, "Find a source-backed reference class and state why it applies.")


def market_subtype(question: str) -> str:
    """Coarse structural sub-type within a family.

    Round-of-32 platform recap (2026-07-04) showed opposite biases pooled
    inside single families: cards comparisons were well-calibrated (+3.3 RBP
    vs crowd) while total-cards thresholds were the worst category (-2.8), and
    total-SOT under-forecasts cancelled comparison over-forecasts. Corrections
    are therefore keyed by family|subtype with the family as fallback.
    """
    q = _normalize(question)
    if re.search(r"\bmore\b.+\bthan\b|\bfewer\b.+\bthan\b", q):
        return "comparison"
    if "total" in q:
        return "total_threshold"
    return "team_or_other"


def threshold_from_question(question: str) -> float | None:
    q = _normalize(question)
    match = re.search(r"(\d+(?:\.\d+)?)\s*\+", q)
    if match:
        return float(match.group(1))
    match = re.search(r"(?:over|under|at least)\s+(\d+(?:\.\d+)?)", q)
    if match:
        return float(match.group(1))
    return None


def subject_key(question: str) -> str:
    q = re.sub(r"[^a-z0-9+ ]+", " ", question.lower())
    q = re.sub(r"\s+", " ", q).strip()
    q = re.sub(r"\bwill\b", "", q).strip()
    q = re.sub(r"\b(have|record|make|score|get|receive|commit|win|draw|advance)\b.*", "", q).strip()
    q = re.sub(r"\b(over|under|at least)\b.*", "", q).strip()
    return q[:80]


def question_subject(question: str) -> str:
    q = re.sub(r"[^a-z0-9+ ]+", " ", question.lower())
    q = re.sub(r"\s+", " ", q).strip()
    q = re.sub(r"^will\s+", "", q)
    return re.split(
        r"\b(have|record|make|score|get|receive|commit|win|draw|advance|be|take)\b",
        q,
        maxsplit=1,
    )[0].strip()


def _normalize(question: str) -> str:
    return re.sub(r"\s+", " ", question.lower()).strip()


def _looks_like_player_prop(question: str, match_name: str) -> bool:
    q = _normalize(question)
    if any(term in q for term in ("player", "any player")):
        return True
    if any(term in q for term in ("team", "both teams", "total goals")):
        return False
    subject = question_subject(question)
    if not subject:
        return False
    teams = {_normalize(team) for team in re.split(r"\s+v(?:s)?\.?\s+|\s+-\s+", match_name, flags=re.I) if team.strip()}
    if subject in teams:
        return False
    if subject and any(subject == team or subject.endswith(f" {team}") for team in teams):
        return False
    if subject in {"there", "either team", "both teams", "neither team", "the match", "match"}:
        return False
    words = [word for word in subject.split() if word not in {"the"}]
    return len(words) >= 2
