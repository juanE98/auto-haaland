"""
Opponent Analysis Features Module

Computes features related to the opponent team's defensive vulnerability and attacking threat.
Total: 24 features
"""

from typing import Any

# Opponent analysis features (24 total)
OPPONENT_FEATURES = [
    # Defensive vulnerability (12)
    "opp_goals_conceded_last_3",
    "opp_goals_conceded_last_5",
    "opp_clean_sheets_last_3",
    "opp_clean_sheets_last_5",
    "opp_clean_sheets_rate",  # Season clean sheet %
    "opp_goals_conceded_home",  # Season avg at home
    "opp_goals_conceded_away",  # Season avg away
    "opp_xgc_per_90",  # Expected goals conceded per 90 (if available)
    "opp_defensive_errors_last_5",  # Own goals, penalty conceded
    "opp_saves_rate",  # GK saves / shots faced
    "opp_big_chances_conceded_last_5",
    "opp_defensive_rating",  # Composite score
    # Attacking threat (8)
    "opp_goals_scored_last_3",
    "opp_goals_scored_last_5",
    "opp_xg_per_90",  # Expected goals per 90 (if available)
    "opp_shots_on_target_last_5",
    "opp_big_chances_last_5",
    "opp_goals_scored_home",  # Season avg at home
    "opp_goals_scored_away",  # Season avg away
    "opp_attacking_rating",  # Composite score
    # Context (4)
    "opp_league_position",
    "opp_form_score",  # Points from last 5 matches
    "opp_days_rest",  # Days since last match
    "opp_fixture_congestion",  # Games in last 14 days
]


def compute_opponent_features(
    opponent_data: dict[str, Any] | None = None,
    opponent_fixtures: list[dict[str, Any]] | None = None,
    opponent_players: list[dict[str, Any]] | None = None,
    is_home: bool = True,
    current_date: str | None = None,
) -> dict[str, float]:
    """
    Compute opponent analysis features.

    Args:
        opponent_data: Opponent team data dict with strength, position info
        opponent_fixtures: Recent fixtures for the opponent
        opponent_players: List of all players in opponent team
        is_home: Whether the player's team is playing at home
        current_date: Current date for days_rest calculation (ISO format)

    Returns:
        Dict mapping opponent feature name to its value
    """
    result: dict[str, float] = {}
    opponent_data = opponent_data or {}
    opponent_fixtures = opponent_fixtures or []
    opponent_players = opponent_players or []
    opponent_id = opponent_data.get("id")

    # Get recent fixtures
    recent_3 = (
        opponent_fixtures[-3:] if len(opponent_fixtures) >= 3 else opponent_fixtures
    )
    recent_5 = (
        opponent_fixtures[-5:] if len(opponent_fixtures) >= 5 else opponent_fixtures
    )

    # === Defensive Vulnerability (12) ===
    result["opp_goals_conceded_last_3"] = _avg_or_zero(
        [_get_goals_against(f, opponent_id) for f in recent_3]
    )
    result["opp_goals_conceded_last_5"] = _avg_or_zero(
        [_get_goals_against(f, opponent_id) for f in recent_5]
    )
    result["opp_clean_sheets_last_3"] = _avg_or_zero(
        [1.0 if _get_goals_against(f, opponent_id) == 0 else 0.0 for f in recent_3]
    )
    result["opp_clean_sheets_last_5"] = _avg_or_zero(
        [1.0 if _get_goals_against(f, opponent_id) == 0 else 0.0 for f in recent_5]
    )

    # Season clean sheet rate (from team data)
    clean_sheets = opponent_data.get("clean_sheets", 0)
    games_played = opponent_data.get("played", 1) or 1
    result["opp_clean_sheets_rate"] = round(clean_sheets / games_played, 3)

    # Home/away specific conceded goals
    home_fixtures = [f for f in opponent_fixtures if f.get("team_h") == opponent_id]
    away_fixtures = [f for f in opponent_fixtures if f.get("team_a") == opponent_id]

    result["opp_goals_conceded_home"] = _avg_or_zero(
        [_get_goals_against(f, opponent_id) for f in home_fixtures]
    )
    result["opp_goals_conceded_away"] = _avg_or_zero(
        [_get_goals_against(f, opponent_id) for f in away_fixtures]
    )

    # xGC (not always available, default to goals conceded)
    result["opp_xgc_per_90"] = result["opp_goals_conceded_last_5"]

    # Defensive errors (approximate from own goals, penalty conceded)
    if opponent_players:
        own_goals = sum(p.get("own_goals", 0) for p in opponent_players)
        penalties_conceded = sum(p.get("penalties_missed", 0) for p in opponent_players)
        result["opp_defensive_errors_last_5"] = float(own_goals + penalties_conceded)
    else:
        result["opp_defensive_errors_last_5"] = 0.0

    # GK saves rate (from opponent goalkeepers)
    if opponent_players:
        gks = [p for p in opponent_players if p.get("element_type") == 1]
        if gks:
            total_saves = sum(p.get("saves", 0) for p in gks)
            # Estimate shots faced as saves + goals conceded
            goals_conceded = opponent_data.get("goals_against", 0)
            shots_faced = total_saves + goals_conceded
            result["opp_saves_rate"] = (
                round(total_saves / shots_faced, 3) if shots_faced > 0 else 0.0
            )
        else:
            result["opp_saves_rate"] = 0.0
    else:
        result["opp_saves_rate"] = 0.0

    # Big chances conceded (not directly available, estimate from goals)
    result["opp_big_chances_conceded_last_5"] = (
        result["opp_goals_conceded_last_5"] * 1.5
    )

    # Defensive rating (composite: lower is worse for opponent = better for player)
    # Higher goals conceded, lower clean sheet rate = worse defence
    defensive_rating = (
        (2 - result["opp_goals_conceded_last_5"]) * 3
        + result["opp_clean_sheets_rate"] * 2  # Inverted
        + (1 - result["opp_saves_rate"]) * 2  # Inverted
    )
    result["opp_defensive_rating"] = round(defensive_rating, 2)

    # === Attacking Threat (8) ===
    result["opp_goals_scored_last_3"] = _avg_or_zero(
        [_get_goals_for(f, opponent_id) for f in recent_3]
    )
    result["opp_goals_scored_last_5"] = _avg_or_zero(
        [_get_goals_for(f, opponent_id) for f in recent_5]
    )

    # xG (not always available, default to goals scored)
    result["opp_xg_per_90"] = result["opp_goals_scored_last_5"]

    # Shots and big chances (estimate from player data)
    if opponent_players:
        # Approximate shots on target from threat stat (higher threat = more shots)
        total_threat = sum(float(p.get("threat", 0) or 0) for p in opponent_players)
        result["opp_shots_on_target_last_5"] = round(total_threat / 100, 1)
        result["opp_big_chances_last_5"] = result["opp_goals_scored_last_5"] * 1.3
    else:
        result["opp_shots_on_target_last_5"] = 0.0
        result["opp_big_chances_last_5"] = 0.0

    # Home/away specific scored goals
    result["opp_goals_scored_home"] = _avg_or_zero(
        [_get_goals_for(f, opponent_id) for f in home_fixtures]
    )
    result["opp_goals_scored_away"] = _avg_or_zero(
        [_get_goals_for(f, opponent_id) for f in away_fixtures]
    )

    # Attacking rating (composite: higher is more dangerous opponent)
    attacking_rating = (
        result["opp_goals_scored_last_5"] * 3
        + result["opp_shots_on_target_last_5"] * 0.1
        + result["opp_big_chances_last_5"] * 2
    )
    result["opp_attacking_rating"] = round(attacking_rating, 2)

    # === Context (4) ===
    result["opp_league_position"] = float(opponent_data.get("position", 10))

    # Form score (points from last 5: W=3, D=1, L=0)
    form_points = sum(_get_match_points(f, opponent_id) for f in recent_5)
    result["opp_form_score"] = float(form_points)

    # Days rest (requires current_date and fixture dates)
    if opponent_fixtures and current_date:
        result["opp_days_rest"] = _calculate_days_rest(opponent_fixtures, current_date)
    else:
        result["opp_days_rest"] = 7.0  # Default assumption

    # Fixture congestion (games in last 14 days)
    if opponent_fixtures and current_date:
        result["opp_fixture_congestion"] = _count_recent_games(
            opponent_fixtures, current_date, days=14
        )
    else:
        result["opp_fixture_congestion"] = 2.0  # Default: ~2 games per 14 days

    return result


def _avg_or_zero(values: list[float]) -> float:
    """Return average of values, or 0.0 if empty."""
    return round(sum(values) / len(values), 2) if values else 0.0


def _get_goals_for(fixture: dict[str, Any], team_id: int | None) -> float:
    """Get goals scored by team in fixture."""
    if team_id is None:
        return 0.0
    if fixture.get("team_h") == team_id:
        return float(fixture.get("team_h_score") or 0)
    elif fixture.get("team_a") == team_id:
        return float(fixture.get("team_a_score") or 0)
    return 0.0


def _get_goals_against(fixture: dict[str, Any], team_id: int | None) -> float:
    """Get goals conceded by team in fixture."""
    if team_id is None:
        return 0.0
    if fixture.get("team_h") == team_id:
        return float(fixture.get("team_a_score") or 0)
    elif fixture.get("team_a") == team_id:
        return float(fixture.get("team_h_score") or 0)
    return 0.0


def _get_match_points(fixture: dict[str, Any], team_id: int | None) -> int:
    """Get match points (W=3, D=1, L=0) for team in fixture."""
    if team_id is None:
        return 0
    goals_for = _get_goals_for(fixture, team_id)
    goals_against = _get_goals_against(fixture, team_id)
    if goals_for > goals_against:
        return 3
    elif goals_for == goals_against:
        return 1
    return 0


def _calculate_days_rest(fixtures: list[dict[str, Any]], current_date: str) -> float:
    """Calculate days since last fixture."""
    from datetime import datetime

    try:
        current = datetime.fromisoformat(current_date.replace("Z", "+00:00"))
        for fixture in reversed(fixtures):
            kickoff = fixture.get("kickoff_time")
            if kickoff:
                fixture_date = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
                if fixture_date < current:
                    return float((current - fixture_date).days)
    except (ValueError, TypeError):
        pass
    return 7.0  # Default


def _count_recent_games(
    fixtures: list[dict[str, Any]], current_date: str, days: int = 14
) -> float:
    """Count games played in the last N days."""
    from datetime import datetime, timedelta

    try:
        current = datetime.fromisoformat(current_date.replace("Z", "+00:00"))
        cutoff = current - timedelta(days=days)
        count = 0
        for fixture in fixtures:
            kickoff = fixture.get("kickoff_time")
            if kickoff:
                fixture_date = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
                if cutoff <= fixture_date < current:
                    count += 1
        return float(count)
    except (ValueError, TypeError):
        pass
    return 2.0  # Default assumption
