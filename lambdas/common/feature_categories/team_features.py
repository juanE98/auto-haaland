"""
Team Context Features Module

Computes features related to the player's team performance, strength, and context.
Total: 28 features
"""

from typing import Any

# Team context features (28 total)
TEAM_FEATURES = [
    # Team form (10)
    "team_goals_scored_last_3",
    "team_goals_scored_last_5",
    "team_goals_conceded_last_3",
    "team_goals_conceded_last_5",
    "team_clean_sheets_last_3",
    "team_clean_sheets_last_5",
    "team_wins_last_3",
    "team_wins_last_5",
    "team_form_score",  # Points from last 5 matches
    "team_form_trend",  # Recent vs earlier form
    # Team strength (8)
    "team_strength_overall",
    "team_strength_attack_home",
    "team_strength_attack_away",
    "team_strength_defence_home",
    "team_strength_defence_away",
    "team_attack_vs_opp_defence",  # Ratio
    "team_defence_vs_opp_attack",  # Ratio
    "strength_differential",  # Team overall - opponent overall
    # League position (4)
    "team_league_position",
    "team_points",
    "team_goal_difference",
    "team_position_change_last_5",
    # Player context within team (6)
    "team_total_points_avg",  # Average points of all team players
    "player_share_of_team_points",  # Player points / team total
    "player_share_of_team_goals",  # Player goals / team goals
    "team_avg_ict",  # Average ICT of team
    "team_players_available",  # Count of available players
    "squad_depth_at_position",  # Players at same position
]


def compute_team_features(
    player: dict[str, Any],
    team_data: dict[str, Any] | None = None,
    team_players: list[dict[str, Any]] | None = None,
    team_fixtures: list[dict[str, Any]] | None = None,
    opponent_data: dict[str, Any] | None = None,
) -> dict[str, float]:
    """
    Compute team context features for a player.

    Args:
        player: Player element dict from FPL API
        team_data: Team data dict with strength, league position info
        team_players: List of all players in the same team
        team_fixtures: Recent fixtures for the team
        opponent_data: Opponent team data for differential calculations

    Returns:
        Dict mapping team feature name to its value
    """
    result: dict[str, float] = {}
    team_data = team_data or {}
    team_players = team_players or []
    team_fixtures = team_fixtures or []
    opponent_data = opponent_data or {}

    # === Team Form (10) ===
    # Calculate from recent fixtures
    recent_3 = team_fixtures[-3:] if len(team_fixtures) >= 3 else team_fixtures
    recent_5 = team_fixtures[-5:] if len(team_fixtures) >= 5 else team_fixtures

    result["team_goals_scored_last_3"] = _avg_or_zero(
        [_get_goals_for(f, player.get("team")) for f in recent_3]
    )
    result["team_goals_scored_last_5"] = _avg_or_zero(
        [_get_goals_for(f, player.get("team")) for f in recent_5]
    )
    result["team_goals_conceded_last_3"] = _avg_or_zero(
        [_get_goals_against(f, player.get("team")) for f in recent_3]
    )
    result["team_goals_conceded_last_5"] = _avg_or_zero(
        [_get_goals_against(f, player.get("team")) for f in recent_5]
    )
    result["team_clean_sheets_last_3"] = _avg_or_zero(
        [
            1.0 if _get_goals_against(f, player.get("team")) == 0 else 0.0
            for f in recent_3
        ]
    )
    result["team_clean_sheets_last_5"] = _avg_or_zero(
        [
            1.0 if _get_goals_against(f, player.get("team")) == 0 else 0.0
            for f in recent_5
        ]
    )
    result["team_wins_last_3"] = _avg_or_zero(
        [1.0 if _is_win(f, player.get("team")) else 0.0 for f in recent_3]
    )
    result["team_wins_last_5"] = _avg_or_zero(
        [1.0 if _is_win(f, player.get("team")) else 0.0 for f in recent_5]
    )

    # Team form score (points from last 5: W=3, D=1, L=0)
    form_points = sum(_get_match_points(f, player.get("team")) for f in recent_5)
    result["team_form_score"] = float(form_points)

    # Form trend (recent 2 vs earlier 3)
    if len(team_fixtures) >= 5:
        recent_form = sum(
            _get_match_points(f, player.get("team")) for f in team_fixtures[-2:]
        )
        earlier_form = sum(
            _get_match_points(f, player.get("team")) for f in team_fixtures[-5:-2]
        )
        result["team_form_trend"] = round(recent_form - earlier_form / 3 * 2, 2)
    else:
        result["team_form_trend"] = 0.0

    # === Team Strength (8) ===
    result["team_strength_overall"] = (
        float(
            team_data.get("strength_overall_home", 0)
            + team_data.get("strength_overall_away", 0)
        )
        / 2
    )
    result["team_strength_attack_home"] = float(
        team_data.get("strength_attack_home", 1200)
    )
    result["team_strength_attack_away"] = float(
        team_data.get("strength_attack_away", 1200)
    )
    result["team_strength_defence_home"] = float(
        team_data.get("strength_defence_home", 1200)
    )
    result["team_strength_defence_away"] = float(
        team_data.get("strength_defence_away", 1200)
    )

    # Strength differentials (require opponent data)
    opp_def = (
        opponent_data.get("strength_defence_home", 1200)
        + opponent_data.get("strength_defence_away", 1200)
    ) / 2
    opp_atk = (
        opponent_data.get("strength_attack_home", 1200)
        + opponent_data.get("strength_attack_away", 1200)
    ) / 2
    team_atk = (
        team_data.get("strength_attack_home", 1200)
        + team_data.get("strength_attack_away", 1200)
    ) / 2
    team_def = (
        team_data.get("strength_defence_home", 1200)
        + team_data.get("strength_defence_away", 1200)
    ) / 2

    result["team_attack_vs_opp_defence"] = (
        round(team_atk / opp_def, 3) if opp_def > 0 else 1.0
    )
    result["team_defence_vs_opp_attack"] = (
        round(team_def / opp_atk, 3) if opp_atk > 0 else 1.0
    )

    team_overall = result["team_strength_overall"]
    opp_overall = (
        float(
            opponent_data.get("strength_overall_home", 0)
            + opponent_data.get("strength_overall_away", 0)
        )
        / 2
    )
    result["strength_differential"] = round(team_overall - opp_overall, 1)

    # === League Position (4) ===
    result["team_league_position"] = float(team_data.get("position", 10))
    result["team_points"] = float(team_data.get("points", 0))
    result["team_goal_difference"] = float(team_data.get("goal_difference", 0))

    # Position change (from team data if available, otherwise 0)
    result["team_position_change_last_5"] = float(team_data.get("position_change", 0))

    # === Player Context (6) ===
    if team_players:
        total_team_points = sum(p.get("total_points", 0) for p in team_players)
        total_team_goals = sum(p.get("goals_scored", 0) for p in team_players)
        avg_points = total_team_points / len(team_players) if team_players else 0

        result["team_total_points_avg"] = round(avg_points, 2)

        player_points = player.get("total_points", 0)
        player_goals = player.get("goals_scored", 0)

        result["player_share_of_team_points"] = (
            round(player_points / total_team_points, 4)
            if total_team_points > 0
            else 0.0
        )
        result["player_share_of_team_goals"] = (
            round(player_goals / total_team_goals, 4) if total_team_goals > 0 else 0.0
        )

        # Average ICT
        avg_ict = sum(float(p.get("ict_index", 0) or 0) for p in team_players) / len(
            team_players
        )
        result["team_avg_ict"] = round(avg_ict, 2)

        # Players available (status == 'a')
        available = sum(1 for p in team_players if p.get("status") == "a")
        result["team_players_available"] = float(available)

        # Squad depth at position
        player_position = player.get("element_type", 0)
        same_position = sum(
            1 for p in team_players if p.get("element_type") == player_position
        )
        result["squad_depth_at_position"] = float(same_position)
    else:
        result["team_total_points_avg"] = 0.0
        result["player_share_of_team_points"] = 0.0
        result["player_share_of_team_goals"] = 0.0
        result["team_avg_ict"] = 0.0
        result["team_players_available"] = 0.0
        result["squad_depth_at_position"] = 0.0

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


def _is_win(fixture: dict[str, Any], team_id: int | None) -> bool:
    """Check if team won the fixture."""
    if team_id is None:
        return False
    goals_for = _get_goals_for(fixture, team_id)
    goals_against = _get_goals_against(fixture, team_id)
    return goals_for > goals_against


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
