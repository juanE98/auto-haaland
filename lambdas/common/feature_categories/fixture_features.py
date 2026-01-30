"""
Fixture Context Features Module

Computes features related to fixture difficulty, double/blank gameweeks, and timing.
Total: 16 features
"""

from datetime import datetime, timedelta
from typing import Any

# Fixture context features (16 total)
FIXTURE_FEATURES = [
    # Difficulty (6)
    "fdr_current",  # FPL's fixture difficulty rating for current fixture
    "fdr_next_3_avg",  # Average FDR of next 3 fixtures
    "fdr_next_5_avg",  # Average FDR of next 5 fixtures
    "fixture_swing",  # Change in FDR (upcoming vs previous)
    "is_tough_fixture",  # FDR >= 4
    "is_easy_fixture",  # FDR <= 2
    # DGW/BGW (4)
    "is_double_gameweek",  # Player has 2+ fixtures this GW
    "is_blank_gameweek",  # Player has 0 fixtures this GW
    "dgw_fixture_count",  # Number of fixtures in DGW (0, 1, 2+)
    "next_is_dgw",  # Next gameweek is a DGW for this player
    # Timing (6)
    "days_since_last_game",  # Days rest for the player
    "fixture_congestion_7d",  # Games in last 7 days
    "fixture_congestion_14d",  # Games in last 14 days
    "kickoff_hour",  # Hour of kickoff (0-23)
    "is_weekend_game",  # Saturday or Sunday
    "is_evening_kickoff",  # Kickoff after 17:00
]


def compute_fixture_features(
    player: dict[str, Any] | None = None,
    current_fixture: dict[str, Any] | None = None,
    upcoming_fixtures: list[dict[str, Any]] | None = None,
    past_fixtures: list[dict[str, Any]] | None = None,
    current_gw_fixtures: list[dict[str, Any]] | None = None,
    next_gw_fixtures: list[dict[str, Any]] | None = None,
    current_date: str | None = None,
) -> dict[str, float]:
    """
    Compute fixture context features.

    Args:
        player: Player element dict from FPL API
        current_fixture: The fixture being predicted
        upcoming_fixtures: Future fixtures for difficulty calculation
        past_fixtures: Recent fixtures for congestion calculation
        current_gw_fixtures: All fixtures for the player this gameweek
        next_gw_fixtures: All fixtures for the player next gameweek
        current_date: Current date for timing calculations (ISO format)

    Returns:
        Dict mapping fixture feature name to its value
    """
    result: dict[str, float] = {}
    player = player or {}
    current_fixture = current_fixture or {}
    upcoming_fixtures = upcoming_fixtures or []
    past_fixtures = past_fixtures or []
    current_gw_fixtures = current_gw_fixtures or []
    next_gw_fixtures = next_gw_fixtures or []

    team_id = player.get("team")

    # === Difficulty (6) ===
    # Current fixture difficulty
    fdr = _get_fixture_difficulty(current_fixture, team_id)
    result["fdr_current"] = float(fdr)

    # Average FDR of upcoming fixtures
    upcoming_fdrs = [_get_fixture_difficulty(f, team_id) for f in upcoming_fixtures]
    result["fdr_next_3_avg"] = _avg_or_default(upcoming_fdrs[:3], 3.0)
    result["fdr_next_5_avg"] = _avg_or_default(upcoming_fdrs[:5], 3.0)

    # Fixture swing (upcoming vs past)
    past_fdrs = [_get_fixture_difficulty(f, team_id) for f in past_fixtures[-3:]]
    past_avg = _avg_or_default(past_fdrs, 3.0)
    next_avg = result["fdr_next_3_avg"]
    result["fixture_swing"] = round(next_avg - past_avg, 2)

    # Difficulty flags
    result["is_tough_fixture"] = 1.0 if fdr >= 4 else 0.0
    result["is_easy_fixture"] = 1.0 if fdr <= 2 else 0.0

    # === DGW/BGW (4) ===
    fixture_count = len(current_gw_fixtures)
    result["is_double_gameweek"] = 1.0 if fixture_count >= 2 else 0.0
    result["is_blank_gameweek"] = 1.0 if fixture_count == 0 else 0.0
    result["dgw_fixture_count"] = float(fixture_count)

    next_fixture_count = len(next_gw_fixtures)
    result["next_is_dgw"] = 1.0 if next_fixture_count >= 2 else 0.0

    # === Timing (6) ===
    # Days since last game
    if past_fixtures and current_date:
        result["days_since_last_game"] = _calculate_days_since(
            past_fixtures, current_date
        )
    else:
        result["days_since_last_game"] = 7.0  # Default

    # Fixture congestion
    if past_fixtures and current_date:
        result["fixture_congestion_7d"] = _count_recent_games(
            past_fixtures, current_date, days=7
        )
        result["fixture_congestion_14d"] = _count_recent_games(
            past_fixtures, current_date, days=14
        )
    else:
        result["fixture_congestion_7d"] = 1.0
        result["fixture_congestion_14d"] = 2.0

    # Kickoff timing
    kickoff_time = current_fixture.get("kickoff_time")
    if kickoff_time:
        kickoff_hour, is_weekend = _parse_kickoff_time(kickoff_time)
        result["kickoff_hour"] = float(kickoff_hour)
        result["is_weekend_game"] = 1.0 if is_weekend else 0.0
        result["is_evening_kickoff"] = 1.0 if kickoff_hour >= 17 else 0.0
    else:
        result["kickoff_hour"] = 15.0  # Default 3pm
        result["is_weekend_game"] = 1.0  # Default Saturday
        result["is_evening_kickoff"] = 0.0

    return result


def _get_fixture_difficulty(fixture: dict[str, Any], team_id: int | None) -> int:
    """Get fixture difficulty rating for a team."""
    if team_id is None or not fixture:
        return 3  # Default medium difficulty

    if fixture.get("team_h") == team_id:
        # Home team - use away team's difficulty for home games
        return fixture.get("team_h_difficulty", 3)
    elif fixture.get("team_a") == team_id:
        # Away team - use home team's difficulty for away games
        return fixture.get("team_a_difficulty", 3)
    return 3


def _avg_or_default(values: list[int | float], default: float) -> float:
    """Return average or default if empty."""
    if not values:
        return default
    return round(sum(values) / len(values), 2)


def _calculate_days_since(fixtures: list[dict[str, Any]], current_date: str) -> float:
    """Calculate days since last fixture."""
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
    return 1.0 if days == 7 else 2.0  # Default


def _parse_kickoff_time(kickoff_time: str) -> tuple[int, bool]:
    """Parse kickoff time to extract hour and weekend flag."""
    try:
        dt = datetime.fromisoformat(kickoff_time.replace("Z", "+00:00"))
        is_weekend = dt.weekday() >= 5  # Saturday=5, Sunday=6
        return dt.hour, is_weekend
    except (ValueError, TypeError):
        return 15, True  # Default 3pm Saturday
