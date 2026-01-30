"""
Position-Specific Features Module

Computes features tailored to each position (GK, DEF, MID, FWD).
Total: 8 features
"""

from typing import Any

# Position-specific features (8 total)
POSITION_FEATURES = [
    # GK (2)
    "gk_saves_per_90",  # Saves per 90 minutes
    "gk_penalty_save_rate",  # Penalties saved / penalties faced
    # DEF (2)
    "def_clean_sheet_rate",  # Clean sheets / games started
    "def_goal_involvement",  # (Goals + assists) / games
    # MID (2)
    "mid_goal_involvement_rate",  # (Goals + assists) / 90 minutes
    "mid_creativity_threat_ratio",  # Creativity / (Threat + 1)
    # FWD (2)
    "fwd_shots_per_90",  # Shots (estimated from threat) per 90
    "fwd_conversion_rate",  # Goals / shots (estimated)
]


def compute_position_features(
    player: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    """
    Compute position-specific features for a player.

    These features are only relevant for certain positions. For players
    of other positions, default values (0.0) are returned.

    Args:
        player: Player element dict from FPL API
        history: Player's recent game history

    Returns:
        Dict mapping position feature name to its value
    """
    result: dict[str, float] = {}
    player = player or {}
    history = history or []

    position = player.get("element_type", 0)
    total_minutes = player.get("minutes", 0)
    games_started = _count_starts(history)

    # === GK Features (2) ===
    if position == 1:  # Goalkeeper
        saves = player.get("saves", 0)
        penalties_saved = player.get("penalties_saved", 0)
        # Estimate penalties faced from saves and goals conceded stats
        # FPL doesn't directly provide penalties faced, so estimate
        penalties_faced = penalties_saved + _count_penalty_goals_conceded(history)

        result["gk_saves_per_90"] = (
            round(saves / total_minutes * 90, 2) if total_minutes > 0 else 0.0
        )
        result["gk_penalty_save_rate"] = (
            round(penalties_saved / penalties_faced, 3) if penalties_faced > 0 else 0.0
        )
    else:
        result["gk_saves_per_90"] = 0.0
        result["gk_penalty_save_rate"] = 0.0

    # === DEF Features (2) ===
    if position == 2:  # Defender
        clean_sheets = player.get("clean_sheets", 0)
        goals = player.get("goals_scored", 0)
        assists = player.get("assists", 0)

        result["def_clean_sheet_rate"] = (
            round(clean_sheets / games_started, 3) if games_started > 0 else 0.0
        )
        result["def_goal_involvement"] = (
            round((goals + assists) / games_started, 3) if games_started > 0 else 0.0
        )
    else:
        result["def_clean_sheet_rate"] = 0.0
        result["def_goal_involvement"] = 0.0

    # === MID Features (2) ===
    if position == 3:  # Midfielder
        goals = player.get("goals_scored", 0)
        assists = player.get("assists", 0)
        creativity = float(player.get("creativity", 0) or 0)
        threat = float(player.get("threat", 0) or 0)

        result["mid_goal_involvement_rate"] = (
            round((goals + assists) / total_minutes * 90, 3)
            if total_minutes > 0
            else 0.0
        )
        result["mid_creativity_threat_ratio"] = round(
            creativity / (threat + 1), 3
        )  # +1 to avoid division by zero
    else:
        result["mid_goal_involvement_rate"] = 0.0
        result["mid_creativity_threat_ratio"] = 0.0

    # === FWD Features (2) ===
    if position == 4:  # Forward
        goals = player.get("goals_scored", 0)
        threat = float(player.get("threat", 0) or 0)
        # Estimate shots from threat (threat ~10-20 per shot on average)
        estimated_shots = threat / 15 if threat > 0 else 0

        result["fwd_shots_per_90"] = (
            round(estimated_shots / total_minutes * 90, 2) if total_minutes > 0 else 0.0
        )
        result["fwd_conversion_rate"] = (
            round(goals / estimated_shots, 3) if estimated_shots > 0 else 0.0
        )
    else:
        result["fwd_shots_per_90"] = 0.0
        result["fwd_conversion_rate"] = 0.0

    return result


def _count_starts(history: list[dict[str, Any]]) -> int:
    """Count games where player started (played 60+ minutes or was in starting XI)."""
    count = 0
    for game in history:
        minutes = game.get("minutes", 0)
        # Consider a start if played 60+ minutes
        if minutes >= 60:
            count += 1
    return count


def _count_penalty_goals_conceded(history: list[dict[str, Any]]) -> int:
    """
    Estimate penalty goals conceded from history.

    FPL doesn't directly track this, so return a default estimate.
    """
    # Rough estimate: assume ~5% of goals conceded are penalties
    total_goals_conceded = sum(
        game.get("goals_conceded", 0) for game in history if game.get("minutes", 0) > 0
    )
    return max(1, int(total_goals_conceded * 0.05))  # At least 1 to avoid div/0
