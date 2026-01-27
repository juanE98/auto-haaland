"""
Feature Configuration Module

Single source of truth for ML feature definitions and computation functions.
All feature engineering code (feature_processor, backfill, import_historical,
inference, training) imports from here to avoid duplication and drift.
"""

import math
from typing import Any

# Rolling stat definitions: each entry maps a stat name to the history field
# it is derived from, plus which rolling windows to compute.
ROLLING_STATS: list[dict[str, Any]] = [
    {"stat": "points", "field": "total_points", "windows": [1, 3, 5]},
    {"stat": "goals", "field": "goals_scored", "windows": [1, 3, 5]},
    {"stat": "assists", "field": "assists", "windows": [1, 3, 5]},
    {"stat": "clean_sheets", "field": "clean_sheets", "windows": [1, 3, 5]},
    {"stat": "bps", "field": "bps", "windows": [1, 3, 5]},
    {
        "stat": "ict_index",
        "field": "ict_index",
        "windows": [1, 3, 5],
        "coerce": float,
    },
    {"stat": "threat", "field": "threat", "windows": [1, 3, 5], "coerce": float},
    {
        "stat": "creativity",
        "field": "creativity",
        "windows": [1, 3, 5],
        "coerce": float,
    },
    {
        "stat": "influence",
        "field": "influence",
        "windows": [1, 3, 5],
        "coerce": float,
    },
    {"stat": "bonus", "field": "bonus", "windows": [1, 3, 5]},
    {"stat": "yellow_cards", "field": "yellow_cards", "windows": [3, 5]},
    {"stat": "saves", "field": "saves", "windows": [3, 5]},
    {
        "stat": "transfers_balance",
        "field": None,
        "windows": [3, 5],
        "derived_from": ("transfers_in", "transfers_out"),
    },
]

STATIC_FEATURES = [
    "form_score",
    "opponent_strength",
    "home_away",
    "chance_of_playing",
    "position",
    "opponent_attack_strength",
    "opponent_defence_strength",
    "selected_by_percent",
    "now_cost",
]

DERIVED_FEATURES = [
    "minutes_pct",
    "form_x_difficulty",
    "points_per_90",
    "goal_contributions_last_3",
    "points_volatility",
]

TARGET_COL = "actual_points"


def _generate_rolling_names() -> list[str]:
    """Generate rolling feature column names from ROLLING_STATS config."""
    names = []
    for entry in ROLLING_STATS:
        for w in entry["windows"]:
            names.append(f"{entry['stat']}_last_{w}")
    return names


ROLLING_FEATURE_NAMES = _generate_rolling_names()

# The canonical list of 50 feature columns used for training and inference.
FEATURE_COLS = ROLLING_FEATURE_NAMES + STATIC_FEATURES + DERIVED_FEATURES


# ---------------------------------------------------------------------------
# Shared computation functions
# ---------------------------------------------------------------------------


def calculate_rolling_average(values: list[float], window: int) -> float:
    """
    Calculate rolling average over the last `window` values.

    Args:
        values: List of values (most recent last)
        window: Number of values to average

    Returns:
        Rolling average, or average of available data if fewer values exist
    """
    if not values:
        return 0.0
    recent = values[-window:] if len(values) >= window else values
    return sum(recent) / len(recent)


def calculate_rolling_stddev(values: list[float], window: int) -> float:
    """
    Calculate rolling standard deviation over the last `window` values.

    Args:
        values: List of values (most recent last)
        window: Number of values to consider

    Returns:
        Population standard deviation of the recent window, or 0.0 if
        insufficient data
    """
    if len(values) < 2:
        return 0.0
    recent = values[-window:] if len(values) >= window else values
    if len(recent) < 2:
        return 0.0
    mean = sum(recent) / len(recent)
    variance = sum((x - mean) ** 2 for x in recent) / len(recent)
    return math.sqrt(variance)


def calculate_minutes_pct(history: list[dict[str, Any]], window: int = 5) -> float:
    """
    Calculate minutes played percentage over recent games.

    Args:
        history: List of gameweek data dicts with 'minutes' key
                 (most recent last)
        window: Number of games to consider

    Returns:
        Average minutes percentage (0-1 scale)
    """
    if not history:
        return 0.0
    recent = history[-window:] if len(history) >= window else history
    total_minutes = sum(h.get("minutes", 0) for h in recent)
    max_minutes = 90 * len(recent)
    return total_minutes / max_minutes if max_minutes > 0 else 0.0


def extract_values(
    history: list[dict[str, Any]],
    field: str,
    coerce: type | None = None,
) -> list[float]:
    """
    Extract a list of numeric values from history dicts.

    Handles string-to-float coercion for ICT fields that the FPL API
    returns as strings.

    Args:
        history: List of gameweek data dicts
        field: Key to extract from each dict
        coerce: Optional type to coerce values through (e.g. float)

    Returns:
        List of numeric values
    """
    values = []
    for h in history:
        raw = h.get(field, 0)
        if coerce is not None:
            values.append(coerce(raw or 0))
        else:
            values.append(raw or 0)
    return values


def compute_rolling_features(history: list[dict[str, Any]]) -> dict[str, float]:
    """
    Compute all 36 rolling features from a player's gameweek history.

    Args:
        history: List of gameweek data dicts (most recent last).
                 Expected keys: total_points, goals_scored, assists,
                 clean_sheets, bps, ict_index, threat, creativity,
                 influence, bonus, yellow_cards, saves, transfers_in,
                 transfers_out.

    Returns:
        Dict mapping feature name (e.g. 'points_last_3') to its value.
    """
    result: dict[str, float] = {}

    for entry in ROLLING_STATS:
        stat = entry["stat"]
        field = entry["field"]
        coerce = entry.get("coerce")
        derived_from = entry.get("derived_from")

        if derived_from:
            # Compute derived stat (e.g. transfers_balance = in - out)
            field_a, field_b = derived_from
            values = [
                (h.get(field_a, 0) or 0) - (h.get(field_b, 0) or 0) for h in history
            ]
        else:
            values = extract_values(history, field, coerce)

        for w in entry["windows"]:
            feature_name = f"{stat}_last_{w}"
            result[feature_name] = round(calculate_rolling_average(values, w), 2)

    return result


def compute_derived_features(
    history: list[dict[str, Any]],
    rolling: dict[str, float],
    static: dict[str, Any],
) -> dict[str, float]:
    """
    Compute the 5 derived features from history, rolling, and static data.

    Args:
        history: Player gameweek history (most recent last)
        rolling: Already-computed rolling features dict
        static: Static feature values dict (must contain 'form_score',
                'opponent_strength')

    Returns:
        Dict with keys: minutes_pct, form_x_difficulty, points_per_90,
        goal_contributions_last_3, points_volatility
    """
    # minutes_pct: avg(minutes / 90) over last 5 games
    minutes_pct = calculate_minutes_pct(history, window=5)

    # form_x_difficulty: form_score * opponent_strength
    form_score = static.get("form_score", 0.0)
    opponent_strength = static.get("opponent_strength", 3)
    form_x_difficulty = form_score * opponent_strength

    # points_per_90: total_points / minutes * 90 over last 5 games
    if history:
        recent = history[-5:] if len(history) >= 5 else history
        total_pts = sum(h.get("total_points", 0) for h in recent)
        total_mins = sum(h.get("minutes", 0) for h in recent)
        points_per_90 = (total_pts / total_mins * 90) if total_mins > 0 else 0.0
    else:
        points_per_90 = 0.0

    # goal_contributions_last_3: goals_last_3 + assists_last_3
    goal_contributions_last_3 = rolling.get("goals_last_3", 0.0) + rolling.get(
        "assists_last_3", 0.0
    )

    # points_volatility: stddev of points over last 5 games
    points_values = extract_values(history, "total_points")
    points_volatility = calculate_rolling_stddev(points_values, window=5)

    return {
        "minutes_pct": round(minutes_pct, 3),
        "form_x_difficulty": round(form_x_difficulty, 2),
        "points_per_90": round(points_per_90, 2),
        "goal_contributions_last_3": round(goal_contributions_last_3, 2),
        "points_volatility": round(points_volatility, 2),
    }
