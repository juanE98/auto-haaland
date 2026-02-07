"""
Feature Configuration Module

Single source of truth for ML feature definitions and computation functions.
All feature engineering code (feature_processor, backfill, import_historical,
inference, training) imports from here to avoid duplication and drift.

IMPORTANT: When updating features, also update sagemaker/train.py FEATURE_COLS
and bump FEATURE_VERSION to ensure consistency across the pipeline.
"""

import math
from typing import Any

from common.feature_categories.fixture_features import FIXTURE_FEATURES
from common.feature_categories.interaction_features import INTERACTION_FEATURES
from common.feature_categories.opponent_features import OPPONENT_FEATURES
from common.feature_categories.position_features import POSITION_FEATURES
from common.feature_categories.team_features import TEAM_FEATURES

# Feature version for tracking compatibility between components
FEATURE_VERSION = "2.4.0"

# Rolling stat definitions: each entry maps a stat name to the history field
# it is derived from, plus which rolling windows to compute.
# Total: 73 rolling features (36 original + 37 new in Phase 1)
ROLLING_STATS: list[dict[str, Any]] = [
    # === Core stats with standard windows (1, 3, 5) ===
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
    # === New xG/xA stats with standard windows (1, 3, 5) ===
    {
        "stat": "expected_goals",
        "field": "expected_goals",
        "windows": [1, 3, 5],
        "coerce": float,
    },
    {
        "stat": "expected_assists",
        "field": "expected_assists",
        "windows": [1, 3, 5],
        "coerce": float,
    },
    # === New minutes/starts stats with standard windows (1, 3, 5) ===
    {"stat": "minutes", "field": "minutes", "windows": [1, 3, 5]},
    {"stat": "starts", "field": "starts", "windows": [1, 3, 5]},
    # === Stats with medium/long windows (3, 5) ===
    {"stat": "yellow_cards", "field": "yellow_cards", "windows": [3, 5]},
    {"stat": "saves", "field": "saves", "windows": [3, 5]},
    {
        "stat": "transfers_balance",
        "field": None,
        "windows": [3, 5],
        "derived_from": ("transfers_in", "transfers_out"),
    },
    # === New rarer event stats with longer windows (3, 5, 10) ===
    {"stat": "red_cards", "field": "red_cards", "windows": [3, 5, 10]},
    {"stat": "own_goals", "field": "own_goals", "windows": [3, 5, 10]},
    {"stat": "penalties_saved", "field": "penalties_saved", "windows": [3, 5, 10]},
    {"stat": "penalties_missed", "field": "penalties_missed", "windows": [3, 5, 10]},
    # === Extended window (10) for key existing stats ===
    {"stat": "points", "field": "total_points", "windows": [10]},
    {"stat": "goals", "field": "goals_scored", "windows": [10]},
    {"stat": "assists", "field": "assists", "windows": [10]},
    {"stat": "clean_sheets", "field": "clean_sheets", "windows": [10]},
    {"stat": "bps", "field": "bps", "windows": [10]},
    {"stat": "ict_index", "field": "ict_index", "windows": [10], "coerce": float},
    {"stat": "threat", "field": "threat", "windows": [10], "coerce": float},
    {"stat": "creativity", "field": "creativity", "windows": [10], "coerce": float},
    {"stat": "influence", "field": "influence", "windows": [10], "coerce": float},
    {"stat": "bonus", "field": "bonus", "windows": [10]},
    {"stat": "yellow_cards", "field": "yellow_cards", "windows": [10]},
    {"stat": "saves", "field": "saves", "windows": [10]},
    {
        "stat": "transfers_balance",
        "field": None,
        "windows": [10],
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

# Bootstrap features extracted directly from FPL API player data (Phase 2)
# Total: 32 new features
BOOTSTRAP_FEATURES = [
    # FPL Expected Points & Value (8)
    "ep_this",  # Expected points this gameweek
    "ep_next",  # Expected points next gameweek
    "points_per_game",  # FPL's PPG calculation
    "value_form",  # Points per million (form)
    "value_season",  # Points per million (season)
    "cost_change_start",  # Price change from season start
    "cost_change_event",  # Price change this gameweek
    "cost_change_event_fall",  # Price drops this gameweek
    # Availability & Status (6)
    "status_available",  # 1 if status is 'a', else 0
    "status_injured",  # 1 if status is 'i', else 0
    "status_suspended",  # 1 if status is 's', else 0
    "status_doubtful",  # 1 if status is 'd', else 0
    "has_news",  # 1 if there is news text, else 0
    "news_injury_flag",  # 1 if news contains injury keywords
    # Dream Team & Recognition (4)
    "dreamteam_count",  # Times in dreamteam this season
    "in_dreamteam",  # 1 if currently in dreamteam
    "dreamteam_rate",  # dreamteam_count / games played
    "bonus_rate",  # Total bonus / games played
    # Transfer Momentum (6)
    "transfers_in_event",  # Transfers in this gameweek
    "transfers_out_event",  # Transfers out this gameweek
    "net_transfers_event",  # transfers_in - transfers_out
    "transfer_momentum",  # Net transfers / total transfers ratio
    "transfers_in_rank",  # Rank among all players for transfers in
    "ownership_change_rate",  # Change in ownership % from last GW
    # Set Piece Responsibility (4)
    "corners_and_indirect_freekicks_order",  # 1-5 rank, 0 if not
    "direct_freekicks_order",  # 1-5 rank, 0 if not
    "penalties_order",  # 1-5 rank, 0 if not
    "set_piece_taker",  # 1 if any set piece order <= 2
    # Season Totals Normalised (4)
    "total_points_rank_pct",  # Rank percentile among position
    "goals_per_90_season",  # Goals per 90 mins this season
    "assists_per_90_season",  # Assists per 90 mins this season
    "ict_per_90_season",  # ICT index per 90 mins this season
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
    """
    Generate rolling feature column names from ROLLING_STATS config.

    Deduplicates names when the same stat appears with different window sets
    (e.g., points with [1,3,5] and points with [10]).

    Returns:
        Deduplicated list of feature names in order of first appearance.
    """
    seen = set()
    names = []
    for entry in ROLLING_STATS:
        for w in entry["windows"]:
            name = f"{entry['stat']}_last_{w}"
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


ROLLING_FEATURE_NAMES = _generate_rolling_names()

# The canonical list of feature columns used for training and inference.
# Total: 73 rolling + 9 static + 32 bootstrap + 35 team + 24 opponent
#        + 16 fixture + 8 position + 5 interaction + 5 derived
#      = 207 features
FEATURE_COLS = (
    ROLLING_FEATURE_NAMES
    + STATIC_FEATURES
    + BOOTSTRAP_FEATURES
    + TEAM_FEATURES
    + OPPONENT_FEATURES
    + FIXTURE_FEATURES
    + POSITION_FEATURES
    + INTERACTION_FEATURES
    + DERIVED_FEATURES
)


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
    Compute all 73 rolling features from a player's gameweek history.

    Args:
        history: List of gameweek data dicts (most recent last).
                 Expected keys: total_points, goals_scored, assists,
                 clean_sheets, bps, ict_index, threat, creativity,
                 influence, bonus, yellow_cards, saves, transfers_in,
                 transfers_out, expected_goals, expected_assists,
                 minutes, starts, red_cards, own_goals, penalties_saved,
                 penalties_missed.

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


def compute_bootstrap_features(
    player: dict[str, Any],
    all_players: list[dict[str, Any]] | None = None,
    prev_ownership: float | None = None,
) -> dict[str, float]:
    """
    Compute the 32 bootstrap features from FPL API player data.

    Args:
        player: Single player element from FPL bootstrap-static API response
        all_players: Optional list of all players for ranking calculations
        prev_ownership: Optional previous gameweek's selected_by_percent

    Returns:
        Dict mapping bootstrap feature name to its value
    """
    result: dict[str, float] = {}

    # --- FPL Expected Points & Value (8) ---
    result["ep_this"] = float(player.get("ep_this") or 0)
    result["ep_next"] = float(player.get("ep_next") or 0)
    result["points_per_game"] = float(player.get("points_per_game") or 0)
    result["value_form"] = float(player.get("value_form") or 0)
    result["value_season"] = float(player.get("value_season") or 0)
    result["cost_change_start"] = float(player.get("cost_change_start") or 0)
    result["cost_change_event"] = float(player.get("cost_change_event") or 0)
    result["cost_change_event_fall"] = float(player.get("cost_change_event_fall") or 0)

    # --- Availability & Status (6) ---
    status = player.get("status", "a")
    result["status_available"] = 1.0 if status == "a" else 0.0
    result["status_injured"] = 1.0 if status == "i" else 0.0
    result["status_suspended"] = 1.0 if status == "s" else 0.0
    result["status_doubtful"] = 1.0 if status == "d" else 0.0

    news = player.get("news", "") or ""
    result["has_news"] = 1.0 if news.strip() else 0.0
    injury_keywords = [
        "injury",
        "injured",
        "knock",
        "muscle",
        "hamstring",
        "ankle",
        "knee",
        "illness",
        "ill",
        "sick",
        "strain",
        "sprain",
    ]
    result["news_injury_flag"] = (
        1.0 if any(kw in news.lower() for kw in injury_keywords) else 0.0
    )

    # --- Dream Team & Recognition (4) ---
    dreamteam_count = int(player.get("dreamteam_count") or 0)
    in_dreamteam = player.get("in_dreamteam", False)
    total_bonus = int(player.get("bonus") or 0)
    minutes_played = int(player.get("minutes") or 0)
    games_played = max(1, minutes_played // 60) if minutes_played > 0 else 1

    result["dreamteam_count"] = float(dreamteam_count)
    result["in_dreamteam"] = 1.0 if in_dreamteam else 0.0
    result["dreamteam_rate"] = round(dreamteam_count / games_played, 4)
    result["bonus_rate"] = round(total_bonus / games_played, 2)

    # --- Transfer Momentum (6) ---
    transfers_in_event = int(player.get("transfers_in_event") or 0)
    transfers_out_event = int(player.get("transfers_out_event") or 0)
    net_transfers = transfers_in_event - transfers_out_event
    total_transfers = transfers_in_event + transfers_out_event

    result["transfers_in_event"] = float(transfers_in_event)
    result["transfers_out_event"] = float(transfers_out_event)
    result["net_transfers_event"] = float(net_transfers)

    # Transfer momentum: ratio of net transfers to total (bounded -1 to 1)
    if total_transfers > 0:
        result["transfer_momentum"] = round(net_transfers / total_transfers, 4)
    else:
        result["transfer_momentum"] = 0.0

    # Transfers in rank (requires all_players)
    if all_players:
        sorted_by_transfers = sorted(
            all_players, key=lambda p: p.get("transfers_in_event", 0), reverse=True
        )
        player_id = player.get("id")
        rank = 1
        for i, p in enumerate(sorted_by_transfers):
            if p.get("id") == player_id:
                rank = i + 1
                break
        result["transfers_in_rank"] = float(rank)
    else:
        result["transfers_in_rank"] = 0.0

    # Ownership change rate
    current_ownership = float(player.get("selected_by_percent") or 0)
    if prev_ownership is not None:
        result["ownership_change_rate"] = round(current_ownership - prev_ownership, 2)
    else:
        result["ownership_change_rate"] = 0.0

    # --- Set Piece Responsibility (4) ---
    corners_order = player.get("corners_and_indirect_freekicks_order") or 0
    freekicks_order = player.get("direct_freekicks_order") or 0
    penalties_order = player.get("penalties_order") or 0

    result["corners_and_indirect_freekicks_order"] = float(corners_order)
    result["direct_freekicks_order"] = float(freekicks_order)
    result["penalties_order"] = float(penalties_order)

    # Set piece taker if any order <= 2 (first or second choice)
    is_set_piece_taker = any(
        0 < order <= 2 for order in [corners_order, freekicks_order, penalties_order]
    )
    result["set_piece_taker"] = 1.0 if is_set_piece_taker else 0.0

    # --- Season Totals Normalised (4) ---
    goals = int(player.get("goals_scored") or 0)
    assists = int(player.get("assists") or 0)
    ict_index = float(player.get("ict_index") or 0)
    element_type = player.get("element_type", 3)

    # Total points rank percentile (requires all_players)
    if all_players:
        same_position = [
            p for p in all_players if p.get("element_type") == element_type
        ]
        sorted_by_points = sorted(
            same_position, key=lambda p: p.get("total_points", 0), reverse=True
        )
        player_id = player.get("id")
        rank = 1
        for i, p in enumerate(sorted_by_points):
            if p.get("id") == player_id:
                rank = i + 1
                break
        # Convert rank to percentile (0-100, higher is better)
        if len(same_position) > 1:
            result["total_points_rank_pct"] = round(
                100 * (1 - (rank - 1) / (len(same_position) - 1)), 1
            )
        else:
            result["total_points_rank_pct"] = 100.0
    else:
        result["total_points_rank_pct"] = 0.0

    # Per-90 stats for the season
    if minutes_played > 0:
        result["goals_per_90_season"] = round(goals / minutes_played * 90, 3)
        result["assists_per_90_season"] = round(assists / minutes_played * 90, 3)
        result["ict_per_90_season"] = round(ict_index / minutes_played * 90, 2)
    else:
        result["goals_per_90_season"] = 0.0
        result["assists_per_90_season"] = 0.0
        result["ict_per_90_season"] = 0.0

    return result
