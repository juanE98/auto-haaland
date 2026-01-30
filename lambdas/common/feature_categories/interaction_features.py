"""
Interaction Features Module

Computes derived features that combine multiple base features.
Total: 5 features
"""

from typing import Any

# Interaction features (5 total)
INTERACTION_FEATURES = [
    "form_x_fixture_difficulty",  # Form score * inverse FDR
    "ict_x_minutes",  # ICT index * minutes percentage
    "ownership_x_form",  # Ownership * form score
    "value_x_form",  # Value (cost) * form score
    "momentum_score",  # Composite of transfers, form, and recent points
]


def compute_interaction_features(
    player: dict[str, Any] | None = None,
    rolling_features: dict[str, float] | None = None,
    fixture_features: dict[str, float] | None = None,
    bootstrap_features: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Compute interaction features from base features.

    Args:
        player: Player element dict from FPL API
        rolling_features: Already computed rolling features
        fixture_features: Already computed fixture features
        bootstrap_features: Already computed bootstrap features

    Returns:
        Dict mapping interaction feature name to its value
    """
    result: dict[str, float] = {}
    player = player or {}
    rolling_features = rolling_features or {}
    fixture_features = fixture_features or {}
    bootstrap_features = bootstrap_features or {}

    # Form score (from player data, higher is better)
    form = float(player.get("form", 0) or 0)

    # ICT index from rolling features or player
    ict = rolling_features.get(
        "ict_index_last_5", float(player.get("ict_index", 0) or 0)
    )

    # Fixture difficulty (1-5 scale, lower is easier)
    fdr = fixture_features.get("fdr_current", 3.0)

    # Minutes percentage from rolling or estimate
    minutes_pct = (
        rolling_features.get("minutes_last_5", 0) / 90
        if rolling_features.get("minutes_last_5")
        else 0
    )
    if minutes_pct == 0:
        # Fallback: estimate from total minutes
        total_minutes = player.get("minutes", 0)
        games_played = player.get("starts", 0) or 1
        minutes_pct = (
            min(1.0, total_minutes / (games_played * 90)) if games_played > 0 else 0
        )

    # === Interaction Features ===

    # Form x Fixture Difficulty (higher form + easier fixture = better)
    # Invert FDR so easier fixtures (low FDR) give higher values
    inverse_fdr = 6 - fdr  # FDR 1 -> 5, FDR 5 -> 1
    result["form_x_fixture_difficulty"] = round(form * inverse_fdr, 2)

    # ICT x Minutes (high ICT is only valuable if player plays)
    result["ict_x_minutes"] = round(ict * minutes_pct, 2)

    # Ownership x Form (popular + in-form players)
    ownership = float(player.get("selected_by_percent", 0) or 0)
    result["ownership_x_form"] = round(ownership * form, 2)

    # Value x Form (expensive + in-form players = premium picks)
    cost = float(player.get("now_cost", 0) or 0) / 10  # Convert to millions
    result["value_x_form"] = round(cost * form, 2)

    # Momentum Score (composite indicator)
    # Combines: net transfers (popularity trend), form, and recent points
    net_transfers = bootstrap_features.get("net_transfers_event", 0)
    points_last_3 = rolling_features.get("points_last_3", 0)

    # Normalise components to similar scales
    transfer_signal = (
        min(1.0, max(-1.0, net_transfers / 100000)) if net_transfers else 0
    )
    form_signal = min(1.0, form / 10) if form else 0
    points_signal = min(1.0, points_last_3 / 20) if points_last_3 else 0

    momentum = (
        transfer_signal * 0.3  # 30% weight on transfer momentum
        + form_signal * 0.4  # 40% weight on form
        + points_signal * 0.3  # 30% weight on recent points
    )
    result["momentum_score"] = round(momentum * 10, 2)  # Scale to 0-10

    return result
