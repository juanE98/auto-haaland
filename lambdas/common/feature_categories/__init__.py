"""
Feature Categories Module

Modular feature computation functions organised by category.
Each submodule handles a specific feature domain (team, opponent, fixture, etc.)
"""

from .fixture_features import FIXTURE_FEATURES, compute_fixture_features
from .interaction_features import INTERACTION_FEATURES, compute_interaction_features
from .opponent_features import OPPONENT_FEATURES, compute_opponent_features
from .position_features import POSITION_FEATURES, compute_position_features
from .team_features import TEAM_FEATURES, compute_team_features

__all__ = [
    "FIXTURE_FEATURES",
    "INTERACTION_FEATURES",
    "OPPONENT_FEATURES",
    "POSITION_FEATURES",
    "TEAM_FEATURES",
    "compute_fixture_features",
    "compute_interaction_features",
    "compute_opponent_features",
    "compute_position_features",
    "compute_team_features",
]
