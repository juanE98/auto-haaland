"""
Unit tests for the feature configuration module.
"""

import math
import os
import re
from pathlib import Path

import pytest

from lambdas.common.feature_categories.fixture_features import FIXTURE_FEATURES
from lambdas.common.feature_categories.interaction_features import INTERACTION_FEATURES
from lambdas.common.feature_categories.opponent_features import OPPONENT_FEATURES
from lambdas.common.feature_categories.position_features import POSITION_FEATURES
from lambdas.common.feature_categories.team_features import TEAM_FEATURES
from lambdas.common.feature_config import (
    BOOTSTRAP_FEATURES,
    DERIVED_FEATURES,
    FEATURE_COLS,
    FEATURE_VERSION,
    ROLLING_FEATURE_NAMES,
    ROLLING_STATS,
    STATIC_FEATURES,
    TARGET_COL,
    calculate_minutes_pct,
    calculate_rolling_average,
    calculate_rolling_stddev,
    compute_bootstrap_features,
    compute_derived_features,
    compute_rolling_features,
    extract_values,
)

# === Feature List Validation ===


@pytest.mark.unit
class TestFeatureList:
    def test_feature_cols_has_200_entries(self):
        """FEATURE_COLS should contain exactly 200 features (Phase 4)."""
        assert len(FEATURE_COLS) == 200

    def test_no_duplicate_feature_names(self):
        """All feature names should be unique."""
        assert len(FEATURE_COLS) == len(set(FEATURE_COLS))

    def test_rolling_feature_names_count(self):
        """Rolling features: 73 total (36 original + 37 new in Phase 1)."""
        assert len(ROLLING_FEATURE_NAMES) == 73

    def test_static_features_count(self):
        """Static features should have 9 entries."""
        assert len(STATIC_FEATURES) == 9

    def test_bootstrap_features_count(self):
        """Bootstrap features should have 32 entries (Phase 2)."""
        assert len(BOOTSTRAP_FEATURES) == 32

    def test_team_features_count(self):
        """Team features should have 28 entries (Phase 3)."""
        assert len(TEAM_FEATURES) == 28

    def test_opponent_features_count(self):
        """Opponent features should have 24 entries (Phase 3)."""
        assert len(OPPONENT_FEATURES) == 24

    def test_fixture_features_count(self):
        """Fixture features should have 16 entries (Phase 4)."""
        assert len(FIXTURE_FEATURES) == 16

    def test_position_features_count(self):
        """Position features should have 8 entries (Phase 4)."""
        assert len(POSITION_FEATURES) == 8

    def test_interaction_features_count(self):
        """Interaction features should have 5 entries (Phase 4)."""
        assert len(INTERACTION_FEATURES) == 5

    def test_derived_features_count(self):
        """Derived features should have 5 entries."""
        assert len(DERIVED_FEATURES) == 5

    def test_feature_cols_composition(self):
        """FEATURE_COLS = all feature categories combined."""
        expected = (
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
        assert FEATURE_COLS == expected

    def test_target_col(self):
        """Target column should be 'actual_points'."""
        assert TARGET_COL == "actual_points"

    def test_rolling_names_format(self):
        """All rolling feature names should match '<stat>_last_<window>'."""
        for name in ROLLING_FEATURE_NAMES:
            parts = name.rsplit("_last_", 1)
            assert len(parts) == 2, f"Invalid rolling feature name: {name}"
            assert parts[1].isdigit(), f"Window is not a digit: {name}"

    def test_feature_version_format(self):
        """Feature version should be a valid semantic version string."""
        parts = FEATURE_VERSION.split(".")
        assert len(parts) == 3, "Version should have 3 parts (major.minor.patch)"
        for part in parts:
            assert part.isdigit(), "Version parts should be numeric"

    def test_key_features_present(self):
        """Verify important features are in the list."""
        expected = [
            # Original core features
            "points_last_1",
            "points_last_3",
            "points_last_5",
            "goals_last_3",
            "assists_last_3",
            "influence_last_3",
            "bonus_last_3",
            "yellow_cards_last_3",
            "saves_last_5",
            "transfers_balance_last_3",
            "form_score",
            "now_cost",
            "minutes_pct",
            "points_per_90",
            "points_volatility",
            # New Phase 1 features
            "expected_goals_last_1",
            "expected_goals_last_3",
            "expected_goals_last_5",
            "expected_assists_last_1",
            "expected_assists_last_3",
            "expected_assists_last_5",
            "minutes_last_1",
            "minutes_last_3",
            "minutes_last_5",
            "starts_last_1",
            "starts_last_3",
            "starts_last_5",
            "red_cards_last_3",
            "red_cards_last_5",
            "red_cards_last_10",
            "own_goals_last_3",
            "penalties_saved_last_5",
            "penalties_missed_last_10",
            # Extended window features
            "points_last_10",
            "goals_last_10",
            "assists_last_10",
            "ict_index_last_10",
            "yellow_cards_last_10",
            "saves_last_10",
            "transfers_balance_last_10",
        ]
        for feat in expected:
            assert feat in FEATURE_COLS, f"Missing feature: {feat}"

    def test_new_xg_xa_features_present(self):
        """Verify xG and xA features are present with all windows."""
        for stat in ["expected_goals", "expected_assists"]:
            for window in [1, 3, 5]:
                feat = f"{stat}_last_{window}"
                assert feat in ROLLING_FEATURE_NAMES, f"Missing: {feat}"

    def test_new_rare_event_features_with_extended_windows(self):
        """Verify rare event stats have windows (3, 5, 10)."""
        for stat in ["red_cards", "own_goals", "penalties_saved", "penalties_missed"]:
            for window in [3, 5, 10]:
                feat = f"{stat}_last_{window}"
                assert feat in ROLLING_FEATURE_NAMES, f"Missing: {feat}"

    def test_extended_window_10_for_key_stats(self):
        """Verify extended window (10) exists for key existing stats."""
        key_stats = [
            "points",
            "goals",
            "assists",
            "clean_sheets",
            "bps",
            "ict_index",
            "threat",
            "creativity",
            "influence",
            "bonus",
            "yellow_cards",
            "saves",
            "transfers_balance",
        ]
        for stat in key_stats:
            feat = f"{stat}_last_10"
            assert feat in ROLLING_FEATURE_NAMES, f"Missing: {feat}"

    def test_bootstrap_features_present(self):
        """Verify all bootstrap feature categories are present."""
        # FPL Expected Points & Value
        assert "ep_this" in BOOTSTRAP_FEATURES
        assert "ep_next" in BOOTSTRAP_FEATURES
        assert "points_per_game" in BOOTSTRAP_FEATURES
        assert "value_form" in BOOTSTRAP_FEATURES

        # Availability & Status
        assert "status_available" in BOOTSTRAP_FEATURES
        assert "status_injured" in BOOTSTRAP_FEATURES
        assert "has_news" in BOOTSTRAP_FEATURES

        # Dream Team & Recognition
        assert "dreamteam_count" in BOOTSTRAP_FEATURES
        assert "bonus_rate" in BOOTSTRAP_FEATURES

        # Transfer Momentum
        assert "transfers_in_event" in BOOTSTRAP_FEATURES
        assert "net_transfers_event" in BOOTSTRAP_FEATURES
        assert "transfer_momentum" in BOOTSTRAP_FEATURES

        # Set Piece Responsibility
        assert "penalties_order" in BOOTSTRAP_FEATURES
        assert "set_piece_taker" in BOOTSTRAP_FEATURES

        # Season Totals Normalised
        assert "total_points_rank_pct" in BOOTSTRAP_FEATURES
        assert "goals_per_90_season" in BOOTSTRAP_FEATURES


# === Rolling Average ===


@pytest.mark.unit
class TestCalculateRollingAverage:
    def test_exact_window(self):
        assert calculate_rolling_average([2.0, 4.0, 6.0], 3) == pytest.approx(4.0)

    def test_window_larger_than_data(self):
        assert calculate_rolling_average([3.0, 5.0], 5) == pytest.approx(4.0)

    def test_window_smaller_than_data(self):
        result = calculate_rolling_average([1.0, 2.0, 3.0, 10.0, 20.0], 3)
        assert result == pytest.approx(11.0)

    def test_empty_list(self):
        assert calculate_rolling_average([], 3) == 0.0

    def test_single_value(self):
        assert calculate_rolling_average([7.0], 3) == pytest.approx(7.0)


# === Rolling Stddev ===


@pytest.mark.unit
class TestCalculateRollingStddev:
    def test_uniform_values(self):
        """All same values should give stddev of 0."""
        assert calculate_rolling_stddev([5.0, 5.0, 5.0], 3) == pytest.approx(0.0)

    def test_known_stddev(self):
        """Known distribution: [2, 4, 6] has mean 4, variance 8/3."""
        result = calculate_rolling_stddev([2.0, 4.0, 6.0], 3)
        expected = math.sqrt(8.0 / 3.0)
        assert result == pytest.approx(expected, rel=0.01)

    def test_fewer_than_two_values(self):
        assert calculate_rolling_stddev([5.0], 3) == 0.0

    def test_empty_list(self):
        assert calculate_rolling_stddev([], 3) == 0.0

    def test_window_limits_data(self):
        """Only the last `window` values should be considered."""
        result = calculate_rolling_stddev([0.0, 0.0, 0.0, 2.0, 4.0, 6.0], 3)
        expected = math.sqrt(8.0 / 3.0)
        assert result == pytest.approx(expected, rel=0.01)


# === Minutes Percentage ===


@pytest.mark.unit
class TestCalculateMinutesPct:
    def test_full_minutes(self):
        history = [{"minutes": 90}, {"minutes": 90}]
        assert calculate_minutes_pct(history, 5) == pytest.approx(1.0)

    def test_empty(self):
        assert calculate_minutes_pct([], 5) == 0.0

    def test_partial(self):
        history = [{"minutes": 45}, {"minutes": 90}]
        assert calculate_minutes_pct(history, 5) == pytest.approx(135 / 180)


# === Extract Values ===


@pytest.mark.unit
class TestExtractValues:
    def test_basic_extraction(self):
        history = [{"total_points": 5}, {"total_points": 8}]
        assert extract_values(history, "total_points") == [5, 8]

    def test_coerce_to_float(self):
        history = [{"ict_index": "85.3"}, {"ict_index": "92.1"}]
        result = extract_values(history, "ict_index", coerce=float)
        assert result == [85.3, 92.1]

    def test_missing_field_defaults_to_zero(self):
        history = [{"other": 5}, {"other": 8}]
        assert extract_values(history, "total_points") == [0, 0]

    def test_none_value_with_coerce(self):
        history = [{"ict_index": None}]
        result = extract_values(history, "ict_index", coerce=float)
        assert result == [0.0]


# === Compute Rolling Features ===


@pytest.mark.unit
class TestComputeRollingFeatures:
    @pytest.fixture
    def sample_history(self):
        return [
            {
                "total_points": 6,
                "goals_scored": 1,
                "assists": 0,
                "clean_sheets": 0,
                "bps": 22,
                "ict_index": "85.0",
                "threat": "45.0",
                "creativity": "55.0",
                "influence": "30.0",
                "bonus": 2,
                "yellow_cards": 0,
                "saves": 0,
                "transfers_in": 5000,
                "transfers_out": 2000,
                "minutes": 90,
                # New Phase 1 fields
                "expected_goals": "0.65",
                "expected_assists": "0.20",
                "starts": 1,
                "red_cards": 0,
                "own_goals": 0,
                "penalties_saved": 0,
                "penalties_missed": 0,
            },
            {
                "total_points": 4,
                "goals_scored": 0,
                "assists": 1,
                "clean_sheets": 0,
                "bps": 18,
                "ict_index": "72.0",
                "threat": "38.0",
                "creativity": "48.0",
                "influence": "25.0",
                "bonus": 0,
                "yellow_cards": 1,
                "saves": 0,
                "transfers_in": 3000,
                "transfers_out": 4000,
                "minutes": 80,
                # New Phase 1 fields
                "expected_goals": "0.35",
                "expected_assists": "0.55",
                "starts": 1,
                "red_cards": 0,
                "own_goals": 0,
                "penalties_saved": 0,
                "penalties_missed": 0,
            },
            {
                "total_points": 10,
                "goals_scored": 1,
                "assists": 1,
                "clean_sheets": 1,
                "bps": 35,
                "ict_index": "92.0",
                "threat": "60.0",
                "creativity": "30.0",
                "influence": "40.0",
                "bonus": 3,
                "yellow_cards": 0,
                "saves": 0,
                "transfers_in": 8000,
                "transfers_out": 1000,
                "minutes": 90,
                # New Phase 1 fields
                "expected_goals": "0.90",
                "expected_assists": "0.45",
                "starts": 1,
                "red_cards": 0,
                "own_goals": 0,
                "penalties_saved": 0,
                "penalties_missed": 1,
            },
        ]

    def test_returns_73_features(self, sample_history):
        result = compute_rolling_features(sample_history)
        assert len(result) == 73

    def test_all_feature_names_present(self, sample_history):
        result = compute_rolling_features(sample_history)
        for name in ROLLING_FEATURE_NAMES:
            assert name in result, f"Missing: {name}"

    def test_points_last_3(self, sample_history):
        result = compute_rolling_features(sample_history)
        # avg(6, 4, 10) = 6.67
        assert result["points_last_3"] == pytest.approx(6.67, abs=0.01)

    def test_points_last_1(self, sample_history):
        result = compute_rolling_features(sample_history)
        # Last game only: 10
        assert result["points_last_1"] == pytest.approx(10.0)

    def test_transfers_balance(self, sample_history):
        result = compute_rolling_features(sample_history)
        # Balance: (5000-2000)=3000, (3000-4000)=-1000, (8000-1000)=7000
        # last_3 avg: (3000 + -1000 + 7000) / 3 = 3000.0
        assert result["transfers_balance_last_3"] == pytest.approx(3000.0, abs=0.01)

    def test_expected_goals_last_3(self, sample_history):
        result = compute_rolling_features(sample_history)
        # avg(0.65, 0.35, 0.90) = 0.633...
        assert result["expected_goals_last_3"] == pytest.approx(0.63, abs=0.01)

    def test_expected_assists_last_1(self, sample_history):
        result = compute_rolling_features(sample_history)
        # Last game only: 0.45
        assert result["expected_assists_last_1"] == pytest.approx(0.45)

    def test_minutes_last_3(self, sample_history):
        result = compute_rolling_features(sample_history)
        # avg(90, 80, 90) = 86.67
        assert result["minutes_last_3"] == pytest.approx(86.67, abs=0.01)

    def test_starts_last_3(self, sample_history):
        result = compute_rolling_features(sample_history)
        # avg(1, 1, 1) = 1.0
        assert result["starts_last_3"] == pytest.approx(1.0)

    def test_penalties_missed_last_3(self, sample_history):
        result = compute_rolling_features(sample_history)
        # avg(0, 0, 1) = 0.33
        assert result["penalties_missed_last_3"] == pytest.approx(0.33, abs=0.01)

    def test_extended_window_10_with_short_history(self, sample_history):
        """Window 10 should work with only 3 games of history."""
        result = compute_rolling_features(sample_history)
        # With 3 games, window 10 uses all available data
        assert result["points_last_10"] == pytest.approx(6.67, abs=0.01)
        assert result["goals_last_10"] == pytest.approx(0.67, abs=0.01)

    def test_empty_history(self):
        result = compute_rolling_features([])
        assert all(v == 0.0 for v in result.values())


# === Compute Derived Features ===


@pytest.mark.unit
class TestComputeDerivedFeatures:
    def test_all_derived_keys(self):
        history = [
            {"total_points": 6, "minutes": 90},
            {"total_points": 8, "minutes": 90},
            {"total_points": 4, "minutes": 90},
        ]
        rolling = {"goals_last_3": 0.67, "assists_last_3": 0.33}
        static = {"form_score": 6.0, "opponent_strength": 4}

        result = compute_derived_features(history, rolling, static)

        expected_keys = {
            "minutes_pct",
            "form_x_difficulty",
            "points_per_90",
            "goal_contributions_last_3",
            "points_volatility",
        }
        assert set(result.keys()) == expected_keys

    def test_form_x_difficulty(self):
        result = compute_derived_features(
            [],
            {"goals_last_3": 0, "assists_last_3": 0},
            {"form_score": 8.5, "opponent_strength": 4},
        )
        assert result["form_x_difficulty"] == pytest.approx(34.0)

    def test_goal_contributions(self):
        result = compute_derived_features(
            [],
            {"goals_last_3": 0.67, "assists_last_3": 0.33},
            {"form_score": 0, "opponent_strength": 3},
        )
        assert result["goal_contributions_last_3"] == pytest.approx(1.0)

    def test_minutes_pct(self):
        history = [{"minutes": 90, "total_points": 5}]
        result = compute_derived_features(
            history,
            {"goals_last_3": 0, "assists_last_3": 0},
            {"form_score": 0, "opponent_strength": 3},
        )
        assert result["minutes_pct"] == pytest.approx(1.0)

    def test_points_per_90(self):
        history = [
            {"total_points": 9, "minutes": 90},
            {"total_points": 6, "minutes": 90},
        ]
        result = compute_derived_features(
            history,
            {"goals_last_3": 0, "assists_last_3": 0},
            {"form_score": 0, "opponent_strength": 3},
        )
        # (9+6) / (90+90) * 90 = 15/180*90 = 7.5
        assert result["points_per_90"] == pytest.approx(7.5)

    def test_points_volatility(self):
        history = [
            {"total_points": 2, "minutes": 90},
            {"total_points": 2, "minutes": 90},
            {"total_points": 2, "minutes": 90},
        ]
        result = compute_derived_features(
            history,
            {"goals_last_3": 0, "assists_last_3": 0},
            {"form_score": 0, "opponent_strength": 3},
        )
        assert result["points_volatility"] == pytest.approx(0.0)

    def test_empty_history_defaults(self):
        result = compute_derived_features(
            [],
            {"goals_last_3": 0, "assists_last_3": 0},
            {"form_score": 0, "opponent_strength": 3},
        )
        assert result["minutes_pct"] == 0.0
        assert result["points_per_90"] == 0.0
        assert result["points_volatility"] == 0.0


# === Compute Bootstrap Features ===


@pytest.mark.unit
class TestComputeBootstrapFeatures:
    @pytest.fixture
    def sample_player(self):
        """Sample FPL API player element."""
        return {
            "id": 350,
            "web_name": "Salah",
            "element_type": 3,  # MID
            "ep_this": "8.5",
            "ep_next": "7.2",
            "points_per_game": "6.8",
            "value_form": "1.2",
            "value_season": "1.1",
            "cost_change_start": 5,
            "cost_change_event": 1,
            "cost_change_event_fall": 0,
            "status": "a",
            "news": "",
            "dreamteam_count": 5,
            "in_dreamteam": True,
            "bonus": 25,
            "minutes": 1800,  # 20 games worth
            "transfers_in_event": 50000,
            "transfers_out_event": 20000,
            "selected_by_percent": "45.3",
            "corners_and_indirect_freekicks_order": 1,
            "direct_freekicks_order": 2,
            "penalties_order": 1,
            "total_points": 120,
            "goals_scored": 12,
            "assists": 8,
            "ict_index": "250.5",
        }

    def test_returns_32_features(self, sample_player):
        result = compute_bootstrap_features(sample_player)
        assert len(result) == 32

    def test_all_feature_names_present(self, sample_player):
        result = compute_bootstrap_features(sample_player)
        for name in BOOTSTRAP_FEATURES:
            assert name in result, f"Missing: {name}"

    def test_ep_this_and_ep_next(self, sample_player):
        result = compute_bootstrap_features(sample_player)
        assert result["ep_this"] == pytest.approx(8.5)
        assert result["ep_next"] == pytest.approx(7.2)

    def test_status_flags(self, sample_player):
        result = compute_bootstrap_features(sample_player)
        assert result["status_available"] == 1.0
        assert result["status_injured"] == 0.0
        assert result["status_suspended"] == 0.0
        assert result["status_doubtful"] == 0.0

    def test_injured_status(self, sample_player):
        sample_player["status"] = "i"
        result = compute_bootstrap_features(sample_player)
        assert result["status_available"] == 0.0
        assert result["status_injured"] == 1.0

    def test_news_injury_detection(self, sample_player):
        sample_player["news"] = "Hamstring injury - expected back in 2 weeks"
        result = compute_bootstrap_features(sample_player)
        assert result["has_news"] == 1.0
        assert result["news_injury_flag"] == 1.0

    def test_dreamteam_metrics(self, sample_player):
        result = compute_bootstrap_features(sample_player)
        assert result["dreamteam_count"] == 5.0
        assert result["in_dreamteam"] == 1.0
        # dreamteam_rate = 5 / (1800 // 60) = 5 / 30 = 0.1667
        assert result["dreamteam_rate"] == pytest.approx(0.1667, abs=0.001)

    def test_transfer_momentum(self, sample_player):
        result = compute_bootstrap_features(sample_player)
        assert result["transfers_in_event"] == 50000.0
        assert result["transfers_out_event"] == 20000.0
        assert result["net_transfers_event"] == 30000.0
        # momentum = (50000 - 20000) / (50000 + 20000) = 30000 / 70000 = 0.4286
        assert result["transfer_momentum"] == pytest.approx(0.4286, abs=0.001)

    def test_set_piece_taker(self, sample_player):
        result = compute_bootstrap_features(sample_player)
        assert result["corners_and_indirect_freekicks_order"] == 1.0
        assert result["penalties_order"] == 1.0
        assert result["set_piece_taker"] == 1.0

    def test_not_set_piece_taker(self, sample_player):
        sample_player["corners_and_indirect_freekicks_order"] = 0
        sample_player["direct_freekicks_order"] = 0
        sample_player["penalties_order"] = 5
        result = compute_bootstrap_features(sample_player)
        assert result["set_piece_taker"] == 0.0

    def test_per_90_stats(self, sample_player):
        result = compute_bootstrap_features(sample_player)
        # goals_per_90 = 12 / 1800 * 90 = 0.6
        assert result["goals_per_90_season"] == pytest.approx(0.6, abs=0.01)
        # assists_per_90 = 8 / 1800 * 90 = 0.4
        assert result["assists_per_90_season"] == pytest.approx(0.4, abs=0.01)
        # ict_per_90 = 250.5 / 1800 * 90 = 12.525
        assert result["ict_per_90_season"] == pytest.approx(12.53, abs=0.01)

    def test_zero_minutes_defaults(self, sample_player):
        sample_player["minutes"] = 0
        result = compute_bootstrap_features(sample_player)
        assert result["goals_per_90_season"] == 0.0
        assert result["assists_per_90_season"] == 0.0
        assert result["ict_per_90_season"] == 0.0

    def test_empty_player(self):
        """Test with minimal player data."""
        result = compute_bootstrap_features({})
        assert len(result) == 32
        assert result["ep_this"] == 0.0
        assert result["status_available"] == 1.0  # default status is 'a'

    def test_with_all_players_for_ranking(self, sample_player):
        """Test ranking features when all_players is provided."""
        all_players = [
            {
                "id": 350,
                "element_type": 3,
                "total_points": 120,
                "transfers_in_event": 50000,
            },
            {
                "id": 351,
                "element_type": 3,
                "total_points": 150,
                "transfers_in_event": 80000,
            },
            {
                "id": 352,
                "element_type": 3,
                "total_points": 100,
                "transfers_in_event": 30000,
            },
        ]
        result = compute_bootstrap_features(sample_player, all_players=all_players)

        # Player 350 has 2nd most points among MIDs (150 > 120 > 100)
        # rank = 2, percentile = 100 * (1 - (2-1)/(3-1)) = 100 * 0.5 = 50.0
        assert result["total_points_rank_pct"] == pytest.approx(50.0)

        # Player 350 has 2nd most transfers in (80000 > 50000 > 30000)
        assert result["transfers_in_rank"] == 2.0

    def test_ownership_change_rate(self, sample_player):
        """Test ownership change calculation."""
        result = compute_bootstrap_features(sample_player, prev_ownership=42.0)
        # 45.3 - 42.0 = 3.3
        assert result["ownership_change_rate"] == pytest.approx(3.3, abs=0.01)


# === Lambda Import Validation ===


@pytest.mark.unit
class TestLambdaImports:
    """
    Ensure imports in lambdas/ directory are Lambda-compatible.

    Lambda functions have a different directory structure than local development.
    Imports using 'from lambdas.common...' will fail in Lambda because the
    'lambdas' package doesn't exist in the Lambda environment.
    """

    def test_no_lambdas_prefix_imports(self):
        """
        Verify no Python files in lambdas/ use 'from lambdas.' imports.

        These imports work locally but fail in Lambda deployment because
        the directory structure is different.
        """
        lambdas_dir = Path(__file__).parent.parent.parent / "lambdas"
        assert lambdas_dir.exists(), f"lambdas directory not found at {lambdas_dir}"

        # Pattern to match problematic imports
        bad_import_pattern = re.compile(
            r"^\s*(from\s+lambdas\.|import\s+lambdas\.)", re.MULTILINE
        )

        violations = []

        for py_file in lambdas_dir.rglob("*.py"):
            content = py_file.read_text()
            matches = bad_import_pattern.findall(content)
            if matches:
                rel_path = py_file.relative_to(lambdas_dir)
                violations.append(f"{rel_path}: {matches}")

        assert not violations, (
            "Found imports using 'lambdas.' prefix which will fail in Lambda:\n"
            + "\n".join(violations)
            + "\n\nUse 'from common.' instead of 'from lambdas.common.'"
        )
