"""
Unit tests for the feature configuration module.
"""

import math

import pytest

from lambdas.common.feature_config import (
    DERIVED_FEATURES,
    FEATURE_COLS,
    ROLLING_FEATURE_NAMES,
    ROLLING_STATS,
    STATIC_FEATURES,
    TARGET_COL,
    calculate_minutes_pct,
    calculate_rolling_average,
    calculate_rolling_stddev,
    compute_derived_features,
    compute_rolling_features,
    extract_values,
)

# === Feature List Validation ===


@pytest.mark.unit
class TestFeatureList:
    def test_feature_cols_has_50_entries(self):
        """FEATURE_COLS should contain exactly 50 features."""
        assert len(FEATURE_COLS) == 50

    def test_no_duplicate_feature_names(self):
        """All feature names should be unique."""
        assert len(FEATURE_COLS) == len(set(FEATURE_COLS))

    def test_rolling_feature_names_count(self):
        """Rolling features: 10 stats x 3 windows + 3 stats x 2 windows = 36."""
        assert len(ROLLING_FEATURE_NAMES) == 36

    def test_static_features_count(self):
        """Static features should have 9 entries."""
        assert len(STATIC_FEATURES) == 9

    def test_derived_features_count(self):
        """Derived features should have 5 entries."""
        assert len(DERIVED_FEATURES) == 5

    def test_feature_cols_composition(self):
        """FEATURE_COLS = rolling + static + derived."""
        expected = ROLLING_FEATURE_NAMES + STATIC_FEATURES + DERIVED_FEATURES
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

    def test_key_features_present(self):
        """Verify important features are in the list."""
        expected = [
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
        ]
        for feat in expected:
            assert feat in FEATURE_COLS, f"Missing feature: {feat}"


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
            },
        ]

    def test_returns_36_features(self, sample_history):
        result = compute_rolling_features(sample_history)
        assert len(result) == 36

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
