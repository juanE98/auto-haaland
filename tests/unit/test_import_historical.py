"""
Unit tests for the historical data import script.
"""

import pandas as pd
import pytest

from scripts.import_historical import (
    build_team_strength_map,
    calculate_minutes_pct,
    calculate_rolling_average,
    convert_season_format,
    engineer_historical_features,
)

# === Season Format Conversion ===


@pytest.mark.unit
class TestConvertSeasonFormat:
    def test_standard_conversion(self):
        assert convert_season_format("2023-24") == "2023_24"

    def test_earlier_season(self):
        assert convert_season_format("2021-22") == "2021_22"

    def test_no_dash(self):
        """A season without a dash should pass through unchanged."""
        assert convert_season_format("2023_24") == "2023_24"


# === Rolling Average ===


@pytest.mark.unit
class TestRollingAverage:
    def test_exact_window(self):
        result = calculate_rolling_average([2.0, 4.0, 6.0], 3)
        assert result == pytest.approx(4.0)

    def test_window_larger_than_data(self):
        """When fewer values than window, average all available data."""
        result = calculate_rolling_average([3.0, 5.0], 5)
        assert result == pytest.approx(4.0)

    def test_window_smaller_than_data(self):
        """Only the last `window` values should be used."""
        result = calculate_rolling_average([1.0, 2.0, 3.0, 10.0, 20.0], 3)
        assert result == pytest.approx(11.0)

    def test_empty_list(self):
        assert calculate_rolling_average([], 3) == 0.0

    def test_single_value(self):
        assert calculate_rolling_average([7.0], 3) == pytest.approx(7.0)

    def test_window_of_one(self):
        result = calculate_rolling_average([1.0, 2.0, 5.0], 1)
        assert result == pytest.approx(5.0)


# === Minutes Percentage ===


@pytest.mark.unit
class TestMinutesPct:
    def test_full_minutes(self):
        result = calculate_minutes_pct([90, 90, 90], 3)
        assert result == pytest.approx(1.0)

    def test_no_minutes(self):
        result = calculate_minutes_pct([0, 0, 0], 3)
        assert result == pytest.approx(0.0)

    def test_partial_minutes(self):
        result = calculate_minutes_pct([45, 90, 90], 3)
        assert result == pytest.approx(225 / 270)

    def test_empty_list(self):
        assert calculate_minutes_pct([], 5) == 0.0

    def test_window_larger_than_data(self):
        result = calculate_minutes_pct([90, 45], 5)
        assert result == pytest.approx(135 / 180)

    def test_window_limits_data(self):
        """Only the last `window` entries should be considered."""
        result = calculate_minutes_pct([0, 0, 0, 90, 90], 2)
        assert result == pytest.approx(1.0)


# === Team Strength Map ===


@pytest.mark.unit
class TestBuildTeamStrengthMap:
    def test_basic_mapping(self):
        teams_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Arsenal", "Aston Villa", "Bournemouth"],
                "strength": [4, 3, 2],
            }
        )
        result = build_team_strength_map(teams_df)
        assert result == {1: 4, 2: 3, 3: 2}

    def test_missing_strength_defaults_to_3(self):
        teams_df = pd.DataFrame(
            {
                "id": [1],
                "name": ["Arsenal"],
            }
        )
        result = build_team_strength_map(teams_df)
        assert result[1] == 3

    def test_empty_dataframe(self):
        teams_df = pd.DataFrame(columns=["id", "name", "strength"])
        result = build_team_strength_map(teams_df)
        assert result == {}


# === Feature Engineering ===


@pytest.mark.unit
class TestEngineerHistoricalFeatures:
    def _make_gw_df(self, rows):
        """Helper to create a gameweek DataFrame."""
        return pd.DataFrame(rows)

    def test_basic_features(self):
        """Features should be calculated from player history."""
        gw_df = self._make_gw_df(
            [
                {
                    "element": 100,
                    "name": "Salah",
                    "team": 10,
                    "element_type": 3,
                    "opponent_team": 5,
                    "was_home": True,
                    "total_points": 8,
                    "minutes": 90,
                }
            ]
        )

        player_history = {
            100: [
                {
                    "total_points": 6,
                    "minutes": 90,
                    "round": 1,
                    "goals_scored": 1,
                    "assists": 0,
                    "clean_sheets": 0,
                    "bps": 22,
                    "ict_index": 85.3,
                    "threat": 45.0,
                    "creativity": 55.0,
                    "selected": 100000,
                },
                {
                    "total_points": 4,
                    "minutes": 80,
                    "round": 2,
                    "goals_scored": 0,
                    "assists": 1,
                    "clean_sheets": 0,
                    "bps": 18,
                    "ict_index": 72.1,
                    "threat": 38.0,
                    "creativity": 48.0,
                    "selected": 110000,
                },
                {
                    "total_points": 10,
                    "minutes": 90,
                    "round": 3,
                    "goals_scored": 1,
                    "assists": 1,
                    "clean_sheets": 1,
                    "bps": 35,
                    "ict_index": 92.0,
                    "threat": 60.0,
                    "creativity": 30.0,
                    "selected": 120000,
                },
            ]
        }

        team_strength = {5: 4, 10: 4}

        result = engineer_historical_features(gw_df, 4, player_history, team_strength)

        assert len(result) == 1
        row = result.iloc[0]

        assert row["player_id"] == 100
        assert row["player_name"] == "Salah"
        assert row["gameweek"] == 4
        assert row["actual_points"] == 8
        assert row["home_away"] == 1
        assert row["opponent_strength"] == 4
        assert row["chance_of_playing"] == 100
        # points_last_3 = avg(6, 4, 10) = 6.67
        assert row["points_last_3"] == pytest.approx(6.67, abs=0.01)
        # points_last_5 = avg(6, 4, 10) = 6.67 (only 3 available)
        assert row["points_last_5"] == pytest.approx(6.67, abs=0.01)

    def test_no_history(self):
        """Player with no history should get zero features."""
        gw_df = self._make_gw_df(
            [
                {
                    "element": 200,
                    "name": "NewPlayer",
                    "team": 5,
                    "element_type": 2,
                    "opponent_team": 10,
                    "was_home": False,
                    "total_points": 2,
                    "minutes": 60,
                }
            ]
        )

        result = engineer_historical_features(gw_df, 4, {}, {10: 4})

        row = result.iloc[0]
        assert row["points_last_3"] == 0.0
        assert row["points_last_5"] == 0.0
        assert row["minutes_pct"] == 0.0
        assert row["form_score"] == 0.0
        assert row["home_away"] == 0
        assert row["actual_points"] == 2

    def test_unknown_opponent_defaults_to_3(self):
        """Unknown opponent team should default to strength 3."""
        gw_df = self._make_gw_df(
            [
                {
                    "element": 100,
                    "name": "Player",
                    "team": 1,
                    "element_type": 3,
                    "opponent_team": 99,
                    "was_home": True,
                    "total_points": 5,
                    "minutes": 90,
                }
            ]
        )

        result = engineer_historical_features(gw_df, 4, {}, {})
        assert result.iloc[0]["opponent_strength"] == 3

    def test_double_gameweek_players(self):
        """Players appearing twice in a GW should produce two rows."""
        gw_df = self._make_gw_df(
            [
                {
                    "element": 100,
                    "name": "Player",
                    "team": 1,
                    "element_type": 3,
                    "opponent_team": 5,
                    "was_home": True,
                    "total_points": 8,
                    "minutes": 90,
                },
                {
                    "element": 100,
                    "name": "Player",
                    "team": 1,
                    "element_type": 3,
                    "opponent_team": 10,
                    "was_home": False,
                    "total_points": 3,
                    "minutes": 70,
                },
            ]
        )

        result = engineer_historical_features(gw_df, 5, {}, {5: 4, 10: 3})
        assert len(result) == 2

    def test_output_schema(self):
        """Output should contain all expected columns."""
        gw_df = self._make_gw_df(
            [
                {
                    "element": 100,
                    "name": "Player",
                    "team": 1,
                    "element_type": 3,
                    "opponent_team": 5,
                    "was_home": True,
                    "total_points": 5,
                    "minutes": 90,
                }
            ]
        )

        result = engineer_historical_features(gw_df, 4, {}, {5: 3})

        expected_cols = {
            "player_id",
            "player_name",
            "team_id",
            "position",
            "gameweek",
            "points_last_3",
            "points_last_5",
            "minutes_pct",
            "form_score",
            "opponent_strength",
            "home_away",
            "chance_of_playing",
            "form_x_difficulty",
            "goals_last_3",
            "assists_last_3",
            "clean_sheets_last_3",
            "bps_last_3",
            "ict_index_last_3",
            "threat_last_3",
            "creativity_last_3",
            "opponent_attack_strength",
            "opponent_defence_strength",
            "selected_by_percent",
            "actual_points",
        }
        assert set(result.columns) == expected_cols

    def test_form_x_difficulty_calculation(self):
        """form_x_difficulty should be form_score * opponent_strength."""
        gw_df = self._make_gw_df(
            [
                {
                    "element": 100,
                    "name": "Player",
                    "team": 1,
                    "element_type": 3,
                    "opponent_team": 5,
                    "was_home": True,
                    "total_points": 5,
                    "minutes": 90,
                }
            ]
        )

        player_history = {
            100: [
                {
                    "total_points": 8,
                    "minutes": 90,
                    "round": 1,
                    "goals_scored": 1,
                    "assists": 0,
                    "clean_sheets": 0,
                    "bps": 28,
                    "ict_index": 85.0,
                    "threat": 50.0,
                    "creativity": 40.0,
                    "selected": 90000,
                },
                {
                    "total_points": 6,
                    "minutes": 90,
                    "round": 2,
                    "goals_scored": 0,
                    "assists": 1,
                    "clean_sheets": 0,
                    "bps": 22,
                    "ict_index": 72.0,
                    "threat": 35.0,
                    "creativity": 50.0,
                    "selected": 95000,
                },
                {
                    "total_points": 4,
                    "minutes": 90,
                    "round": 3,
                    "goals_scored": 0,
                    "assists": 0,
                    "clean_sheets": 1,
                    "bps": 18,
                    "ict_index": 60.0,
                    "threat": 25.0,
                    "creativity": 45.0,
                    "selected": 88000,
                },
            ]
        }

        result = engineer_historical_features(gw_df, 4, player_history, {5: 4})
        row = result.iloc[0]

        # form_score = points_last_5 = avg(8,6,4) = 6.0
        # form_x_difficulty = 6.0 * 4 = 24.0
        assert row["form_x_difficulty"] == pytest.approx(24.0, abs=0.01)

    def test_was_home_string_handling(self):
        """was_home as string 'True'/'False' should be handled correctly."""
        gw_df = self._make_gw_df(
            [
                {
                    "element": 100,
                    "name": "Player",
                    "team": 1,
                    "element_type": 3,
                    "opponent_team": 5,
                    "was_home": "True",
                    "total_points": 5,
                    "minutes": 90,
                },
                {
                    "element": 200,
                    "name": "Player2",
                    "team": 2,
                    "element_type": 4,
                    "opponent_team": 5,
                    "was_home": "False",
                    "total_points": 3,
                    "minutes": 60,
                },
            ]
        )

        result = engineer_historical_features(gw_df, 4, {}, {5: 3})
        assert result.iloc[0]["home_away"] == 1
        assert result.iloc[1]["home_away"] == 0
