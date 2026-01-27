"""
Unit tests for the current season backfill script.
"""

import pytest

from scripts.backfill_current_season import (
    calculate_minutes_pct,
    calculate_rolling_average,
    engineer_backfill_features,
    filter_history_before_gameweek,
    get_finished_gameweeks,
    get_fixture_info,
    get_gameweek_entry,
    get_team_strength_map,
)

# === Data Leakage Prevention ===


@pytest.mark.unit
class TestFilterHistoryBeforeGameweek:
    def test_filters_future_gameweeks(self):
        history = [
            {"round": 1, "total_points": 5},
            {"round": 2, "total_points": 8},
            {"round": 3, "total_points": 3},
            {"round": 4, "total_points": 10},
        ]
        result = filter_history_before_gameweek(history, 3)
        assert len(result) == 2
        assert all(h["round"] < 3 for h in result)

    def test_excludes_target_gameweek(self):
        """The target gameweek itself must not be included."""
        history = [
            {"round": 1, "total_points": 5},
            {"round": 2, "total_points": 8},
        ]
        result = filter_history_before_gameweek(history, 2)
        assert len(result) == 1
        assert result[0]["round"] == 1

    def test_empty_history(self):
        result = filter_history_before_gameweek([], 5)
        assert result == []

    def test_all_future(self):
        history = [
            {"round": 5, "total_points": 3},
            {"round": 6, "total_points": 7},
        ]
        result = filter_history_before_gameweek(history, 5)
        assert result == []


# === Gameweek Entry Lookup ===


@pytest.mark.unit
class TestGetGameweekEntry:
    def test_found(self):
        history = [
            {"round": 1, "total_points": 5},
            {"round": 2, "total_points": 8},
        ]
        result = get_gameweek_entry(history, 2)
        assert result["total_points"] == 8

    def test_not_found(self):
        history = [{"round": 1, "total_points": 5}]
        result = get_gameweek_entry(history, 3)
        assert result is None


# === Finished Gameweeks ===


@pytest.mark.unit
class TestGetFinishedGameweeks:
    def test_basic(self):
        events = [
            {"id": 1, "finished": True},
            {"id": 2, "finished": True},
            {"id": 3, "finished": False},
            {"id": 4, "finished": False},
        ]
        result = get_finished_gameweeks(events)
        assert result == [1, 2]

    def test_no_finished(self):
        events = [{"id": 1, "finished": False}]
        result = get_finished_gameweeks(events)
        assert result == []

    def test_sorted_output(self):
        events = [
            {"id": 3, "finished": True},
            {"id": 1, "finished": True},
            {"id": 2, "finished": True},
        ]
        result = get_finished_gameweeks(events)
        assert result == [1, 2, 3]


# === Team Strength Map ===


@pytest.mark.unit
class TestGetTeamStrengthMap:
    def test_basic(self):
        teams = [
            {"id": 1, "name": "Arsenal", "strength": 4},
            {"id": 2, "name": "Villa", "strength": 3},
        ]
        result = get_team_strength_map(teams)
        assert result == {1: 4, 2: 3}

    def test_missing_strength(self):
        teams = [{"id": 1, "name": "Arsenal"}]
        result = get_team_strength_map(teams)
        assert result[1] == 3


# === Fixture Info ===


@pytest.mark.unit
class TestGetFixtureInfo:
    def test_home_team(self):
        fixtures = [{"team_h": 10, "team_a": 5}]
        strength_map = {5: 4, 10: 3}
        strength, home, opp_attack, opp_defence = get_fixture_info(
            10, fixtures, strength_map
        )
        assert strength == 4
        assert home == 1
        assert opp_attack == 1200  # Default (no attack/defence map)
        assert opp_defence == 1200

    def test_away_team(self):
        fixtures = [{"team_h": 10, "team_a": 5}]
        strength_map = {5: 4, 10: 3}
        strength, home, opp_attack, opp_defence = get_fixture_info(
            5, fixtures, strength_map
        )
        assert strength == 3
        assert home == 0
        assert opp_attack == 1200
        assert opp_defence == 1200

    def test_no_fixture_defaults(self):
        strength, home, opp_attack, opp_defence = get_fixture_info(99, [], {})
        assert strength == 3
        assert home == 0
        assert opp_attack == 1200
        assert opp_defence == 1200

    def test_home_team_with_attack_defence_map(self):
        fixtures = [{"team_h": 10, "team_a": 5}]
        strength_map = {5: 4, 10: 3}
        ad_map = {
            5: {
                "attack_home": 1300,
                "attack_away": 1250,
                "defence_home": 1280,
                "defence_away": 1230,
            },
        }
        strength, home, opp_attack, opp_defence = get_fixture_info(
            10, fixtures, strength_map, ad_map
        )
        assert strength == 4
        assert home == 1
        # Opponent (team 5) is away, so use away variants
        assert opp_attack == 1250
        assert opp_defence == 1230


# === Feature Engineering ===


@pytest.mark.unit
class TestEngineerBackfillFeatures:
    def _make_players(self):
        return [
            {
                "id": 100,
                "web_name": "Salah",
                "team": 10,
                "element_type": 3,
                "selected_by_percent": "45.3",
            },
            {
                "id": 200,
                "web_name": "Haaland",
                "team": 5,
                "element_type": 4,
                "selected_by_percent": "52.1",
            },
        ]

    def _make_histories(self):
        return {
            100: [
                {
                    "round": 1,
                    "total_points": 6,
                    "minutes": 90,
                    "ict_index": "85.3",
                    "threat": "45.0",
                    "creativity": "55.0",
                },
                {
                    "round": 2,
                    "total_points": 8,
                    "minutes": 80,
                    "ict_index": "72.1",
                    "threat": "38.0",
                    "creativity": "48.0",
                },
                {
                    "round": 3,
                    "total_points": 4,
                    "minutes": 90,
                    "ict_index": "92.0",
                    "threat": "60.0",
                    "creativity": "30.0",
                },
                {
                    "round": 4,
                    "total_points": 10,
                    "minutes": 90,
                    "ict_index": "88.0",
                    "threat": "55.0",
                    "creativity": "40.0",
                },
            ],
            200: [
                {
                    "round": 1,
                    "total_points": 12,
                    "minutes": 90,
                    "ict_index": "95.0",
                    "threat": "70.0",
                    "creativity": "25.0",
                },
                {
                    "round": 2,
                    "total_points": 2,
                    "minutes": 60,
                    "ict_index": "40.0",
                    "threat": "20.0",
                    "creativity": "18.0",
                },
                {
                    "round": 3,
                    "total_points": 7,
                    "minutes": 90,
                    "ict_index": "78.0",
                    "threat": "50.0",
                    "creativity": "30.0",
                },
                {
                    "round": 4,
                    "total_points": 5,
                    "minutes": 75,
                    "ict_index": "62.0",
                    "threat": "35.0",
                    "creativity": "28.0",
                },
            ],
        }

    def test_actual_points_included(self):
        """Output should include actual_points as the training target."""
        result = engineer_backfill_features(
            players=self._make_players(),
            all_histories=self._make_histories(),
            fixtures=[{"team_h": 10, "team_a": 5}],
            team_strength_map={5: 4, 10: 3},
            gameweek=4,
        )

        assert "actual_points" in result.columns
        # Salah scored 10 in GW4
        salah_row = result[result["player_id"] == 100].iloc[0]
        assert salah_row["actual_points"] == 10

    def test_data_leakage_prevention(self):
        """Features for GW4 must only use GW1-3 history."""
        result = engineer_backfill_features(
            players=self._make_players(),
            all_histories=self._make_histories(),
            fixtures=[{"team_h": 10, "team_a": 5}],
            team_strength_map={5: 4, 10: 3},
            gameweek=4,
        )

        salah = result[result["player_id"] == 100].iloc[0]
        # points_last_3 from GW1-3: avg(6, 8, 4) = 6.0
        assert salah["points_last_3"] == pytest.approx(6.0, abs=0.01)
        # Should NOT include GW4 score of 10
        assert salah["points_last_3"] != pytest.approx(7.33, abs=0.1)

    def test_skips_players_without_gw_entry(self):
        """Players who did not play in the target GW should be excluded."""
        players = [
            {
                "id": 100,
                "web_name": "Salah",
                "team": 10,
                "element_type": 3,
                "selected_by_percent": "45.3",
            },
            {
                "id": 300,
                "web_name": "Ghost",
                "team": 1,
                "element_type": 2,
                "selected_by_percent": "5.0",
            },
        ]
        histories = {
            100: [
                {"round": 4, "total_points": 5, "minutes": 90},
            ],
            300: [
                {"round": 1, "total_points": 2, "minutes": 45},
            ],
        }
        result = engineer_backfill_features(
            players=players,
            all_histories=histories,
            fixtures=[{"team_h": 10, "team_a": 5}],
            team_strength_map={5: 3, 10: 4},
            gameweek=4,
        )

        # Only Salah has a GW4 entry
        assert len(result) == 1
        assert result.iloc[0]["player_id"] == 100

    def test_output_schema(self):
        """Output should have all expected columns."""
        result = engineer_backfill_features(
            players=self._make_players(),
            all_histories=self._make_histories(),
            fixtures=[{"team_h": 10, "team_a": 5}],
            team_strength_map={5: 4, 10: 3},
            gameweek=4,
        )

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


# === Rolling Calculations ===


@pytest.mark.unit
class TestRollingCalculations:
    def test_rolling_average_basic(self):
        assert calculate_rolling_average([2.0, 4.0, 6.0], 3) == pytest.approx(4.0)

    def test_rolling_average_empty(self):
        assert calculate_rolling_average([], 3) == 0.0

    def test_minutes_pct_full(self):
        history = [
            {"minutes": 90},
            {"minutes": 90},
        ]
        assert calculate_minutes_pct(history, 5) == pytest.approx(1.0)

    def test_minutes_pct_empty(self):
        assert calculate_minutes_pct([], 5) == 0.0
