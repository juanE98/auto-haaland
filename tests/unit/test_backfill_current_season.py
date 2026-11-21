"""
Unit tests for the current season backfill script.
"""

from unittest.mock import Mock

import pytest

from lambdas.common.feature_config import FEATURE_COLS
from scripts.backfill_current_season import (
    engineer_backfill_features,
    fetch_all_player_histories,
    filter_history_before_gameweek,
    get_finished_gameweeks,
    get_fixture_info,
    get_gameweek_entry,
    get_team_strength_map,
)


@pytest.mark.unit
def test_fetch_player_histories_retains_previous_season_summary():
    api_client = Mock()
    api_client.get_player_summary.return_value = {
        "history": [{"round": 1, "total_points": 5}],
        "history_past": [{"season_name": "2025/26", "total_points": 180}],
    }

    result = fetch_all_player_histories(api_client, [100], batch_delay=0)

    assert result[100] == api_client.get_player_summary.return_value


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

    def test_double_gameweek_entries_are_summed(self):
        history = [
            {"round": 2, "total_points": 8, "minutes": 90, "value": 75},
            {"round": 2, "total_points": 3, "minutes": 70, "value": 76},
        ]

        result = get_gameweek_entry(history, 2)

        assert result["total_points"] == 11
        assert result["minutes"] == 160
        assert result["value"] == 76
        assert result["fixture_count"] == 2


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

    def test_fixture_difficulty_overrides_team_strength(self):
        fixtures = [
            {
                "team_h": 10,
                "team_a": 5,
                "team_h_difficulty": 5,
                "team_a_difficulty": 2,
            }
        ]

        strength, home, _, _ = get_fixture_info(10, fixtures, {5: 2})

        assert strength == 5
        assert home == 1

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
        histories = {
            100: [
                {
                    "round": 1,
                    "total_points": 6,
                    "minutes": 90,
                    "goals_scored": 1,
                    "assists": 0,
                    "clean_sheets": 0,
                    "bps": 22,
                    "ict_index": "85.3",
                    "threat": "45.0",
                    "creativity": "55.0",
                    "influence": "30.0",
                    "bonus": 2,
                    "yellow_cards": 0,
                    "saves": 0,
                    "transfers_in": 5000,
                    "transfers_out": 2000,
                },
                {
                    "round": 2,
                    "total_points": 8,
                    "minutes": 80,
                    "goals_scored": 0,
                    "assists": 1,
                    "clean_sheets": 0,
                    "bps": 18,
                    "ict_index": "72.1",
                    "threat": "38.0",
                    "creativity": "48.0",
                    "influence": "25.0",
                    "bonus": 0,
                    "yellow_cards": 1,
                    "saves": 0,
                    "transfers_in": 3000,
                    "transfers_out": 4000,
                },
                {
                    "round": 3,
                    "total_points": 4,
                    "minutes": 90,
                    "goals_scored": 0,
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
                },
                {
                    "round": 4,
                    "total_points": 10,
                    "minutes": 90,
                    "goals_scored": 1,
                    "assists": 1,
                    "clean_sheets": 0,
                    "bps": 30,
                    "ict_index": "88.0",
                    "threat": "55.0",
                    "creativity": "40.0",
                    "influence": "35.0",
                    "bonus": 2,
                    "yellow_cards": 0,
                    "saves": 0,
                    "transfers_in": 10000,
                    "transfers_out": 1000,
                },
            ],
            200: [
                {
                    "round": 1,
                    "total_points": 12,
                    "minutes": 90,
                    "goals_scored": 2,
                    "assists": 0,
                    "clean_sheets": 0,
                    "bps": 40,
                    "ict_index": "95.0",
                    "threat": "70.0",
                    "creativity": "25.0",
                    "influence": "45.0",
                    "bonus": 3,
                    "yellow_cards": 0,
                    "saves": 0,
                    "transfers_in": 15000,
                    "transfers_out": 500,
                },
                {
                    "round": 2,
                    "total_points": 2,
                    "minutes": 60,
                    "goals_scored": 0,
                    "assists": 0,
                    "clean_sheets": 0,
                    "bps": 10,
                    "ict_index": "40.0",
                    "threat": "20.0",
                    "creativity": "18.0",
                    "influence": "15.0",
                    "bonus": 0,
                    "yellow_cards": 1,
                    "saves": 0,
                    "transfers_in": 2000,
                    "transfers_out": 8000,
                },
                {
                    "round": 3,
                    "total_points": 7,
                    "minutes": 90,
                    "goals_scored": 1,
                    "assists": 0,
                    "clean_sheets": 0,
                    "bps": 25,
                    "ict_index": "78.0",
                    "threat": "50.0",
                    "creativity": "30.0",
                    "influence": "32.0",
                    "bonus": 1,
                    "yellow_cards": 0,
                    "saves": 0,
                    "transfers_in": 7000,
                    "transfers_out": 3000,
                },
                {
                    "round": 4,
                    "total_points": 5,
                    "minutes": 75,
                    "goals_scored": 0,
                    "assists": 1,
                    "clean_sheets": 0,
                    "bps": 18,
                    "ict_index": "62.0",
                    "threat": "35.0",
                    "creativity": "28.0",
                    "influence": "22.0",
                    "bonus": 0,
                    "yellow_cards": 0,
                    "saves": 0,
                    "transfers_in": 4000,
                    "transfers_out": 5000,
                },
            ],
        }
        for entry in histories[100]:
            entry.update({"opponent_team": 5, "was_home": True})
        for entry in histories[200]:
            entry.update({"opponent_team": 10, "was_home": False})
        return histories

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
        haaland_row = result[result["player_id"] == 200].iloc[0]
        assert haaland_row["prior_fwd_ict_per_90"] == 0.0

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
        # Current bootstrap totals already include GW4 and are not model inputs.
        assert "points_per_game" not in result.columns

    def test_adds_early_season_prior_and_history_count(self):
        histories = self._make_histories()
        summaries = {
            player_id: {
                "history": history,
                "history_past": [
                    {
                        "season_name": "2025/26",
                        "total_points": 200,
                        "minutes": 2250,
                        "goals_scored": 20,
                        "assists": 10,
                        "bonus": 30,
                        "ict_index": "180.0",
                    }
                ],
            }
            for player_id, history in histories.items()
        }

        result = engineer_backfill_features(
            players=self._make_players(),
            all_histories=summaries,
            fixtures=[{"team_h": 10, "team_a": 5}],
            team_strength_map={5: 4, 10: 3},
            gameweek=4,
        )

        salah = result[result["player_id"] == 100].iloc[0]
        assert salah["history_game_count"] == 3
        assert salah["prior_points_per_90"] == pytest.approx(8.0)
        assert salah["prior_fwd_ict_per_90"] == 0.0
        assert salah["prior_season_available"] == 1.0
        haaland = result[result["player_id"] == 200].iloc[0]
        assert haaland["prior_fwd_ict_per_90"] == pytest.approx(7.2)

    def test_backfill_team_priors_match_live_dgw_aggregation(self):
        players = [
            {"id": 10, "web_name": "Target", "team": 1, "element_type": 3},
            {"id": 20, "web_name": "Opponent A", "team": 2, "element_type": 2},
            {"id": 30, "web_name": "Opponent B", "team": 3, "element_type": 2},
            {"id": 40, "web_name": "No Prior", "team": 4, "element_type": 4},
        ]
        prior_rows = {
            10: {"minutes": 1800, "goals_scored": 38, "goals_conceded": 30},
            20: {"minutes": 900, "goals_scored": 10, "goals_conceded": 18},
            30: {"minutes": 1800, "goals_scored": 20, "goals_conceded": 20},
        }
        fixture_sides = {
            10: {"opponent_team": 2, "was_home": True},
            20: {"opponent_team": 1, "was_home": False},
            30: {"opponent_team": 1, "was_home": True},
            40: {"opponent_team": 5, "was_home": True},
        }
        summaries = {
            player["id"]: {
                "history": [
                    {
                        "round": 6,
                        "total_points": 2,
                        "minutes": 90,
                        "value": 50,
                        **fixture_sides[player["id"]],
                    }
                ],
                "history_past": (
                    [prior_rows[player["id"]]] if player["id"] in prior_rows else []
                ),
            }
            for player in players
        }
        fixtures = [
            {"team_h": 1, "team_a": 2},
            {"team_h": 3, "team_a": 1},
            {"team_h": 4, "team_a": 5},
        ]

        result = engineer_backfill_features(
            players, summaries, fixtures, {1: 3, 2: 3, 3: 3, 4: 3}, 6
        )

        target = result[result["player_id"] == 10].iloc[0]
        assert target["prior_team_goals_per_match"] == pytest.approx(1.0)
        assert target["prior_team_squad_minutes"] == 1800
        assert target["prior_team_context_available"] == 1.0
        assert target["prior_opponent_goals_conceded_per_90"] == pytest.approx(1.4)
        assert target["prior_opponent_squad_minutes"] == pytest.approx(1350)
        assert target["prior_opponent_context_available"] == 1.0

        missing = result[result["player_id"] == 40].iloc[0]
        assert (
            missing[
                [
                    "prior_team_goals_per_match",
                    "prior_team_squad_minutes",
                    "prior_team_context_available",
                    "prior_opponent_goals_conceded_per_90",
                    "prior_opponent_squad_minutes",
                    "prior_opponent_context_available",
                ]
            ].tolist()
            == [0.0] * 6
        )

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
                {
                    "round": 4,
                    "opponent_team": 5,
                    "was_home": True,
                    "total_points": 5,
                    "minutes": 90,
                },
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

        # Metadata columns + feature columns + target
        metadata_cols = {
            "player_id",
            "player_name",
            "team_id",
            "gameweek",
            "chance_of_playing",
        }
        expected_cols = metadata_cols | set(FEATURE_COLS) | {"actual_points"}
        assert set(result.columns) == expected_cols

    def test_team_priors_use_target_fixture_roster_not_current_bootstrap(self):
        players = [
            {"id": 10, "web_name": "Arrival", "team": 2, "element_type": 4},
            {"id": 20, "web_name": "Departure", "team": 2, "element_type": 4},
            {"id": 30, "web_name": "Incumbent", "team": 2, "element_type": 3},
        ]
        summaries = {
            10: {
                "history": [{"round": 20, "fixture": 200, "total_points": 2}],
                "history_past": [
                    {"minutes": 1800, "goals_scored": 380, "goals_conceded": 20}
                ],
            },
            20: {
                "history": [
                    {
                        "round": 1,
                        "fixture": 101,
                        "opponent_team": 3,
                        "was_home": True,
                        "total_points": 2,
                        "minutes": 90,
                    }
                ],
                "history_past": [
                    {"minutes": 1800, "goals_scored": 38, "goals_conceded": 20}
                ],
            },
            30: {
                "history": [
                    {
                        "round": 1,
                        "fixture": 102,
                        "opponent_team": 4,
                        "was_home": True,
                        "total_points": 2,
                        "minutes": 90,
                    }
                ],
                "history_past": [
                    {"minutes": 1800, "goals_scored": 10, "goals_conceded": 20}
                ],
            },
        }
        fixtures = [
            {"id": 101, "event": 1, "team_h": 1, "team_a": 3},
            {"id": 102, "event": 1, "team_h": 2, "team_a": 4},
        ]

        result = engineer_backfill_features(
            players, summaries, fixtures, {1: 3, 2: 3, 3: 3, 4: 3}, 1
        ).set_index("player_id")

        assert 10 not in result.index
        assert result.loc[20, "team_id"] == 1
        assert result.loc[20, "prior_team_goals_per_match"] == pytest.approx(1.0)
        assert result.loc[30, "prior_team_goals_per_match"] == pytest.approx(10 / 38)

    def test_conflicting_dgw_team_derivation_skips_player(self, caplog):
        players = [{"id": 10, "web_name": "Mover", "team": 2, "element_type": 3}]
        histories = {
            10: {
                "history": [
                    {
                        "round": 5,
                        "fixture": 101,
                        "opponent_team": 3,
                        "was_home": True,
                        "total_points": 2,
                    },
                    {
                        "round": 5,
                        "fixture": 102,
                        "opponent_team": 4,
                        "was_home": True,
                        "total_points": 3,
                    },
                ],
                "history_past": [],
            }
        }
        fixtures = [
            {"id": 101, "event": 5, "team_h": 1, "team_a": 3},
            {"id": 102, "event": 5, "team_h": 2, "team_a": 4},
        ]

        result = engineer_backfill_features(players, histories, fixtures, {}, 5)

        assert result.empty
        assert "Could not reconstruct GW5 team for player 10" in caplog.text
