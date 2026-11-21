"""
Unit tests for the historical data import script.
"""

from unittest.mock import patch

import pandas as pd
import pytest

import scripts.import_historical as import_historical
from lambdas.common.feature_config import FEATURE_COLS
from scripts.import_historical import (
    aggregate_gameweek_rows,
    build_team_strength_map,
    convert_season_format,
    engineer_historical_features,
    process_season,
)


@pytest.mark.unit
def test_historical_source_is_pinned_to_immutable_commit():
    assert import_historical.GITHUB_RAW_BASE == (
        "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
        "9779cdbc0c07f6c900c2d0c181ddf6bb9c800f88/data"
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


@pytest.mark.unit
class TestPriorFeatureMap:
    def test_joins_previous_season_by_stable_player_code(self):
        current = pd.DataFrame(
            [{"id": 10, "code": 1234, "total_points": 999, "minutes": 9999}]
        )
        previous = pd.DataFrame(
            [
                {
                    "id": 99,
                    "code": 1234,
                    "total_points": 180,
                    "minutes": 2250,
                    "goals_scored": 15,
                    "assists": 10,
                    "bonus": 25,
                }
            ]
        )

        result = import_historical.build_prior_feature_map(current, previous)

        assert result[10]["prior_points_per_90"] == pytest.approx(7.2)
        assert result[10]["prior_goals_per_90"] == pytest.approx(0.6)
        assert result[10]["prior_assists_per_90"] == pytest.approx(0.4)
        assert result[10]["prior_bonus_per_90"] == pytest.approx(1.0)
        assert result[10]["prior_season_minutes"] == 2250
        assert result[10]["prior_season_available"] == 1.0

    def test_player_without_previous_season_has_missing_prior(self):
        current = pd.DataFrame([{"id": 10, "code": 1234}])
        previous = pd.DataFrame([{"id": 99, "code": 5678}])

        result = import_historical.build_prior_feature_map(current, previous)

        assert result[10]["prior_season_available"] == 0.0

    def test_prior_team_summary_joins_by_code_without_target_season_totals(self):
        current = pd.DataFrame(
            [
                {
                    "id": 10,
                    "code": 1234,
                    "team": 7,
                    "minutes": 9999,
                    "goals_scored": 999,
                    "goals_conceded": 999,
                }
            ]
        )
        previous = pd.DataFrame(
            [
                {
                    "id": 99,
                    "code": 1234,
                    "team": 2,
                    "minutes": 1800,
                    "goals_scored": 38,
                    "goals_conceded": 30,
                }
            ]
        )

        summaries = import_historical.build_prior_summary_map(current, previous)
        context = import_historical.compute_prior_team_context({10: 7}, summaries)

        assert context[7] == {
            "goals_per_match": 1.0,
            "goals_conceded_per_90": 1.5,
            "squad_minutes": 1800.0,
            "context_available": 1.0,
        }
        assert 2 not in context


@pytest.mark.unit
def test_conflicting_dgw_team_derivation_skips_historical_player(caplog):
    rows = pd.DataFrame(
        [
            {"element": 10, "team": "Old Club"},
            {"element": 10, "team": "New Club"},
        ]
    )

    result = import_historical.get_gameweek_player_team_ids(
        rows, {"Old Club": 1, "New Club": 2}
    )

    assert result == {}
    assert "Conflicting gameweek teams for player 10" in caplog.text


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
                    "influence": 30.0,
                    "bonus": 2,
                    "yellow_cards": 0,
                    "saves": 0,
                    "transfers_in": 5000,
                    "transfers_out": 2000,
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
                    "influence": 25.0,
                    "bonus": 0,
                    "yellow_cards": 1,
                    "saves": 0,
                    "transfers_in": 3000,
                    "transfers_out": 4000,
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
                    "influence": 40.0,
                    "bonus": 3,
                    "yellow_cards": 0,
                    "saves": 0,
                    "transfers_in": 8000,
                    "transfers_out": 1000,
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
        assert row["prior_fwd_ict_per_90"] == 0.0

    def test_adds_history_count_and_previous_season_features(self):
        gw_df = self._make_gw_df(
            [
                {
                    "element": 200,
                    "name": "Player",
                    "team": 5,
                    "element_type": 4,
                    "opponent_team": 10,
                    "total_points": 2,
                }
            ]
        )
        history = {200: [{"round": gw, "total_points": gw} for gw in range(1, 13)]}
        priors = {
            200: {
                "prior_ict_per_90": 9.0,
                "prior_points_per_90": 8.0,
                "prior_goals_per_90": 0.8,
                "prior_assists_per_90": 0.4,
                "prior_bonus_per_90": 1.2,
                "prior_season_minutes": 2250.0,
                "prior_season_available": 1.0,
            }
        }

        result = engineer_historical_features(
            gw_df,
            13,
            history,
            {10: 4},
            prior_features_by_player_id=priors,
        )

        row = result.iloc[0]
        assert row["history_game_count"] == 10
        assert row["prior_points_per_90"] == 8.0
        assert row["prior_fwd_ict_per_90"] == 9.0
        assert row["prior_season_available"] == 1.0

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

    def test_uses_player_fixture_difficulty_when_available(self):
        gw_df = self._make_gw_df(
            [
                {
                    "element": 100,
                    "name": "Player",
                    "team": 1,
                    "element_type": 3,
                    "opponent_team": 5,
                    "difficulty": 5,
                    "total_points": 2,
                }
            ]
        )

        result = engineer_historical_features(gw_df, 4, {}, {5: 2})

        assert result.iloc[0]["opponent_strength"] == 5

    def test_double_gameweek_players(self):
        """A double gameweek should remain one player-week training sample."""
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
        assert len(result) == 1
        assert result.iloc[0]["actual_points"] == 11
        assert result.iloc[0]["dgw_fixture_count"] == 2

    def test_double_gameweek_averages_every_prior_opponent(self):
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
        context = {
            1: {
                "goals_per_match": 1.5,
                "squad_minutes": 3000.0,
                "context_available": 1.0,
            },
            5: {
                "goals_conceded_per_90": 1.0,
                "squad_minutes": 1800.0,
                "context_available": 1.0,
            },
            10: {
                "goals_conceded_per_90": 2.0,
                "squad_minutes": 900.0,
                "context_available": 1.0,
            },
        }

        result = engineer_historical_features(
            gw_df,
            5,
            {},
            {5: 4, 10: 3},
            prior_team_context=context,
        )

        row = result.iloc[0]
        assert row["prior_team_goals_per_match"] == 1.5
        assert row["prior_opponent_goals_conceded_per_90"] == 1.5
        assert row["prior_opponent_squad_minutes"] == 1350

    def test_missing_historical_team_priors_are_finite_zero(self):
        gw_df = self._make_gw_df(
            [
                {
                    "element": 100,
                    "name": "Player",
                    "team": 1,
                    "element_type": 3,
                    "opponent_team": 5,
                    "total_points": 2,
                }
            ]
        )

        result = engineer_historical_features(
            gw_df, 1, {}, {5: 3}, prior_team_context={}
        )

        assert (
            result.iloc[0][
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

    def test_gameweek_aggregation_sums_match_stats_not_snapshots(self):
        gw_df = self._make_gw_df(
            [
                {
                    "element": 100,
                    "total_points": 8,
                    "minutes": 90,
                    "transfers_in": 120,
                    "selected": 5000,
                },
                {
                    "element": 100,
                    "total_points": 3,
                    "minutes": 70,
                    "transfers_in": 140,
                    "selected": 5100,
                },
            ]
        )

        result = aggregate_gameweek_rows(gw_df)

        assert len(result) == 1
        assert result.iloc[0]["total_points"] == 11
        assert result.iloc[0]["minutes"] == 160
        assert result.iloc[0]["transfers_in"] == 140
        assert result.iloc[0]["selected"] == 5100
        assert result.iloc[0]["fixture_count"] == 2

    def test_aggregation_preserves_existing_fixture_count(self):
        """Aggregation is idempotent after double-gameweek rows collapse."""
        gw_df = self._make_gw_df(
            {
                "element": [100, 100],
                "total_points": [5, 6],
                "minutes": [70, 90],
            }
        )

        result = aggregate_gameweek_rows(aggregate_gameweek_rows(gw_df))

        assert result.iloc[0]["fixture_count"] == 2
        assert result.iloc[0]["total_points"] == 11

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
                    "influence": 35.0,
                    "bonus": 2,
                    "yellow_cards": 0,
                    "saves": 0,
                    "transfers_in": 6000,
                    "transfers_out": 2000,
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
                    "influence": 28.0,
                    "bonus": 1,
                    "yellow_cards": 1,
                    "saves": 0,
                    "transfers_in": 4000,
                    "transfers_out": 3000,
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
                    "influence": 22.0,
                    "bonus": 0,
                    "yellow_cards": 0,
                    "saves": 0,
                    "transfers_in": 2000,
                    "transfers_out": 5000,
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


@pytest.mark.unit
class TestProcessSeason:
    @patch("scripts.import_historical.time.sleep")
    @patch("scripts.import_historical.fetch_gameweek_csv")
    @patch("scripts.import_historical.fetch_players_raw_csv")
    @patch("scripts.import_historical.fetch_teams_csv")
    def test_default_import_generates_gw1_with_previous_season_prior(
        self,
        mock_fetch_teams,
        mock_fetch_players,
        mock_fetch_gameweek,
        _mock_sleep,
        tmp_path,
    ):
        mock_fetch_teams.return_value = pd.DataFrame(
            [
                {
                    "id": 1,
                    "name": "Team A",
                    "short_name": "AAA",
                    "strength": 3,
                    "strength_attack_home": 1200,
                    "strength_attack_away": 1200,
                    "strength_defence_home": 1200,
                    "strength_defence_away": 1200,
                },
                {
                    "id": 2,
                    "name": "Team B",
                    "short_name": "BBB",
                    "strength": 3,
                    "strength_attack_home": 1200,
                    "strength_attack_away": 1200,
                    "strength_defence_home": 1200,
                    "strength_defence_away": 1200,
                },
            ]
        )
        mock_fetch_players.side_effect = [
            pd.DataFrame(
                [
                    {
                        "id": 10,
                        "code": 1234,
                        "team": 1,
                        "goals_scored": 999,
                        "goals_conceded": 999,
                    }
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "id": 99,
                        "code": 1234,
                        "team": 2,
                        "total_points": 180,
                        "minutes": 2250,
                        "goals_scored": 15,
                        "goals_conceded": 40,
                        "assists": 10,
                        "bonus": 25,
                    }
                ]
            ),
        ]
        mock_fetch_gameweek.return_value = pd.DataFrame(
            [
                {
                    "element": 10,
                    "name": "Player",
                    "team": "Team A",
                    "position": "FWD",
                    "opponent_team": 2,
                    "was_home": True,
                    "total_points": 6,
                    "minutes": 90,
                    "value": 100,
                }
            ]
        )

        files = process_season("2025-26", str(tmp_path), max_gameweek=1)

        assert len(files) == 1
        row = pd.read_parquet(files[0]).iloc[0]
        assert row["gameweek"] == 1
        assert row["history_game_count"] == 0
        assert row["prior_points_per_90"] == pytest.approx(7.2)
        assert row["prior_team_goals_per_match"] == pytest.approx(15 / 38)
        assert row["prior_team_squad_minutes"] == 2250
        assert row["prior_team_context_available"] == 1.0

    @patch("scripts.import_historical.time.sleep")
    @patch("scripts.import_historical.fetch_gameweek_csv")
    @patch("scripts.import_historical.fetch_players_raw_csv")
    @patch("scripts.import_historical.fetch_teams_csv")
    def test_gw1_team_priors_use_gw1_roster_not_final_roster(
        self,
        mock_fetch_teams,
        mock_fetch_players,
        mock_fetch_gameweek,
        _mock_sleep,
        tmp_path,
    ):
        mock_fetch_teams.return_value = pd.DataFrame(
            [
                {"id": 1, "name": "Old Club", "short_name": "OLD", "strength": 3},
                {
                    "id": 2,
                    "name": "Future Club",
                    "short_name": "NEW",
                    "strength": 3,
                },
                {"id": 3, "name": "Opponent", "short_name": "OPP", "strength": 3},
            ]
        )
        mock_fetch_players.side_effect = [
            pd.DataFrame(
                [
                    {"id": 10, "code": 1010, "team": 2},  # January arrival
                    {"id": 20, "code": 2020, "team": 2},  # January departure
                    {"id": 30, "code": 3030, "team": 2},
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "id": 110,
                        "code": 1010,
                        "minutes": 1800,
                        "goals_scored": 380,
                        "goals_conceded": 20,
                    },
                    {
                        "id": 120,
                        "code": 2020,
                        "minutes": 1800,
                        "goals_scored": 38,
                        "goals_conceded": 20,
                    },
                    {
                        "id": 130,
                        "code": 3030,
                        "minutes": 1800,
                        "goals_scored": 10,
                        "goals_conceded": 20,
                    },
                ]
            ),
        ]
        mock_fetch_gameweek.return_value = pd.DataFrame(
            [
                {
                    "element": 20,
                    "name": "Departure",
                    "team": "Old Club",
                    "position": "FWD",
                    "opponent_team": 3,
                    "was_home": True,
                    "total_points": 2,
                    "minutes": 90,
                    "value": 50,
                },
                {
                    "element": 30,
                    "name": "Incumbent",
                    "team": "Future Club",
                    "position": "FWD",
                    "opponent_team": 3,
                    "was_home": False,
                    "total_points": 2,
                    "minutes": 90,
                    "value": 50,
                },
            ]
        )

        files = process_season("2025-26", str(tmp_path), max_gameweek=1)

        rows = pd.read_parquet(files[0]).set_index("player_id")
        assert rows.loc[20, "team_id"] == 1
        assert rows.loc[20, "prior_team_goals_per_match"] == pytest.approx(1.0)
        assert rows.loc[30, "prior_team_goals_per_match"] == pytest.approx(10 / 38)
