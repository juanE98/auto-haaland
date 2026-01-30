"""
Integration tests for feature_processor with S3.

These tests run against LocalStack when AWS_ENDPOINT_URL is set.
Mark these tests with @pytest.mark.integration
"""

import io
import json
from unittest.mock import patch

import pandas as pd
import pytest

from lambdas.common.feature_config import FEATURE_COLS
from lambdas.feature_processor.handler import (
    engineer_features,
    handler,
    load_bootstrap_from_s3,
    load_fixtures_from_s3,
    load_player_histories_from_s3,
    save_features_to_s3,
)


@pytest.mark.integration
class TestFeatureProcessorS3Integration:
    """Integration tests for feature processor S3 operations."""

    def test_load_bootstrap_from_s3(self, localstack_s3_client, clean_s3_bucket):
        """Test loading bootstrap data from S3."""
        s3_client = localstack_s3_client
        bucket = clean_s3_bucket

        # Setup: save bootstrap data to S3
        bootstrap_data = {
            "events": [{"id": 20, "finished": True}],
            "elements": [{"id": 350, "web_name": "Salah", "team": 10}],
            "teams": [{"id": 10, "name": "Liverpool", "strength": 5}],
        }

        s3_client.put_object(
            Bucket=bucket,
            Key="raw/season_2024_25/gw20_bootstrap.json",
            Body=json.dumps(bootstrap_data),
        )

        # Load using our function
        result = load_bootstrap_from_s3(
            s3_client=s3_client, bucket=bucket, gameweek=20, season="2024_25"
        )

        # Verify
        assert result == bootstrap_data
        assert len(result["elements"]) == 1
        assert result["elements"][0]["web_name"] == "Salah"

    def test_load_fixtures_from_s3(self, localstack_s3_client, clean_s3_bucket):
        """Test loading fixtures data from S3."""
        s3_client = localstack_s3_client
        bucket = clean_s3_bucket

        fixtures_data = [{"team_h": 10, "team_a": 1, "event": 20}]

        s3_client.put_object(
            Bucket=bucket,
            Key="raw/season_2024_25/gw20_fixtures.json",
            Body=json.dumps(fixtures_data),
        )

        result = load_fixtures_from_s3(
            s3_client=s3_client, bucket=bucket, gameweek=20, season="2024_25"
        )

        assert result == fixtures_data
        assert len(result) == 1
        assert result[0]["team_h"] == 10

    def test_load_player_histories_missing_gracefully(
        self, localstack_s3_client, clean_s3_bucket
    ):
        """Test loading player histories when directory doesn't exist."""
        s3_client = localstack_s3_client
        bucket = clean_s3_bucket

        # Don't create any player files - should return empty dict
        result = load_player_histories_from_s3(
            s3_client=s3_client, bucket=bucket, gameweek=20, season="2024_25"
        )

        assert result == {}

    def test_load_player_histories_with_data(
        self, localstack_s3_client, clean_s3_bucket
    ):
        """Test loading player histories when files exist."""
        s3_client = localstack_s3_client
        bucket = clean_s3_bucket

        # Setup: save player history files
        salah_history = {
            "history": [
                {"total_points": 8, "minutes": 90, "round": 18},
                {"total_points": 12, "minutes": 90, "round": 19},
            ]
        }
        haaland_history = {
            "history": [{"total_points": 15, "minutes": 90, "round": 18}]
        }

        s3_client.put_object(
            Bucket=bucket,
            Key="raw/season_2024_25/gw20_players/player_350.json",
            Body=json.dumps(salah_history),
        )
        s3_client.put_object(
            Bucket=bucket,
            Key="raw/season_2024_25/gw20_players/player_328.json",
            Body=json.dumps(haaland_history),
        )

        result = load_player_histories_from_s3(
            s3_client=s3_client, bucket=bucket, gameweek=20, season="2024_25"
        )

        assert len(result) == 2
        assert 350 in result
        assert 328 in result
        assert len(result[350]) == 2
        assert result[350][0]["total_points"] == 8

    def test_save_features_parquet_to_s3(self, localstack_s3_client, clean_s3_bucket):
        """Test saving features DataFrame as Parquet to S3."""
        s3_client = localstack_s3_client
        bucket = clean_s3_bucket

        # Create test DataFrame
        df = pd.DataFrame(
            {
                "player_id": [350, 328],
                "player_name": ["Salah", "Haaland"],
                "points_last_3": [8.5, 7.2],
                "form_score": [8.5, 7.2],
            }
        )

        # Save to S3
        key = save_features_to_s3(
            s3_client=s3_client,
            df=df,
            bucket=bucket,
            gameweek=20,
            season="2024_25",
            mode="historical",
        )

        # Verify key format
        assert key == "processed/season_2024_25/gw20_features_training.parquet"

        # Read back and verify
        response = s3_client.get_object(Bucket=bucket, Key=key)

        # Parse Parquet
        buffer = io.BytesIO(response["Body"].read())
        loaded_df = pd.read_parquet(buffer)

        assert len(loaded_df) == 2
        assert list(loaded_df.columns) == [
            "player_id",
            "player_name",
            "points_last_3",
            "form_score",
        ]
        assert loaded_df.iloc[0]["player_name"] == "Salah"

    def test_save_features_prediction_mode(self, localstack_s3_client, clean_s3_bucket):
        """Test saving features in prediction mode uses correct filename."""
        s3_client = localstack_s3_client
        bucket = clean_s3_bucket

        df = pd.DataFrame({"player_id": [350], "points_last_3": [8.5]})

        key = save_features_to_s3(
            s3_client=s3_client,
            df=df,
            bucket=bucket,
            gameweek=21,
            season="2024_25",
            mode="prediction",
        )

        assert key == "processed/season_2024_25/gw21_features_prediction.parquet"

        # Verify file exists
        response = s3_client.list_objects_v2(
            Bucket=bucket, Prefix="processed/season_2024_25/gw21"
        )
        assert response["KeyCount"] == 1

    def test_end_to_end_historical_mode(self, localstack_s3_client, clean_s3_bucket):
        """Test complete feature processing flow in historical mode."""
        s3_client = localstack_s3_client
        bucket = clean_s3_bucket

        # Setup: create all required input files
        bootstrap_data = {
            "events": [{"id": 20, "finished": True}],
            "elements": [
                {
                    "id": 350,
                    "web_name": "Salah",
                    "team": 10,
                    "element_type": 3,
                    "form": "8.5",
                    "chance_of_playing_next_round": 100,
                    "minutes": 900,
                    "selected_by_percent": "45.3",
                    "now_cost": 130,
                }
            ],
            "teams": [
                {
                    "id": 1,
                    "name": "Arsenal",
                    "strength": 4,
                    "strength_attack_home": 1300,
                    "strength_attack_away": 1250,
                    "strength_defence_home": 1280,
                    "strength_defence_away": 1230,
                },
                {
                    "id": 10,
                    "name": "Liverpool",
                    "strength": 5,
                    "strength_attack_home": 1350,
                    "strength_attack_away": 1300,
                    "strength_defence_home": 1320,
                    "strength_defence_away": 1270,
                },
            ],
        }

        fixtures_data = [
            {"team_h": 10, "team_a": 1, "event": 20}  # Liverpool vs Arsenal at home
        ]

        player_history = {
            "history": [
                {
                    "total_points": 8,
                    "minutes": 90,
                    "round": 18,
                    "goals_scored": 1,
                    "assists": 0,
                    "clean_sheets": 0,
                    "bps": 28,
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
                    "total_points": 12,
                    "minutes": 90,
                    "round": 19,
                    "goals_scored": 1,
                    "assists": 1,
                    "clean_sheets": 0,
                    "bps": 35,
                    "ict_index": "92.1",
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
                    "total_points": 6,
                    "minutes": 90,
                    "round": 20,
                    "goals_scored": 0,
                    "assists": 1,
                    "clean_sheets": 1,
                    "bps": 22,
                    "ict_index": "78.5",
                    "threat": "38.0",
                    "creativity": "62.0",
                    "influence": "25.0",
                    "bonus": 1,
                    "yellow_cards": 1,
                    "saves": 0,
                    "transfers_in": 3000,
                    "transfers_out": 4000,
                },
            ]
        }

        # Save input data to S3
        s3_client.put_object(
            Bucket=bucket,
            Key="raw/season_2024_25/gw20_bootstrap.json",
            Body=json.dumps(bootstrap_data),
        )
        s3_client.put_object(
            Bucket=bucket,
            Key="raw/season_2024_25/gw20_fixtures.json",
            Body=json.dumps(fixtures_data),
        )
        s3_client.put_object(
            Bucket=bucket,
            Key="raw/season_2024_25/gw20_players/player_350.json",
            Body=json.dumps(player_history),
        )

        # Invoke handler
        with patch(
            "lambdas.feature_processor.handler.get_s3_client", return_value=s3_client
        ):
            event = {"gameweek": 20, "season": "2024_25", "mode": "historical"}
            result = handler(event, None)

        # Verify response
        assert result["statusCode"] == 200
        assert result["gameweek"] == 20
        assert result["mode"] == "historical"
        assert result["rows_processed"] == 1

        # Verify output file was created
        output_key = result["features_file"]
        assert output_key == "processed/season_2024_25/gw20_features_training.parquet"

        # Read and verify output
        response = s3_client.get_object(Bucket=bucket, Key=output_key)
        buffer = io.BytesIO(response["Body"].read())
        df = pd.read_parquet(buffer)

        assert len(df) == 1
        assert df.iloc[0]["player_id"] == 350
        assert df.iloc[0]["player_name"] == "Salah"
        assert df.iloc[0]["points_last_3"] == pytest.approx(8.67, rel=0.01)
        assert df.iloc[0]["home_away"] == 1  # Home game
        assert df.iloc[0]["opponent_strength"] == 4  # Arsenal's strength
        assert "actual_points" in df.columns  # Historical mode includes target

    def test_end_to_end_prediction_mode(self, localstack_s3_client, clean_s3_bucket):
        """Test complete feature processing flow in prediction mode."""
        s3_client = localstack_s3_client
        bucket = clean_s3_bucket

        # Setup: create input files (no player histories)
        bootstrap_data = {
            "events": [],
            "elements": [
                {
                    "id": 350,
                    "web_name": "Salah",
                    "team": 10,
                    "element_type": 3,
                    "form": "8.5",
                    "chance_of_playing_next_round": 100,
                    "minutes": 1800,
                    "selected_by_percent": "45.3",
                    "now_cost": 130,
                }
            ],
            "teams": [
                {
                    "id": 1,
                    "name": "Arsenal",
                    "strength": 4,
                    "strength_attack_home": 1300,
                    "strength_attack_away": 1250,
                    "strength_defence_home": 1280,
                    "strength_defence_away": 1230,
                },
                {
                    "id": 10,
                    "name": "Liverpool",
                    "strength": 5,
                    "strength_attack_home": 1350,
                    "strength_attack_away": 1300,
                    "strength_defence_home": 1320,
                    "strength_defence_away": 1270,
                },
            ],
        }

        fixtures_data = [
            {"team_h": 1, "team_a": 10, "event": 21}  # Arsenal vs Liverpool (away)
        ]

        s3_client.put_object(
            Bucket=bucket,
            Key="raw/season_2024_25/gw21_bootstrap.json",
            Body=json.dumps(bootstrap_data),
        )
        s3_client.put_object(
            Bucket=bucket,
            Key="raw/season_2024_25/gw21_fixtures.json",
            Body=json.dumps(fixtures_data),
        )

        # Invoke handler in prediction mode (no player histories)
        with patch(
            "lambdas.feature_processor.handler.get_s3_client", return_value=s3_client
        ):
            event = {"gameweek": 21, "season": "2024_25", "mode": "prediction"}
            result = handler(event, None)

        # Verify response
        assert result["statusCode"] == 200
        assert result["mode"] == "prediction"
        assert result["rows_processed"] == 1

        # Verify output
        output_key = result["features_file"]
        assert "prediction" in output_key

        response = s3_client.get_object(Bucket=bucket, Key=output_key)
        buffer = io.BytesIO(response["Body"].read())
        df = pd.read_parquet(buffer)

        assert len(df) == 1
        assert df.iloc[0]["home_away"] == 0  # Away game
        assert df.iloc[0]["opponent_strength"] == 4  # Arsenal's strength
        # Uses form as fallback since no history
        assert df.iloc[0]["points_last_3"] == 8.5
        assert "actual_points" not in df.columns  # Prediction mode excludes target

    def test_end_to_end_produces_all_200_features(
        self, localstack_s3_client, clean_s3_bucket
    ):
        """Integration test verifying all 200 ML features are produced.

        This is a regression test to ensure the feature processor computes
        all feature categories in the full end-to-end flow with real S3.
        """
        s3_client = localstack_s3_client
        bucket = clean_s3_bucket

        # Setup: comprehensive bootstrap data with multiple players and teams
        bootstrap_data = {
            "events": [{"id": 20, "finished": True}],
            "elements": [
                {
                    "id": 350,
                    "web_name": "Salah",
                    "team": 10,
                    "element_type": 3,  # Midfielder
                    "form": "8.5",
                    "chance_of_playing_next_round": 100,
                    "minutes": 1800,
                    "selected_by_percent": "45.3",
                    "now_cost": 130,
                    "total_points": 150,
                    "goals_scored": 12,
                    "assists": 8,
                    "ict_index": "250.5",
                    "creativity": "800.0",
                    "threat": "600.0",
                    "influence": "500.0",
                    "bonus": 15,
                    "clean_sheets": 5,
                    "saves": 0,
                    "status": "a",
                    "news": "",
                    "dreamteam_count": 3,
                    "in_dreamteam": False,
                    "transfers_in_event": 50000,
                    "transfers_out_event": 10000,
                    "ep_this": "6.5",
                    "ep_next": "7.0",
                    "points_per_game": "7.5",
                    "value_form": "0.65",
                    "value_season": "11.5",
                    "cost_change_start": 5,
                    "cost_change_event": 1,
                    "cost_change_event_fall": 0,
                    "corners_and_indirect_freekicks_order": 1,
                    "direct_freekicks_order": 2,
                    "penalties_order": None,
                    "starts": 20,
                },
            ],
            "teams": [
                {
                    "id": 1,
                    "name": "Arsenal",
                    "strength": 4,
                    "strength_overall_home": 1280,
                    "strength_overall_away": 1250,
                    "strength_attack_home": 1300,
                    "strength_attack_away": 1250,
                    "strength_defence_home": 1280,
                    "strength_defence_away": 1230,
                    "position": 3,
                    "points": 45,
                    "played": 20,
                    "clean_sheets": 8,
                },
                {
                    "id": 10,
                    "name": "Liverpool",
                    "strength": 5,
                    "strength_overall_home": 1350,
                    "strength_overall_away": 1300,
                    "strength_attack_home": 1350,
                    "strength_attack_away": 1300,
                    "strength_defence_home": 1320,
                    "strength_defence_away": 1270,
                    "position": 1,
                    "points": 52,
                    "played": 20,
                    "clean_sheets": 10,
                },
            ],
        }

        fixtures_data = [
            {
                "team_h": 10,
                "team_a": 1,
                "event": 20,
                "team_h_difficulty": 2,
                "team_a_difficulty": 4,
                "kickoff_time": "2025-01-18T15:00:00Z",
                "team_h_score": 2,
                "team_a_score": 1,
            },
            # Past fixtures for form calculation
            {
                "team_h": 10,
                "team_a": 5,
                "event": 19,
                "team_h_score": 3,
                "team_a_score": 0,
            },
            {
                "team_h": 6,
                "team_a": 10,
                "event": 18,
                "team_h_score": 1,
                "team_a_score": 2,
            },
            # Future fixtures for FDR calculation
            {
                "team_h": 1,
                "team_a": 10,
                "event": 21,
                "team_h_difficulty": 4,
                "team_a_difficulty": 3,
            },
            {
                "team_h": 10,
                "team_a": 3,
                "event": 22,
                "team_h_difficulty": 2,
                "team_a_difficulty": 4,
            },
        ]

        player_history = {
            "history": [
                {
                    "total_points": 8,
                    "minutes": 90,
                    "round": 18,
                    "goals_scored": 1,
                    "assists": 0,
                    "clean_sheets": 0,
                    "bps": 28,
                    "ict_index": "85.3",
                    "threat": "45.0",
                    "creativity": "55.0",
                    "influence": "30.0",
                    "bonus": 2,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "saves": 0,
                    "transfers_in": 5000,
                    "transfers_out": 2000,
                    "expected_goals": "0.8",
                    "expected_assists": "0.3",
                    "starts": 1,
                    "own_goals": 0,
                    "penalties_saved": 0,
                    "penalties_missed": 0,
                },
                {
                    "total_points": 12,
                    "minutes": 90,
                    "round": 19,
                    "goals_scored": 1,
                    "assists": 1,
                    "clean_sheets": 1,
                    "bps": 35,
                    "ict_index": "92.1",
                    "threat": "60.0",
                    "creativity": "30.0",
                    "influence": "40.0",
                    "bonus": 3,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "saves": 0,
                    "transfers_in": 8000,
                    "transfers_out": 1000,
                    "expected_goals": "1.2",
                    "expected_assists": "0.5",
                    "starts": 1,
                    "own_goals": 0,
                    "penalties_saved": 0,
                    "penalties_missed": 0,
                },
                {
                    "total_points": 6,
                    "minutes": 90,
                    "round": 20,
                    "goals_scored": 0,
                    "assists": 1,
                    "clean_sheets": 1,
                    "bps": 22,
                    "ict_index": "78.5",
                    "threat": "38.0",
                    "creativity": "62.0",
                    "influence": "25.0",
                    "bonus": 1,
                    "yellow_cards": 1,
                    "red_cards": 0,
                    "saves": 0,
                    "transfers_in": 3000,
                    "transfers_out": 4000,
                    "expected_goals": "0.5",
                    "expected_assists": "0.8",
                    "starts": 1,
                    "own_goals": 0,
                    "penalties_saved": 0,
                    "penalties_missed": 0,
                },
            ]
        }

        # Save input data to S3
        s3_client.put_object(
            Bucket=bucket,
            Key="raw/season_2024_25/gw20_bootstrap.json",
            Body=json.dumps(bootstrap_data),
        )
        s3_client.put_object(
            Bucket=bucket,
            Key="raw/season_2024_25/gw20_fixtures.json",
            Body=json.dumps(fixtures_data),
        )
        s3_client.put_object(
            Bucket=bucket,
            Key="raw/season_2024_25/gw20_players/player_350.json",
            Body=json.dumps(player_history),
        )

        # Invoke handler
        with patch(
            "lambdas.feature_processor.handler.get_s3_client", return_value=s3_client
        ):
            event = {"gameweek": 20, "season": "2024_25", "mode": "historical"}
            result = handler(event, None)

        # Verify success
        assert result["statusCode"] == 200

        # Read output and verify all 200 features
        output_key = result["features_file"]
        response = s3_client.get_object(Bucket=bucket, Key=output_key)
        buffer = io.BytesIO(response["Body"].read())
        df = pd.read_parquet(buffer)

        # Check all 200 features are present
        missing_features = set(FEATURE_COLS) - set(df.columns)
        assert len(missing_features) == 0, (
            f"Missing {len(missing_features)} features in output: "
            f"{sorted(missing_features)[:20]}..."
        )

        # Verify metadata columns
        assert "player_id" in df.columns
        assert "player_name" in df.columns
        assert "gameweek" in df.columns

        # Verify target column in historical mode
        assert "actual_points" in df.columns

        # Spot check some features from each category
        salah = df[df["player_id"] == 350].iloc[0]

        # Rolling features (73)
        assert "points_last_3" in df.columns
        assert "expected_goals_last_3" in df.columns

        # Static features (9)
        assert salah["form_score"] == 8.5
        assert salah["home_away"] == 1

        # Bootstrap features (32)
        assert "ep_this" in df.columns
        assert "transfers_in_event" in df.columns
        assert "set_piece_taker" in df.columns

        # Team features (28)
        assert "team_strength_overall" in df.columns
        assert "team_form_score" in df.columns
        assert "player_share_of_team_points" in df.columns

        # Opponent features (24)
        assert "opp_goals_conceded_last_3" in df.columns
        assert "opp_defensive_rating" in df.columns

        # Fixture features (16)
        assert "fdr_current" in df.columns
        assert "is_double_gameweek" in df.columns

        # Position features (8)
        assert "mid_goal_involvement_rate" in df.columns

        # Interaction features (5)
        assert "form_x_fixture_difficulty" in df.columns
        assert "momentum_score" in df.columns

        # Derived features (5)
        assert "minutes_pct" in df.columns
        assert "points_per_90" in df.columns
