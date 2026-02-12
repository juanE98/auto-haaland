"""
Integration tests for data_fetcher with S3.

These tests run against LocalStack when AWS_ENDPOINT_URL is set.
Mark these tests with @pytest.mark.integration
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from lambdas.data_fetcher.handler import handler, save_to_s3


@pytest.mark.integration
class TestDataFetcherS3Integration:
    """Integration tests for data fetcher S3 operations."""

    def test_save_bootstrap_to_s3(self, localstack_s3_client, clean_s3_bucket):
        """Test saving bootstrap data to S3."""
        s3_client = localstack_s3_client
        bucket = clean_s3_bucket

        # Test data
        bootstrap_data = {
            "events": [{"id": 1, "name": "Gameweek 1", "finished": True}],
            "elements": [{"id": 350, "web_name": "Salah", "team": 10}],
        }

        # Save to S3
        save_to_s3(
            s3_client=s3_client,
            bucket=bucket,
            key="raw/season_2024_25/gw1_bootstrap.json",
            data=bootstrap_data,
        )

        # Verify data was saved
        response = s3_client.get_object(
            Bucket=bucket, Key="raw/season_2024_25/gw1_bootstrap.json"
        )

        # Read and parse JSON
        saved_data = json.loads(response["Body"].read().decode("utf-8"))

        # Assertions
        assert saved_data == bootstrap_data
        assert len(saved_data["events"]) == 1
        assert saved_data["elements"][0]["web_name"] == "Salah"

    def test_save_fixtures_to_s3(self, localstack_s3_client, clean_s3_bucket):
        """Test saving fixtures data to S3."""
        s3_client = localstack_s3_client
        bucket = clean_s3_bucket

        fixtures_data = [
            {
                "id": 1,
                "event": 20,
                "team_h": 1,
                "team_a": 2,
                "team_h_score": 3,
                "team_a_score": 1,
            }
        ]

        save_to_s3(
            s3_client=s3_client,
            bucket=bucket,
            key="raw/season_2024_25/gw20_fixtures.json",
            data=fixtures_data,
        )

        # Verify
        response = s3_client.get_object(
            Bucket=bucket, Key="raw/season_2024_25/gw20_fixtures.json"
        )

        saved_data = json.loads(response["Body"].read())
        assert saved_data == fixtures_data

    @patch("lambdas.data_fetcher.handler.FPLApiClient")
    def test_handler_end_to_end_with_s3(
        self, mock_fpl_class, localstack_s3_client, clean_s3_bucket
    ):
        """Test complete handler flow with S3."""
        s3_client = localstack_s3_client
        bucket = clean_s3_bucket

        # Mock FPL API client
        mock_fpl = MagicMock()
        mock_fpl.get_season_string.return_value = "2024_25"
        mock_fpl.get_bootstrap_static.return_value = {
            "events": [{"id": 20, "finished": False}],
            "elements": [{"id": 350, "web_name": "Salah"}],
        }
        mock_fpl.get_fixtures.return_value = [{"id": 1, "event": 20}]
        mock_fpl.__enter__.return_value = mock_fpl
        mock_fpl.__exit__.return_value = None
        mock_fpl_class.return_value = mock_fpl

        # Patch get_s3_client to return our client
        with patch(
            "lambdas.data_fetcher.handler.get_s3_client", return_value=s3_client
        ):
            # Invoke handler
            event = {"gameweek": 20, "fetch_player_details": False}

            result = handler(event, None)

        # Assertions on response
        assert result["statusCode"] == 200
        assert result["gameweek"] == 20
        assert result["files_count"] == 2

        # Verify bootstrap was saved to S3
        bootstrap_response = s3_client.get_object(
            Bucket=bucket, Key="raw/season_2024_25/gw20_bootstrap.json"
        )
        bootstrap_data = json.loads(bootstrap_response["Body"].read())
        assert len(bootstrap_data["elements"]) == 1
        assert bootstrap_data["elements"][0]["web_name"] == "Salah"

        # Verify fixtures were saved to S3
        fixtures_response = s3_client.get_object(
            Bucket=bucket, Key="raw/season_2024_25/gw20_fixtures.json"
        )
        fixtures_data = json.loads(fixtures_response["Body"].read())
        assert len(fixtures_data) == 1
        assert fixtures_data[0]["event"] == 20

    @patch("lambdas.data_fetcher.handler.FPLApiClient")
    def test_handler_with_player_details_s3(
        self, mock_fpl_class, localstack_s3_client, clean_s3_bucket
    ):
        """Test handler saves player details to S3."""
        s3_client = localstack_s3_client
        bucket = clean_s3_bucket

        # Mock FPL API
        mock_fpl = MagicMock()
        mock_fpl.get_season_string.return_value = "2024_25"
        mock_fpl.get_bootstrap_static.return_value = {
            "events": [],
            "elements": [
                {"id": 350, "web_name": "Salah"},
                {"id": 328, "web_name": "Haaland"},
            ],
        }
        mock_fpl.get_fixtures.return_value = []
        mock_fpl.get_player_summary.side_effect = [
            {"history": [{"total_points": 12}]},  # Salah
            {"history": [{"total_points": 15}]},  # Haaland
        ]
        mock_fpl.__enter__.return_value = mock_fpl
        mock_fpl.__exit__.return_value = None
        mock_fpl_class.return_value = mock_fpl

        # Invoke handler
        with patch(
            "lambdas.data_fetcher.handler.get_s3_client", return_value=s3_client
        ):
            event = {"gameweek": 20, "fetch_player_details": True}

            result = handler(event, None)

        # Verify combined histories file was saved
        assert result["files_count"] == 3  # bootstrap + fixtures + combined histories

        # Check combined histories file
        histories_response = s3_client.get_object(
            Bucket=bucket,
            Key="raw/season_2024_25/gw20_player_histories.json",
        )
        histories_data = json.loads(histories_response["Body"].read())

        # Verify both players present (keys are string in JSON)
        assert "350" in histories_data
        assert "328" in histories_data
        assert histories_data["350"]["history"][0]["total_points"] == 12
        assert histories_data["328"]["history"][0]["total_points"] == 15

    def test_s3_list_objects(self, localstack_s3_client, clean_s3_bucket):
        """Test listing saved objects in S3."""
        s3_client = localstack_s3_client
        bucket = clean_s3_bucket

        # Save multiple files
        for gw in [1, 2, 3]:
            save_to_s3(
                s3_client=s3_client,
                bucket=bucket,
                key=f"raw/season_2024_25/gw{gw}_bootstrap.json",
                data={"gameweek": gw},
            )

        # List objects
        response = s3_client.list_objects_v2(
            Bucket=bucket, Prefix="raw/season_2024_25/"
        )

        # Verify
        assert response["KeyCount"] == 3
        keys = [obj["Key"] for obj in response["Contents"]]
        assert "raw/season_2024_25/gw1_bootstrap.json" in keys
        assert "raw/season_2024_25/gw2_bootstrap.json" in keys
        assert "raw/season_2024_25/gw3_bootstrap.json" in keys
