"""
Integration tests for data_fetcher with S3.

These tests use moto to mock AWS S3 and verify actual boto3 integration.
Mark these tests with @pytest.mark.integration
"""
import json
import pytest
from unittest.mock import patch, Mock, MagicMock

from lambdas.data_fetcher.handler import handler, save_to_s3


@pytest.mark.integration
class TestDataFetcherS3Integration:
    """Integration tests for data fetcher S3 operations."""

    def test_save_bootstrap_to_s3(self, s3_client):
        """Test saving bootstrap data to mocked S3."""
        # Test data
        bootstrap_data = {
            "events": [
                {"id": 1, "name": "Gameweek 1", "finished": True}
            ],
            "elements": [
                {"id": 350, "web_name": "Salah", "team": 10}
            ]
        }

        # Save to S3
        save_to_s3(
            s3_client=s3_client,
            bucket="fpl-ml-data",
            key="raw/season_2024_25/gw1_bootstrap.json",
            data=bootstrap_data
        )

        # Verify data was saved
        response = s3_client.get_object(
            Bucket="fpl-ml-data",
            Key="raw/season_2024_25/gw1_bootstrap.json"
        )

        # Read and parse JSON
        saved_data = json.loads(response["Body"].read().decode("utf-8"))

        # Assertions
        assert saved_data == bootstrap_data
        assert len(saved_data["events"]) == 1
        assert saved_data["elements"][0]["web_name"] == "Salah"

    def test_save_fixtures_to_s3(self, s3_client):
        """Test saving fixtures data to mocked S3."""
        fixtures_data = [
            {
                "id": 1,
                "event": 20,
                "team_h": 1,
                "team_a": 2,
                "team_h_score": 3,
                "team_a_score": 1
            }
        ]

        save_to_s3(
            s3_client=s3_client,
            bucket="fpl-ml-data",
            key="raw/season_2024_25/gw20_fixtures.json",
            data=fixtures_data
        )

        # Verify
        response = s3_client.get_object(
            Bucket="fpl-ml-data",
            Key="raw/season_2024_25/gw20_fixtures.json"
        )

        saved_data = json.loads(response["Body"].read())
        assert saved_data == fixtures_data

    @patch('lambdas.data_fetcher.handler.FPLApiClient')
    def test_handler_end_to_end_with_s3(self, mock_fpl_class, s3_client):
        """Test complete handler flow with mocked S3."""
        # Mock FPL API client
        mock_fpl = MagicMock()
        mock_fpl.get_season_string.return_value = "2024_25"
        mock_fpl.get_bootstrap_static.return_value = {
            "events": [{"id": 20, "finished": False}],
            "elements": [{"id": 350, "web_name": "Salah"}]
        }
        mock_fpl.get_fixtures.return_value = [
            {"id": 1, "event": 20}
        ]
        mock_fpl.__enter__.return_value = mock_fpl
        mock_fpl.__exit__.return_value = None
        mock_fpl_class.return_value = mock_fpl

        # Patch get_s3_client to return our mocked client
        with patch('lambdas.data_fetcher.handler.get_s3_client', return_value=s3_client):
            # Invoke handler
            event = {
                "gameweek": 20,
                "fetch_player_details": False
            }

            result = handler(event, None)

        # Assertions on response
        assert result["statusCode"] == 200
        assert result["gameweek"] == 20
        assert result["files_count"] == 2

        # Verify bootstrap was saved to S3
        bootstrap_response = s3_client.get_object(
            Bucket="fpl-ml-data",
            Key="raw/season_2024_25/gw20_bootstrap.json"
        )
        bootstrap_data = json.loads(bootstrap_response["Body"].read())
        assert len(bootstrap_data["elements"]) == 1
        assert bootstrap_data["elements"][0]["web_name"] == "Salah"

        # Verify fixtures were saved to S3
        fixtures_response = s3_client.get_object(
            Bucket="fpl-ml-data",
            Key="raw/season_2024_25/gw20_fixtures.json"
        )
        fixtures_data = json.loads(fixtures_response["Body"].read())
        assert len(fixtures_data) == 1
        assert fixtures_data[0]["event"] == 20

    @patch('lambdas.data_fetcher.handler.FPLApiClient')
    def test_handler_with_player_details_s3(self, mock_fpl_class, s3_client):
        """Test handler saves player details to S3."""
        # Mock FPL API
        mock_fpl = MagicMock()
        mock_fpl.get_season_string.return_value = "2024_25"
        mock_fpl.get_bootstrap_static.return_value = {
            "events": [],
            "elements": [
                {"id": 350, "web_name": "Salah"},
                {"id": 328, "web_name": "Haaland"}
            ]
        }
        mock_fpl.get_fixtures.return_value = []
        mock_fpl.get_player_summary.side_effect = [
            {"history": [{"total_points": 12}]},  # Salah
            {"history": [{"total_points": 15}]}   # Haaland
        ]
        mock_fpl.__enter__.return_value = mock_fpl
        mock_fpl.__exit__.return_value = None
        mock_fpl_class.return_value = mock_fpl

        # Invoke handler
        with patch('lambdas.data_fetcher.handler.get_s3_client', return_value=s3_client):
            event = {
                "gameweek": 20,
                "fetch_player_details": True
            }

            result = handler(event, None)

        # Verify player files were saved
        assert result["files_count"] == 4  # bootstrap + fixtures + 2 players

        # Check Salah's data
        salah_response = s3_client.get_object(
            Bucket="fpl-ml-data",
            Key="raw/season_2024_25/gw20_players/player_350.json"
        )
        salah_data = json.loads(salah_response["Body"].read())
        assert salah_data["history"][0]["total_points"] == 12

        # Check Haaland's data
        haaland_response = s3_client.get_object(
            Bucket="fpl-ml-data",
            Key="raw/season_2024_25/gw20_players/player_328.json"
        )
        haaland_data = json.loads(haaland_response["Body"].read())
        assert haaland_data["history"][0]["total_points"] == 15

    def test_s3_list_objects(self, s3_client):
        """Test listing saved objects in S3."""
        # Save multiple files
        for gw in [1, 2, 3]:
            save_to_s3(
                s3_client=s3_client,
                bucket="fpl-ml-data",
                key=f"raw/season_2024_25/gw{gw}_bootstrap.json",
                data={"gameweek": gw}
            )

        # List objects
        response = s3_client.list_objects_v2(
            Bucket="fpl-ml-data",
            Prefix="raw/season_2024_25/"
        )

        # Verify
        assert response["KeyCount"] == 3
        keys = [obj["Key"] for obj in response["Contents"]]
        assert "raw/season_2024_25/gw1_bootstrap.json" in keys
        assert "raw/season_2024_25/gw2_bootstrap.json" in keys
        assert "raw/season_2024_25/gw3_bootstrap.json" in keys
