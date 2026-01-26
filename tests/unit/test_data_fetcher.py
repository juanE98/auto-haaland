"""
Unit tests for data_fetcher Lambda handler.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock, call

from lambdas.data_fetcher.handler import handler, save_to_s3


class TestDataFetcherHandler:
    """Tests for data_fetcher Lambda handler."""

    @patch("lambdas.data_fetcher.handler.get_s3_client")
    @patch("lambdas.data_fetcher.handler.FPLApiClient")
    def test_handler_with_explicit_gameweek(self, mock_fpl_class, mock_get_s3):
        """Test handler with explicit gameweek in event."""
        # Mock S3 client
        mock_s3 = Mock()
        mock_get_s3.return_value = mock_s3

        # Mock FPL client
        mock_fpl = MagicMock()
        mock_fpl.get_season_string.return_value = "2024_25"
        mock_fpl.get_bootstrap_static.return_value = {"events": [], "elements": []}
        mock_fpl.get_fixtures.return_value = [{"id": 1}]
        mock_fpl.__enter__.return_value = mock_fpl
        mock_fpl.__exit__.return_value = None
        mock_fpl_class.return_value = mock_fpl

        # Test event
        event = {"gameweek": 20, "fetch_player_details": False}

        # Invoke handler
        result = handler(event, None)

        # Assertions
        assert result["statusCode"] == 200
        assert result["gameweek"] == 20
        assert result["season"] == "2024_25"
        assert result["files_count"] == 2  # bootstrap + fixtures
        assert len(result["files_saved"]) == 2

        # Verify S3 calls
        assert mock_s3.put_object.call_count == 2

        # Verify bootstrap was saved
        bootstrap_call = mock_s3.put_object.call_args_list[0]
        assert bootstrap_call[1]["Key"] == "raw/season_2024_25/gw20_bootstrap.json"

        # Verify fixtures were saved
        fixtures_call = mock_s3.put_object.call_args_list[1]
        assert fixtures_call[1]["Key"] == "raw/season_2024_25/gw20_fixtures.json"

    @patch("lambdas.data_fetcher.handler.get_s3_client")
    @patch("lambdas.data_fetcher.handler.FPLApiClient")
    def test_handler_auto_detect_gameweek(self, mock_fpl_class, mock_get_s3):
        """Test handler auto-detects current gameweek if not provided."""
        # Mock S3 client
        mock_s3 = Mock()
        mock_get_s3.return_value = mock_s3

        # Mock FPL client
        mock_fpl = MagicMock()
        mock_fpl.get_season_string.return_value = "2024_25"
        mock_fpl.get_current_gameweek.return_value = 22  # Auto-detected
        mock_fpl.get_bootstrap_static.return_value = {"events": []}
        mock_fpl.get_fixtures.return_value = []
        mock_fpl.__enter__.return_value = mock_fpl
        mock_fpl.__exit__.return_value = None
        mock_fpl_class.return_value = mock_fpl

        # Test event (no gameweek specified)
        event = {}

        # Invoke handler
        result = handler(event, None)

        # Assertions
        assert result["statusCode"] == 200
        assert result["gameweek"] == 22
        mock_fpl.get_current_gameweek.assert_called_once()

    @patch("lambdas.data_fetcher.handler.get_s3_client")
    @patch("lambdas.data_fetcher.handler.FPLApiClient")
    def test_handler_with_player_details(self, mock_fpl_class, mock_get_s3):
        """Test handler fetches individual player details when requested."""
        # Mock S3 client
        mock_s3 = Mock()
        mock_get_s3.return_value = mock_s3

        # Mock FPL client
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
        mock_fpl.get_player_summary.return_value = {"history": []}
        mock_fpl.__enter__.return_value = mock_fpl
        mock_fpl.__exit__.return_value = None
        mock_fpl_class.return_value = mock_fpl

        # Test event
        event = {"gameweek": 20, "fetch_player_details": True}

        # Invoke handler
        result = handler(event, None)

        # Assertions
        assert result["statusCode"] == 200
        # bootstrap + fixtures + 2 players
        assert result["files_count"] == 4

        # Verify player summaries were fetched
        assert mock_fpl.get_player_summary.call_count == 2
        mock_fpl.get_player_summary.assert_any_call(350)
        mock_fpl.get_player_summary.assert_any_call(328)

    @patch("lambdas.data_fetcher.handler.get_s3_client")
    @patch("lambdas.data_fetcher.handler.FPLApiClient")
    def test_handler_fpl_api_error(self, mock_fpl_class, mock_get_s3):
        """Test handler handles FPL API errors gracefully."""
        from lambdas.common.fpl_api import FPLApiError

        # Mock S3 client
        mock_s3 = Mock()
        mock_get_s3.return_value = mock_s3

        # Mock FPL client to raise error
        mock_fpl = MagicMock()
        mock_fpl.get_season_string.side_effect = FPLApiError("API unavailable")
        mock_fpl.__enter__.return_value = mock_fpl
        mock_fpl.__exit__.return_value = None
        mock_fpl_class.return_value = mock_fpl

        # Test event
        event = {"gameweek": 20}

        # Invoke handler
        result = handler(event, None)

        # Assertions
        assert result["statusCode"] == 500
        assert "error" in result
        assert "FPL API error" in result["error"]

    @patch("lambdas.data_fetcher.handler.get_s3_client")
    @patch("lambdas.data_fetcher.handler.FPLApiClient")
    def test_handler_no_current_gameweek(self, mock_fpl_class, mock_get_s3):
        """Test handler when current gameweek cannot be determined."""
        # Mock S3 client
        mock_s3 = Mock()
        mock_get_s3.return_value = mock_s3

        # Mock FPL client
        mock_fpl = MagicMock()
        mock_fpl.get_season_string.return_value = "2024_25"
        mock_fpl.get_current_gameweek.return_value = None  # Season not started
        mock_fpl.__enter__.return_value = mock_fpl
        mock_fpl.__exit__.return_value = None
        mock_fpl_class.return_value = mock_fpl

        # Test event (no gameweek)
        event = {}

        # Invoke handler
        result = handler(event, None)

        # Assertions
        assert result["statusCode"] == 400
        assert "Could not determine current gameweek" in result["error"]


class TestSaveToS3:
    """Tests for save_to_s3 helper function."""

    def test_save_to_s3_success(self):
        """Test successful S3 save."""
        # Mock S3 client
        mock_s3 = Mock()

        # Test data
        test_data = {"player_id": 350, "points": 12}

        # Call function
        save_to_s3(
            s3_client=mock_s3, bucket="test-bucket", key="test/key.json", data=test_data
        )

        # Assertions
        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args[1]

        assert call_kwargs["Bucket"] == "test-bucket"
        assert call_kwargs["Key"] == "test/key.json"
        assert call_kwargs["ContentType"] == "application/json"

        # Verify JSON content
        saved_json = call_kwargs["Body"]
        assert json.loads(saved_json) == test_data

    def test_save_to_s3_with_nested_data(self):
        """Test S3 save with nested JSON structure."""
        mock_s3 = Mock()

        nested_data = {
            "events": [{"id": 1, "name": "GW1"}, {"id": 2, "name": "GW2"}],
            "metadata": {"season": "2024_25"},
        }

        save_to_s3(
            s3_client=mock_s3, bucket="test-bucket", key="nested.json", data=nested_data
        )

        # Verify structure is preserved
        call_kwargs = mock_s3.put_object.call_args[1]
        saved_json = json.loads(call_kwargs["Body"])

        assert len(saved_json["events"]) == 2
        assert saved_json["metadata"]["season"] == "2024_25"
