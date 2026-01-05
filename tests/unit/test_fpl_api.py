"""
Unit tests for FPL API client.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import httpx

from lambdas.common.fpl_api import FPLApiClient, FPLApiError


class TestFPLApiClient:
    """Tests for FPLApiClient."""

    def test_initialization(self):
        """Test client initialization with custom parameters."""
        client = FPLApiClient(timeout=60, max_retries=5, base_delay=2.0)

        assert client.timeout == 60
        assert client.max_retries == 5
        assert client.base_delay == 2.0
        assert client.BASE_URL == "https://fantasy.premierleague.com/api"

    @patch('lambdas.common.fpl_api.httpx.Client')
    def test_get_bootstrap_static_success(self, mock_client_class):
        """Test successful bootstrap-static fetch."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "events": [{"id": 1, "name": "Gameweek 1"}],
            "elements": [{"id": 350, "web_name": "Salah"}]
        }

        # Mock client
        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Test
        client = FPLApiClient()
        result = client.get_bootstrap_static()

        # Assertions
        assert "events" in result
        assert "elements" in result
        assert result["events"][0]["name"] == "Gameweek 1"
        mock_client.get.assert_called_once_with(
            "https://fantasy.premierleague.com/api/bootstrap-static/",
            params=None
        )

    @patch('lambdas.common.fpl_api.httpx.Client')
    @patch('lambdas.common.fpl_api.time.sleep')  # Mock sleep to speed up tests
    def test_rate_limit_retry(self, mock_sleep, mock_client_class):
        """Test that rate limiting triggers retry with backoff."""
        # First response: 429 (rate limited)
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {"Retry-After": "2"}

        # Second response: Success
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {"events": []}

        # Mock client
        mock_client = Mock()
        mock_client.get.side_effect = [rate_limit_response, success_response]
        mock_client_class.return_value = mock_client

        # Test
        client = FPLApiClient()
        result = client.get_bootstrap_static()

        # Assertions
        assert result == {"events": []}
        assert mock_client.get.call_count == 2
        mock_sleep.assert_called_once_with(2)  # Should sleep for Retry-After duration

    @patch('lambdas.common.fpl_api.httpx.Client')
    @patch('lambdas.common.fpl_api.time.sleep')
    def test_http_error_with_retries(self, mock_sleep, mock_client_class):
        """Test that HTTP errors trigger exponential backoff."""
        # Mock error responses
        error_response = Mock()
        error_response.status_code = 500
        error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Server Error",
            request=Mock(),
            response=error_response
        )

        mock_client = Mock()
        mock_client.get.return_value = error_response
        mock_client_class.return_value = mock_client

        # Test
        client = FPLApiClient(max_retries=3, base_delay=1.0)

        with pytest.raises(FPLApiError) as exc_info:
            client.get_bootstrap_static()

        # Assertions
        assert "Failed to fetch" in str(exc_info.value)
        assert mock_client.get.call_count == 3  # Should retry 3 times
        # Check exponential backoff: 1s, 2s
        assert mock_sleep.call_count == 2

    @patch('lambdas.common.fpl_api.httpx.Client')
    def test_get_fixtures_with_gameweek(self, mock_client_class):
        """Test fetching fixtures for specific gameweek."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": 1, "event": 20, "team_h": 1, "team_a": 2}
        ]

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Test
        client = FPLApiClient()
        result = client.get_fixtures(gameweek=20)

        # Assertions
        assert len(result) == 1
        assert result[0]["event"] == 20
        mock_client.get.assert_called_once_with(
            "https://fantasy.premierleague.com/api/fixtures/",
            params={"event": 20}
        )

    @patch('lambdas.common.fpl_api.httpx.Client')
    def test_get_player_summary(self, mock_client_class):
        """Test fetching player summary."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "history": [{"total_points": 8, "minutes": 90}],
            "fixtures": [{"event": 21}]
        }

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Test
        client = FPLApiClient()
        result = client.get_player_summary(player_id=350)

        # Assertions
        assert "history" in result
        assert "fixtures" in result
        mock_client.get.assert_called_once_with(
            "https://fantasy.premierleague.com/api/element-summary/350/",
            params=None
        )

    @patch('lambdas.common.fpl_api.httpx.Client')
    def test_get_current_gameweek(self, mock_client_class):
        """Test getting current gameweek."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "events": [
                {"id": 19, "is_current": False},
                {"id": 20, "is_current": True},
                {"id": 21, "is_current": False}
            ]
        }

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Test
        client = FPLApiClient()
        current_gw = client.get_current_gameweek()

        # Assertions
        assert current_gw == 20

    @patch('lambdas.common.fpl_api.httpx.Client')
    def test_is_gameweek_finished(self, mock_client_class):
        """Test checking if gameweek is finished."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "events": [
                {"id": 19, "finished": True},
                {"id": 20, "finished": False}
            ]
        }

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Test
        client = FPLApiClient()

        assert client.is_gameweek_finished(19) is True
        assert client.is_gameweek_finished(20) is False

    @patch('lambdas.common.fpl_api.datetime')
    def test_get_season_string_august_onwards(self, mock_datetime):
        """Test season string generation for months Aug-Dec."""
        # Mock current date: August 15, 2024
        mock_datetime.now.return_value = Mock(year=2024, month=8)

        client = FPLApiClient()
        season = client.get_season_string()

        assert season == "2024_25"

    @patch('lambdas.common.fpl_api.datetime')
    def test_get_season_string_january_july(self, mock_datetime):
        """Test season string generation for months Jan-Jul."""
        # Mock current date: January 15, 2025
        mock_datetime.now.return_value = Mock(year=2025, month=1)

        client = FPLApiClient()
        season = client.get_season_string()

        assert season == "2024_25"

    def test_context_manager(self):
        """Test client works as context manager."""
        with FPLApiClient() as client:
            assert client is not None
            assert hasattr(client, 'client')

        # Client should be closed after exiting context
        # (We can't easily test this without mocking)
