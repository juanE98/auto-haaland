"""
Unit tests for api_handler Lambda using AWS Lambda Powertools.
"""

import json
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from lambdas.api_handler.handler import VALID_POSITIONS, app, decimal_to_float, handler


class MockLambdaContext:
    """Mock Lambda context for testing."""

    def __init__(self):
        self.function_name = "test-function"
        self.function_version = "$LATEST"
        self.invoked_function_arn = (
            "arn:aws:lambda:us-east-1:123456789012:function:test"
        )
        self.memory_limit_in_mb = 128
        self.aws_request_id = "test-request-id"
        self.log_group_name = "/aws/lambda/test"
        self.log_stream_name = "2024/01/01/[$LATEST]test"

    def get_remaining_time_in_millis(self):
        return 30000


@pytest.fixture
def mock_context():
    """Provide a mock Lambda context."""
    return MockLambdaContext()


def make_api_event(
    method: str = "GET",
    path: str = "/",
    query_params: dict = None,
    path_params: dict = None,
) -> dict:
    """Create a mock API Gateway event."""
    return {
        "httpMethod": method,
        "path": path,
        "resource": path.split("?")[0],
        "queryStringParameters": query_params,
        "pathParameters": path_params,
        "headers": {"Content-Type": "application/json"},
        "requestContext": {
            "stage": "test",
            "requestId": "test-request-id",
        },
        "body": None,
        "isBase64Encoded": False,
    }


class TestDecimalToFloat:
    """Tests for decimal_to_float function."""

    def test_converts_decimal_to_float(self):
        """Test Decimal values are converted to floats."""
        result = decimal_to_float(Decimal("8.5"))
        assert result == 8.5
        assert isinstance(result, float)

    def test_converts_nested_dict(self):
        """Test nested dict with Decimals."""
        data = {
            "player": {
                "predicted_points": Decimal("12.34"),
                "form": Decimal("9.8"),
            }
        }
        result = decimal_to_float(data)
        assert result["player"]["predicted_points"] == 12.34
        assert result["player"]["form"] == 9.8

    def test_converts_list_of_decimals(self):
        """Test list containing Decimals."""
        data = [Decimal("1.1"), Decimal("2.2"), Decimal("3.3")]
        result = decimal_to_float(data)
        assert result == [1.1, 2.2, 3.3]

    def test_preserves_non_decimal_types(self):
        """Test non-Decimal types are preserved."""
        data = {"name": "Salah", "id": 350, "active": True}
        result = decimal_to_float(data)
        assert result == data


class TestValidPositions:
    """Tests for valid positions constant."""

    def test_all_positions_defined(self):
        """Test all FPL positions are defined."""
        assert "GKP" in VALID_POSITIONS
        assert "DEF" in VALID_POSITIONS
        assert "MID" in VALID_POSITIONS
        assert "FWD" in VALID_POSITIONS
        assert len(VALID_POSITIONS) == 4


class TestGetPredictionsEndpoint:
    """Tests for GET /predictions endpoint."""

    @patch("lambdas.api_handler.handler.get_table")
    def test_missing_gameweek_returns_400(self, mock_get_table, mock_context):
        """Test missing gameweek returns 400."""
        event = make_api_event(path="/predictions", query_params=None)

        response = handler(event, mock_context)

        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "gameweek" in body["message"]

    @patch("lambdas.api_handler.handler.get_table")
    def test_invalid_gameweek_returns_400(self, mock_get_table, mock_context):
        """Test invalid gameweek returns 400."""
        event = make_api_event(path="/predictions", query_params={"gameweek": "abc"})

        response = handler(event, mock_context)

        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "integer" in body["message"]

    @patch("lambdas.api_handler.handler.get_table")
    def test_valid_request_returns_predictions(self, mock_get_table, mock_context):
        """Test valid request returns predictions."""
        mock_table = MagicMock()
        mock_table.query.return_value = {
            "Items": [
                {"player_id": 350, "predicted_points": Decimal("8.5")},
                {"player_id": 328, "predicted_points": Decimal("12.3")},
            ]
        }
        mock_get_table.return_value = mock_table

        event = make_api_event(path="/predictions", query_params={"gameweek": "20"})

        response = handler(event, mock_context)

        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["gameweek"] == 20
        assert body["count"] == 2
        assert len(body["predictions"]) == 2

    @patch("lambdas.api_handler.handler.get_table")
    def test_respects_limit_parameter(self, mock_get_table, mock_context):
        """Test limit parameter is passed to query."""
        mock_table = MagicMock()
        mock_table.query.return_value = {"Items": []}
        mock_get_table.return_value = mock_table

        event = make_api_event(
            path="/predictions", query_params={"gameweek": "20", "limit": "5"}
        )

        handler(event, mock_context)

        call_kwargs = mock_table.query.call_args[1]
        assert call_kwargs["Limit"] == 5


class TestGetPlayerPredictionEndpoint:
    """Tests for GET /predictions/{player_id} endpoint."""

    @patch("lambdas.api_handler.handler.get_table")
    def test_player_not_found_returns_404(self, mock_get_table, mock_context):
        """Test player not found returns 404."""
        mock_table = MagicMock()
        mock_table.get_item.return_value = {}
        mock_get_table.return_value = mock_table

        event = make_api_event(
            path="/predictions/999",
            path_params={"player_id": "999"},
            query_params={"gameweek": "20"},
        )
        event["resource"] = "/predictions/{player_id}"

        response = handler(event, mock_context)

        assert response["statusCode"] == 404

    @patch("lambdas.api_handler.handler.get_table")
    def test_valid_request_returns_prediction(self, mock_get_table, mock_context):
        """Test valid request returns prediction."""
        mock_table = MagicMock()
        mock_table.get_item.return_value = {
            "Item": {
                "player_id": 350,
                "gameweek": 20,
                "predicted_points": Decimal("8.5"),
            }
        }
        mock_get_table.return_value = mock_table

        event = make_api_event(
            path="/predictions/350",
            path_params={"player_id": "350"},
            query_params={"gameweek": "20"},
        )
        event["resource"] = "/predictions/{player_id}"

        response = handler(event, mock_context)

        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["player_id"] == 350
        assert body["predicted_points"] == 8.5

    @patch("lambdas.api_handler.handler.get_table")
    def test_supports_gw_query_param(self, mock_get_table, mock_context):
        """Test 'gw' query param works as alias for 'gameweek'."""
        mock_table = MagicMock()
        mock_table.get_item.return_value = {
            "Item": {
                "player_id": 350,
                "gameweek": 20,
                "predicted_points": Decimal("8.5"),
            }
        }
        mock_get_table.return_value = mock_table

        event = make_api_event(
            path="/predictions/350",
            path_params={"player_id": "350"},
            query_params={"gw": "20"},
        )
        event["resource"] = "/predictions/{player_id}"

        response = handler(event, mock_context)

        assert response["statusCode"] == 200

    @patch("lambdas.api_handler.handler.get_table")
    def test_returns_all_gameweeks_without_gw_param(self, mock_get_table, mock_context):
        """Test returns all gameweeks when no gameweek specified."""
        mock_table = MagicMock()
        mock_table.query.return_value = {
            "Items": [
                {"player_id": 350, "gameweek": 21, "predicted_points": Decimal("7.2")},
                {"player_id": 350, "gameweek": 20, "predicted_points": Decimal("8.5")},
            ]
        }
        mock_get_table.return_value = mock_table

        event = make_api_event(
            path="/predictions/350",
            path_params={"player_id": "350"},
            query_params=None,
        )
        event["resource"] = "/predictions/{player_id}"

        response = handler(event, mock_context)

        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["player_id"] == 350
        assert body["count"] == 2


class TestGetTopPredictionsEndpoint:
    """Tests for GET /top endpoint."""

    @patch("lambdas.api_handler.handler.get_table")
    def test_missing_gameweek_returns_400(self, mock_get_table, mock_context):
        """Test missing gameweek returns 400."""
        event = make_api_event(path="/top", query_params=None)

        response = handler(event, mock_context)

        assert response["statusCode"] == 400

    @patch("lambdas.api_handler.handler.get_table")
    def test_invalid_position_returns_400(self, mock_get_table, mock_context):
        """Test invalid position returns 400."""
        event = make_api_event(
            path="/top", query_params={"gameweek": "20", "position": "INVALID"}
        )

        response = handler(event, mock_context)

        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "Invalid position" in body["message"]

    @patch("lambdas.api_handler.handler.get_table")
    def test_valid_positions_accepted(self, mock_get_table, mock_context):
        """Test all valid positions are accepted."""
        mock_table = MagicMock()
        mock_table.query.return_value = {"Items": []}
        mock_get_table.return_value = mock_table

        for pos in ["GKP", "DEF", "MID", "FWD"]:
            event = make_api_event(
                path="/top", query_params={"gameweek": "20", "position": pos}
            )

            response = handler(event, mock_context)
            assert response["statusCode"] == 200

    @patch("lambdas.api_handler.handler.get_table")
    def test_position_is_case_insensitive(self, mock_get_table, mock_context):
        """Test position parameter is case insensitive."""
        mock_table = MagicMock()
        mock_table.query.return_value = {"Items": []}
        mock_get_table.return_value = mock_table

        event = make_api_event(
            path="/top", query_params={"gameweek": "20", "position": "mid"}
        )

        response = handler(event, mock_context)

        assert response["statusCode"] == 200

    @patch("lambdas.api_handler.handler.get_table")
    def test_default_limit_is_10(self, mock_get_table, mock_context):
        """Test default limit is 10."""
        mock_table = MagicMock()
        mock_table.query.return_value = {"Items": []}
        mock_get_table.return_value = mock_table

        event = make_api_event(path="/top", query_params={"gameweek": "20"})

        response = handler(event, mock_context)

        body = json.loads(response["body"])
        assert body["limit"] == 10


class TestComparePlayersEndpoint:
    """Tests for GET /compare endpoint."""

    @patch("lambdas.api_handler.handler.get_table")
    def test_missing_players_returns_400(self, mock_get_table, mock_context):
        """Test missing players returns 400."""
        event = make_api_event(path="/compare", query_params={"gameweek": "20"})

        response = handler(event, mock_context)

        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "players" in body["message"]

    @patch("lambdas.api_handler.handler.get_table")
    def test_missing_gameweek_returns_400(self, mock_get_table, mock_context):
        """Test missing gameweek returns 400."""
        event = make_api_event(path="/compare", query_params={"players": "350,328"})

        response = handler(event, mock_context)

        assert response["statusCode"] == 400

    @patch("lambdas.api_handler.handler.get_table")
    def test_invalid_players_format_returns_400(self, mock_get_table, mock_context):
        """Test invalid players format returns 400."""
        event = make_api_event(
            path="/compare", query_params={"players": "abc,def", "gameweek": "20"}
        )

        response = handler(event, mock_context)

        assert response["statusCode"] == 400

    @patch("lambdas.api_handler.handler.get_table")
    def test_too_many_players_returns_400(self, mock_get_table, mock_context):
        """Test more than 15 players returns 400."""
        players = ",".join(str(i) for i in range(20))
        event = make_api_event(
            path="/compare", query_params={"players": players, "gameweek": "20"}
        )

        response = handler(event, mock_context)

        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "15" in body["message"]

    @patch("lambdas.api_handler.handler.get_table")
    def test_valid_request_returns_comparison(self, mock_get_table, mock_context):
        """Test valid request returns comparison."""
        mock_table = MagicMock()
        mock_table.get_item.side_effect = [
            {"Item": {"player_id": 350, "predicted_points": Decimal("8.5")}},
            {"Item": {"player_id": 328, "predicted_points": Decimal("12.3")}},
        ]
        mock_get_table.return_value = mock_table

        event = make_api_event(
            path="/compare", query_params={"players": "350,328", "gameweek": "20"}
        )

        response = handler(event, mock_context)

        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["found"] == 2
        assert len(body["predictions"]) == 2

    @patch("lambdas.api_handler.handler.get_table")
    def test_results_sorted_by_points_descending(self, mock_get_table, mock_context):
        """Test results are sorted by predicted_points descending."""
        mock_table = MagicMock()
        mock_table.get_item.side_effect = [
            {"Item": {"player_id": 350, "predicted_points": Decimal("8.5")}},
            {"Item": {"player_id": 328, "predicted_points": Decimal("12.3")}},
        ]
        mock_get_table.return_value = mock_table

        event = make_api_event(
            path="/compare", query_params={"players": "350,328", "gameweek": "20"}
        )

        response = handler(event, mock_context)

        body = json.loads(response["body"])
        assert body["predictions"][0]["player_id"] == 328  # Higher points first
        assert body["predictions"][1]["player_id"] == 350

    @patch("lambdas.api_handler.handler.get_table")
    def test_reports_missing_players(self, mock_get_table, mock_context):
        """Test missing players are reported."""
        mock_table = MagicMock()
        mock_table.get_item.side_effect = [
            {"Item": {"player_id": 350, "predicted_points": Decimal("8.5")}},
            {},  # Not found
        ]
        mock_get_table.return_value = mock_table

        event = make_api_event(
            path="/compare", query_params={"players": "350,999", "gameweek": "20"}
        )

        response = handler(event, mock_context)

        body = json.loads(response["body"])
        assert body["found"] == 1
        assert 999 in body["missing"]


class TestGetLatestGameweek:
    """Tests for GET /gameweek/latest endpoint."""

    @patch("lambdas.api_handler.handler.get_table")
    def test_returns_latest_gameweek(self, mock_get_table, mock_context):
        """Test returns the highest gameweek with predictions."""
        mock_table = MagicMock()
        # Queries from GW38 downward; GW38-23 empty, GW22 has predictions
        responses = [{"Items": []}] * 16 + [{"Items": [{"gameweek": 22}]}]
        mock_table.query.side_effect = responses
        mock_get_table.return_value = mock_table

        event = make_api_event(path="/gameweek/latest")

        response = handler(event, mock_context)

        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["gameweek"] == 22

    @patch("lambdas.api_handler.handler.get_table")
    def test_no_predictions_returns_404(self, mock_get_table, mock_context):
        """Test empty table returns 404."""
        mock_table = MagicMock()
        mock_table.query.return_value = {"Items": []}
        mock_get_table.return_value = mock_table

        event = make_api_event(path="/gameweek/latest")

        response = handler(event, mock_context)

        assert response["statusCode"] == 404

    @patch("lambdas.api_handler.handler.get_table")
    def test_handles_single_gameweek(self, mock_get_table, mock_context):
        """Test works with only one gameweek available."""
        mock_table = MagicMock()
        # GW38-16 empty, GW15 has predictions
        responses = [{"Items": []}] * 23 + [{"Items": [{"gameweek": 15}]}]
        mock_table.query.side_effect = responses
        mock_get_table.return_value = mock_table

        event = make_api_event(path="/gameweek/latest")

        response = handler(event, mock_context)

        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["gameweek"] == 15


class TestCORSHeaders:
    """Tests for CORS headers."""

    @patch("lambdas.api_handler.handler.get_table")
    def test_cors_headers_present(self, mock_get_table, mock_context):
        """Test CORS headers are present in response."""
        mock_table = MagicMock()
        mock_table.query.return_value = {"Items": []}
        mock_get_table.return_value = mock_table

        event = make_api_event(path="/predictions", query_params={"gameweek": "20"})
        event["headers"]["Origin"] = "http://localhost:3000"

        response = handler(event, mock_context)

        assert "Access-Control-Allow-Origin" in response["multiValueHeaders"]


class TestHandler:
    """Tests for main handler function."""

    @patch("lambdas.api_handler.handler.get_table")
    def test_unknown_path_returns_404(self, mock_get_table, mock_context):
        """Test unknown path returns 404."""
        event = make_api_event(path="/unknown")

        response = handler(event, mock_context)

        assert response["statusCode"] == 404
