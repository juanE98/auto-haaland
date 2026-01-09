"""
Integration tests for api_handler with DynamoDB.
"""

import json
from decimal import Decimal

import boto3
import pytest
from moto import mock_aws

from lambdas.api_handler.handler import handler


@pytest.fixture
def sample_predictions():
    """Sample predictions to load into DynamoDB."""
    return [
        {
            "player_id": 350,
            "gameweek": 20,
            "player_name": "Salah",
            "team_id": 10,
            "position": "MID",
            "predicted_points": Decimal("8.5"),
            "season": "2024_25",
        },
        {
            "player_id": 328,
            "gameweek": 20,
            "player_name": "Haaland",
            "team_id": 11,
            "position": "FWD",
            "predicted_points": Decimal("12.3"),
            "season": "2024_25",
        },
        {
            "player_id": 233,
            "gameweek": 20,
            "player_name": "Saka",
            "team_id": 1,
            "position": "MID",
            "predicted_points": Decimal("6.8"),
            "season": "2024_25",
        },
        {
            "player_id": 412,
            "gameweek": 20,
            "player_name": "Martinez",
            "team_id": 2,
            "position": "GKP",
            "predicted_points": Decimal("5.2"),
            "season": "2024_25",
        },
        {
            "player_id": 567,
            "gameweek": 20,
            "player_name": "Van Dijk",
            "team_id": 10,
            "position": "DEF",
            "predicted_points": Decimal("4.8"),
            "season": "2024_25",
        },
        # GW21 predictions for the same players
        {
            "player_id": 350,
            "gameweek": 21,
            "player_name": "Salah",
            "team_id": 10,
            "position": "MID",
            "predicted_points": Decimal("7.2"),
            "season": "2024_25",
        },
        {
            "player_id": 328,
            "gameweek": 21,
            "player_name": "Haaland",
            "team_id": 11,
            "position": "FWD",
            "predicted_points": Decimal("10.5"),
            "season": "2024_25",
        },
    ]


@pytest.fixture
def dynamodb_with_predictions(sample_predictions):
    """Create DynamoDB table and load sample predictions."""
    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name="ap-southeast-2")

        # Create table with GSIs
        table = dynamodb.create_table(
            TableName="fpl-predictions",
            KeySchema=[
                {"AttributeName": "player_id", "KeyType": "HASH"},
                {"AttributeName": "gameweek", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "player_id", "AttributeType": "N"},
                {"AttributeName": "gameweek", "AttributeType": "N"},
                {"AttributeName": "predicted_points", "AttributeType": "N"},
                {"AttributeName": "position", "AttributeType": "S"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "gameweek-points-index",
                    "KeySchema": [
                        {"AttributeName": "gameweek", "KeyType": "HASH"},
                        {"AttributeName": "predicted_points", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                },
                {
                    "IndexName": "position-points-index",
                    "KeySchema": [
                        {"AttributeName": "position", "KeyType": "HASH"},
                        {"AttributeName": "predicted_points", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                },
            ],
            BillingMode="PROVISIONED",
            ProvisionedThroughput={
                "ReadCapacityUnits": 5,
                "WriteCapacityUnits": 5,
            },
        )

        # Load predictions
        with table.batch_writer() as writer:
            for prediction in sample_predictions:
                writer.put_item(Item=prediction)

        yield table


class TestGetPredictionsEndpoint:
    """Integration tests for GET /predictions endpoint."""

    @mock_aws
    def test_get_predictions_by_gameweek(self, sample_predictions):
        """Test getting all predictions for a gameweek."""
        # Setup
        dynamodb = boto3.resource("dynamodb", region_name="ap-southeast-2")
        table = dynamodb.create_table(
            TableName="fpl-predictions",
            KeySchema=[
                {"AttributeName": "player_id", "KeyType": "HASH"},
                {"AttributeName": "gameweek", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "player_id", "AttributeType": "N"},
                {"AttributeName": "gameweek", "AttributeType": "N"},
                {"AttributeName": "predicted_points", "AttributeType": "N"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "gameweek-points-index",
                    "KeySchema": [
                        {"AttributeName": "gameweek", "KeyType": "HASH"},
                        {"AttributeName": "predicted_points", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                },
            ],
            BillingMode="PROVISIONED",
            ProvisionedThroughput={
                "ReadCapacityUnits": 5,
                "WriteCapacityUnits": 5,
            },
        )

        with table.batch_writer() as writer:
            for prediction in sample_predictions:
                writer.put_item(Item=prediction)

        # Call handler
        event = {
            "httpMethod": "GET",
            "path": "/predictions",
            "resource": "/predictions",
            "queryStringParameters": {"gameweek": "20"},
        }

        response = handler(event, None)
        body = json.loads(response["body"])

        assert response["statusCode"] == 200
        assert body["gameweek"] == 20
        assert body["count"] == 5  # 5 predictions for GW20

    @mock_aws
    def test_predictions_sorted_by_points_descending(self, sample_predictions):
        """Test predictions are sorted by predicted_points descending."""
        dynamodb = boto3.resource("dynamodb", region_name="ap-southeast-2")
        table = dynamodb.create_table(
            TableName="fpl-predictions",
            KeySchema=[
                {"AttributeName": "player_id", "KeyType": "HASH"},
                {"AttributeName": "gameweek", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "player_id", "AttributeType": "N"},
                {"AttributeName": "gameweek", "AttributeType": "N"},
                {"AttributeName": "predicted_points", "AttributeType": "N"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "gameweek-points-index",
                    "KeySchema": [
                        {"AttributeName": "gameweek", "KeyType": "HASH"},
                        {"AttributeName": "predicted_points", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                },
            ],
            BillingMode="PROVISIONED",
            ProvisionedThroughput={
                "ReadCapacityUnits": 5,
                "WriteCapacityUnits": 5,
            },
        )

        with table.batch_writer() as writer:
            for prediction in sample_predictions:
                writer.put_item(Item=prediction)

        event = {
            "httpMethod": "GET",
            "path": "/predictions",
            "resource": "/predictions",
            "queryStringParameters": {"gameweek": "20"},
        }

        response = handler(event, None)
        body = json.loads(response["body"])

        points = [p["predicted_points"] for p in body["predictions"]]
        assert points == sorted(points, reverse=True)


class TestGetPlayerPredictionEndpoint:
    """Integration tests for GET /predictions/{player_id} endpoint."""

    @mock_aws
    def test_get_player_prediction_specific_gameweek(self, sample_predictions):
        """Test getting prediction for specific player and gameweek."""
        dynamodb = boto3.resource("dynamodb", region_name="ap-southeast-2")
        table = dynamodb.create_table(
            TableName="fpl-predictions",
            KeySchema=[
                {"AttributeName": "player_id", "KeyType": "HASH"},
                {"AttributeName": "gameweek", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "player_id", "AttributeType": "N"},
                {"AttributeName": "gameweek", "AttributeType": "N"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        with table.batch_writer() as writer:
            for prediction in sample_predictions:
                writer.put_item(Item=prediction)

        event = {
            "httpMethod": "GET",
            "path": "/predictions/350",
            "resource": "/predictions/{player_id}",
            "pathParameters": {"player_id": "350"},
            "queryStringParameters": {"gameweek": "20"},
        }

        response = handler(event, None)
        body = json.loads(response["body"])

        assert response["statusCode"] == 200
        assert body["player_id"] == 350
        assert body["player_name"] == "Salah"
        assert body["predicted_points"] == 8.5

    @mock_aws
    def test_get_player_all_gameweeks(self, sample_predictions):
        """Test getting all predictions for a player."""
        dynamodb = boto3.resource("dynamodb", region_name="ap-southeast-2")
        table = dynamodb.create_table(
            TableName="fpl-predictions",
            KeySchema=[
                {"AttributeName": "player_id", "KeyType": "HASH"},
                {"AttributeName": "gameweek", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "player_id", "AttributeType": "N"},
                {"AttributeName": "gameweek", "AttributeType": "N"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        with table.batch_writer() as writer:
            for prediction in sample_predictions:
                writer.put_item(Item=prediction)

        event = {
            "httpMethod": "GET",
            "path": "/predictions/350",
            "resource": "/predictions/{player_id}",
            "pathParameters": {"player_id": "350"},
            "queryStringParameters": None,
        }

        response = handler(event, None)
        body = json.loads(response["body"])

        assert response["statusCode"] == 200
        assert body["player_id"] == 350
        assert body["count"] == 2  # GW20 and GW21


class TestTopPredictionsEndpoint:
    """Integration tests for GET /top endpoint."""

    @mock_aws
    def test_get_top_predictions(self, sample_predictions):
        """Test getting top predictions for a gameweek."""
        dynamodb = boto3.resource("dynamodb", region_name="ap-southeast-2")
        table = dynamodb.create_table(
            TableName="fpl-predictions",
            KeySchema=[
                {"AttributeName": "player_id", "KeyType": "HASH"},
                {"AttributeName": "gameweek", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "player_id", "AttributeType": "N"},
                {"AttributeName": "gameweek", "AttributeType": "N"},
                {"AttributeName": "predicted_points", "AttributeType": "N"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "gameweek-points-index",
                    "KeySchema": [
                        {"AttributeName": "gameweek", "KeyType": "HASH"},
                        {"AttributeName": "predicted_points", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                },
            ],
            BillingMode="PROVISIONED",
            ProvisionedThroughput={
                "ReadCapacityUnits": 5,
                "WriteCapacityUnits": 5,
            },
        )

        with table.batch_writer() as writer:
            for prediction in sample_predictions:
                writer.put_item(Item=prediction)

        event = {
            "httpMethod": "GET",
            "path": "/top",
            "resource": "/top",
            "queryStringParameters": {"gameweek": "20", "limit": "3"},
        }

        response = handler(event, None)
        body = json.loads(response["body"])

        assert response["statusCode"] == 200
        assert len(body["predictions"]) == 3
        # Top 3 should be Haaland (12.3), Salah (8.5), Saka (6.8)
        assert body["predictions"][0]["player_name"] == "Haaland"

    @mock_aws
    def test_get_top_by_position(self, sample_predictions):
        """Test getting top predictions filtered by position."""
        dynamodb = boto3.resource("dynamodb", region_name="ap-southeast-2")
        table = dynamodb.create_table(
            TableName="fpl-predictions",
            KeySchema=[
                {"AttributeName": "player_id", "KeyType": "HASH"},
                {"AttributeName": "gameweek", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "player_id", "AttributeType": "N"},
                {"AttributeName": "gameweek", "AttributeType": "N"},
                {"AttributeName": "predicted_points", "AttributeType": "N"},
                {"AttributeName": "position", "AttributeType": "S"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "gameweek-points-index",
                    "KeySchema": [
                        {"AttributeName": "gameweek", "KeyType": "HASH"},
                        {"AttributeName": "predicted_points", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                },
                {
                    "IndexName": "position-points-index",
                    "KeySchema": [
                        {"AttributeName": "position", "KeyType": "HASH"},
                        {"AttributeName": "predicted_points", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                },
            ],
            BillingMode="PROVISIONED",
            ProvisionedThroughput={
                "ReadCapacityUnits": 5,
                "WriteCapacityUnits": 5,
            },
        )

        with table.batch_writer() as writer:
            for prediction in sample_predictions:
                writer.put_item(Item=prediction)

        event = {
            "httpMethod": "GET",
            "path": "/top",
            "resource": "/top",
            "queryStringParameters": {"gameweek": "20", "position": "MID"},
        }

        response = handler(event, None)
        body = json.loads(response["body"])

        assert response["statusCode"] == 200
        # Should only return midfielders
        for pred in body["predictions"]:
            assert pred["position"] == "MID"


class TestComparePlayersEndpoint:
    """Integration tests for GET /compare endpoint."""

    @mock_aws
    def test_compare_multiple_players(self, sample_predictions):
        """Test comparing multiple players."""
        dynamodb = boto3.resource("dynamodb", region_name="ap-southeast-2")
        table = dynamodb.create_table(
            TableName="fpl-predictions",
            KeySchema=[
                {"AttributeName": "player_id", "KeyType": "HASH"},
                {"AttributeName": "gameweek", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "player_id", "AttributeType": "N"},
                {"AttributeName": "gameweek", "AttributeType": "N"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        with table.batch_writer() as writer:
            for prediction in sample_predictions:
                writer.put_item(Item=prediction)

        event = {
            "httpMethod": "GET",
            "path": "/compare",
            "resource": "/compare",
            "queryStringParameters": {
                "players": "350,328,233",
                "gameweek": "20",
            },
        }

        response = handler(event, None)
        body = json.loads(response["body"])

        assert response["statusCode"] == 200
        assert body["found"] == 3
        assert len(body["predictions"]) == 3
        # Sorted by points: Haaland, Salah, Saka
        assert body["predictions"][0]["player_name"] == "Haaland"
        assert body["predictions"][1]["player_name"] == "Salah"
        assert body["predictions"][2]["player_name"] == "Saka"

    @mock_aws
    def test_compare_with_missing_players(self, sample_predictions):
        """Test comparing players when some are not found."""
        dynamodb = boto3.resource("dynamodb", region_name="ap-southeast-2")
        table = dynamodb.create_table(
            TableName="fpl-predictions",
            KeySchema=[
                {"AttributeName": "player_id", "KeyType": "HASH"},
                {"AttributeName": "gameweek", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "player_id", "AttributeType": "N"},
                {"AttributeName": "gameweek", "AttributeType": "N"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        with table.batch_writer() as writer:
            for prediction in sample_predictions:
                writer.put_item(Item=prediction)

        event = {
            "httpMethod": "GET",
            "path": "/compare",
            "resource": "/compare",
            "queryStringParameters": {
                "players": "350,999,888",  # 999 and 888 don't exist
                "gameweek": "20",
            },
        }

        response = handler(event, None)
        body = json.loads(response["body"])

        assert response["statusCode"] == 200
        assert body["found"] == 1
        assert 999 in body["missing"]
        assert 888 in body["missing"]
