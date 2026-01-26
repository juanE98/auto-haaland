"""
Shared pytest fixtures for Auto-Haaland FPL ML System tests.
"""

import os

import boto3
import pytest
from moto import mock_aws

# === Environment Setup ===


@pytest.fixture(scope="session", autouse=True)
def aws_credentials():
    """Set up fake AWS credentials for testing."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "ap-southeast-2"


# === S3 Fixtures ===


@pytest.fixture
def s3_client(aws_credentials):
    """Create a mocked S3 client with the fpl-ml-data bucket."""
    with mock_aws():
        client = boto3.client("s3", region_name="ap-southeast-2")
        client.create_bucket(
            Bucket="fpl-ml-data",
            CreateBucketConfiguration={"LocationConstraint": "ap-southeast-2"},
        )
        yield client


@pytest.fixture
def s3_resource(aws_credentials):
    """Create a mocked S3 resource with the fpl-ml-data bucket."""
    with mock_aws():
        resource = boto3.resource("s3", region_name="ap-southeast-2")
        bucket = resource.create_bucket(
            Bucket="fpl-ml-data",
            CreateBucketConfiguration={"LocationConstraint": "ap-southeast-2"},
        )
        yield resource


# === Lambda Context Fixture ===


class MockLambdaContext:
    """Mock AWS Lambda context for testing."""

    function_name = "test-function"
    memory_limit_in_mb = 128
    invoked_function_arn = "arn:aws:lambda:ap-southeast-2:123456789:function:test"
    aws_request_id = "test-request-id"


@pytest.fixture
def lambda_context():
    """Provide a mock Lambda context for handler tests."""
    return MockLambdaContext()


# === DynamoDB Fixtures ===


@pytest.fixture
def dynamodb_client(aws_credentials):
    """Create a mocked DynamoDB client."""
    with mock_aws():
        client = boto3.client("dynamodb", region_name="ap-southeast-2")
        yield client


@pytest.fixture
def dynamodb_table(aws_credentials):
    """Create a mocked DynamoDB table for predictions."""
    with mock_aws():
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
            ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
        )
        yield table


# === Sample Data Fixtures ===


@pytest.fixture
def sample_fpl_bootstrap_data():
    """Sample FPL bootstrap-static API response."""
    return {
        "events": [
            {
                "id": 1,
                "name": "Gameweek 1",
                "deadline_time": "2024-08-16T17:30:00Z",
                "finished": True,
            }
        ],
        "teams": [{"id": 1, "name": "Arsenal", "short_name": "ARS", "strength": 4}],
        "elements": [
            {
                "id": 350,
                "web_name": "Salah",
                "first_name": "Mohamed",
                "second_name": "Salah",
                "team": 10,
                "element_type": 3,
                "now_cost": 130,
                "form": "8.5",
                "points_per_game": "7.2",
                "selected_by_percent": "45.3",
                "chance_of_playing_next_round": 100,
                "minutes": 90,
            }
        ],
    }


@pytest.fixture
def sample_player_history():
    """Sample player history data."""
    return [
        {
            "element": 350,
            "fixture": 1,
            "opponent_team": 5,
            "total_points": 8,
            "was_home": True,
            "kickoff_time": "2024-08-17T11:30:00Z",
            "minutes": 90,
            "goals_scored": 1,
            "assists": 1,
            "clean_sheets": 0,
        },
        {
            "element": 350,
            "fixture": 2,
            "opponent_team": 12,
            "total_points": 12,
            "was_home": False,
            "kickoff_time": "2024-08-24T11:30:00Z",
            "minutes": 90,
            "goals_scored": 2,
            "assists": 0,
            "clean_sheets": 0,
        },
    ]


@pytest.fixture
def sample_features_dataframe():
    """Sample engineered features DataFrame."""
    import pandas as pd

    return pd.DataFrame(
        {
            "player_id": [350, 328, 233],
            "gameweek": [20, 20, 20],
            "points_last_3": [7.3, 5.2, 8.0],
            "points_last_5": [6.8, 4.9, 7.5],
            "minutes_pct": [0.95, 0.88, 1.0],
            "form_score": [8.5, 5.8, 7.9],
            "opponent_strength": [3, 4, 2],
            "home_away": [1, 0, 1],
            "chance_of_playing": [100, 75, 100],
            "form_x_difficulty": [25.5, 23.2, 15.8],
        }
    )
