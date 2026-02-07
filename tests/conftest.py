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
        resource.create_bucket(
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


# === Feature DataFrame Generator ===


def generate_training_dataframe(n_samples: int = 10):
    """
    Generate a valid training DataFrame with all required feature columns.

    This generates synthetic data for all features defined in FEATURE_COLS,
    making tests resilient to feature changes.

    Args:
        n_samples: Number of sample rows to generate

    Returns:
        DataFrame with all FEATURE_COLS plus actual_points target
    """
    import numpy as np
    import pandas as pd

    from lambdas.common.feature_config import FEATURE_COLS, TARGET_COL

    np.random.seed(42)
    data = {}

    for col in FEATURE_COLS:
        # Generate appropriate random data based on feature name
        if "last_" in col:
            # Rolling features - small positive numbers
            if "transfers_balance" in col:
                data[col] = np.random.randint(-5000, 10000, n_samples).astype(float)
            elif any(
                x in col
                for x in [
                    "goals",
                    "assists",
                    "clean_sheets",
                    "bonus",
                    "yellow_cards",
                    "red_cards",
                    "own_goals",
                    "penalties_saved",
                    "penalties_missed",
                    "starts",
                ]
            ):
                data[col] = np.random.uniform(0, 2, n_samples)
            elif "expected" in col:
                data[col] = np.random.uniform(0, 1.5, n_samples)
            elif "minutes" in col:
                data[col] = np.random.uniform(0, 90, n_samples)
            elif any(
                x in col for x in ["ict_index", "threat", "creativity", "influence"]
            ):
                data[col] = np.random.uniform(0, 100, n_samples)
            elif "bps" in col:
                data[col] = np.random.uniform(0, 40, n_samples)
            elif "saves" in col:
                data[col] = np.random.uniform(0, 5, n_samples)
            else:
                # points_last_* etc
                data[col] = np.random.uniform(0, 15, n_samples)
        elif col == "form_score":
            data[col] = np.random.uniform(0, 10, n_samples)
        elif col == "opponent_strength":
            data[col] = np.random.randint(1, 6, n_samples)
        elif col == "home_away":
            data[col] = np.random.randint(0, 2, n_samples)
        elif col == "chance_of_playing":
            data[col] = np.random.choice([0, 25, 50, 75, 100], n_samples)
        elif col == "position":
            data[col] = np.random.randint(1, 5, n_samples)
        elif col in ["opponent_attack_strength", "opponent_defence_strength"]:
            data[col] = np.random.randint(1000, 1500, n_samples)
        elif col == "selected_by_percent":
            data[col] = np.random.uniform(0, 100, n_samples)
        elif col == "now_cost":
            data[col] = np.random.randint(40, 150, n_samples)
        elif col == "minutes_pct":
            data[col] = np.random.uniform(0, 1, n_samples)
        elif col == "form_x_difficulty":
            data[col] = np.random.uniform(0, 50, n_samples)
        elif col == "points_per_90":
            data[col] = np.random.uniform(0, 10, n_samples)
        elif col == "goal_contributions_last_3":
            data[col] = np.random.uniform(0, 3, n_samples)
        elif col == "points_volatility":
            data[col] = np.random.uniform(0, 5, n_samples)
        # Bootstrap features (Phase 2)
        elif col in ["ep_this", "ep_next", "points_per_game"]:
            data[col] = np.random.uniform(0, 10, n_samples)
        elif col in ["value_form", "value_season"]:
            data[col] = np.random.uniform(0, 2, n_samples)
        elif col.startswith("cost_change"):
            data[col] = np.random.randint(-20, 20, n_samples).astype(float)
        elif col.startswith("status_") or col in [
            "has_news",
            "news_injury_flag",
            "in_dreamteam",
            "set_piece_taker",
        ]:
            data[col] = np.random.randint(0, 2, n_samples).astype(float)
        elif col == "dreamteam_count":
            data[col] = np.random.randint(0, 5, n_samples).astype(float)
        elif col in ["dreamteam_rate", "bonus_rate"]:
            data[col] = np.random.uniform(0, 1, n_samples)
        elif col in ["transfers_in_event", "transfers_out_event"]:
            data[col] = np.random.randint(0, 100000, n_samples).astype(float)
        elif col == "net_transfers_event":
            data[col] = np.random.randint(-50000, 50000, n_samples).astype(float)
        elif col == "transfer_momentum":
            data[col] = np.random.uniform(-1, 1, n_samples)
        elif col == "transfers_in_rank":
            data[col] = np.random.randint(1, 500, n_samples).astype(float)
        elif col == "ownership_change_rate":
            data[col] = np.random.uniform(-5, 5, n_samples)
        elif col.endswith("_order"):
            data[col] = np.random.choice([0, 1, 2, 3, 4, 5], n_samples).astype(float)
        elif col == "total_points_rank_pct":
            data[col] = np.random.uniform(0, 100, n_samples)
        elif col.endswith("_per_90_season"):
            data[col] = np.random.uniform(0, 1, n_samples)
        # Team features (Phase 3)
        elif col.startswith("team_goals_") or col.startswith("team_clean_sheets"):
            data[col] = np.random.uniform(0, 3, n_samples)
        elif col.startswith("team_wins"):
            data[col] = np.random.uniform(0, 1, n_samples)
        elif col == "team_form_score":
            data[col] = np.random.randint(0, 15, n_samples).astype(float)
        elif col == "team_form_trend":
            data[col] = np.random.uniform(-5, 5, n_samples)
        elif col.startswith("team_strength"):
            data[col] = np.random.randint(1000, 1500, n_samples).astype(float)
        elif col in ["team_attack_vs_opp_defence", "team_defence_vs_opp_attack"]:
            data[col] = np.random.uniform(0.8, 1.2, n_samples)
        elif col == "strength_differential":
            data[col] = np.random.uniform(-500, 500, n_samples)
        elif col == "team_league_position":
            data[col] = np.random.randint(1, 21, n_samples).astype(float)
        elif col == "team_points":
            data[col] = np.random.randint(0, 100, n_samples).astype(float)
        elif col == "team_goal_difference":
            data[col] = np.random.randint(-30, 50, n_samples).astype(float)
        elif col == "team_position_change_last_5":
            data[col] = np.random.randint(-5, 5, n_samples).astype(float)
        elif col == "team_total_points_avg":
            data[col] = np.random.uniform(30, 100, n_samples)
        elif col.startswith("player_share_"):
            data[col] = np.random.uniform(0, 0.3, n_samples)
        elif col == "team_avg_ict":
            data[col] = np.random.uniform(50, 150, n_samples)
        elif col == "team_players_available":
            data[col] = np.random.randint(15, 25, n_samples).astype(float)
        elif col == "squad_depth_at_position":
            data[col] = np.random.randint(2, 6, n_samples).astype(float)
        elif col.startswith("player_rank_in_team"):
            data[col] = np.random.randint(1, 15, n_samples).astype(float)
        elif col == "player_minutes_share":
            data[col] = np.random.uniform(0.3, 1.5, n_samples)
        elif col == "player_points_vs_position_avg":
            data[col] = np.random.uniform(-30, 30, n_samples)
        elif col == "games_at_current_team":
            data[col] = np.random.randint(1, 38, n_samples).astype(float)
        # Opponent features (Phase 3)
        elif col.startswith("opp_goals_"):
            data[col] = np.random.uniform(0, 3, n_samples)
        elif col.startswith("opp_clean_sheets"):
            data[col] = np.random.uniform(0, 1, n_samples)
        elif col in ["opp_xgc_per_90", "opp_xg_per_90"]:
            data[col] = np.random.uniform(0.5, 2.5, n_samples)
        elif col == "opp_defensive_errors_last_5":
            data[col] = np.random.randint(0, 3, n_samples).astype(float)
        elif col == "opp_saves_rate":
            data[col] = np.random.uniform(0.5, 0.9, n_samples)
        elif col.startswith("opp_big_chances"):
            data[col] = np.random.uniform(0, 5, n_samples)
        elif col in ["opp_defensive_rating", "opp_attacking_rating"]:
            data[col] = np.random.uniform(0, 15, n_samples)
        elif col == "opp_shots_on_target_last_5":
            data[col] = np.random.uniform(0, 20, n_samples)
        elif col == "opp_league_position":
            data[col] = np.random.randint(1, 21, n_samples).astype(float)
        elif col == "opp_form_score":
            data[col] = np.random.randint(0, 15, n_samples).astype(float)
        elif col == "opp_days_rest":
            data[col] = np.random.randint(2, 10, n_samples).astype(float)
        elif col == "opp_fixture_congestion":
            data[col] = np.random.randint(1, 5, n_samples).astype(float)
        # Fixture features (Phase 4)
        elif col == "fdr_current":
            data[col] = np.random.randint(1, 6, n_samples).astype(float)
        elif col in ["fdr_next_3_avg", "fdr_next_5_avg"]:
            data[col] = np.random.uniform(1, 5, n_samples)
        elif col == "fixture_swing":
            data[col] = np.random.uniform(-3, 3, n_samples)
        elif col in [
            "is_tough_fixture",
            "is_easy_fixture",
            "is_double_gameweek",
            "is_blank_gameweek",
            "next_is_dgw",
            "is_weekend_game",
            "is_evening_kickoff",
        ]:
            data[col] = np.random.randint(0, 2, n_samples).astype(float)
        elif col == "dgw_fixture_count":
            data[col] = np.random.choice([0, 1, 2], n_samples).astype(float)
        elif col == "days_since_last_game":
            data[col] = np.random.randint(3, 10, n_samples).astype(float)
        elif col == "fixture_congestion_7d":
            data[col] = np.random.randint(0, 3, n_samples).astype(float)
        elif col == "fixture_congestion_14d":
            data[col] = np.random.randint(1, 5, n_samples).astype(float)
        elif col == "kickoff_hour":
            data[col] = np.random.choice([12, 14, 15, 17, 20], n_samples).astype(float)
        # Position-specific features (Phase 4)
        elif col.startswith("gk_"):
            data[col] = np.random.uniform(0, 5, n_samples)
        elif col.startswith("def_"):
            data[col] = np.random.uniform(0, 1, n_samples)
        elif col.startswith("mid_"):
            data[col] = np.random.uniform(0, 2, n_samples)
        elif col.startswith("fwd_"):
            data[col] = np.random.uniform(0, 3, n_samples)
        # Interaction features (Phase 4)
        elif col == "form_x_fixture_difficulty":
            data[col] = np.random.uniform(0, 50, n_samples)
        elif col == "ict_x_minutes":
            data[col] = np.random.uniform(0, 100, n_samples)
        elif col == "ownership_x_form":
            data[col] = np.random.uniform(0, 500, n_samples)
        elif col == "value_x_form":
            data[col] = np.random.uniform(0, 100, n_samples)
        elif col == "momentum_score":
            data[col] = np.random.uniform(-10, 10, n_samples)
        else:
            # Default fallback
            data[col] = np.random.uniform(0, 10, n_samples)

    # Add target column
    data[TARGET_COL] = np.random.randint(0, 15, n_samples)

    # Add temporal metadata columns for temporal split support
    seasons = ["2022-23", "2023-24"]
    data["season"] = [seasons[i % len(seasons)] for i in range(n_samples)]
    data["gameweek"] = [(i % 38) + 1 for i in range(n_samples)]

    return pd.DataFrame(data)


@pytest.fixture
def training_dataframe_10():
    """Generate a training DataFrame with 10 samples and all features."""
    return generate_training_dataframe(10)


@pytest.fixture
def training_dataframe_5():
    """Generate a training DataFrame with 5 samples and all features."""
    return generate_training_dataframe(5)


@pytest.fixture
def training_dataframe_100():
    """Generate a training DataFrame with 100 samples and all features."""
    return generate_training_dataframe(100)


@pytest.fixture
def inference_features_df():
    """Generate a features DataFrame for inference testing (no target column)."""
    from lambdas.common.feature_config import TARGET_COL

    df = generate_training_dataframe(3)
    # Remove target and add metadata columns
    df = df.drop(columns=[TARGET_COL])
    df["player_id"] = [100, 200, 300]
    df["player_name"] = ["Salah", "Haaland", "Saka"]
    df["team_id"] = [10, 5, 1]
    df["gameweek"] = [20, 20, 20]
    return df
