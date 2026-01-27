"""
Unit tests for the inference Lambda handler.
"""

import io
import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from moto import mock_aws

from lambdas.inference.handler import (
    FEATURE_COLS,
    POSITION_MAP,
    handler,
    load_features_from_s3,
    load_model_from_s3,
    run_inference,
    save_predictions_to_s3,
    validate_features,
)


@pytest.fixture
def trained_model(tmp_path):
    """Create a simple trained XGBoost model for testing."""
    np.random.seed(42)
    n_samples = 100
    X = pd.DataFrame({col: np.random.rand(n_samples) for col in FEATURE_COLS})
    y = np.random.rand(n_samples) * 10  # Points 0-10

    model = xgb.XGBRegressor(
        n_estimators=10,
        max_depth=3,
        random_state=42,
    )
    model.fit(X, y)

    model_path = tmp_path / "model.xgb"
    model.save_model(str(model_path))

    return model, str(model_path)


@pytest.fixture
def sample_features_df():
    """Sample features DataFrame for inference."""
    return pd.DataFrame(
        {
            "player_id": [100, 200, 300],
            "player_name": ["Salah", "Haaland", "Saka"],
            "team_id": [10, 5, 1],
            "position": [3, 4, 3],
            "gameweek": [20, 20, 20],
            "points_last_3": [7.3, 5.2, 8.0],
            "points_last_5": [6.8, 4.9, 7.5],
            "minutes_pct": [0.95, 0.88, 1.0],
            "form_score": [8.5, 5.8, 7.9],
            "opponent_strength": [3, 4, 2],
            "home_away": [1, 0, 1],
            "chance_of_playing": [100, 75, 100],
            "form_x_difficulty": [25.5, 23.2, 15.8],
            "goals_last_3": [0.67, 1.0, 0.33],
            "assists_last_3": [0.33, 0.0, 0.67],
            "clean_sheets_last_3": [0.0, 0.0, 0.33],
            "bps_last_3": [28.0, 32.0, 25.0],
            "ict_index_last_3": [85.3, 92.1, 78.5],
            "threat_last_3": [45.0, 60.0, 38.0],
            "creativity_last_3": [55.0, 30.0, 62.0],
            "opponent_attack_strength": [1200, 1350, 1100],
            "opponent_defence_strength": [1250, 1300, 1150],
            "selected_by_percent": [45.3, 52.1, 38.7],
        }
    )


# === Feature Validation ===


@pytest.mark.unit
class TestValidateFeatures:
    def test_valid_features(self, sample_features_df):
        """No exception for valid features."""
        validate_features(sample_features_df)

    def test_missing_single_feature(self, sample_features_df):
        df = sample_features_df.drop(columns=["form_score"])
        with pytest.raises(ValueError, match="Missing feature columns"):
            validate_features(df)

    def test_missing_multiple_features(self):
        df = pd.DataFrame({"player_id": [1], "points_last_3": [5.0]})
        with pytest.raises(ValueError, match="Missing feature columns"):
            validate_features(df)

    def test_empty_dataframe_with_correct_columns(self):
        df = pd.DataFrame(columns=FEATURE_COLS)
        validate_features(df)  # Should not raise


# === Model Loading ===


@pytest.mark.unit
class TestLoadModelFromS3:
    @mock_aws
    def test_load_model_success(self, trained_model):
        """Model should load successfully from S3."""
        import boto3

        model, model_path = trained_model

        # Set up mock S3
        s3 = boto3.client("s3", region_name="ap-southeast-2")
        s3.create_bucket(
            Bucket="test-bucket",
            CreateBucketConfiguration={"LocationConstraint": "ap-southeast-2"},
        )
        s3.upload_file(model_path, "test-bucket", "models/model.xgb")

        # Clear cache
        import lambdas.inference.handler as handler_module

        handler_module._cached_model = None
        handler_module._cached_model_key = None

        loaded = load_model_from_s3(s3, "test-bucket", "models/model.xgb")
        assert isinstance(loaded, xgb.XGBRegressor)

    @mock_aws
    def test_model_not_found_raises(self):
        """Missing model should raise FileNotFoundError."""
        import boto3

        s3 = boto3.client("s3", region_name="ap-southeast-2")
        s3.create_bucket(
            Bucket="test-bucket",
            CreateBucketConfiguration={"LocationConstraint": "ap-southeast-2"},
        )

        # Clear cache
        import lambdas.inference.handler as handler_module

        handler_module._cached_model = None
        handler_module._cached_model_key = None

        with pytest.raises(FileNotFoundError, match="Model not found"):
            load_model_from_s3(s3, "test-bucket", "models/nonexistent.xgb")

    def test_cached_model_returned(self):
        """Cached model should be returned without hitting S3."""
        import lambdas.inference.handler as handler_module

        fake_model = xgb.XGBRegressor()
        handler_module._cached_model = fake_model
        handler_module._cached_model_key = "models/model.xgb"

        result = load_model_from_s3(None, "any-bucket", "models/model.xgb")
        assert result is fake_model

        # Clean up
        handler_module._cached_model = None
        handler_module._cached_model_key = None


# === Features Loading ===


@pytest.mark.unit
class TestLoadFeaturesFromS3:
    @mock_aws
    def test_load_features_success(self, sample_features_df):
        """Features should load correctly from S3 Parquet."""
        import boto3

        s3 = boto3.client("s3", region_name="ap-southeast-2")
        s3.create_bucket(
            Bucket="test-bucket",
            CreateBucketConfiguration={"LocationConstraint": "ap-southeast-2"},
        )

        # Upload features as Parquet
        buffer = io.BytesIO()
        sample_features_df.to_parquet(buffer, engine="pyarrow", index=False)
        buffer.seek(0)
        s3.put_object(
            Bucket="test-bucket",
            Key="features.parquet",
            Body=buffer.getvalue(),
        )

        result = load_features_from_s3(s3, "test-bucket", "features.parquet")
        assert len(result) == 3
        assert "player_id" in result.columns

    @mock_aws
    def test_features_not_found_raises(self):
        """Missing features file should raise FileNotFoundError."""
        import boto3

        s3 = boto3.client("s3", region_name="ap-southeast-2")
        s3.create_bucket(
            Bucket="test-bucket",
            CreateBucketConfiguration={"LocationConstraint": "ap-southeast-2"},
        )

        with pytest.raises(FileNotFoundError, match="Features not found"):
            load_features_from_s3(s3, "test-bucket", "missing.parquet")


# === Inference ===


@pytest.mark.unit
class TestRunInference:
    def test_predictions_output_schema(self, trained_model, sample_features_df):
        """Predictions should have the correct schema."""
        model, _ = trained_model
        result = run_inference(model, sample_features_df, 20, "2024_25")

        expected_cols = {
            "player_id",
            "gameweek",
            "predicted_points",
            "position",
            "player_name",
            "team_id",
            "season",
        }
        assert set(result.columns) == expected_cols

    def test_predictions_count_matches_input(self, trained_model, sample_features_df):
        """One prediction per input row."""
        model, _ = trained_model
        result = run_inference(model, sample_features_df, 20, "2024_25")
        assert len(result) == len(sample_features_df)

    def test_position_mapping(self, trained_model, sample_features_df):
        """Element types should be mapped to position strings."""
        model, _ = trained_model
        result = run_inference(model, sample_features_df, 20, "2024_25")

        positions = result["position"].tolist()
        # Player 100: element_type=3 -> MID, Player 200: 4 -> FWD, Player 300: 3 -> MID
        assert positions == ["MID", "FWD", "MID"]

    def test_gameweek_and_season_set(self, trained_model, sample_features_df):
        """Gameweek and season should be set on all rows."""
        model, _ = trained_model
        result = run_inference(model, sample_features_df, 20, "2024_25")

        assert all(result["gameweek"] == 20)
        assert all(result["season"] == "2024_25")

    def test_predicted_points_are_numeric(self, trained_model, sample_features_df):
        """Predictions should be numeric values."""
        model, _ = trained_model
        result = run_inference(model, sample_features_df, 20, "2024_25")
        assert result["predicted_points"].dtype in [np.float64, np.float32]


# === Predictions Saving ===


@pytest.mark.unit
class TestSavePredictionsToS3:
    @mock_aws
    def test_save_and_verify(self, trained_model, sample_features_df):
        """Predictions should be saved to S3 as Parquet."""
        import boto3

        model, _ = trained_model
        predictions_df = run_inference(model, sample_features_df, 20, "2024_25")

        s3 = boto3.client("s3", region_name="ap-southeast-2")
        s3.create_bucket(
            Bucket="test-bucket",
            CreateBucketConfiguration={"LocationConstraint": "ap-southeast-2"},
        )

        key = save_predictions_to_s3(
            s3, predictions_df, "test-bucket", "predictions/test.parquet"
        )
        assert key == "predictions/test.parquet"

        # Verify the file exists and is valid Parquet
        response = s3.get_object(Bucket="test-bucket", Key=key)
        loaded = pd.read_parquet(io.BytesIO(response["Body"].read()))
        assert len(loaded) == 3


# === Handler Integration ===


@pytest.mark.unit
class TestHandler:
    def test_missing_gameweek_raises(self):
        """Handler should raise ValueError for missing gameweek."""
        with pytest.raises(ValueError, match="gameweek"):
            handler({"season": "2024_25"}, None)

    def test_missing_season_raises(self):
        """Handler should raise ValueError for missing season."""
        with pytest.raises(ValueError, match="season"):
            handler({"gameweek": 20}, None)

    @mock_aws
    def test_handler_success(self, trained_model, sample_features_df):
        """Full handler should run successfully end-to-end."""
        import boto3

        import lambdas.inference.handler as handler_module

        model, model_path = trained_model

        # Set up mock S3
        s3 = boto3.client("s3", region_name="ap-southeast-2")
        s3.create_bucket(
            Bucket="fpl-ml-data",
            CreateBucketConfiguration={"LocationConstraint": "ap-southeast-2"},
        )

        # Upload model
        s3.upload_file(model_path, "fpl-ml-data", "models/model.xgb")

        # Upload features
        buffer = io.BytesIO()
        sample_features_df.to_parquet(buffer, engine="pyarrow", index=False)
        buffer.seek(0)
        s3.put_object(
            Bucket="fpl-ml-data",
            Key="processed/season_2024_25/gw20_features_prediction.parquet",
            Body=buffer.getvalue(),
        )

        # Clear model cache
        handler_module._cached_model = None
        handler_module._cached_model_key = None

        # Patch get_s3_client to return our mock
        with patch.object(handler_module, "get_s3_client", return_value=s3):
            result = handler(
                {"gameweek": 20, "season": "2024_25"},
                None,
            )

        assert result["gameweek"] == 20
        assert result["season"] == "2024_25"
        assert result["predictions_count"] == 3
        assert "predictions_key" in result
        assert "timestamp" in result

    @mock_aws
    def test_handler_model_not_found(self, sample_features_df):
        """Handler should raise when model is missing."""
        import boto3

        import lambdas.inference.handler as handler_module

        s3 = boto3.client("s3", region_name="ap-southeast-2")
        s3.create_bucket(
            Bucket="fpl-ml-data",
            CreateBucketConfiguration={"LocationConstraint": "ap-southeast-2"},
        )

        # Clear cache
        handler_module._cached_model = None
        handler_module._cached_model_key = None

        with patch.object(handler_module, "get_s3_client", return_value=s3):
            with pytest.raises(FileNotFoundError, match="Model not found"):
                handler(
                    {"gameweek": 20, "season": "2024_25"},
                    None,
                )
