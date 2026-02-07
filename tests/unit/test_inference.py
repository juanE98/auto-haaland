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
    TEAM_DIVERSITY_MAX,
    TEAM_DIVERSITY_PENALTY,
    apply_team_diversification,
    handler,
    load_features_from_s3,
    load_model_from_s3,
    run_inference,
    save_predictions_to_s3,
    validate_features,
)


@pytest.fixture
def trained_model(tmp_path, training_dataframe_100):
    """Create a simple trained XGBoost model for testing."""
    from lambdas.common.feature_config import TARGET_COL

    np.random.seed(42)
    X = training_dataframe_100[FEATURE_COLS]
    y = training_dataframe_100[TARGET_COL]

    model = xgb.XGBRegressor(
        n_estimators=10,
        max_depth=3,
        random_state=42,
    )
    model.fit(X, y)

    model_path = tmp_path / "model.xgb"
    model.save_model(str(model_path))

    return model, str(model_path)


# === Feature Validation ===


@pytest.mark.unit
class TestValidateFeatures:
    def test_valid_features(self, inference_features_df):
        """No exception for valid features."""
        validate_features(inference_features_df)

    def test_missing_single_feature(self, inference_features_df):
        df = inference_features_df.drop(columns=["form_score"])
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
    def test_load_features_success(self, inference_features_df):
        """Features should load correctly from S3 Parquet."""
        import boto3

        s3 = boto3.client("s3", region_name="ap-southeast-2")
        s3.create_bucket(
            Bucket="test-bucket",
            CreateBucketConfiguration={"LocationConstraint": "ap-southeast-2"},
        )

        # Upload features as Parquet
        buffer = io.BytesIO()
        inference_features_df.to_parquet(buffer, engine="pyarrow", index=False)
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
    def test_predictions_output_schema(self, trained_model, inference_features_df):
        """Predictions should have the correct schema."""
        model, _ = trained_model
        result = run_inference(model, inference_features_df, 20, "2024_25")

        expected_cols = {
            "player_id",
            "gameweek",
            "predicted_points",
            "position",
            "player_name",
            "team_id",
            "season",
            "chance_of_playing",
            "haul_probability",
        }
        assert set(result.columns) == expected_cols

    def test_predictions_count_matches_input(
        self, trained_model, inference_features_df
    ):
        """One prediction per input row."""
        model, _ = trained_model
        result = run_inference(model, inference_features_df, 20, "2024_25")
        assert len(result) == len(inference_features_df)

    def test_position_mapping(self, trained_model, inference_features_df):
        """Element types should be mapped to position strings."""
        model, _ = trained_model
        # Set specific positions for testing
        inference_features_df["position"] = [3, 4, 3]  # MID, FWD, MID
        result = run_inference(model, inference_features_df, 20, "2024_25")

        positions = result["position"].tolist()
        assert positions == ["MID", "FWD", "MID"]

    def test_gameweek_and_season_set(self, trained_model, inference_features_df):
        """Gameweek and season should be set on all rows."""
        model, _ = trained_model
        result = run_inference(model, inference_features_df, 20, "2024_25")

        assert all(result["gameweek"] == 20)
        assert all(result["season"] == "2024_25")

    def test_predicted_points_are_numeric(self, trained_model, inference_features_df):
        """Predictions should be numeric values."""
        model, _ = trained_model
        result = run_inference(model, inference_features_df, 20, "2024_25")
        assert result["predicted_points"].dtype in [np.float64, np.float32]


# === Predictions Saving ===


@pytest.mark.unit
class TestSavePredictionsToS3:
    @mock_aws
    def test_save_and_verify(self, trained_model, inference_features_df):
        """Predictions should be saved to S3 as Parquet."""
        import boto3

        model, _ = trained_model
        predictions_df = run_inference(model, inference_features_df, 20, "2024_25")

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
    def test_handler_success(self, trained_model, inference_features_df):
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
        inference_features_df.to_parquet(buffer, engine="pyarrow", index=False)
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
    def test_handler_model_not_found(self, inference_features_df):
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


# === Team Diversification ===


@pytest.mark.unit
class TestTeamDiversification:
    def test_penalty_applied_beyond_cap(self):
        """4th and 5th players from the same team should get 0.85x penalty."""
        df = pd.DataFrame(
            {
                "player_id": [1, 2, 3, 4, 5],
                "team_id": [10, 10, 10, 10, 10],
                "predicted_points": [5.0, 4.0, 3.0, 2.0, 1.0],
            }
        )
        result = apply_team_diversification(df)

        # Top 3 keep full score
        assert result.loc[0, "predicted_points"] == 5.0
        assert result.loc[1, "predicted_points"] == 4.0
        assert result.loc[2, "predicted_points"] == 3.0
        # 4th and 5th get penalty
        assert result.loc[3, "predicted_points"] == pytest.approx(
            round(2.0 * TEAM_DIVERSITY_PENALTY, 2)
        )
        assert result.loc[4, "predicted_points"] == pytest.approx(
            round(1.0 * TEAM_DIVERSITY_PENALTY, 2)
        )

    def test_no_penalty_for_small_teams(self):
        """Teams with <= 3 players should keep full scores."""
        df = pd.DataFrame(
            {
                "player_id": [1, 2, 3],
                "team_id": [10, 10, 10],
                "predicted_points": [5.0, 4.0, 3.0],
            }
        )
        result = apply_team_diversification(df)

        assert result["predicted_points"].tolist() == [5.0, 4.0, 3.0]

    def test_diversification_across_multiple_teams(self):
        """Penalty is applied independently per team."""
        df = pd.DataFrame(
            {
                "player_id": [1, 2, 3, 4, 5, 6, 7, 8],
                "team_id": [10, 10, 10, 10, 20, 20, 20, 20],
                "predicted_points": [5.0, 4.0, 3.0, 2.0, 6.0, 5.0, 4.0, 3.0],
            }
        )
        result = apply_team_diversification(df)

        # Team 10: 4th player penalised
        assert result.loc[0, "predicted_points"] == 5.0
        assert result.loc[3, "predicted_points"] == pytest.approx(
            round(2.0 * TEAM_DIVERSITY_PENALTY, 2)
        )
        # Team 20: 4th player penalised
        assert result.loc[4, "predicted_points"] == 6.0
        assert result.loc[7, "predicted_points"] == pytest.approx(
            round(3.0 * TEAM_DIVERSITY_PENALTY, 2)
        )
