"""
Integration tests for SageMaker training with S3.

When AWS_ENDPOINT_URL is set, these tests run against LocalStack.
Otherwise, they use moto mocks.
"""

import io
import json
import os

import boto3
import pandas as pd
import pytest

from sagemaker.train_local import (
    FEATURE_COLS,
    TARGET_COL,
    evaluate_model,
    load_model,
    save_model,
    train_model,
)

# Skip moto-based tests when running against LocalStack
use_localstack = os.environ.get("AWS_ENDPOINT_URL") is not None


@pytest.fixture
def training_features_dataframe():
    """Sample training features DataFrame with all 200 features and actual_points."""
    from tests.conftest import generate_training_dataframe

    df = generate_training_dataframe(10)
    # Add metadata columns for tests that need them
    df["player_id"] = [350, 328, 233, 412, 567, 189, 234, 456, 789, 321]
    df["player_name"] = [
        "Salah",
        "Haaland",
        "Saka",
        "Palmer",
        "Son",
        "Isak",
        "Gordon",
        "Watkins",
        "Solanke",
        "Cunha",
    ]
    df["team_id"] = [10, 11, 1, 4, 17, 14, 14, 2, 3, 20]
    df["gameweek"] = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    return df


@pytest.mark.integration
class TestS3DataLoading:
    """Tests for loading training data from S3."""

    def test_load_parquet_from_s3(
        self, training_features_dataframe, localstack_s3_client, clean_s3_bucket
    ):
        """Test loading Parquet file from S3."""
        s3 = localstack_s3_client
        bucket = clean_s3_bucket

        # Upload Parquet
        buffer = io.BytesIO()
        training_features_dataframe.to_parquet(buffer, engine="pyarrow", index=False)
        buffer.seek(0)

        s3.put_object(
            Bucket=bucket,
            Key="processed/season_2024_25/gw20_features_training.parquet",
            Body=buffer.getvalue(),
        )

        # Download and verify
        response = s3.get_object(
            Bucket=bucket,
            Key="processed/season_2024_25/gw20_features_training.parquet",
        )
        df = pd.read_parquet(io.BytesIO(response["Body"].read()))

        assert len(df) == 10
        assert TARGET_COL in df.columns
        for col in FEATURE_COLS:
            assert col in df.columns

    def test_load_multiple_seasons(
        self, training_features_dataframe, localstack_s3_client, clean_s3_bucket
    ):
        """Test loading training data from multiple seasons."""
        s3 = localstack_s3_client
        bucket = clean_s3_bucket

        # Upload data for two seasons
        seasons = ["2023_24", "2024_25"]
        for season in seasons:
            buffer = io.BytesIO()
            training_features_dataframe.to_parquet(
                buffer, engine="pyarrow", index=False
            )
            buffer.seek(0)

            s3.put_object(
                Bucket=bucket,
                Key=f"processed/season_{season}/gw20_features_training.parquet",
                Body=buffer.getvalue(),
            )

        # List and verify
        response = s3.list_objects_v2(Bucket=bucket, Prefix="processed/")

        assert len(response["Contents"]) == 2


@pytest.mark.integration
class TestS3ModelSaving:
    """Tests for saving models to S3."""

    def test_save_model_to_s3(
        self,
        training_features_dataframe,
        tmp_path,
        localstack_s3_client,
        clean_s3_bucket,
    ):
        """Test saving trained model to S3."""
        s3 = localstack_s3_client
        bucket = clean_s3_bucket

        # Train model locally
        model, _, _ = train_model(training_features_dataframe)

        # Save locally first
        local_path = save_model(model, str(tmp_path))

        # Upload to S3
        with open(local_path, "rb") as f:
            s3.put_object(
                Bucket=bucket,
                Key="models/season_2024_25/model.xgb",
                Body=f.read(),
            )

        # Verify upload
        response = s3.head_object(Bucket=bucket, Key="models/season_2024_25/model.xgb")

        assert response["ContentLength"] > 0

    def test_load_model_from_s3(
        self,
        training_features_dataframe,
        tmp_path,
        localstack_s3_client,
        clean_s3_bucket,
    ):
        """Test loading model from S3 and making predictions."""
        s3 = localstack_s3_client
        bucket = clean_s3_bucket

        # Train and save locally
        model, X_test, y_test = train_model(training_features_dataframe)
        local_path = save_model(model, str(tmp_path))

        # Upload to S3
        with open(local_path, "rb") as f:
            s3.put_object(
                Bucket=bucket,
                Key="models/season_2024_25/model.xgb",
                Body=f.read(),
            )

        # Download from S3
        response = s3.get_object(Bucket=bucket, Key="models/season_2024_25/model.xgb")

        download_path = tmp_path / "downloaded_model.xgb"
        with open(download_path, "wb") as f:
            f.write(response["Body"].read())

        # Load and predict
        import numpy as np

        loaded_model = load_model(str(download_path))
        predictions = loaded_model.predict(X_test)

        assert len(predictions) == len(X_test)
        assert all(np.issubdtype(type(p), np.number) for p in predictions)


@pytest.mark.integration
class TestEndToEndTraining:
    """End-to-end training pipeline tests."""

    def test_full_training_pipeline(
        self,
        training_features_dataframe,
        tmp_path,
        localstack_s3_client,
        clean_s3_bucket,
    ):
        """Test complete training pipeline: load -> train -> evaluate -> save."""
        s3 = localstack_s3_client
        bucket = clean_s3_bucket

        # Upload training data
        buffer = io.BytesIO()
        training_features_dataframe.to_parquet(buffer, engine="pyarrow", index=False)
        buffer.seek(0)

        s3.put_object(
            Bucket=bucket,
            Key="processed/season_2024_25/gw20_features_training.parquet",
            Body=buffer.getvalue(),
        )

        # Download training data
        response = s3.get_object(
            Bucket=bucket,
            Key="processed/season_2024_25/gw20_features_training.parquet",
        )
        df = pd.read_parquet(io.BytesIO(response["Body"].read()))

        # Train model
        model, X_test, y_test = train_model(df, test_size=0.2)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert metrics["mae"] >= 0

        # Save model locally
        model_path = save_model(model, str(tmp_path))

        # Upload model to S3
        with open(model_path, "rb") as f:
            s3.put_object(
                Bucket=bucket,
                Key="models/season_2024_25/model.xgb",
                Body=f.read(),
            )

        # Upload metrics to S3
        s3.put_object(
            Bucket=bucket,
            Key="models/season_2024_25/metrics.json",
            Body=json.dumps(metrics),
        )

        # Verify all artifacts exist
        objects = s3.list_objects_v2(Bucket=bucket, Prefix="models/")

        keys = [obj["Key"] for obj in objects["Contents"]]
        assert "models/season_2024_25/model.xgb" in keys
        assert "models/season_2024_25/metrics.json" in keys

    def test_cross_season_training(
        self,
        training_features_dataframe,
        tmp_path,
        localstack_s3_client,
        clean_s3_bucket,
    ):
        """Test training on data from multiple seasons."""
        s3 = localstack_s3_client
        bucket = clean_s3_bucket

        # Upload data for multiple gameweeks across seasons
        seasons_gws = [
            ("2023_24", 38),
            ("2023_24", 37),
            ("2024_25", 1),
            ("2024_25", 2),
        ]

        for season, gw in seasons_gws:
            buffer = io.BytesIO()
            training_features_dataframe.to_parquet(
                buffer, engine="pyarrow", index=False
            )
            buffer.seek(0)

            s3.put_object(
                Bucket=bucket,
                Key=f"processed/season_{season}/gw{gw}_features_training.parquet",
                Body=buffer.getvalue(),
            )

        # Combine all training data
        all_dfs = []
        response = s3.list_objects_v2(Bucket=bucket, Prefix="processed/")

        for obj in response["Contents"]:
            data = s3.get_object(Bucket=bucket, Key=obj["Key"])
            df = pd.read_parquet(io.BytesIO(data["Body"].read()))
            all_dfs.append(df)

        combined_df = pd.concat(all_dfs, ignore_index=True)

        assert len(combined_df) == 40  # 10 players x 4 gameweeks

        # Train on combined data
        model, X_test, y_test = train_model(combined_df)
        metrics = evaluate_model(model, X_test, y_test)

        assert metrics["mae"] >= 0


@pytest.mark.integration
class TestPredictionSaving:
    """Tests for saving predictions to S3."""

    def test_save_predictions_parquet(
        self,
        training_features_dataframe,
        tmp_path,
        localstack_s3_client,
        clean_s3_bucket,
    ):
        """Test saving predictions as Parquet to S3."""
        s3 = localstack_s3_client
        bucket = clean_s3_bucket

        # Train model
        model, X_test, y_test = train_model(training_features_dataframe)

        # Make predictions on all data
        X_all = training_features_dataframe[FEATURE_COLS]
        predictions = model.predict(X_all)

        # Create predictions DataFrame
        pred_df = training_features_dataframe[
            ["player_id", "player_name", "gameweek"]
        ].copy()
        pred_df["predicted_points"] = predictions

        # Upload predictions
        buffer = io.BytesIO()
        pred_df.to_parquet(buffer, engine="pyarrow", index=False)
        buffer.seek(0)

        s3.put_object(
            Bucket=bucket,
            Key="predictions/season_2024_25/gw21_predictions.parquet",
            Body=buffer.getvalue(),
        )

        # Download and verify
        response = s3.get_object(
            Bucket=bucket,
            Key="predictions/season_2024_25/gw21_predictions.parquet",
        )
        loaded_pred_df = pd.read_parquet(io.BytesIO(response["Body"].read()))

        assert len(loaded_pred_df) == 10
        assert "predicted_points" in loaded_pred_df.columns
        assert "player_id" in loaded_pred_df.columns
