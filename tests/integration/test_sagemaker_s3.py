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
    """Sample training features DataFrame with actual_points."""
    return pd.DataFrame(
        {
            "player_id": [350, 328, 233, 412, 567, 189, 234, 456, 789, 321],
            "player_name": [
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
            ],
            "team_id": [10, 11, 1, 4, 17, 14, 14, 2, 3, 20],
            "position": [3, 4, 3, 3, 3, 4, 3, 4, 4, 4],
            "gameweek": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
            "points_last_3": [7.3, 9.2, 8.0, 6.5, 5.8, 7.1, 4.5, 6.2, 5.0, 4.8],
            "points_last_5": [6.8, 8.9, 7.5, 6.2, 5.5, 6.8, 4.2, 5.9, 4.8, 4.5],
            "minutes_pct": [0.95, 1.0, 0.92, 0.88, 0.85, 0.90, 0.78, 0.95, 0.88, 0.82],
            "form_score": [8.5, 9.8, 7.9, 6.8, 5.5, 7.2, 4.8, 6.5, 5.2, 4.9],
            "opponent_strength": [3, 2, 4, 3, 5, 3, 2, 4, 3, 4],
            "home_away": [1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            "chance_of_playing": [100, 100, 100, 100, 75, 100, 100, 100, 50, 100],
            "form_x_difficulty": [
                25.5,
                19.6,
                31.6,
                20.4,
                27.5,
                21.6,
                9.6,
                26.0,
                15.6,
                19.6,
            ],
            "position": [3, 4, 3, 3, 3, 4, 3, 4, 4, 4],
            "goals_last_3": [0.67, 1.33, 0.33, 0.67, 0.33, 1.0, 0.0, 0.67, 0.33, 0.33],
            "assists_last_3": [0.33, 0.0, 0.67, 0.33, 0.33, 0.0, 0.33, 0.0, 0.0, 0.33],
            "clean_sheets_last_3": [0.0, 0.0, 0.33, 0.0, 0.0, 0.0, 0.33, 0.0, 0.0, 0.0],
            "bps_last_3": [28.0, 35.0, 25.0, 22.0, 20.0, 30.0, 18.0, 24.0, 19.0, 17.0],
            "actual_points": [8, 15, 6, 9, 2, 11, 5, 4, 7, 3],
        }
    )


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
