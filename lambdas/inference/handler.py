"""
Inference Lambda Handler

Loads a trained XGBoost model from S3, runs predictions on feature data,
and saves prediction results as Parquet to S3 for the prediction loader.
"""

import io
import logging
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import xgboost as xgb
from botocore.exceptions import ClientError

from common.aws_clients import get_s3_client

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variables
BUCKET_NAME = os.getenv("BUCKET_NAME", "fpl-ml-data")
MODEL_KEY = os.getenv("MODEL_KEY", "models/model.xgb")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")  # For LocalStack

# Feature columns (must match training)
FEATURE_COLS = [
    "points_last_3",
    "points_last_5",
    "minutes_pct",
    "form_score",
    "opponent_strength",
    "home_away",
    "chance_of_playing",
    "form_x_difficulty",
    "position",
    "goals_last_3",
    "assists_last_3",
    "clean_sheets_last_3",
    "bps_last_3",
]

# Position mapping (element_type to string)
POSITION_MAP = {
    1: "GKP",
    2: "DEF",
    3: "MID",
    4: "FWD",
}

# Cache model across warm Lambda invocations
_cached_model = None
_cached_model_key = None


def load_model_from_s3(
    s3_client,
    bucket: str,
    model_key: str,
) -> xgb.XGBRegressor:
    """
    Load XGBoost model from S3, caching in /tmp for warm invocations.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        model_key: S3 key for the model file

    Returns:
        Loaded XGBoost model

    Raises:
        FileNotFoundError: If model file not found in S3
        RuntimeError: If model fails to load
    """
    global _cached_model, _cached_model_key

    # Return cached model if same key
    if _cached_model is not None and _cached_model_key == model_key:
        logger.info("Using cached model")
        return _cached_model

    local_path = "/tmp/model.xgb"

    try:
        logger.info(f"Downloading model from s3://{bucket}/{model_key}")
        s3_client.download_file(bucket, model_key, local_path)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("404", "NoSuchKey"):
            raise FileNotFoundError(
                f"Model not found: s3://{bucket}/{model_key}"
            ) from e
        raise

    try:
        model = xgb.XGBRegressor()
        model.load_model(local_path)
        logger.info("Model loaded successfully")

        # Cache for warm invocations
        _cached_model = model
        _cached_model_key = model_key

        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {local_path}: {e}") from e


def load_features_from_s3(
    s3_client,
    bucket: str,
    features_key: str,
) -> pd.DataFrame:
    """
    Load feature Parquet file from S3.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        features_key: S3 key for the features file

    Returns:
        DataFrame with features

    Raises:
        FileNotFoundError: If features file not found
    """
    try:
        logger.info(f"Loading features from s3://{bucket}/{features_key}")
        response = s3_client.get_object(Bucket=bucket, Key=features_key)
        df = pd.read_parquet(io.BytesIO(response["Body"].read()))
        logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
        return df
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("NoSuchKey",):
            raise FileNotFoundError(
                f"Features not found: s3://{bucket}/{features_key}"
            ) from e
        raise


def validate_features(df: pd.DataFrame) -> None:
    """
    Validate that the DataFrame contains all required feature columns.

    Args:
        df: Features DataFrame

    Raises:
        ValueError: If required columns are missing
    """
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")


def run_inference(
    model: xgb.XGBRegressor,
    features_df: pd.DataFrame,
    gameweek: int,
    season: str,
) -> pd.DataFrame:
    """
    Run model predictions on features.

    Args:
        model: Trained XGBoost model
        features_df: DataFrame with feature columns
        gameweek: Current gameweek
        season: Season string

    Returns:
        Predictions DataFrame with schema:
        player_id, gameweek, predicted_points, position, player_name, team_id, season
    """
    X = features_df[FEATURE_COLS]
    predictions = model.predict(X)

    results = pd.DataFrame(
        {
            "player_id": features_df["player_id"].astype(int),
            "gameweek": gameweek,
            "predicted_points": predictions.round(2),
            "position": features_df["position"].map(
                lambda x: POSITION_MAP.get(int(x), "UNK") if pd.notna(x) else "UNK"
            ),
            "player_name": features_df.get(
                "player_name", pd.Series([""] * len(features_df))
            ),
            "team_id": features_df.get(
                "team_id", pd.Series([0] * len(features_df))
            ).astype(int),
            "season": season,
        }
    )

    logger.info(
        f"Generated {len(results)} predictions. "
        f"Mean: {results['predicted_points'].mean():.2f}, "
        f"Max: {results['predicted_points'].max():.2f}"
    )

    return results


def save_predictions_to_s3(
    s3_client,
    predictions_df: pd.DataFrame,
    bucket: str,
    predictions_key: str,
) -> str:
    """
    Save predictions DataFrame to S3 as Parquet.

    Args:
        s3_client: boto3 S3 client
        predictions_df: Predictions DataFrame
        bucket: S3 bucket name
        predictions_key: S3 key for the output file

    Returns:
        S3 key where predictions were saved
    """
    buffer = io.BytesIO()
    predictions_df.to_parquet(buffer, engine="pyarrow", index=False)
    buffer.seek(0)

    logger.info(f"Saving predictions to s3://{bucket}/{predictions_key}")
    s3_client.put_object(
        Bucket=bucket,
        Key=predictions_key,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream",
    )

    logger.info(f"Saved {len(predictions_df)} predictions to {predictions_key}")
    return predictions_key


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    Lambda handler for running model inference.

    Expected event format (from ProcessFeatures step):
    {
        "gameweek": 20,
        "season": "2024_25",
        "features_file": "processed/season_2024_25/gw20_features_prediction.parquet"
    }

    Returns:
    {
        "gameweek": 20,
        "season": "2024_25",
        "predictions_key": "predictions/season_2024_25/gw20_predictions.parquet",
        "predictions_count": 450,
        "model_key": "models/model.xgb",
        "timestamp": "2024-01-05T10:00:00+00:00"
    }

    Raises exceptions (not error dicts) for Step Functions catch/retry.
    """
    logger.info(f"Inference handler invoked with event: {event}")

    # Parse input
    gameweek = event.get("gameweek")
    season = event.get("season")
    features_file = event.get("features_file")
    model_key = event.get("model_key", MODEL_KEY)

    if not gameweek:
        raise ValueError("Missing required parameter: gameweek")

    if not season:
        raise ValueError("Missing required parameter: season")

    if not features_file:
        # Default features key from feature processor
        features_file = (
            f"processed/season_{season}/gw{gameweek}_features_prediction.parquet"
        )

    # Build predictions output key
    predictions_key = f"predictions/season_{season}/gw{gameweek}_predictions.parquet"

    # Initialise S3 client
    s3_client = get_s3_client(endpoint_url=AWS_ENDPOINT_URL)

    # Load model
    model = load_model_from_s3(s3_client, BUCKET_NAME, model_key)

    # Load and validate features
    features_df = load_features_from_s3(s3_client, BUCKET_NAME, features_file)
    validate_features(features_df)

    # Run inference
    predictions_df = run_inference(model, features_df, gameweek, season)

    # Save predictions to S3
    save_predictions_to_s3(s3_client, predictions_df, BUCKET_NAME, predictions_key)

    return {
        "gameweek": gameweek,
        "season": season,
        "predictions_key": predictions_key,
        "predictions_count": len(predictions_df),
        "model_key": model_key,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# For local testing
if __name__ == "__main__":
    import json

    test_event = {
        "gameweek": 20,
        "season": "2024_25",
        "features_file": "processed/season_2024_25/gw20_features_prediction.parquet",
    }

    os.environ["AWS_ENDPOINT_URL"] = "http://localhost:4566"

    result = handler(test_event, None)
    print(json.dumps(result, indent=2))
