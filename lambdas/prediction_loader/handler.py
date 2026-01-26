"""
Prediction Loader Lambda Handler

Reads prediction Parquet files from S3 and batch writes them to DynamoDB.
This enables fast queries by gameweek, position, and predicted points.
"""

import io
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from common.aws_clients import get_dynamodb_resource, get_s3_client

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# Environment variables
BUCKET_NAME = os.getenv("BUCKET_NAME", "fpl-ml-data")
TABLE_NAME = os.getenv("TABLE_NAME", "fpl-predictions")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")  # For LocalStack


# Position mapping (element_type to position string)
POSITION_MAP = {
    1: "GKP",
    2: "DEF",
    3: "MID",
    4: "FWD",
}


def load_predictions_from_s3(
    s3_client,
    bucket: str,
    key: str,
) -> pd.DataFrame:
    """
    Load predictions Parquet file from S3.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key for the predictions file

    Returns:
        DataFrame with predictions
    """
    logger.info(f"Loading predictions from s3://{bucket}/{key}")

    response = s3_client.get_object(Bucket=bucket, Key=key)
    df = pd.read_parquet(io.BytesIO(response["Body"].read()))

    logger.info(f"Loaded {len(df)} predictions from S3")
    return df


def validate_predictions(df: pd.DataFrame) -> None:
    """
    Validate that predictions DataFrame has required columns.

    Args:
        df: Predictions DataFrame

    Raises:
        ValueError: If required columns are missing
    """
    required_cols = ["player_id", "gameweek", "predicted_points"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def convert_to_dynamodb_item(row: pd.Series) -> dict[str, Any]:
    """
    Convert a DataFrame row to a DynamoDB item.

    Args:
        row: Pandas Series representing a prediction row

    Returns:
        Dictionary formatted for DynamoDB
    """
    # Get position string from element_type if available
    position = row.get("position")
    if isinstance(position, (int, float)):
        position = POSITION_MAP.get(int(position), "UNK")
    elif position is None:
        position = "UNK"

    item = {
        "player_id": int(row["player_id"]),
        "gameweek": int(row["gameweek"]),
        "predicted_points": Decimal(str(round(float(row["predicted_points"]), 2))),
        "position": str(position),
    }

    # Add optional fields if present
    if "player_name" in row and pd.notna(row["player_name"]):
        item["player_name"] = str(row["player_name"])

    if "team_id" in row and pd.notna(row["team_id"]):
        item["team_id"] = int(row["team_id"])

    if "season" in row and pd.notna(row["season"]):
        item["season"] = str(row["season"])

    return item


def batch_write_predictions(
    table,
    predictions: list[dict[str, Any]],
    batch_size: int = 25,
) -> dict[str, int]:
    """
    Batch write predictions to DynamoDB.

    Args:
        table: DynamoDB table resource
        predictions: List of prediction items to write
        batch_size: Number of items per batch (max 25 for DynamoDB)

    Returns:
        Dictionary with write statistics
    """
    total_written = 0
    total_batches = 0

    for i in range(0, len(predictions), batch_size):
        batch = predictions[i : i + batch_size]

        with table.batch_writer() as writer:
            for item in batch:
                writer.put_item(Item=item)

        total_written += len(batch)
        total_batches += 1

        if total_batches % 10 == 0:
            logger.info(f"Written {total_written}/{len(predictions)} items")

    logger.info(f"Completed: {total_written} items in {total_batches} batches")

    return {
        "items_written": total_written,
        "batches": total_batches,
    }


def delete_gameweek_predictions(
    table,
    gameweek: int,
) -> int:
    """
    Delete existing predictions for a gameweek before loading new ones.

    Args:
        table: DynamoDB table resource
        gameweek: Gameweek number to delete predictions for

    Returns:
        Number of items deleted
    """
    logger.info(f"Deleting existing predictions for gameweek {gameweek}")

    # Query using the GSI to find all items for this gameweek
    response = table.query(
        IndexName="gameweek-points-index",
        KeyConditionExpression=boto3.dynamodb.conditions.Key("gameweek").eq(gameweek),
    )

    items = response.get("Items", [])
    deleted_count = 0

    # Delete each item using its primary key
    with table.batch_writer() as writer:
        for item in items:
            writer.delete_item(
                Key={
                    "player_id": item["player_id"],
                    "gameweek": item["gameweek"],
                }
            )
            deleted_count += 1

    # Handle pagination if there are more items
    while "LastEvaluatedKey" in response:
        response = table.query(
            IndexName="gameweek-points-index",
            KeyConditionExpression=boto3.dynamodb.conditions.Key("gameweek").eq(
                gameweek
            ),
            ExclusiveStartKey=response["LastEvaluatedKey"],
        )

        with table.batch_writer() as writer:
            for item in response.get("Items", []):
                writer.delete_item(
                    Key={
                        "player_id": item["player_id"],
                        "gameweek": item["gameweek"],
                    }
                )
                deleted_count += 1

    logger.info(f"Deleted {deleted_count} existing predictions for gameweek {gameweek}")
    return deleted_count


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    Lambda handler for loading predictions to DynamoDB.

    Expected event format:
    {
        "gameweek": 20,
        "season": "2024_25",
        "predictions_key": "predictions/season_2024_25/gw20_predictions.parquet",  # Optional
        "replace_existing": true  # Optional: delete existing predictions first
    }

    Returns:
        {
            "statusCode": 200,
            "gameweek": 20,
            "season": "2024_25",
            "items_written": 450,
            "timestamp": "2024-01-05T10:00:00Z"
        }
    """
    logger.info(f"Prediction loader invoked with event: {event}")

    # Parse input
    gameweek = event.get("gameweek")
    season = event.get("season")
    predictions_key = event.get("predictions_key")
    replace_existing = event.get("replace_existing", True)

    if not gameweek:
        raise ValueError("Missing required parameter: gameweek")

    if not season:
        raise ValueError("Missing required parameter: season")

    # Build predictions key if not provided
    if not predictions_key:
        predictions_key = (
            f"predictions/season_{season}/gw{gameweek}_predictions.parquet"
        )

    try:
        # Initialize clients
        s3_client = get_s3_client(endpoint_url=AWS_ENDPOINT_URL)
        dynamodb = get_dynamodb_resource(endpoint_url=AWS_ENDPOINT_URL)
        table = dynamodb.Table(TABLE_NAME)

        # Load predictions from S3
        df = load_predictions_from_s3(s3_client, BUCKET_NAME, predictions_key)
        validate_predictions(df)

        # Add season to DataFrame if not present
        if "season" not in df.columns:
            df["season"] = season

        # Convert to DynamoDB items
        predictions = [convert_to_dynamodb_item(row) for _, row in df.iterrows()]

        # Optionally delete existing predictions
        deleted_count = 0
        if replace_existing:
            deleted_count = delete_gameweek_predictions(table, gameweek)

        # Batch write to DynamoDB
        write_stats = batch_write_predictions(table, predictions)

        return {
            "statusCode": 200,
            "gameweek": gameweek,
            "season": season,
            "predictions_key": predictions_key,
            "items_deleted": deleted_count,
            "items_written": write_stats["items_written"],
            "batches": write_stats["batches"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "NoSuchKey":
            logger.error(
                f"Predictions file not found: s3://{BUCKET_NAME}/{predictions_key}"
            )
            raise FileNotFoundError(
                f"Predictions file not found: {predictions_key}"
            ) from e
        logger.error(f"AWS error: {e}", exc_info=True)
        raise

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise

    except Exception as e:
        logger.error(f"Error loading predictions: {e}", exc_info=True)
        raise
