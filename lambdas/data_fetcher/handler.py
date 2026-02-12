"""
Data Fetcher Lambda Handler

Fetches raw data from FPL API and stores it in S3.
This is the first step in the data pipeline.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict

from common.aws_clients import get_s3_client
from common.fpl_api import FPLApiClient, FPLApiError

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# Environment variables
BUCKET_NAME = os.getenv("BUCKET_NAME", "fpl-ml-data")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")  # For LocalStack

# Batching constants for player history fetches
PLAYER_BATCH_SIZE = 50
PLAYER_BATCH_DELAY = 1.0  # Seconds between batches


def save_to_s3(s3_client, bucket: str, key: str, data: Dict[str, Any]) -> None:
    """
    Save JSON data to S3.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key
        data: Data to save (will be JSON serialized)
    """
    logger.info(f"Saving to s3://{bucket}/{key}")

    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, indent=2),
        ContentType="application/json",
    )

    logger.info(f"Successfully saved to s3://{bucket}/{key}")


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for fetching FPL data.

    Expected event format:
    {
        "gameweek": 20,          # Optional: specific gameweek
        "fetch_player_details": true  # Optional: fetch individual player histories
    }

    Returns:
        {
            "statusCode": 200,
            "gameweek": 20,
            "season": "2024_25",
            "files_saved": [...],
            "timestamp": "2024-01-05T10:00:00Z"
        }
    """
    logger.info(f"Data fetcher invoked with event: {json.dumps(event)}")

    # Parse input
    gameweek = event.get("gameweek")
    fetch_player_details = event.get("fetch_player_details", True)

    # Initialise clients
    s3_client = get_s3_client(endpoint_url=AWS_ENDPOINT_URL)
    files_saved = []

    try:
        with FPLApiClient() as fpl:
            # Get season string
            season = fpl.get_season_string()
            logger.info(f"Processing season: {season}")

            # If no gameweek specified, get current one
            if gameweek is None:
                gameweek = fpl.get_current_gameweek()
                if gameweek is None:
                    return {
                        "statusCode": 400,
                        "error": "Could not determine current gameweek",
                    }
                logger.info(f"Auto-detected gameweek: {gameweek}")

            # === 1. Fetch and save bootstrap-static ===
            logger.info("Fetching bootstrap-static data...")
            bootstrap = fpl.get_bootstrap_static()

            bootstrap_key = f"raw/season_{season}/gw{gameweek}_bootstrap.json"
            save_to_s3(s3_client, BUCKET_NAME, bootstrap_key, bootstrap)
            files_saved.append(bootstrap_key)

            # === 2. Fetch and save fixtures ===
            logger.info(f"Fetching fixtures for gameweek {gameweek}...")
            fixtures = fpl.get_fixtures(gameweek=gameweek)

            fixtures_key = f"raw/season_{season}/gw{gameweek}_fixtures.json"
            save_to_s3(s3_client, BUCKET_NAME, fixtures_key, fixtures)
            files_saved.append(fixtures_key)

            # === 3. Optionally fetch individual player histories ===
            if fetch_player_details:
                logger.info("Fetching individual player histories...")
                players = bootstrap.get("elements", [])
                player_count = len(players)

                logger.info(f"Processing {player_count} players...")

                # Fetch all players in batches with rate limiting
                all_histories = {}
                for batch_start in range(0, player_count, PLAYER_BATCH_SIZE):
                    batch_end = min(batch_start + PLAYER_BATCH_SIZE, player_count)
                    batch = players[batch_start:batch_end]
                    batch_num = batch_start // PLAYER_BATCH_SIZE + 1
                    total_batches = (
                        player_count + PLAYER_BATCH_SIZE - 1
                    ) // PLAYER_BATCH_SIZE

                    logger.info(
                        f"Fetching batch {batch_num}/{total_batches} "
                        f"(players {batch_start + 1}-{batch_end})"
                    )

                    for player in batch:
                        player_id = player["id"]
                        try:
                            player_summary = fpl.get_player_summary(player_id)
                            all_histories[str(player_id)] = player_summary
                        except FPLApiError as e:
                            logger.warning(f"Failed to fetch player {player_id}: {e}")
                            continue

                    # Rate limit between batches (skip delay after last batch)
                    if batch_end < player_count:
                        time.sleep(PLAYER_BATCH_DELAY)

                # Save combined histories as a single file
                histories_key = (
                    f"raw/season_{season}/gw{gameweek}_player_histories.json"
                )
                save_to_s3(s3_client, BUCKET_NAME, histories_key, all_histories)
                files_saved.append(histories_key)
                logger.info(
                    f"Saved combined histories for " f"{len(all_histories)} players"
                )

            # === Success response ===
            return {
                "statusCode": 200,
                "gameweek": gameweek,
                "season": season,
                "files_saved": files_saved,
                "files_count": len(files_saved),
                "timestamp": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            }

    except FPLApiError as e:
        logger.error(f"FPL API error: {e}")
        return {"statusCode": 500, "error": "FPL API error", "message": str(e)}

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {"statusCode": 500, "error": "Internal error", "message": str(e)}


# For local testing
if __name__ == "__main__":
    # Test event
    test_event = {
        "gameweek": 20,
        "fetch_player_details": True,
    }

    # Set LocalStack endpoint for local testing
    os.environ["AWS_ENDPOINT_URL"] = "http://localhost:4566"

    result = handler(test_event, None)
    print(json.dumps(result, indent=2))
