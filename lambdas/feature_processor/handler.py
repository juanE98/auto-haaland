"""
Feature Processor Lambda Handler

Reads raw FPL data from S3, engineers ML features, and outputs Parquet files
for model training and predictions.
"""

import io
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import pandas as pd
from botocore.exceptions import ClientError

from common.aws_clients import get_s3_client
from common.feature_config import (
    FEATURE_COLS,
    compute_derived_features,
    compute_rolling_features,
)

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# Environment variables
BUCKET_NAME = os.getenv("BUCKET_NAME", "fpl-ml-data")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")  # For LocalStack


def load_json_from_s3(s3_client, bucket: str, key: str) -> Dict[str, Any]:
    """Load JSON data from S3."""
    logger.info(f"Loading from s3://{bucket}/{key}")
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read().decode("utf-8"))


def load_bootstrap_from_s3(
    s3_client, bucket: str, gameweek: int, season: str
) -> Dict[str, Any]:
    """Load bootstrap-static data from S3."""
    key = f"raw/season_{season}/gw{gameweek}_bootstrap.json"
    return load_json_from_s3(s3_client, bucket, key)


def load_fixtures_from_s3(
    s3_client, bucket: str, gameweek: int, season: str
) -> List[Dict[str, Any]]:
    """Load fixtures data from S3."""
    key = f"raw/season_{season}/gw{gameweek}_fixtures.json"
    return load_json_from_s3(s3_client, bucket, key)


def load_player_histories_from_s3(
    s3_client, bucket: str, gameweek: int, season: str
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Load player history data from S3.

    Returns a dict mapping player_id -> list of gameweek history entries.
    Returns empty dict if player histories don't exist.
    """
    prefix = f"raw/season_{season}/gw{gameweek}_players/"
    histories = {}

    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

        if "Contents" not in response:
            logger.warning(f"No player histories found at {prefix}")
            return histories

        for obj in response["Contents"]:
            key = obj["Key"]
            if key.endswith(".json"):
                try:
                    data = load_json_from_s3(s3_client, bucket, key)
                    # Extract player_id from filename: player_350.json
                    filename = key.split("/")[-1]
                    player_id = int(
                        filename.replace("player_", "").replace(".json", "")
                    )
                    # Player summary contains "history" key with list of gameweek data
                    histories[player_id] = data.get("history", [])
                except Exception as e:
                    logger.warning(f"Failed to load {key}: {e}")
                    continue

        logger.info(f"Loaded histories for {len(histories)} players")
        return histories

    except Exception as e:
        logger.warning(f"Failed to load player histories: {e}")
        return histories


def get_team_strength(teams: List[Dict], team_id: int) -> int:
    """Get team strength (1-5) for a given team ID."""
    for team in teams:
        if team["id"] == team_id:
            return team.get("strength", 3)
    return 3  # Default to medium strength


def get_opponent_info(
    player_team_id: int, fixtures: List[Dict], teams: List[Dict]
) -> Tuple[int, int, int, int]:
    """
    Get opponent strength and home/away status from fixtures.

    Args:
        player_team_id: The player's team ID
        fixtures: List of fixture data
        teams: List of team data

    Returns:
        Tuple of (opponent_strength, is_home, opp_attack_strength,
        opp_defence_strength) where is_home is 1 or 0
    """
    for fixture in fixtures:
        if fixture.get("team_h") == player_team_id:
            # Player's team is home, opponent is away
            opponent_id = fixture.get("team_a")
            opp_attack = 1200
            opp_defence = 1200
            for team in teams:
                if team["id"] == opponent_id:
                    opp_attack = team.get("strength_attack_away", 1200)
                    opp_defence = team.get("strength_defence_away", 1200)
                    break
            return get_team_strength(teams, opponent_id), 1, opp_attack, opp_defence
        elif fixture.get("team_a") == player_team_id:
            # Player's team is away, opponent is home
            opponent_id = fixture.get("team_h")
            opp_attack = 1200
            opp_defence = 1200
            for team in teams:
                if team["id"] == opponent_id:
                    opp_attack = team.get("strength_attack_home", 1200)
                    opp_defence = team.get("strength_defence_home", 1200)
                    break
            return get_team_strength(teams, opponent_id), 0, opp_attack, opp_defence

    # No fixture found (could be blank gameweek)
    return 3, 0, 1200, 1200


def engineer_features(
    bootstrap: Dict[str, Any],
    fixtures: List[Dict[str, Any]],
    player_histories: Dict[int, List[Dict[str, Any]]],
    mode: str,
    gameweek: int,
) -> pd.DataFrame:
    """
    Engineer ML features from raw FPL data.

    Args:
        bootstrap: Bootstrap-static data (players, teams, events)
        fixtures: Fixtures for the gameweek
        player_histories: Dict mapping player_id to history (may be empty)
        mode: "historical" (include actual_points) or "prediction" (no target)
        gameweek: Current gameweek number

    Returns:
        DataFrame with engineered features
    """
    players = bootstrap.get("elements", [])
    teams = bootstrap.get("teams", [])

    features = []

    for player in players:
        player_id = player["id"]
        team_id = player["team"]

        # Get player history if available
        history = player_histories.get(player_id, [])

        # Calculate rolling features from history
        if history:
            rolling = compute_rolling_features(history)
        else:
            # Fallback: use bootstrap form for points, zeros for rest
            form = float(player.get("form", 0) or 0)
            rolling = {name: 0.0 for name in FEATURE_COLS if "_last_" in name}
            rolling["points_last_1"] = form
            rolling["points_last_3"] = form
            rolling["points_last_5"] = form

        # Form score from bootstrap
        form_score = float(player.get("form", 0) or 0)

        # Selected by percent (ownership)
        selected_by_percent = float(player.get("selected_by_percent", 0) or 0)

        # Chance of playing
        chance_of_playing = player.get("chance_of_playing_next_round")
        if chance_of_playing is None:
            chance_of_playing = 100  # Assume available if not specified

        # Get opponent info from fixtures
        opponent_strength, home_away, opp_attack_strength, opp_defence_strength = (
            get_opponent_info(team_id, fixtures, teams)
        )

        # Static features
        static = {
            "form_score": form_score,
            "opponent_strength": opponent_strength,
            "home_away": home_away,
            "chance_of_playing": chance_of_playing,
            "position": player.get("element_type", 0),
            "opponent_attack_strength": opp_attack_strength,
            "opponent_defence_strength": opp_defence_strength,
            "selected_by_percent": selected_by_percent,
            "now_cost": player.get("now_cost", 0),
        }

        # Derived features
        derived = compute_derived_features(history, rolling, static)

        # Build row from rolling + static + derived
        row = {
            "player_id": player_id,
            "player_name": player.get("web_name", ""),
            "team_id": team_id,
            "gameweek": gameweek,
        }
        row.update(rolling)
        row.update(static)
        row.update(derived)

        # Add actual points for historical mode (training target)
        if mode == "historical" and history:
            # Get actual points for this gameweek from history
            gw_data = [h for h in history if h.get("round") == gameweek]
            if gw_data:
                row["actual_points"] = gw_data[0].get("total_points", 0)
            else:
                # Use last known points
                row["actual_points"] = (
                    history[-1].get("total_points", 0) if history else 0
                )

        features.append(row)

    df = pd.DataFrame(features)
    logger.info(f"Engineered features for {len(df)} players")

    return df


def save_features_to_s3(
    s3_client, df: pd.DataFrame, bucket: str, gameweek: int, season: str, mode: str
) -> str:
    """
    Save features DataFrame to S3 as Parquet.

    Returns:
        S3 key where the file was saved
    """
    suffix = "training" if mode == "historical" else "prediction"
    key = f"processed/season_{season}/gw{gameweek}_features_{suffix}.parquet"

    logger.info(f"Saving features to s3://{bucket}/{key}")

    # Write to buffer
    buffer = io.BytesIO()
    df.to_parquet(buffer, engine="pyarrow", index=False)
    buffer.seek(0)

    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream",
    )

    logger.info(f"Successfully saved {len(df)} rows to {key}")
    return key


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for feature engineering.

    Expected event format:
    {
        "gameweek": 20,
        "season": "2024_25",
        "mode": "historical" | "prediction"  # default: "historical"
    }

    Returns:
    {
        "statusCode": 200,
        "gameweek": 20,
        "season": "2024_25",
        "mode": "historical",
        "features_file": "processed/season_2024_25/gw20_features_training.parquet",
        "rows_processed": 450,
        "timestamp": "2024-01-05T10:00:00Z"
    }
    """
    logger.info(f"Feature processor invoked with event: {json.dumps(event)}")

    # Parse input
    gameweek = event.get("gameweek")
    season = event.get("season")
    mode = event.get("mode", "historical")

    # Validate inputs
    if gameweek is None:
        return {"statusCode": 400, "error": "Missing required field: gameweek"}

    if season is None:
        return {"statusCode": 400, "error": "Missing required field: season"}

    if mode not in ("historical", "prediction"):
        return {
            "statusCode": 400,
            "error": f"Invalid mode: {mode}. Must be 'historical' or 'prediction'",
        }

    # Initialise S3 client
    s3_client = get_s3_client(endpoint_url=AWS_ENDPOINT_URL)

    try:
        # Load raw data from S3
        logger.info(f"Loading raw data for gameweek {gameweek}, season {season}")

        bootstrap = load_bootstrap_from_s3(s3_client, BUCKET_NAME, gameweek, season)
        fixtures = load_fixtures_from_s3(s3_client, BUCKET_NAME, gameweek, season)
        player_histories = load_player_histories_from_s3(
            s3_client, BUCKET_NAME, gameweek, season
        )

        # Engineer features
        logger.info(f"Engineering features in {mode} mode...")
        df = engineer_features(bootstrap, fixtures, player_histories, mode, gameweek)

        # Save to S3
        features_file = save_features_to_s3(
            s3_client, df, BUCKET_NAME, gameweek, season, mode
        )

        return {
            "statusCode": 200,
            "gameweek": gameweek,
            "season": season,
            "mode": mode,
            "features_file": features_file,
            "rows_processed": len(df),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "NoSuchKey":
            logger.error(f"S3 key not found: {e}")
            return {
                "statusCode": 404,
                "error": "Required data not found in S3",
                "message": str(e),
            }
        logger.error(f"AWS error: {e}", exc_info=True)
        return {"statusCode": 500, "error": "AWS error", "message": str(e)}

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {"statusCode": 500, "error": "Internal error", "message": str(e)}


# For local testing
if __name__ == "__main__":
    test_event = {"gameweek": 20, "season": "2024_25", "mode": "historical"}

    os.environ["AWS_ENDPOINT_URL"] = "http://localhost:4566"

    result = handler(test_event, None)
    print(json.dumps(result, indent=2))
