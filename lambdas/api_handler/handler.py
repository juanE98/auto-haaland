"""
API Handler Lambda

Provides REST API endpoints for querying FPL predictions from DynamoDB.
Uses AWS Lambda Powertools for routing and structured logging.

Endpoints:
    GET /predictions?gameweek=20           - All predictions for a gameweek
    GET /predictions/{player_id}?gw=20     - Prediction for specific player
    GET /top?gameweek=20&position=MID&limit=10  - Top predicted scorers
    GET /compare?players=350,328&gameweek=20    - Compare multiple players
"""

import os
from decimal import Decimal
from typing import Any

from aws_lambda_powertools import Logger
from aws_lambda_powertools.event_handler import APIGatewayRestResolver, CORSConfig
from aws_lambda_powertools.event_handler.exceptions import (
    BadRequestError,
    NotFoundError,
)
from aws_lambda_powertools.utilities.typing import LambdaContext
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

from common.aws_clients import get_dynamodb_resource

logger = Logger(service="fpl-api")
cors_config = CORSConfig(
    allow_origin="*",
    allow_headers=["Content-Type"],
    max_age=300,
)
app = APIGatewayRestResolver(cors=cors_config)

TABLE_NAME = os.getenv("TABLE_NAME", "fpl-predictions")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")
VALID_POSITIONS = ["GKP", "DEF", "MID", "FWD"]


def get_table():
    """Get DynamoDB table resource."""
    dynamodb = get_dynamodb_resource(endpoint_url=AWS_ENDPOINT_URL)
    return dynamodb.Table(TABLE_NAME)


def decimal_to_float(obj: Any) -> Any:
    """
    Recursively convert Decimal values to float for JSON serialization.

    Args:
        obj: Object to convert (dict, list, or scalar)

    Returns:
        Object with Decimals converted to floats
    """
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decimal_to_float(item) for item in obj]
    return obj


@app.get("/predictions")
def get_predictions():
    """
    Get all predictions for a gameweek.

    Query params:
        gameweek (required): Gameweek number
        limit (optional): Max results (default 100)
    """
    gameweek = app.current_event.get_query_string_value("gameweek")
    if not gameweek:
        raise BadRequestError("Missing required parameter: gameweek")

    try:
        gameweek = int(gameweek)
    except ValueError:
        raise BadRequestError("gameweek must be an integer")

    limit = int(app.current_event.get_query_string_value("limit", "100"))

    logger.info("Getting predictions", extra={"gameweek": gameweek, "limit": limit})

    table = get_table()
    response = table.query(
        IndexName="gameweek-points-index",
        KeyConditionExpression=Key("gameweek").eq(gameweek),
        ScanIndexForward=False,
        Limit=limit,
    )

    predictions = decimal_to_float(response.get("Items", []))

    return {
        "gameweek": gameweek,
        "count": len(predictions),
        "predictions": predictions,
    }


@app.get("/predictions/<player_id>")
def get_player_prediction(player_id: str):
    """
    Get prediction for a specific player.

    Path params:
        player_id: Player ID

    Query params:
        gameweek/gw (optional): Specific gameweek
    """
    try:
        player_id = int(player_id)
    except ValueError:
        raise BadRequestError("player_id must be an integer")

    gameweek = app.current_event.get_query_string_value(
        "gameweek"
    ) or app.current_event.get_query_string_value("gw")

    logger.info(
        "Getting player prediction",
        extra={"player_id": player_id, "gameweek": gameweek},
    )

    table = get_table()

    if gameweek:
        try:
            gameweek = int(gameweek)
        except ValueError:
            raise BadRequestError("gameweek must be an integer")

        response = table.get_item(Key={"player_id": player_id, "gameweek": gameweek})
        item = response.get("Item")

        if not item:
            raise NotFoundError(f"Player {player_id} not found for gameweek {gameweek}")

        return decimal_to_float(item)
    else:
        response = table.query(
            KeyConditionExpression=Key("player_id").eq(player_id),
            ScanIndexForward=False,
        )

        items = response.get("Items", [])
        if not items:
            raise NotFoundError(f"Player {player_id} not found")

        return {
            "player_id": player_id,
            "count": len(items),
            "predictions": decimal_to_float(items),
        }


@app.get("/top")
def get_top_predictions():
    """
    Get top predicted scorers for a gameweek.

    Query params:
        gameweek (required): Gameweek number
        position (optional): Filter by position (GKP, DEF, MID, FWD)
        limit (optional): Max results (default 10)
        sort_by (optional): Sort by 'points' (default) or 'haul'
        available_only (optional): Exclude unavailable players (default true)
    """
    gameweek = app.current_event.get_query_string_value("gameweek")
    if not gameweek:
        raise BadRequestError("Missing required parameter: gameweek")

    try:
        gameweek = int(gameweek)
    except ValueError:
        raise BadRequestError("gameweek must be an integer")

    position = app.current_event.get_query_string_value("position")
    if position:
        position = position.upper()
        if position not in VALID_POSITIONS:
            raise BadRequestError(
                f"Invalid position. Must be one of: {VALID_POSITIONS}"
            )

    limit = int(app.current_event.get_query_string_value("limit", "10"))

    # Sort by predicted_points (default) or haul_probability
    sort_by = app.current_event.get_query_string_value("sort_by", "points")
    if sort_by not in ("points", "haul"):
        raise BadRequestError("sort_by must be 'points' or 'haul'")

    # Filter out unavailable players by default
    available_only_param = app.current_event.get_query_string_value(
        "available_only", "true"
    )
    available_only = available_only_param.lower() != "false"

    logger.info(
        "Getting top predictions",
        extra={
            "gameweek": gameweek,
            "position": position,
            "limit": limit,
            "sort_by": sort_by,
            "available_only": available_only,
        },
    )

    table = get_table()

    # Choose index based on sort order
    if sort_by == "haul":
        index_name = "gameweek-haul-index"
    else:
        index_name = "gameweek-points-index"

    # Fetch more items to account for filtering (unavailable players, position)
    fetch_limit = limit * 3

    if position:
        # Position filter requires fetching from gameweek index and filtering
        response = table.query(
            IndexName=index_name,
            KeyConditionExpression=Key("gameweek").eq(gameweek),
            ScanIndexForward=False,
            Limit=fetch_limit * 3,  # Fetch more since filtering by position
        )
        items = [
            item
            for item in response.get("Items", [])
            if item.get("position") == position
        ]
    else:
        response = table.query(
            IndexName=index_name,
            KeyConditionExpression=Key("gameweek").eq(gameweek),
            ScanIndexForward=False,
            Limit=fetch_limit,
        )
        items = response.get("Items", [])

    # Filter out unavailable players (chance_of_playing = 0)
    if available_only:
        items = [item for item in items if item.get("chance_of_playing", 100) > 0]

    # Apply final limit
    items = items[:limit]
    predictions = decimal_to_float(items)

    return {
        "gameweek": gameweek,
        "position": position,
        "limit": limit,
        "sort_by": sort_by,
        "available_only": available_only,
        "count": len(predictions),
        "predictions": predictions,
    }


@app.get("/compare")
def compare_players():
    """
    Compare predictions for multiple players.

    Query params:
        players (required): Comma-separated player IDs
        gameweek (required): Gameweek number
    """
    players_str = app.current_event.get_query_string_value("players")
    if not players_str:
        raise BadRequestError("Missing required parameter: players")

    gameweek = app.current_event.get_query_string_value("gameweek")
    if not gameweek:
        raise BadRequestError("Missing required parameter: gameweek")

    try:
        gameweek = int(gameweek)
    except ValueError:
        raise BadRequestError("gameweek must be an integer")

    try:
        player_ids = [int(p.strip()) for p in players_str.split(",")]
    except ValueError:
        raise BadRequestError("players must be comma-separated integers")

    if len(player_ids) > 15:
        raise BadRequestError("Maximum 15 players allowed for comparison")

    logger.info(
        "Comparing players",
        extra={"player_ids": player_ids, "gameweek": gameweek},
    )

    table = get_table()
    predictions = []

    for player_id in player_ids:
        response = table.get_item(Key={"player_id": player_id, "gameweek": gameweek})
        item = response.get("Item")
        if item:
            predictions.append(item)

    predictions.sort(key=lambda x: float(x.get("predicted_points", 0)), reverse=True)

    found_ids = [p["player_id"] for p in predictions]
    missing_ids = [pid for pid in player_ids if pid not in found_ids]

    return {
        "gameweek": gameweek,
        "requested_players": player_ids,
        "found": len(predictions),
        "missing": missing_ids,
        "predictions": decimal_to_float(predictions),
    }


@logger.inject_lambda_context
def handler(event: dict, context: LambdaContext) -> dict:
    """Lambda handler for API Gateway requests."""
    logger.info("API request received")

    try:
        return app.resolve(event, context)
    except ClientError as e:
        logger.error("DynamoDB error", extra={"error": str(e)})
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": '{"error": "Database error"}',
        }
