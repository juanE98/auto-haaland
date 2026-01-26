"""
Integration tests for prediction_loader with DynamoDB.
"""

import io
from decimal import Decimal

import pandas as pd
import pytest
from boto3.dynamodb.conditions import Key

from lambdas.prediction_loader.handler import (
    convert_to_dynamodb_item,
    batch_write_predictions,
    delete_gameweek_predictions,
    handler,
)


@pytest.fixture
def predictions_dataframe():
    """Sample predictions DataFrame."""
    return pd.DataFrame({
        "player_id": [350, 328, 233, 412, 567],
        "player_name": ["Salah", "Haaland", "Saka", "Palmer", "Son"],
        "team_id": [10, 11, 1, 4, 17],
        "position": [3, 4, 3, 3, 3],
        "gameweek": [20, 20, 20, 20, 20],
        "predicted_points": [8.5, 12.3, 6.8, 9.2, 5.5],
        "season": ["2024_25", "2024_25", "2024_25", "2024_25", "2024_25"],
    })


class TestDynamoDBBatchWrite:
    """Integration tests for batch writing to DynamoDB."""

    def test_batch_write_creates_items(
        self, predictions_dataframe, clean_dynamodb_table
    ):
        """Test batch write creates items in DynamoDB."""
        # Convert and write predictions
        predictions = [
            convert_to_dynamodb_item(row)
            for _, row in predictions_dataframe.iterrows()
        ]

        result = batch_write_predictions(clean_dynamodb_table, predictions)

        assert result["items_written"] == 5

        # Verify items exist
        response = clean_dynamodb_table.scan()
        assert response["Count"] == 5

    def test_batch_write_preserves_data(
        self, predictions_dataframe, clean_dynamodb_table
    ):
        """Test batch write preserves all data fields."""
        predictions = [
            convert_to_dynamodb_item(row)
            for _, row in predictions_dataframe.iterrows()
        ]

        batch_write_predictions(clean_dynamodb_table, predictions)

        # Get specific item
        response = clean_dynamodb_table.get_item(
            Key={"player_id": 350, "gameweek": 20}
        )
        item = response["Item"]

        assert item["player_id"] == 350
        assert item["gameweek"] == 20
        assert item["predicted_points"] == Decimal("8.5")
        assert item["player_name"] == "Salah"
        assert item["team_id"] == 10
        assert item["position"] == "MID"
        assert item["season"] == "2024_25"


class TestDynamoDBQuery:
    """Integration tests for querying DynamoDB."""

    def test_query_by_gameweek_gsi(
        self, predictions_dataframe, clean_dynamodb_table
    ):
        """Test querying predictions by gameweek using GSI."""
        # Write predictions
        predictions = [
            convert_to_dynamodb_item(row)
            for _, row in predictions_dataframe.iterrows()
        ]
        batch_write_predictions(clean_dynamodb_table, predictions)

        # Query by gameweek
        response = clean_dynamodb_table.query(
            IndexName="gameweek-points-index",
            KeyConditionExpression=Key("gameweek").eq(20),
            ScanIndexForward=False,  # Descending order by predicted_points
        )

        assert response["Count"] == 5
        # Verify sorted by predicted_points descending
        points = [item["predicted_points"] for item in response["Items"]]
        assert points == sorted(points, reverse=True)

    def test_query_by_position_gsi(
        self, predictions_dataframe, clean_dynamodb_table
    ):
        """Test querying predictions by position using GSI."""
        predictions = [
            convert_to_dynamodb_item(row)
            for _, row in predictions_dataframe.iterrows()
        ]
        batch_write_predictions(clean_dynamodb_table, predictions)

        # Query midfielders
        response = clean_dynamodb_table.query(
            IndexName="position-points-index",
            KeyConditionExpression=Key("position").eq("MID"),
            ScanIndexForward=False,
        )

        # 4 midfielders in sample data (Salah, Saka, Palmer, Son)
        assert response["Count"] == 4
        for item in response["Items"]:
            assert item["position"] == "MID"


class TestDeleteGameweekPredictions:
    """Integration tests for deleting gameweek predictions."""

    def test_delete_existing_predictions(
        self, predictions_dataframe, clean_dynamodb_table
    ):
        """Test deleting existing predictions for a gameweek."""
        # Write initial predictions
        predictions = [
            convert_to_dynamodb_item(row)
            for _, row in predictions_dataframe.iterrows()
        ]
        batch_write_predictions(clean_dynamodb_table, predictions)

        # Verify items exist
        response = clean_dynamodb_table.scan()
        assert response["Count"] == 5

        # Delete predictions for gameweek 20
        deleted = delete_gameweek_predictions(clean_dynamodb_table, 20)

        assert deleted == 5

        # Verify all deleted
        response = clean_dynamodb_table.scan()
        assert response["Count"] == 0

    def test_delete_only_target_gameweek(
        self, predictions_dataframe, clean_dynamodb_table
    ):
        """Test delete only removes target gameweek, not others."""
        # Write GW20 predictions
        predictions_gw20 = [
            convert_to_dynamodb_item(row)
            for _, row in predictions_dataframe.iterrows()
        ]
        batch_write_predictions(clean_dynamodb_table, predictions_gw20)

        # Write GW21 predictions
        df_gw21 = predictions_dataframe.copy()
        df_gw21["gameweek"] = 21
        predictions_gw21 = [
            convert_to_dynamodb_item(row) for _, row in df_gw21.iterrows()
        ]
        batch_write_predictions(clean_dynamodb_table, predictions_gw21)

        # Verify 10 items total
        response = clean_dynamodb_table.scan()
        assert response["Count"] == 10

        # Delete only GW20
        deleted = delete_gameweek_predictions(clean_dynamodb_table, 20)

        assert deleted == 5

        # Verify GW21 still exists
        response = clean_dynamodb_table.scan()
        assert response["Count"] == 5
        for item in response["Items"]:
            assert item["gameweek"] == 21


class TestEndToEndHandler:
    """End-to-end integration tests for the handler."""

    def test_full_load_pipeline(
        self,
        predictions_dataframe,
        localstack_s3_client,
        clean_s3_bucket,
        clean_dynamodb_table,
    ):
        """Test complete load pipeline with S3 and DynamoDB."""
        # Upload predictions to S3
        buffer = io.BytesIO()
        predictions_dataframe.to_parquet(buffer, engine="pyarrow", index=False)
        buffer.seek(0)

        localstack_s3_client.put_object(
            Bucket=clean_s3_bucket,
            Key="predictions/season_2024_25/gw20_predictions.parquet",
            Body=buffer.getvalue(),
        )

        # Call handler
        event = {
            "gameweek": 20,
            "season": "2024_25",
            "replace_existing": False,
        }

        result = handler(event, None)

        assert result["statusCode"] == 200
        assert result["items_written"] == 5

        # Verify data in DynamoDB
        response = clean_dynamodb_table.scan()
        assert response["Count"] == 5

        # Verify specific item
        item_response = clean_dynamodb_table.get_item(
            Key={"player_id": 328, "gameweek": 20}
        )
        item = item_response["Item"]
        assert item["player_name"] == "Haaland"
        assert item["predicted_points"] == Decimal("12.3")
