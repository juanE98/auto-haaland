"""
Unit tests for prediction_loader Lambda handler.
"""

from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
import io

import pandas as pd
import pytest

from lambdas.prediction_loader.handler import (
    POSITION_MAP,
    load_predictions_from_s3,
    validate_predictions,
    convert_to_dynamodb_item,
    batch_write_predictions,
    handler,
)


class TestPositionMap:
    """Tests for position mapping."""

    def test_position_map_has_all_positions(self):
        """Verify all FPL positions are mapped."""
        assert len(POSITION_MAP) == 4

    def test_goalkeeper_mapping(self):
        """Test goalkeeper position mapping."""
        assert POSITION_MAP[1] == "GKP"

    def test_defender_mapping(self):
        """Test defender position mapping."""
        assert POSITION_MAP[2] == "DEF"

    def test_midfielder_mapping(self):
        """Test midfielder position mapping."""
        assert POSITION_MAP[3] == "MID"

    def test_forward_mapping(self):
        """Test forward position mapping."""
        assert POSITION_MAP[4] == "FWD"


class TestValidatePredictions:
    """Tests for validate_predictions function."""

    def test_valid_dataframe_passes(self):
        """Verify valid DataFrame passes validation."""
        df = pd.DataFrame({
            "player_id": [350, 328],
            "gameweek": [20, 20],
            "predicted_points": [8.5, 12.3],
        })
        # Should not raise
        validate_predictions(df)

    def test_missing_player_id_raises(self):
        """Verify missing player_id raises ValueError."""
        df = pd.DataFrame({
            "gameweek": [20],
            "predicted_points": [8.5],
        })
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_predictions(df)

    def test_missing_gameweek_raises(self):
        """Verify missing gameweek raises ValueError."""
        df = pd.DataFrame({
            "player_id": [350],
            "predicted_points": [8.5],
        })
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_predictions(df)

    def test_missing_predicted_points_raises(self):
        """Verify missing predicted_points raises ValueError."""
        df = pd.DataFrame({
            "player_id": [350],
            "gameweek": [20],
        })
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_predictions(df)

    def test_multiple_missing_columns(self):
        """Verify multiple missing columns are reported."""
        df = pd.DataFrame({"player_id": [350]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_predictions(df)


class TestConvertToDynamoDBItem:
    """Tests for convert_to_dynamodb_item function."""

    def test_basic_conversion(self):
        """Test basic row conversion to DynamoDB item."""
        row = pd.Series({
            "player_id": 350,
            "gameweek": 20,
            "predicted_points": 8.5,
            "position": 3,
        })

        item = convert_to_dynamodb_item(row)

        assert item["player_id"] == 350
        assert item["gameweek"] == 20
        assert item["predicted_points"] == Decimal("8.5")
        assert item["position"] == "MID"

    def test_position_string_passthrough(self):
        """Test position string is passed through."""
        row = pd.Series({
            "player_id": 350,
            "gameweek": 20,
            "predicted_points": 8.5,
            "position": "FWD",
        })

        item = convert_to_dynamodb_item(row)
        assert item["position"] == "FWD"

    def test_missing_position_defaults_to_unk(self):
        """Test missing position defaults to UNK."""
        row = pd.Series({
            "player_id": 350,
            "gameweek": 20,
            "predicted_points": 8.5,
        })

        item = convert_to_dynamodb_item(row)
        assert item["position"] == "UNK"

    def test_unknown_position_number_defaults_to_unk(self):
        """Test unknown position number defaults to UNK."""
        row = pd.Series({
            "player_id": 350,
            "gameweek": 20,
            "predicted_points": 8.5,
            "position": 99,
        })

        item = convert_to_dynamodb_item(row)
        assert item["position"] == "UNK"

    def test_optional_player_name_included(self):
        """Test optional player_name is included when present."""
        row = pd.Series({
            "player_id": 350,
            "gameweek": 20,
            "predicted_points": 8.5,
            "position": 3,
            "player_name": "Salah",
        })

        item = convert_to_dynamodb_item(row)
        assert item["player_name"] == "Salah"

    def test_optional_team_id_included(self):
        """Test optional team_id is included when present."""
        row = pd.Series({
            "player_id": 350,
            "gameweek": 20,
            "predicted_points": 8.5,
            "position": 3,
            "team_id": 10,
        })

        item = convert_to_dynamodb_item(row)
        assert item["team_id"] == 10

    def test_optional_season_included(self):
        """Test optional season is included when present."""
        row = pd.Series({
            "player_id": 350,
            "gameweek": 20,
            "predicted_points": 8.5,
            "position": 3,
            "season": "2024_25",
        })

        item = convert_to_dynamodb_item(row)
        assert item["season"] == "2024_25"

    def test_predicted_points_rounded_to_two_decimals(self):
        """Test predicted_points is rounded to 2 decimal places."""
        row = pd.Series({
            "player_id": 350,
            "gameweek": 20,
            "predicted_points": 8.54321,
            "position": 3,
        })

        item = convert_to_dynamodb_item(row)
        assert item["predicted_points"] == Decimal("8.54")

    def test_nan_optional_fields_excluded(self):
        """Test NaN optional fields are excluded."""
        import numpy as np

        row = pd.Series({
            "player_id": 350,
            "gameweek": 20,
            "predicted_points": 8.5,
            "position": 3,
            "player_name": np.nan,
            "team_id": np.nan,
        })

        item = convert_to_dynamodb_item(row)
        assert "player_name" not in item
        assert "team_id" not in item


class TestBatchWritePredictions:
    """Tests for batch_write_predictions function."""

    def test_writes_all_items(self):
        """Test all items are written."""
        mock_table = MagicMock()
        mock_batch_writer = MagicMock()
        mock_table.batch_writer.return_value.__enter__ = Mock(
            return_value=mock_batch_writer
        )
        mock_table.batch_writer.return_value.__exit__ = Mock(return_value=False)

        predictions = [
            {"player_id": 350, "gameweek": 20, "predicted_points": Decimal("8.5")},
            {"player_id": 328, "gameweek": 20, "predicted_points": Decimal("12.3")},
        ]

        result = batch_write_predictions(mock_table, predictions)

        assert result["items_written"] == 2
        assert mock_batch_writer.put_item.call_count == 2

    def test_respects_batch_size(self):
        """Test batching respects batch_size parameter."""
        mock_table = MagicMock()
        mock_batch_writer = MagicMock()
        mock_table.batch_writer.return_value.__enter__ = Mock(
            return_value=mock_batch_writer
        )
        mock_table.batch_writer.return_value.__exit__ = Mock(return_value=False)

        # Create 30 predictions
        predictions = [
            {"player_id": i, "gameweek": 20, "predicted_points": Decimal("5.0")}
            for i in range(30)
        ]

        result = batch_write_predictions(mock_table, predictions, batch_size=10)

        assert result["items_written"] == 30
        assert result["batches"] == 3  # 30 items / 10 batch_size

    def test_empty_predictions_list(self):
        """Test handling of empty predictions list."""
        mock_table = MagicMock()

        result = batch_write_predictions(mock_table, [])

        assert result["items_written"] == 0
        assert result["batches"] == 0


class TestLoadPredictionsFromS3:
    """Tests for load_predictions_from_s3 function."""

    def test_loads_parquet_from_s3(self):
        """Test loading Parquet file from S3."""
        # Create sample predictions DataFrame
        df = pd.DataFrame({
            "player_id": [350, 328],
            "gameweek": [20, 20],
            "predicted_points": [8.5, 12.3],
        })

        # Mock S3 response
        buffer = io.BytesIO()
        df.to_parquet(buffer, engine="pyarrow", index=False)
        buffer.seek(0)

        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": io.BytesIO(buffer.getvalue())}

        result = load_predictions_from_s3(
            mock_s3, "fpl-ml-data", "predictions/gw20.parquet"
        )

        assert len(result) == 2
        assert list(result.columns) == ["player_id", "gameweek", "predicted_points"]


class TestHandler:
    """Tests for Lambda handler function."""

    @patch("lambdas.prediction_loader.handler.get_s3_client")
    @patch("lambdas.prediction_loader.handler.get_dynamodb_resource")
    def test_missing_gameweek_returns_400(self, mock_dynamodb, mock_s3):
        """Test missing gameweek returns 400 error."""
        event = {"season": "2024_25"}

        result = handler(event, None)

        assert result["statusCode"] == 400
        assert "gameweek" in result["error"]

    @patch("lambdas.prediction_loader.handler.get_s3_client")
    @patch("lambdas.prediction_loader.handler.get_dynamodb_resource")
    def test_missing_season_returns_400(self, mock_dynamodb, mock_s3):
        """Test missing season returns 400 error."""
        event = {"gameweek": 20}

        result = handler(event, None)

        assert result["statusCode"] == 400
        assert "season" in result["error"]

    @patch("lambdas.prediction_loader.handler.get_s3_client")
    @patch("lambdas.prediction_loader.handler.get_dynamodb_resource")
    @patch("lambdas.prediction_loader.handler.load_predictions_from_s3")
    @patch("lambdas.prediction_loader.handler.delete_gameweek_predictions")
    @patch("lambdas.prediction_loader.handler.batch_write_predictions")
    def test_successful_load(
        self,
        mock_batch_write,
        mock_delete,
        mock_load,
        mock_dynamodb,
        mock_s3,
    ):
        """Test successful prediction loading."""
        # Setup mocks
        mock_load.return_value = pd.DataFrame({
            "player_id": [350, 328],
            "gameweek": [20, 20],
            "predicted_points": [8.5, 12.3],
            "position": [3, 4],
        })
        mock_delete.return_value = 0
        mock_batch_write.return_value = {"items_written": 2, "batches": 1}

        mock_table = MagicMock()
        mock_dynamodb.return_value.Table.return_value = mock_table

        event = {"gameweek": 20, "season": "2024_25"}

        result = handler(event, None)

        assert result["statusCode"] == 200
        assert result["gameweek"] == 20
        assert result["season"] == "2024_25"
        assert result["items_written"] == 2

    @patch("lambdas.prediction_loader.handler.get_s3_client")
    @patch("lambdas.prediction_loader.handler.get_dynamodb_resource")
    @patch("lambdas.prediction_loader.handler.load_predictions_from_s3")
    def test_validation_error_returns_400(
        self,
        mock_load,
        mock_dynamodb,
        mock_s3,
    ):
        """Test validation error returns 400."""
        # Return DataFrame missing required column
        mock_load.return_value = pd.DataFrame({
            "player_id": [350],
            "gameweek": [20],
            # Missing predicted_points
        })

        event = {"gameweek": 20, "season": "2024_25"}

        result = handler(event, None)

        assert result["statusCode"] == 400
        assert "Missing required columns" in result["error"]

    @patch("lambdas.prediction_loader.handler.get_s3_client")
    @patch("lambdas.prediction_loader.handler.get_dynamodb_resource")
    @patch("lambdas.prediction_loader.handler.load_predictions_from_s3")
    @patch("lambdas.prediction_loader.handler.delete_gameweek_predictions")
    @patch("lambdas.prediction_loader.handler.batch_write_predictions")
    def test_replace_existing_calls_delete(
        self,
        mock_batch_write,
        mock_delete,
        mock_load,
        mock_dynamodb,
        mock_s3,
    ):
        """Test replace_existing=True calls delete."""
        mock_load.return_value = pd.DataFrame({
            "player_id": [350],
            "gameweek": [20],
            "predicted_points": [8.5],
        })
        mock_delete.return_value = 5
        mock_batch_write.return_value = {"items_written": 1, "batches": 1}

        mock_table = MagicMock()
        mock_dynamodb.return_value.Table.return_value = mock_table

        event = {"gameweek": 20, "season": "2024_25", "replace_existing": True}

        result = handler(event, None)

        assert result["statusCode"] == 200
        assert result["items_deleted"] == 5
        mock_delete.assert_called_once()

    @patch("lambdas.prediction_loader.handler.get_s3_client")
    @patch("lambdas.prediction_loader.handler.get_dynamodb_resource")
    @patch("lambdas.prediction_loader.handler.load_predictions_from_s3")
    @patch("lambdas.prediction_loader.handler.delete_gameweek_predictions")
    @patch("lambdas.prediction_loader.handler.batch_write_predictions")
    def test_no_replace_skips_delete(
        self,
        mock_batch_write,
        mock_delete,
        mock_load,
        mock_dynamodb,
        mock_s3,
    ):
        """Test replace_existing=False skips delete."""
        mock_load.return_value = pd.DataFrame({
            "player_id": [350],
            "gameweek": [20],
            "predicted_points": [8.5],
        })
        mock_batch_write.return_value = {"items_written": 1, "batches": 1}

        mock_table = MagicMock()
        mock_dynamodb.return_value.Table.return_value = mock_table

        event = {"gameweek": 20, "season": "2024_25", "replace_existing": False}

        result = handler(event, None)

        assert result["statusCode"] == 200
        assert result["items_deleted"] == 0
        mock_delete.assert_not_called()

    @patch("lambdas.prediction_loader.handler.get_s3_client")
    @patch("lambdas.prediction_loader.handler.get_dynamodb_resource")
    @patch("lambdas.prediction_loader.handler.load_predictions_from_s3")
    @patch("lambdas.prediction_loader.handler.delete_gameweek_predictions")
    @patch("lambdas.prediction_loader.handler.batch_write_predictions")
    def test_custom_predictions_key(
        self,
        mock_batch_write,
        mock_delete,
        mock_load,
        mock_dynamodb,
        mock_s3,
    ):
        """Test custom predictions_key is used."""
        mock_load.return_value = pd.DataFrame({
            "player_id": [350],
            "gameweek": [20],
            "predicted_points": [8.5],
        })
        mock_delete.return_value = 0
        mock_batch_write.return_value = {"items_written": 1, "batches": 1}

        mock_table = MagicMock()
        mock_dynamodb.return_value.Table.return_value = mock_table

        custom_key = "custom/path/predictions.parquet"
        event = {
            "gameweek": 20,
            "season": "2024_25",
            "predictions_key": custom_key,
        }

        result = handler(event, None)

        assert result["statusCode"] == 200
        assert result["predictions_key"] == custom_key

    @patch("lambdas.prediction_loader.handler.get_s3_client")
    @patch("lambdas.prediction_loader.handler.get_dynamodb_resource")
    @patch("lambdas.prediction_loader.handler.load_predictions_from_s3")
    @patch("lambdas.prediction_loader.handler.delete_gameweek_predictions")
    @patch("lambdas.prediction_loader.handler.batch_write_predictions")
    def test_default_predictions_key_format(
        self,
        mock_batch_write,
        mock_delete,
        mock_load,
        mock_dynamodb,
        mock_s3,
    ):
        """Test default predictions_key format."""
        mock_load.return_value = pd.DataFrame({
            "player_id": [350],
            "gameweek": [20],
            "predicted_points": [8.5],
        })
        mock_delete.return_value = 0
        mock_batch_write.return_value = {"items_written": 1, "batches": 1}

        mock_table = MagicMock()
        mock_dynamodb.return_value.Table.return_value = mock_table

        event = {"gameweek": 20, "season": "2024_25"}

        result = handler(event, None)

        expected_key = "predictions/season_2024_25/gw20_predictions.parquet"
        assert result["predictions_key"] == expected_key
