"""
Unit tests for SageMaker training module.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import xgboost as xgb

from sagemaker.train_local import (
    DEFAULT_HYPERPARAMS,
    FEATURE_COLS,
    TARGET_COL,
    evaluate_model,
    get_feature_importance,
    load_model,
    load_training_data,
    save_model,
    train_model,
    validate_features,
)


class TestFeatureColumns:
    """Tests for feature column definitions."""

    def test_feature_cols_count(self):
        """Verify correct number of feature columns."""
        assert len(FEATURE_COLS) == 19

    def test_feature_cols_names(self):
        """Verify feature column names match expected."""
        expected = [
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
            "ict_index_last_3",
            "threat_last_3",
            "creativity_last_3",
            "opponent_attack_strength",
            "opponent_defence_strength",
            "selected_by_percent",
        ]
        assert FEATURE_COLS == expected

    def test_target_col_name(self):
        """Verify target column name."""
        assert TARGET_COL == "actual_points"


class TestValidateFeatures:
    """Tests for validate_features function."""

    @pytest.fixture
    def valid_dataframe(self):
        """Create a valid training DataFrame."""
        return pd.DataFrame(
            {
                "points_last_3": [7.3, 5.2, 8.0],
                "points_last_5": [6.8, 4.9, 7.5],
                "minutes_pct": [0.95, 0.88, 1.0],
                "form_score": [8.5, 5.8, 7.9],
                "opponent_strength": [3, 4, 2],
                "home_away": [1, 0, 1],
                "chance_of_playing": [100, 75, 100],
                "form_x_difficulty": [25.5, 23.2, 15.8],
                "position": [3, 4, 3],
                "goals_last_3": [0.67, 1.0, 0.33],
                "assists_last_3": [0.33, 0.0, 0.67],
                "clean_sheets_last_3": [0.0, 0.0, 0.33],
                "bps_last_3": [28.0, 32.0, 25.0],
                "ict_index_last_3": [85.3, 92.1, 78.5],
                "threat_last_3": [45.0, 60.0, 38.0],
                "creativity_last_3": [55.0, 30.0, 62.0],
                "opponent_attack_strength": [1200, 1350, 1100],
                "opponent_defence_strength": [1250, 1300, 1150],
                "selected_by_percent": [45.3, 52.1, 38.7],
                "actual_points": [8, 4, 12],
            }
        )

    def test_valid_dataframe_passes(self, valid_dataframe):
        """Verify valid DataFrame passes validation."""
        # Should not raise
        validate_features(valid_dataframe)

    def test_missing_feature_column_raises(self, valid_dataframe):
        """Verify missing feature column raises ValueError."""
        df = valid_dataframe.drop(columns=["points_last_3"])
        with pytest.raises(ValueError, match="Missing feature columns"):
            validate_features(df)

    def test_missing_target_column_raises(self, valid_dataframe):
        """Verify missing target column raises ValueError."""
        df = valid_dataframe.drop(columns=["actual_points"])
        with pytest.raises(ValueError, match="Missing target column"):
            validate_features(df)

    def test_multiple_missing_columns_raises(self):
        """Verify multiple missing columns are reported."""
        df = pd.DataFrame(
            {
                "points_last_3": [1, 2, 3],
                "actual_points": [4, 5, 6],
            }
        )
        with pytest.raises(ValueError, match="Missing feature columns"):
            validate_features(df)


class TestLoadTrainingData:
    """Tests for load_training_data function."""

    def test_load_single_parquet_file(self, tmp_path):
        """Test loading a single Parquet file."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        parquet_path = tmp_path / "data.parquet"
        df.to_parquet(parquet_path, index=False)

        result = load_training_data(data_path=str(parquet_path))

        assert len(result) == 3
        assert list(result.columns) == ["col1", "col2"]

    def test_load_from_directory(self, tmp_path):
        """Test loading multiple Parquet files from directory."""
        df1 = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        df2 = pd.DataFrame({"col1": [5, 6], "col2": [7, 8]})

        (tmp_path / "gw1_features_training.parquet").write_bytes(
            df1.to_parquet(index=False)
        )
        (tmp_path / "gw2_features_training.parquet").write_bytes(
            df2.to_parquet(index=False)
        )

        result = load_training_data(data_dir=str(tmp_path))

        assert len(result) == 4  # Combined rows

    def test_file_not_found_raises(self):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            load_training_data(data_path="/nonexistent/path.parquet")

    def test_directory_not_found_raises(self):
        """Test FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError, match="Data directory not found"):
            load_training_data(data_dir="/nonexistent/dir")

    def test_no_path_raises(self):
        """Test ValueError when no path provided."""
        with pytest.raises(ValueError, match="Either data_path or data_dir"):
            load_training_data()

    def test_empty_directory_raises(self, tmp_path):
        """Test FileNotFoundError for directory with no parquet files."""
        with pytest.raises(FileNotFoundError, match="No training parquet files"):
            load_training_data(data_dir=str(tmp_path))


class TestTrainModel:
    """Tests for train_model function."""

    @pytest.fixture
    def training_dataframe(self):
        """Create a training DataFrame with enough samples."""
        # Need enough samples for train/test split
        return pd.DataFrame(
            {
                "points_last_3": [7.3, 5.2, 8.0, 4.5, 6.7, 9.1, 3.2, 7.8, 5.5, 8.9],
                "points_last_5": [6.8, 4.9, 7.5, 4.2, 6.1, 8.5, 3.0, 7.2, 5.1, 8.3],
                "minutes_pct": [0.95, 0.88, 1.0, 0.7, 0.9, 1.0, 0.5, 0.95, 0.85, 1.0],
                "form_score": [8.5, 5.8, 7.9, 4.0, 6.5, 9.0, 3.5, 7.5, 5.5, 8.8],
                "opponent_strength": [3, 4, 2, 5, 3, 2, 4, 3, 4, 2],
                "home_away": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
                "chance_of_playing": [100, 75, 100, 50, 100, 100, 25, 100, 75, 100],
                "form_x_difficulty": [
                    25.5,
                    23.2,
                    15.8,
                    20.0,
                    19.5,
                    18.0,
                    14.0,
                    22.5,
                    22.0,
                    17.6,
                ],
                "position": [3, 4, 3, 3, 3, 4, 3, 4, 4, 4],
                "goals_last_3": [
                    0.67,
                    1.0,
                    0.33,
                    0.0,
                    0.33,
                    1.33,
                    0.0,
                    0.67,
                    0.33,
                    0.33,
                ],
                "assists_last_3": [
                    0.33,
                    0.0,
                    0.67,
                    0.33,
                    0.33,
                    0.0,
                    0.33,
                    0.0,
                    0.0,
                    0.33,
                ],
                "clean_sheets_last_3": [
                    0.0,
                    0.0,
                    0.33,
                    0.33,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "bps_last_3": [
                    28.0,
                    32.0,
                    25.0,
                    18.0,
                    22.0,
                    35.0,
                    15.0,
                    24.0,
                    20.0,
                    19.0,
                ],
                "ict_index_last_3": [
                    85.3,
                    92.1,
                    78.5,
                    50.2,
                    65.0,
                    95.0,
                    40.0,
                    80.0,
                    55.0,
                    88.0,
                ],
                "threat_last_3": [
                    45.0,
                    60.0,
                    38.0,
                    25.0,
                    35.0,
                    65.0,
                    20.0,
                    50.0,
                    30.0,
                    55.0,
                ],
                "creativity_last_3": [
                    55.0,
                    30.0,
                    62.0,
                    40.0,
                    48.0,
                    28.0,
                    35.0,
                    42.0,
                    38.0,
                    50.0,
                ],
                "opponent_attack_strength": [
                    1200,
                    1350,
                    1100,
                    1280,
                    1200,
                    1150,
                    1300,
                    1250,
                    1180,
                    1100,
                ],
                "opponent_defence_strength": [
                    1250,
                    1300,
                    1150,
                    1320,
                    1200,
                    1180,
                    1280,
                    1220,
                    1200,
                    1150,
                ],
                "selected_by_percent": [
                    45.3,
                    52.1,
                    38.7,
                    22.0,
                    30.5,
                    55.0,
                    15.0,
                    40.0,
                    25.0,
                    48.0,
                ],
                "actual_points": [8, 4, 12, 2, 6, 15, 1, 9, 5, 11],
            }
        )

    def test_train_returns_model(self, training_dataframe):
        """Verify training returns a valid XGBoost model."""
        model, X_test, y_test = train_model(training_dataframe)

        assert isinstance(model, xgb.XGBRegressor)
        assert hasattr(model, "predict")

    def test_train_returns_test_data(self, training_dataframe):
        """Verify training returns test data for evaluation."""
        model, X_test, y_test = train_model(training_dataframe, test_size=0.2)

        assert len(X_test) == 2  # 20% of 10 samples
        assert len(y_test) == 2

    def test_custom_hyperparameters(self, training_dataframe):
        """Test training with custom hyperparameters."""
        hyperparams = {"n_estimators": 50, "max_depth": 3}
        model, _, _ = train_model(training_dataframe, hyperparams=hyperparams)

        assert model.n_estimators == 50
        assert model.max_depth == 3

    def test_predictions_are_numeric(self, training_dataframe):
        """Verify model produces numeric predictions."""
        import numpy as np

        model, X_test, _ = train_model(training_dataframe)
        predictions = model.predict(X_test)

        assert len(predictions) == len(X_test)
        assert all(np.issubdtype(type(p), np.number) for p in predictions)


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    @pytest.fixture
    def trained_model_and_data(self):
        """Create a trained model and test data."""
        df = pd.DataFrame(
            {
                "points_last_3": [7.3, 5.2, 8.0, 4.5, 6.7, 9.1, 3.2, 7.8, 5.5, 8.9],
                "points_last_5": [6.8, 4.9, 7.5, 4.2, 6.1, 8.5, 3.0, 7.2, 5.1, 8.3],
                "minutes_pct": [0.95, 0.88, 1.0, 0.7, 0.9, 1.0, 0.5, 0.95, 0.85, 1.0],
                "form_score": [8.5, 5.8, 7.9, 4.0, 6.5, 9.0, 3.5, 7.5, 5.5, 8.8],
                "opponent_strength": [3, 4, 2, 5, 3, 2, 4, 3, 4, 2],
                "home_away": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
                "chance_of_playing": [100, 75, 100, 50, 100, 100, 25, 100, 75, 100],
                "form_x_difficulty": [
                    25.5,
                    23.2,
                    15.8,
                    20.0,
                    19.5,
                    18.0,
                    14.0,
                    22.5,
                    22.0,
                    17.6,
                ],
                "position": [3, 4, 3, 3, 3, 4, 3, 4, 4, 4],
                "goals_last_3": [
                    0.67,
                    1.0,
                    0.33,
                    0.0,
                    0.33,
                    1.33,
                    0.0,
                    0.67,
                    0.33,
                    0.33,
                ],
                "assists_last_3": [
                    0.33,
                    0.0,
                    0.67,
                    0.33,
                    0.33,
                    0.0,
                    0.33,
                    0.0,
                    0.0,
                    0.33,
                ],
                "clean_sheets_last_3": [
                    0.0,
                    0.0,
                    0.33,
                    0.33,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "bps_last_3": [
                    28.0,
                    32.0,
                    25.0,
                    18.0,
                    22.0,
                    35.0,
                    15.0,
                    24.0,
                    20.0,
                    19.0,
                ],
                "ict_index_last_3": [
                    85.3,
                    92.1,
                    78.5,
                    50.2,
                    65.0,
                    95.0,
                    40.0,
                    80.0,
                    55.0,
                    88.0,
                ],
                "threat_last_3": [
                    45.0,
                    60.0,
                    38.0,
                    25.0,
                    35.0,
                    65.0,
                    20.0,
                    50.0,
                    30.0,
                    55.0,
                ],
                "creativity_last_3": [
                    55.0,
                    30.0,
                    62.0,
                    40.0,
                    48.0,
                    28.0,
                    35.0,
                    42.0,
                    38.0,
                    50.0,
                ],
                "opponent_attack_strength": [
                    1200,
                    1350,
                    1100,
                    1280,
                    1200,
                    1150,
                    1300,
                    1250,
                    1180,
                    1100,
                ],
                "opponent_defence_strength": [
                    1250,
                    1300,
                    1150,
                    1320,
                    1200,
                    1180,
                    1280,
                    1220,
                    1200,
                    1150,
                ],
                "selected_by_percent": [
                    45.3,
                    52.1,
                    38.7,
                    22.0,
                    30.5,
                    55.0,
                    15.0,
                    40.0,
                    25.0,
                    48.0,
                ],
                "actual_points": [8, 4, 12, 2, 6, 15, 1, 9, 5, 11],
            }
        )
        model, X_test, y_test = train_model(df)
        return model, X_test, y_test

    def test_returns_expected_metrics(self, trained_model_and_data):
        """Verify evaluation returns expected metric keys."""
        model, X_test, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "mean_prediction" in metrics
        assert "mean_actual" in metrics

    def test_metrics_are_numeric(self, trained_model_and_data):
        """Verify all metrics are numeric."""
        model, X_test, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test)

        for key, value in metrics.items():
            assert isinstance(value, (int, float)), f"{key} is not numeric"

    def test_mae_is_non_negative(self, trained_model_and_data):
        """Verify MAE is non-negative."""
        model, X_test, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test)

        assert metrics["mae"] >= 0

    def test_rmse_is_non_negative(self, trained_model_and_data):
        """Verify RMSE is non-negative."""
        model, X_test, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test)

        assert metrics["rmse"] >= 0


class TestGetFeatureImportance:
    """Tests for get_feature_importance function."""

    @pytest.fixture
    def trained_model(self):
        """Create a simple trained model."""
        df = pd.DataFrame(
            {
                "points_last_3": [7.3, 5.2, 8.0, 4.5, 6.7],
                "points_last_5": [6.8, 4.9, 7.5, 4.2, 6.1],
                "minutes_pct": [0.95, 0.88, 1.0, 0.7, 0.9],
                "form_score": [8.5, 5.8, 7.9, 4.0, 6.5],
                "opponent_strength": [3, 4, 2, 5, 3],
                "home_away": [1, 0, 1, 0, 1],
                "chance_of_playing": [100, 75, 100, 50, 100],
                "form_x_difficulty": [25.5, 23.2, 15.8, 20.0, 19.5],
                "position": [3, 4, 3, 3, 3],
                "goals_last_3": [0.67, 1.0, 0.33, 0.0, 0.33],
                "assists_last_3": [0.33, 0.0, 0.67, 0.33, 0.33],
                "clean_sheets_last_3": [0.0, 0.0, 0.33, 0.33, 0.0],
                "bps_last_3": [28.0, 32.0, 25.0, 18.0, 22.0],
                "ict_index_last_3": [85.3, 92.1, 78.5, 50.2, 65.0],
                "threat_last_3": [45.0, 60.0, 38.0, 25.0, 35.0],
                "creativity_last_3": [55.0, 30.0, 62.0, 40.0, 48.0],
                "opponent_attack_strength": [1200, 1350, 1100, 1280, 1200],
                "opponent_defence_strength": [1250, 1300, 1150, 1320, 1200],
                "selected_by_percent": [45.3, 52.1, 38.7, 22.0, 30.5],
                "actual_points": [8, 4, 12, 2, 6],
            }
        )
        model, _, _ = train_model(df, test_size=0.2)
        return model

    def test_returns_all_features(self, trained_model):
        """Verify all feature importances are returned."""
        importance = get_feature_importance(trained_model)

        assert len(importance) == len(FEATURE_COLS)
        for col in FEATURE_COLS:
            assert col in importance

    def test_importance_values_are_numeric(self, trained_model):
        """Verify importance values are numeric."""
        import numpy as np

        importance = get_feature_importance(trained_model)

        for value in importance.values():
            assert np.issubdtype(type(value), np.number)

    def test_importance_sums_to_approximately_one(self, trained_model):
        """Verify feature importances sum to approximately 1."""
        importance = get_feature_importance(trained_model)
        total = sum(importance.values())

        assert total == pytest.approx(1.0, rel=0.01)


class TestSaveLoadModel:
    """Tests for save_model and load_model functions."""

    @pytest.fixture
    def trained_model(self):
        """Create a simple trained model."""
        df = pd.DataFrame(
            {
                "points_last_3": [7.3, 5.2, 8.0, 4.5, 6.7],
                "points_last_5": [6.8, 4.9, 7.5, 4.2, 6.1],
                "minutes_pct": [0.95, 0.88, 1.0, 0.7, 0.9],
                "form_score": [8.5, 5.8, 7.9, 4.0, 6.5],
                "opponent_strength": [3, 4, 2, 5, 3],
                "home_away": [1, 0, 1, 0, 1],
                "chance_of_playing": [100, 75, 100, 50, 100],
                "form_x_difficulty": [25.5, 23.2, 15.8, 20.0, 19.5],
                "position": [3, 4, 3, 3, 3],
                "goals_last_3": [0.67, 1.0, 0.33, 0.0, 0.33],
                "assists_last_3": [0.33, 0.0, 0.67, 0.33, 0.33],
                "clean_sheets_last_3": [0.0, 0.0, 0.33, 0.33, 0.0],
                "bps_last_3": [28.0, 32.0, 25.0, 18.0, 22.0],
                "ict_index_last_3": [85.3, 92.1, 78.5, 50.2, 65.0],
                "threat_last_3": [45.0, 60.0, 38.0, 25.0, 35.0],
                "creativity_last_3": [55.0, 30.0, 62.0, 40.0, 48.0],
                "opponent_attack_strength": [1200, 1350, 1100, 1280, 1200],
                "opponent_defence_strength": [1250, 1300, 1150, 1320, 1200],
                "selected_by_percent": [45.3, 52.1, 38.7, 22.0, 30.5],
                "actual_points": [8, 4, 12, 2, 6],
            }
        )
        model, _, _ = train_model(df, test_size=0.2)
        return model

    def test_save_to_directory(self, trained_model, tmp_path):
        """Test saving model to directory creates model.xgb."""
        output_path = save_model(trained_model, str(tmp_path))

        assert Path(output_path).exists()
        assert output_path.endswith("model.xgb")

    def test_save_to_file_path(self, trained_model, tmp_path):
        """Test saving model to specific file path."""
        model_file = tmp_path / "custom_model.xgb"
        output_path = save_model(trained_model, str(model_file))

        assert Path(output_path).exists()
        assert output_path == str(model_file)

    def test_save_creates_parent_directories(self, trained_model, tmp_path):
        """Test saving creates parent directories if needed."""
        nested_path = tmp_path / "nested" / "dir" / "model.xgb"
        output_path = save_model(trained_model, str(nested_path))

        assert Path(output_path).exists()

    def test_load_model_roundtrip(self, trained_model, tmp_path):
        """Test model can be saved and loaded."""
        model_path = save_model(trained_model, str(tmp_path))
        loaded_model = load_model(model_path)

        assert isinstance(loaded_model, xgb.XGBRegressor)
        assert hasattr(loaded_model, "predict")

    def test_loaded_model_produces_same_predictions(self, trained_model, tmp_path):
        """Test loaded model produces same predictions as original."""
        X = pd.DataFrame(
            {
                "points_last_3": [7.0],
                "points_last_5": [6.5],
                "minutes_pct": [0.9],
                "form_score": [7.5],
                "opponent_strength": [3],
                "home_away": [1],
                "chance_of_playing": [100],
                "form_x_difficulty": [22.5],
                "position": [3],
                "goals_last_3": [0.5],
                "assists_last_3": [0.33],
                "clean_sheets_last_3": [0.0],
                "bps_last_3": [25.0],
                "ict_index_last_3": [80.0],
                "threat_last_3": [45.0],
                "creativity_last_3": [50.0],
                "opponent_attack_strength": [1200],
                "opponent_defence_strength": [1250],
                "selected_by_percent": [40.0],
            }
        )

        original_pred = trained_model.predict(X)

        model_path = save_model(trained_model, str(tmp_path))
        loaded_model = load_model(model_path)
        loaded_pred = loaded_model.predict(X)

        assert original_pred[0] == pytest.approx(loaded_pred[0], rel=0.001)


class TestDefaultHyperparameters:
    """Tests for default hyperparameters."""

    def test_default_objective(self):
        """Verify default objective is regression."""
        assert DEFAULT_HYPERPARAMS["objective"] == "reg:squarederror"

    def test_default_n_estimators(self):
        """Verify default n_estimators."""
        assert DEFAULT_HYPERPARAMS["n_estimators"] == 100

    def test_default_max_depth(self):
        """Verify default max_depth."""
        assert DEFAULT_HYPERPARAMS["max_depth"] == 6

    def test_default_learning_rate(self):
        """Verify default learning_rate."""
        assert DEFAULT_HYPERPARAMS["learning_rate"] == 0.1
