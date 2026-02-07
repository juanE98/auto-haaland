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
    _has_temporal_columns,
    _temporal_train_test_split,
    evaluate_model,
    get_feature_importance,
    load_model,
    load_training_data,
    save_model,
    train_model,
    train_model_temporal,
    tune_hyperparameters,
    validate_features,
)


class TestFeatureColumns:
    """Tests for feature column definitions."""

    def test_feature_cols_count(self):
        """Verify correct number of feature columns (207 features)."""
        assert len(FEATURE_COLS) == 207

    def test_feature_cols_key_names(self):
        """Verify key feature column names are present."""
        expected_subset = [
            "points_last_1",
            "points_last_3",
            "points_last_5",
            "form_score",
            "opponent_strength",
            "home_away",
            "chance_of_playing",
            "position",
            "now_cost",
            "minutes_pct",
            "form_x_difficulty",
            "points_per_90",
            "goal_contributions_last_3",
            "points_volatility",
            # New Phase 1 features
            "expected_goals_last_3",
            "expected_assists_last_3",
            "minutes_last_3",
            "starts_last_3",
            "red_cards_last_5",
            "points_last_10",
            # New Phase 2 bootstrap features
            "ep_this",
            "ep_next",
            "points_per_game",
            "status_available",
            "dreamteam_count",
            "transfers_in_event",
            "penalties_order",
            "total_points_rank_pct",
            # New Phase 3 team/opponent features
            "team_form_score",
            "team_strength_overall",
            "team_league_position",
            "opp_goals_conceded_last_3",
            "opp_clean_sheets_rate",
            "opp_defensive_rating",
            # New Phase 4 fixture/position/interaction features
            "fdr_current",
            "is_double_gameweek",
            "days_since_last_game",
            "gk_saves_per_90",
            "def_clean_sheet_rate",
            "mid_goal_involvement_rate",
            "fwd_conversion_rate",
            "form_x_fixture_difficulty",
            "momentum_score",
        ]
        for col in expected_subset:
            assert col in FEATURE_COLS, f"Missing: {col}"

    def test_target_col_name(self):
        """Verify target column name."""
        assert TARGET_COL == "actual_points"


class TestValidateFeatures:
    """Tests for validate_features function."""

    def test_valid_dataframe_passes(self, training_dataframe_5):
        """Verify valid DataFrame passes validation."""
        # Should not raise
        validate_features(training_dataframe_5)

    def test_missing_feature_column_raises(self, training_dataframe_5):
        """Verify missing feature column raises ValueError."""
        df = training_dataframe_5.drop(columns=["points_last_3"])
        with pytest.raises(ValueError, match="Missing feature columns"):
            validate_features(df)

    def test_missing_target_column_raises(self, training_dataframe_5):
        """Verify missing target column raises ValueError."""
        df = training_dataframe_5.drop(columns=["actual_points"])
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

    def test_train_returns_model(self, training_dataframe_10):
        """Verify training returns a valid XGBoost model."""
        model, X_test, y_test = train_model(training_dataframe_10)

        assert isinstance(model, xgb.XGBRegressor)
        assert hasattr(model, "predict")

    def test_train_returns_test_data(self, training_dataframe_10):
        """Verify training returns test data for evaluation."""
        model, X_test, y_test = train_model(training_dataframe_10, test_size=0.2)

        assert len(X_test) == 2  # 20% of 10 samples
        assert len(y_test) == 2

    def test_custom_hyperparameters(self, training_dataframe_10):
        """Test training with custom hyperparameters."""
        hyperparams = {"n_estimators": 50, "max_depth": 3}
        model, _, _ = train_model(training_dataframe_10, hyperparams=hyperparams)

        assert model.n_estimators == 50
        assert model.max_depth == 3

    def test_predictions_are_numeric(self, training_dataframe_10):
        """Verify model produces numeric predictions."""
        import numpy as np

        model, X_test, _ = train_model(training_dataframe_10)
        predictions = model.predict(X_test)

        assert len(predictions) == len(X_test)
        assert all(np.issubdtype(type(p), np.number) for p in predictions)


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    @pytest.fixture
    def trained_model_and_data(self, training_dataframe_10):
        """Create a trained model and test data."""
        model, X_test, y_test = train_model(training_dataframe_10)
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
    def trained_model(self, training_dataframe_5):
        """Create a simple trained model."""
        model, _, _ = train_model(training_dataframe_5, test_size=0.2)
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
    def trained_model(self, training_dataframe_5):
        """Create a simple trained model."""
        model, _, _ = train_model(training_dataframe_5, test_size=0.2)
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

    def test_loaded_model_produces_same_predictions(
        self, trained_model, training_dataframe_5, tmp_path
    ):
        """Test loaded model produces same predictions as original."""
        # Use first row of training data for prediction test
        X = training_dataframe_5[FEATURE_COLS].iloc[[0]]

        original_pred = trained_model.predict(X)

        model_path = save_model(trained_model, str(tmp_path))
        loaded_model = load_model(model_path)
        loaded_pred = loaded_model.predict(X)

        assert original_pred[0] == pytest.approx(loaded_pred[0], rel=0.001)


class TestDefaultHyperparameters:
    """Tests for default hyperparameters."""

    def test_default_objective(self):
        """Verify default objective is quantile regression."""
        assert DEFAULT_HYPERPARAMS["objective"] == "reg:quantileerror"

    def test_default_quantile_alpha(self):
        """Verify default quantile_alpha targets 75th percentile."""
        assert DEFAULT_HYPERPARAMS["quantile_alpha"] == 0.75

    def test_default_n_estimators(self):
        """Verify default n_estimators."""
        assert DEFAULT_HYPERPARAMS["n_estimators"] == 100

    def test_default_max_depth(self):
        """Verify default max_depth."""
        assert DEFAULT_HYPERPARAMS["max_depth"] == 6

    def test_default_learning_rate(self):
        """Verify default learning_rate."""
        assert DEFAULT_HYPERPARAMS["learning_rate"] == 0.1

    def test_default_subsample(self):
        """Verify default subsample for regularisation."""
        assert DEFAULT_HYPERPARAMS["subsample"] == 0.8

    def test_default_colsample_bytree(self):
        """Verify default colsample_bytree for regularisation."""
        assert DEFAULT_HYPERPARAMS["colsample_bytree"] == 0.5


class TestTemporalHelpers:
    """Tests for temporal split helper functions."""

    def test_has_temporal_columns_true(self, training_dataframe_10):
        """Verify detection of gameweek and season columns."""
        assert _has_temporal_columns(training_dataframe_10)

    def test_has_temporal_columns_false_no_gameweek(self, training_dataframe_10):
        """Verify detection fails when gameweek is missing."""
        df = training_dataframe_10.drop(columns=["gameweek"])
        assert not _has_temporal_columns(df)

    def test_has_temporal_columns_false_no_season(self, training_dataframe_10):
        """Verify detection fails when season is missing."""
        df = training_dataframe_10.drop(columns=["season"])
        assert not _has_temporal_columns(df)

    def test_temporal_split_preserves_all_rows(self, training_dataframe_100):
        """Verify temporal split preserves all data rows."""
        train_df, test_df = _temporal_train_test_split(
            training_dataframe_100, test_fraction=0.2
        )
        assert len(train_df) + len(test_df) == len(training_dataframe_100)

    def test_temporal_split_respects_fraction(self, training_dataframe_100):
        """Verify temporal split produces roughly correct sizes."""
        train_df, test_df = _temporal_train_test_split(
            training_dataframe_100, test_fraction=0.2
        )
        # Gameweek-boundary split won't be exact 80/20 by row count,
        # but should be approximately correct
        total = len(train_df) + len(test_df)
        assert total == len(training_dataframe_100)
        test_ratio = len(test_df) / total
        assert 0.10 <= test_ratio <= 0.35

    def test_temporal_split_no_gameweek_in_both_sets(self, training_dataframe_100):
        """Verify no gameweek appears in both train and test sets."""
        train_df, test_df = _temporal_train_test_split(
            training_dataframe_100, test_fraction=0.2
        )
        train_keys = set(
            train_df["season"].astype(str)
            + "_"
            + train_df["gameweek"].astype(str).str.zfill(2)
        )
        test_keys = set(
            test_df["season"].astype(str)
            + "_"
            + test_df["gameweek"].astype(str).str.zfill(2)
        )
        assert train_keys.isdisjoint(test_keys)

    def test_temporal_split_chronological_order(self, training_dataframe_100):
        """Verify train set precedes test set chronologically."""
        train_df, test_df = _temporal_train_test_split(
            training_dataframe_100, test_fraction=0.2
        )
        # Build sort keys for comparison
        train_max = (
            train_df["season"].astype(str)
            + "_"
            + train_df["gameweek"].astype(str).str.zfill(2)
        ).max()
        test_min = (
            test_df["season"].astype(str)
            + "_"
            + test_df["gameweek"].astype(str).str.zfill(2)
        ).min()
        assert train_max <= test_min


class TestTrainModelTemporal:
    """Tests for train_model_temporal function."""

    def test_returns_model_with_temporal_data(self, training_dataframe_100):
        """Verify temporal training returns a valid model."""
        model, X_test, y_test, split_info = train_model_temporal(training_dataframe_100)

        assert isinstance(model, xgb.XGBRegressor)
        assert hasattr(model, "predict")
        assert split_info["split_type"] == "temporal"

    def test_returns_test_data(self, training_dataframe_100):
        """Verify temporal training returns test data."""
        model, X_test, y_test, split_info = train_model_temporal(
            training_dataframe_100, test_fraction=0.2
        )

        # Gameweek-boundary split produces approximate sizes
        assert len(X_test) > 0
        assert len(X_test) == len(y_test)
        total = split_info["train_samples"] + split_info["test_samples"]
        assert total == len(training_dataframe_100)

    def test_split_info_contains_metadata(self, training_dataframe_100):
        """Verify split_info contains expected metadata."""
        _, _, _, split_info = train_model_temporal(training_dataframe_100)

        assert "split_type" in split_info
        assert "test_fraction" in split_info
        assert "train_samples" in split_info
        assert "test_samples" in split_info
        assert "train_seasons" in split_info
        assert "test_seasons" in split_info

    def test_falls_back_to_random_without_temporal_columns(self, training_dataframe_10):
        """Verify fallback to random split when columns are missing."""
        df = training_dataframe_10.drop(columns=["gameweek", "season"])
        model, X_test, y_test, split_info = train_model_temporal(df)

        assert isinstance(model, xgb.XGBRegressor)
        assert split_info["split_type"] == "random"

    def test_custom_hyperparameters(self, training_dataframe_100):
        """Verify custom hyperparameters are applied."""
        hyperparams = {"n_estimators": 50, "max_depth": 3}
        model, _, _, _ = train_model_temporal(
            training_dataframe_100, hyperparams=hyperparams
        )

        assert model.n_estimators == 50
        assert model.max_depth == 3

    def test_predictions_are_numeric(self, training_dataframe_100):
        """Verify temporally-trained model produces numeric predictions."""
        import numpy as np

        model, X_test, _, _ = train_model_temporal(training_dataframe_100)
        predictions = model.predict(X_test)

        assert len(predictions) == len(X_test)
        assert all(np.issubdtype(type(p), np.number) for p in predictions)


class TestTuneHyperparameters:
    """Tests for tune_hyperparameters function."""

    def test_returns_dict_with_expected_keys(self, training_dataframe_100):
        """Verify tuning returns a dict with required hyperparameter keys."""
        best_params = tune_hyperparameters(
            training_dataframe_100, n_trials=3, temporal=True
        )

        assert isinstance(best_params, dict)
        assert "n_estimators" in best_params
        assert "max_depth" in best_params
        assert "learning_rate" in best_params
        assert "objective" in best_params
        assert best_params["objective"] == "reg:quantileerror"

    def test_returns_valid_hyperparameter_ranges(self, training_dataframe_100):
        """Verify tuned values are within expected search space."""
        best_params = tune_hyperparameters(
            training_dataframe_100, n_trials=3, temporal=True
        )

        assert 100 <= best_params["n_estimators"] <= 500
        assert 3 <= best_params["max_depth"] <= 8
        assert 0.01 <= best_params["learning_rate"] <= 0.3
        assert 0.6 <= best_params["subsample"] <= 1.0
        assert 0.3 <= best_params["colsample_bytree"] <= 1.0

    def test_tuned_params_can_train_model(self, training_dataframe_100):
        """Verify tuned parameters can be used to train a model."""
        best_params = tune_hyperparameters(
            training_dataframe_100, n_trials=3, temporal=True
        )

        model, X_test, y_test, _ = train_model_temporal(
            training_dataframe_100, hyperparams=best_params
        )
        assert isinstance(model, xgb.XGBRegressor)
        assert len(model.predict(X_test)) == len(X_test)

    def test_works_with_random_split(self, training_dataframe_100):
        """Verify tuning works with random split."""
        best_params = tune_hyperparameters(
            training_dataframe_100, n_trials=3, temporal=False
        )

        assert isinstance(best_params, dict)
        assert "n_estimators" in best_params

    def test_falls_back_to_random_without_temporal_columns(
        self, training_dataframe_100
    ):
        """Verify tuning falls back to random split when columns missing."""
        df = training_dataframe_100.drop(columns=["gameweek", "season"])
        best_params = tune_hyperparameters(df, n_trials=3, temporal=True)

        assert isinstance(best_params, dict)
        assert "n_estimators" in best_params
