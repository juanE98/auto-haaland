"""
Unit tests for SageMaker training module.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

import sagemaker.train_local as train_local
from lambdas.common.feature_config import FEATURE_VERSION, MODEL_V5_FEATURES
from sagemaker import train as container_train
from sagemaker.train_local import (
    DEFAULT_HYPERPARAMS,
    FEATURE_COLS,
    TARGET_COL,
    _has_temporal_columns,
    _temporal_train_test_split,
    evaluate_haul_classifier,
    evaluate_model,
    fit_haul_calibration,
    fit_point_calibration,
    get_feature_importance,
    load_model,
    load_training_data,
    save_model,
    select_rank_weight,
    train_model,
    train_model_temporal,
    tune_hyperparameters,
    validate_features,
)


class TestFeatureColumns:
    """Tests for feature column definitions."""

    def test_feature_cols_count(self):
        """Verify the pruned model feature contract."""
        assert len(FEATURE_COLS) == 109

    def test_feature_cols_key_names(self):
        """Verify key feature column names are present."""
        expected_subset = [
            "points_last_1",
            "points_last_3",
            "points_last_5",
            "form_score",
            "opponent_strength",
            "home_away",
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
            # 2026/27 defensive contribution features
            "defensive_contribution_last_3",
            "clearances_blocks_interceptions_last_5",
            "tackles_last_5",
            "recoveries_last_5",
            "is_double_gameweek",
            "dgw_fixture_count",
            "history_game_count",
            "prior_points_per_90",
            "prior_season_available",
            "prior_ict_per_90",
            "prior_fwd_ict_per_90",
            "prior_team_goals_per_match",
            "prior_team_squad_minutes",
            "prior_team_context_available",
            "prior_opponent_goals_conceded_per_90",
            "prior_opponent_squad_minutes",
            "prior_opponent_context_available",
        ]
        for col in expected_subset:
            assert col in FEATURE_COLS, f"Missing: {col}"
        assert "ep_this" not in FEATURE_COLS
        assert "ep_next" not in FEATURE_COLS
        assert "team_form_score" not in FEATURE_COLS
        assert "opponent_attack_strength" not in FEATURE_COLS
        assert "opponent_defence_strength" not in FEATURE_COLS
        assert "selected_by_percent" not in FEATURE_COLS
        prior_ict_index = FEATURE_COLS.index("prior_ict_per_90")
        assert FEATURE_COLS[prior_ict_index + 1] == "prior_fwd_ict_per_90"

    def test_target_col_name(self):
        """Verify target column name."""
        assert TARGET_COL == "actual_points"

    def test_container_feature_contract_matches_shared_config(self):
        """Prevent the isolated SageMaker image from training a stale schema."""
        assert container_train.FEATURE_COLS == FEATURE_COLS
        assert container_train.FEATURE_VERSION == FEATURE_VERSION
        assert container_train.MODEL_CONTRACT_METADATA == {
            "feature_version": FEATURE_VERSION,
            "feature_count": len(FEATURE_COLS),
        }


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

    def test_duplicate_player_week_raises(self, training_dataframe_5):
        training_dataframe_5 = training_dataframe_5.assign(
            season="2025-26", gameweek=range(1, 6), player_id=100
        )
        df = pd.concat(
            [training_dataframe_5, training_dataframe_5.iloc[[0]]],
            ignore_index=True,
        )
        with pytest.raises(ValueError, match="Duplicate player-week rows"):
            validate_features(df)

    def test_missing_feature_values_raise(self, training_dataframe_5):
        df = training_dataframe_5.copy()
        df.loc[0, "points_last_3"] = None
        with pytest.raises(ValueError, match="Missing feature values"):
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
        """Verify point predictions target the conditional mean."""
        assert DEFAULT_HYPERPARAMS["objective"] == "reg:squarederror"

    def test_default_has_no_quantile_alpha(self):
        assert "quantile_alpha" not in DEFAULT_HYPERPARAMS

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


class TestCalibration:
    def test_point_calibration_recovers_linear_bias(self):
        calibration = fit_point_calibration([1, 2, 3], [3, 5, 7])
        assert calibration["point_slope"] == pytest.approx(2.0)
        assert calibration["point_intercept"] == pytest.approx(1.0)

    def test_haul_calibration_returns_finite_parameters(self):
        calibration = fit_haul_calibration([0.05, 0.2, 0.7, 0.9], [0, 0, 1, 1])
        assert calibration["haul_slope"] > 0
        assert abs(calibration["haul_intercept"]) < 10

    def test_rank_weight_selects_best_validation_ranking(self):
        weight = select_rank_weight(
            point_predictions=[3.0, 2.0, 1.0],
            haul_probabilities=[0.0, 1.0, 0.0],
            actual_points=[3.0, 5.0, 1.0],
            gameweeks=[1, 1, 1],
            candidates=(0.0, 1.0, 3.0),
        )
        assert weight == 3.0

    def test_rank_weight_uses_inference_quantisation_for_ties(self):
        weight = select_rank_weight(
            point_predictions=[2.004, 2.0],
            haul_probabilities=[0.0, 0.004],
            actual_points=[0.0, 10.0],
            gameweeks=[1, 1],
            candidates=(0.0, 1.0),
        )

        assert weight == 0.0


class TestRankingMetrics:
    def test_mean_gameweek_ndcg_scores_perfect_order_as_one(self):
        result = train_local.mean_gameweek_ndcg(
            actual_points=[10, 5, 1, 2, 8, 4],
            scores=[10, 5, 1, 2, 8, 4],
            gameweeks=[1, 1, 1, 2, 2, 2],
            k=2,
        )

        assert result == pytest.approx(1.0)

    def test_forward_haul_ndcg_uses_forward_hauls_and_skips_empty_gameweeks(self):
        result = train_local.mean_gameweek_forward_haul_ndcg(
            actual_points=[12, 11, 0, 11, 0],
            scores=[2, 3, 1, 2, 1],
            gameweeks=[1, 1, 1, 2, 2],
            positions=[4, 3, 4, 3, 4],
        )

        assert result == pytest.approx(1 / np.log2(3))

    def test_forward_haul_ndcg_scores_the_full_player_list(self):
        result = train_local.mean_gameweek_forward_haul_ndcg(
            actual_points=[0] * 15 + [10],
            scores=list(range(16, 0, -1)),
            gameweeks=[1] * 16,
            positions=[4] * 16,
        )

        assert result == pytest.approx(1 / np.log2(17))

    def test_haul_recall_at_k_is_averaged_by_gameweek(self):
        result = train_local.haul_recall_at_k(
            actual_points=[12, 10, 2, 11, 1, 0],
            scores=[3, 2, 1, 1, 3, 2],
            gameweeks=[1, 1, 1, 2, 2, 2],
            k=1,
        )

        assert result == pytest.approx(0.25)

    def test_haul_recall_at_k_uses_cohort_for_haul_denominator_only(self):
        result = train_local.haul_recall_at_k(
            actual_points=[12, 10, 2],
            scores=[2, 3, 1],
            gameweeks=[1, 1, 1],
            k=1,
            cohort=[True, False, True],
        )

        assert result == pytest.approx(0.0)

    def test_haul_evaluation_includes_brier_score(self):
        model = MagicMock()
        model.predict.return_value = [0, 1, 1, 0]
        model.predict_proba.return_value = np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.6, 0.4],
            ]
        )

        metrics = evaluate_haul_classifier(
            model,
            pd.DataFrame({"feature": [1, 2, 3, 4]}),
            pd.Series([0, 1, 1, 0]),
        )

        assert metrics["brier_score"] == pytest.approx(0.075)


def quality_gate_frame():
    """Return a minimal six-season frame with four gameweeks per season."""
    rows = []
    for season in [
        "2021-22",
        "2022-23",
        "2023-24",
        "2024-25",
        "2025-26",
        "2026-27",
    ]:
        for gameweek in range(1, 5):
            for player_id, points in enumerate([0, 2, 10, 12], start=1):
                row = {feature: 0.0 for feature in FEATURE_COLS}
                row.update(
                    {
                        "season": season,
                        "gameweek": gameweek,
                        "player_id": player_id,
                        "actual_points": points,
                        "position": 4 if points >= 10 else 3,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


class TestQualityGate:
    def test_quality_gate_uses_fresh_v4_baseline_with_identical_inputs(
        self, monkeypatch
    ):
        calls = []

        def train_candidate(
            train,
            validation,
            completed_test,
            feature_cols,
            hyperparams,
        ):
            calls.append(
                (
                    train,
                    validation,
                    completed_test,
                    feature_cols,
                    hyperparams,
                )
            )
            metrics = {
                "early_mae": 2.0,
                "early_ndcg_at_15": 0.5,
                "early_haul_recall_at_15": 0.4,
                "early_fwd_haul_recall_at_15": 0.4 + 0.1 * len(calls),
                "early_fwd_haul_ndcg": 0.4 + 0.1 * len(calls),
                "mae": 2.0,
                "brier_score": 0.08,
            }
            return metrics, {"rank_haul_weight": 1.5}

        monkeypatch.setattr(train_local, "_train_quality_candidate", train_candidate)
        monkeypatch.setattr(
            train_local,
            "score_early_folds",
            lambda *_: ({}, pd.DataFrame(columns=["round_key"])),
        )
        frame = pd.DataFrame(
            {
                **{feature: [0.0] * 10 for feature in FEATURE_COLS},
                "season": [
                    "2021-22",
                    "2021-22",
                    "2022-23",
                    "2023-24",
                    "2024-25",
                    "2024-25",
                    "2025-26",
                    "2025-26",
                    "2026-27",
                    "2026-27",
                ],
                "gameweek": [1, 4, 1, 1, 1, 4, 1, 4, 1, 2],
                "player_id": [1] * 10,
                "position": [4] * 10,
                "actual_points": [2, 10, 2, 2, 2, 10, 2, 10, 2, 10],
            }
        )
        hyperparams = {"n_estimators": 2, "max_depth": 2}

        train_local.run_quality_gate(frame, hyperparams)

        assert len(calls) == 2
        baseline, candidate = calls
        assert baseline[0].equals(candidate[0])
        assert baseline[1].equals(candidate[1])
        assert baseline[2].equals(candidate[2])
        assert set(baseline[0]["season"]) == {"2021-22", "2022-23", "2023-24"}
        assert set(baseline[1]["season"]) == {"2024-25"}
        assert set(baseline[2]["season"]) == {"2025-26"}
        assert set(baseline[1]["gameweek"]) == {1, 4}
        assert set(baseline[2]["gameweek"]) == {1, 4}
        assert baseline[3] == [
            feature for feature in FEATURE_COLS if feature not in MODEL_V5_FEATURES
        ]
        assert len(baseline[3]) == 101
        assert candidate[3] == FEATURE_COLS
        assert baseline[4] == candidate[4] == hyperparams

    def test_quality_candidates_use_validation_for_early_stopping(self, monkeypatch):
        regressor_params = []
        classifier_params = []
        real_regressor = xgb.XGBRegressor
        real_classifier = xgb.XGBClassifier

        def regressor_factory(**params):
            regressor_params.append(params)
            return real_regressor(**params)

        def classifier_factory(**params):
            classifier_params.append(params)
            return real_classifier(**params)

        monkeypatch.setattr(train_local.xgb, "XGBRegressor", regressor_factory)
        monkeypatch.setattr(train_local.xgb, "XGBClassifier", classifier_factory)

        rows = []
        for season in [
            "2021-22",
            "2022-23",
            "2023-24",
            "2024-25",
            "2025-26",
            "2026-27",
        ]:
            for gameweek in range(1, 5):
                for player_id, points in enumerate([0, 2, 10, 12], start=1):
                    row = {feature: 0.0 for feature in FEATURE_COLS}
                    row.update(
                        {
                            "season": season,
                            "gameweek": gameweek,
                            "player_id": player_id,
                            "actual_points": points,
                            "position": 4 if points >= 10 else 3,
                        }
                    )
                    rows.append(row)

        train_local.run_quality_gate(
            pd.DataFrame(rows), hyperparams={"n_estimators": 2, "max_depth": 2}
        )

        assert all(params["early_stopping_rounds"] == 20 for params in regressor_params)
        assert all(
            params["early_stopping_rounds"] == 20 for params in classifier_params
        )

    def test_split_uses_fixed_train_validation_and_test_seasons(self):
        frame = pd.DataFrame(
            {
                "season": [
                    "2021-22",
                    "2022-23",
                    "2023-24",
                    "2024-25",
                    "2025-26",
                    "2026-27",
                    "2026-27",
                ],
                "gameweek": [1, 1, 1, 1, 1, 1, 2],
            }
        )

        train, validation, completed_test, outer_test = (
            train_local.split_quality_gate_seasons(frame)
        )

        assert set(train["season"]) == {"2021-22", "2022-23", "2023-24"}
        assert set(validation["season"]) == {"2024-25"}
        assert set(completed_test["season"]) == {"2025-26"}
        assert set(outer_test["season"]) == {"2026-27"}

    def test_split_rejects_outer_holdout_before_gameweek_two(self):
        frame = pd.DataFrame(
            {
                "season": [
                    "2021-22",
                    "2022-23",
                    "2023-24",
                    "2024-25",
                    "2025-26",
                    "2026-27",
                ],
                "gameweek": [1, 1, 1, 1, 1, 1],
            }
        )

        with pytest.raises(ValueError, match="GW1 and GW2"):
            train_local.split_quality_gate_seasons(frame)

    def test_split_rejects_missing_mandated_training_season(self):
        frame = pd.DataFrame(
            {
                "season": [
                    "2021-22",
                    "2023-24",
                    "2024-25",
                    "2025-26",
                    "2026-27",
                    "2026-27",
                ],
                "gameweek": [1, 1, 1, 1, 1, 2],
            }
        )

        with pytest.raises(ValueError, match="2021-22 through 2023-24"):
            train_local.split_quality_gate_seasons(frame)

    def test_quality_gate_takes_early_metrics_from_pooled_folds(self, monkeypatch):
        full_metrics = {
            "mae": 1.0,
            "brier_score": 0.01,
            "ndcg_at_15": 0.3,
            "haul_recall_at_15": 0.2,
            "early_mae": 99.0,
            "early_ndcg_at_15": 99.0,
            "early_haul_recall_at_15": 99.0,
            "early_fwd_haul_recall_at_15": 99.0,
            "early_fwd_haul_ndcg": 99.0,
        }
        monkeypatch.setattr(
            train_local,
            "_train_quality_candidate",
            lambda *_: (dict(full_metrics), {"rank_haul_weight": 1.0}),
        )
        pooled = pd.DataFrame(
            {
                "actual": [10.0, 0.0, 10.0, 0.0],
                "rank_score": [2.0, 1.0, 1.0, 2.0],
                "round_key": ["2024-25:1", "2024-25:1", "2025-26:1", "2025-26:1"],
            }
        )
        early = {
            "early_mae": 1.5,
            "early_ndcg_at_15": 0.4,
            "early_haul_recall_at_15": 0.3,
            "early_fwd_haul_recall_at_15": 0.2,
            "early_fwd_haul_ndcg": 0.1,
        }
        monkeypatch.setattr(
            train_local, "score_early_folds", lambda *_: (dict(early), pooled)
        )

        result = train_local.run_quality_gate(quality_gate_frame())

        assert result["candidate"]["mae"] == 1.0
        assert result["candidate"]["early_ndcg_at_15"] == 0.4
        assert result["baseline"]["early_mae"] == 1.5
        assert result["early_gameweeks"] == 2
        assert result["early_ndcg_diff_se"] == 0.0

    def test_early_season_folds_roll_forward_without_leakage(self):
        frame = quality_gate_frame()

        folds = train_local.early_season_folds(frame)

        assert [
            (
                sorted(train["season"].unique()),
                sorted(validation["season"].unique()),
                sorted(test["season"].unique()),
            )
            for train, validation, test in folds
        ] == [
            (["2021-22"], ["2022-23"], ["2023-24"]),
            (["2021-22", "2022-23"], ["2023-24"], ["2024-25"]),
            (["2021-22", "2022-23", "2023-24"], ["2024-25"], ["2025-26"]),
            (
                ["2021-22", "2022-23", "2023-24", "2024-25"],
                ["2025-26"],
                ["2026-27"],
            ),
        ]
        for train, validation, test in folds:
            test_season = test["season"].iat[0]
            assert test_season not in set(train["season"])
            assert test_season not in set(validation["season"])
            assert test["gameweek"].max() <= train_local.QUALITY_EARLY_GAMEWEEK

    def test_predicted_rounds_are_keyed_by_season(self):
        regressor = MagicMock()
        regressor.predict.return_value = np.array([1.0, 2.0])
        classifier = MagicMock()
        classifier.predict_proba.return_value = np.array([[0.9, 0.1], [0.9, 0.1]])
        frame = pd.DataFrame(
            {
                "now_cost": [1.0, 2.0],
                "actual_points": [2, 10],
                "gameweek": [1, 1],
                "position": [3, 4],
                "season": ["2024-25", "2025-26"],
            }
        )

        predicted = train_local._predict_quality_frame(
            regressor,
            classifier,
            frame,
            ["now_cost"],
            {
                "point_slope": 1.0,
                "point_intercept": 0.0,
                "haul_slope": 1.0,
                "haul_intercept": 0.0,
                "rank_haul_weight": 1.0,
            },
        )

        assert predicted["round_key"].tolist() == ["2024-25:1", "2025-26:1"]

    def test_paired_standard_error_needs_two_rounds(self):
        frame = pd.DataFrame(
            {"actual": [10.0, 0.0], "rank_score": [2.0, 1.0], "round_key": [1, 1]}
        )

        assert train_local.paired_ndcg_standard_error(frame, frame) is None

    def test_quality_gate_applies_balanced_release_thresholds(self):
        baseline = {
            "early_mae": 2.0,
            "early_ndcg_at_15": 0.50,
            "early_haul_recall_at_15": 0.40,
            "early_fwd_haul_recall_at_15": 0.40,
            "early_fwd_haul_ndcg": 0.50,
            "mae": 2.0,
            "brier_score": 0.08,
        }
        candidate = {
            "early_mae": 2.05,
            "early_ndcg_at_15": 0.51,
            "early_haul_recall_at_15": 0.41,
            "early_fwd_haul_recall_at_15": 0.40,
            "early_fwd_haul_ndcg": 0.51,
            "mae": 2.02,
            "brier_score": 0.082,
        }

        result = train_local.evaluate_quality_gates(baseline, candidate)

        assert result["passed"] is True
        assert all(result["checks"].values())
        assert result["checks"]["early_fwd_haul_recall_non_regression"] is True
        assert result["checks"]["early_fwd_haul_ndcg_improved"] is True

    def test_quality_gate_rejects_early_ranking_regression(self):
        baseline = {
            "early_mae": 2.0,
            "early_ndcg_at_15": 0.50,
            "early_haul_recall_at_15": 0.40,
            "early_fwd_haul_recall_at_15": 0.40,
            "early_fwd_haul_ndcg": 0.50,
            "mae": 2.0,
            "brier_score": 0.08,
        }
        candidate = {**baseline, "early_ndcg_at_15": 0.49}

        result = train_local.evaluate_quality_gates(baseline, candidate)

        assert result["passed"] is False
        assert result["checks"]["early_ndcg_improved"] is False

    def test_quality_gate_rejects_early_forward_haul_recall_regression(self):
        baseline = {
            "early_mae": 2.0,
            "early_ndcg_at_15": 0.50,
            "early_haul_recall_at_15": 0.40,
            "early_fwd_haul_recall_at_15": 0.40,
            "early_fwd_haul_ndcg": 0.50,
            "mae": 2.0,
            "brier_score": 0.08,
        }
        candidate = {**baseline, "early_fwd_haul_recall_at_15": 0.39}

        result = train_local.evaluate_quality_gates(baseline, candidate)

        assert result["passed"] is False
        assert result["checks"]["early_fwd_haul_recall_non_regression"] is False

    def test_quality_gate_requires_forward_haul_ndcg_improvement(self):
        baseline = {
            "early_mae": 2.0,
            "early_ndcg_at_15": 0.50,
            "early_haul_recall_at_15": 0.40,
            "early_fwd_haul_recall_at_15": 0.40,
            "early_fwd_haul_ndcg": 0.50,
            "mae": 2.0,
            "brier_score": 0.08,
        }
        candidate = {
            **baseline,
            "early_ndcg_at_15": 0.51,
            "early_haul_recall_at_15": 0.41,
        }

        result = train_local.evaluate_quality_gates(baseline, candidate)

        assert result["passed"] is False
        assert result["checks"]["early_fwd_haul_ndcg_improved"] is False

    def test_score_backtest_predictions_reports_early_and_full_metrics(self):
        regressor = MagicMock()
        regressor.predict.return_value = np.array([1.0, 3.0, 2.0, 4.0])
        classifier = MagicMock()
        classifier.predict_proba.return_value = np.array(
            [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]]
        )
        frame = pd.DataFrame(
            {
                **{feature: [0.0] * 4 for feature in FEATURE_COLS},
                "actual_points": [0, 10, 2, 12],
                "gameweek": [1, 1, 4, 4],
                "position": [3, 4, 3, 4],
            }
        )
        metadata = {
            "point_slope": 1.0,
            "point_intercept": 0.0,
            "haul_slope": 1.0,
            "haul_intercept": 0.0,
            "rank_haul_weight": 1.0,
        }

        result = train_local.score_backtest_predictions(
            regressor,
            classifier,
            frame,
            FEATURE_COLS,
            metadata,
        )

        assert result["mae"] == pytest.approx(4.0)
        assert result["early_mae"] == pytest.approx(4.0)
        assert 0 <= result["ndcg_at_15"] <= 1
        assert 0 <= result["early_haul_recall_at_15"] <= 1
        assert result["early_fwd_haul_recall_at_15"] == pytest.approx(1.0)
        assert result["early_fwd_haul_ndcg"] == pytest.approx(1.0)

    def test_backtest_top_fifteen_uses_inference_quantisation_at_boundary(self):
        player_count = 16
        regressor = MagicMock()
        regressor.predict.return_value = np.array([2.004] * 15 + [2.0001], dtype=float)
        classifier = MagicMock()
        probabilities = np.array([0.0] * 15 + [0.004], dtype=float)
        classifier.predict_proba.return_value = np.column_stack(
            [1 - probabilities, probabilities]
        )
        frame = pd.DataFrame(
            {
                **{feature: [0.0] * player_count for feature in FEATURE_COLS},
                "actual_points": [0.0] * 15 + [10.0],
                "gameweek": [1] * player_count,
                "position": [3] * 15 + [4],
            }
        )

        result = train_local.score_backtest_predictions(
            regressor,
            classifier,
            frame,
            FEATURE_COLS,
            {
                "point_slope": 1.0,
                "point_intercept": 0.0,
                "haul_slope": 1.0,
                "haul_intercept": 0.0,
                "rank_haul_weight": 1.0,
            },
        )

        assert result["early_haul_recall_at_15"] == 0.0

    def test_quality_gate_compares_baseline_and_candidate_models(self):
        rows = []
        for season in [
            "2021-22",
            "2022-23",
            "2023-24",
            "2024-25",
            "2025-26",
            "2026-27",
        ]:
            for gameweek in range(1, 5):
                for player_id, points in enumerate([0, 2, 10, 12], start=1):
                    row = {feature: 0.0 for feature in FEATURE_COLS}
                    row.update(
                        {
                            "season": season,
                            "gameweek": gameweek,
                            "player_id": player_id,
                            "actual_points": points,
                            "position": 4 if points >= 10 else 3,
                            "now_cost": float(player_id),
                            "prior_points_per_90": float(points),
                        }
                    )
                    rows.append(row)

        result = train_local.run_quality_gate(
            pd.DataFrame(rows), hyperparams={"n_estimators": 2, "max_depth": 2}
        )

        assert set(result) == {
            "passed",
            "checks",
            "baseline",
            "candidate",
            "early_gameweeks",
            "early_ndcg_diff_se",
        }
        assert result["early_gameweeks"] == 12
        assert "early_ndcg_at_15" in result["baseline"]
        assert "early_fwd_haul_recall_at_15" in result["baseline"]
        assert "early_fwd_haul_ndcg" in result["baseline"]
        assert "metadata" in result["candidate"]
        assert result["candidate"]["metadata"]["regression_n_estimators"] > 0
        assert result["candidate"]["metadata"]["classifier_n_estimators"] > 0


class TestMainQualityGate:
    @staticmethod
    def passing_report():
        return {
            "passed": True,
            "checks": {"early_ndcg_improved": True},
            "baseline": {},
            "candidate": {
                "metadata": {
                    "point_slope": 1.0,
                    "point_intercept": 0.0,
                    "haul_slope": 1.0,
                    "haul_intercept": 0.0,
                    "rank_haul_weight": 1.0,
                    "regression_n_estimators": 2,
                    "classifier_n_estimators": 2,
                }
            },
        }

    def test_passing_gate_only_writes_report_without_training_or_saving(
        self, monkeypatch, tmp_path
    ):
        report = self.passing_report()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_local.py",
                "--data-path",
                "training.parquet",
                "--output-path",
                str(tmp_path),
                "--quality-gate",
                "--train-haul-classifier",
                "--upload-s3",
            ],
        )
        monkeypatch.setattr(
            train_local, "load_training_data", lambda **_: pd.DataFrame()
        )
        monkeypatch.setattr(train_local, "run_quality_gate", lambda *_: report)
        for name in (
            "train_model_temporal",
            "train_model",
            "save_model",
            "train_haul_classifier",
            "save_model_metadata",
            "upload_model_to_s3",
        ):
            monkeypatch.setattr(
                train_local,
                name,
                MagicMock(side_effect=AssertionError(f"{name} must not run")),
            )

        train_local.main()

        assert json.loads((tmp_path / "quality_report.json").read_text()) == report

    def test_failed_gate_still_writes_report_and_exits(self, monkeypatch, tmp_path):
        report = self.passing_report()
        report["passed"] = False
        report["checks"]["early_ndcg_improved"] = False
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_local.py",
                "--data-path",
                "training.parquet",
                "--output-path",
                str(tmp_path),
                "--quality-gate",
            ],
        )
        monkeypatch.setattr(
            train_local, "load_training_data", lambda **_: pd.DataFrame()
        )
        monkeypatch.setattr(train_local, "run_quality_gate", lambda *_: report)
        monkeypatch.setattr(
            train_local,
            "train_model_temporal",
            MagicMock(side_effect=AssertionError("training must not run")),
        )

        with pytest.raises(SystemExit, match="early_ndcg_improved"):
            train_local.main()

        assert json.loads((tmp_path / "quality_report.json").read_text()) == report

    def test_quality_gate_rejects_non_canonical_haul_threshold(
        self, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_local.py",
                "--data-path",
                "training.parquet",
                "--quality-gate",
                "--haul-threshold",
                "12",
            ],
        )
        monkeypatch.setattr(
            train_local,
            "load_training_data",
            MagicMock(side_effect=AssertionError("data must not be loaded")),
        )

        with pytest.raises(SystemExit) as error:
            train_local.main()

        assert error.value.code == 2
        assert "--quality-gate requires --haul-threshold 10" in capsys.readouterr().err

    def test_passing_gate_with_refit_trains_and_saves_release_models(
        self, monkeypatch, tmp_path
    ):
        seasons = [
            "2021-22",
            "2022-23",
            "2023-24",
            "2024-25",
            "2025-26",
            "2026-27",
            "2026-27",
        ]
        frame = pd.DataFrame(
            {
                **{feature: [0.0] * len(seasons) for feature in FEATURE_COLS},
                "actual_points": [10] * len(seasons),
                "season": seasons,
                "gameweek": [1, 1, 1, 1, 1, 1, 2],
                "player_id": list(range(1, len(seasons) + 1)),
            }
        )
        frame["now_cost"] = [1, 2, 3, 4, 5, 90, 91]
        regressor = MagicMock()
        regressor.predict.return_value = np.array([10.0] * 5)
        classifier = MagicMock()
        classifier.predict_proba.return_value = np.array([[0.1, 0.9]] * 5)
        classifier_params = []
        gated_hyperparams = {}
        save_model = MagicMock(return_value=str(tmp_path / "model.xgb"))
        save_metadata = MagicMock(return_value=str(tmp_path / "model_metadata.json"))

        def run_quality_gate(_, hyperparams):
            gated_hyperparams.update(hyperparams)
            return self.passing_report()

        def classifier_factory(**params):
            classifier_params.append(params)
            return classifier

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_local.py",
                "--data-path",
                "training.parquet",
                "--output-path",
                str(tmp_path),
                "--quality-gate",
                "--refit-all",
                "--train-haul-classifier",
                "--n-estimators",
                "17",
                "--max-depth",
                "3",
                "--learning-rate",
                "0.07",
            ],
        )
        monkeypatch.setattr(train_local, "load_training_data", lambda **_: frame)
        monkeypatch.setattr(train_local, "run_quality_gate", run_quality_gate)
        monkeypatch.setattr(train_local.xgb, "XGBRegressor", lambda **_: regressor)
        monkeypatch.setattr(train_local.xgb, "XGBClassifier", classifier_factory)
        monkeypatch.setattr(train_local, "evaluate_model", lambda *_: {"mae": 0.0})
        monkeypatch.setattr(train_local, "get_feature_importance", lambda *_: {})
        monkeypatch.setattr(
            train_local, "evaluate_haul_classifier", lambda *_: {"roc_auc": 1.0}
        )
        monkeypatch.setattr(train_local, "save_model", save_model)
        monkeypatch.setattr(train_local, "save_model_metadata", save_metadata)

        train_local.main()

        regressor.fit.assert_called_once()
        classifier.fit.assert_called_once()
        refit_costs = [1, 2, 3, 4, 5, 90, 91]
        assert list(regressor.fit.call_args.args[0]["now_cost"]) == refit_costs
        assert list(classifier.fit.call_args.args[0]["now_cost"]) == refit_costs
        assert gated_hyperparams == {
            "n_estimators": 17,
            "max_depth": 3,
            "learning_rate": 0.07,
        }
        assert classifier_params == [
            {
                **train_local.HAUL_CLASSIFIER_HYPERPARAMS,
                "n_estimators": 2,
                "max_depth": 3,
                "learning_rate": 0.07,
            }
        ]
        save_model.assert_called_once()
        save_metadata.assert_called_once()
        assert save_metadata.call_args.args[0]["trained_through"] == "2026-27 GW2"

    def test_quality_gate_tuning_excludes_outer_holdout(self, monkeypatch, tmp_path):
        seasons = [
            "2021-22",
            "2022-23",
            "2023-24",
            "2024-25",
            "2025-26",
            "2026-27",
            "2026-27",
        ]
        frame = pd.DataFrame(
            {
                **{feature: [0.0] * len(seasons) for feature in FEATURE_COLS},
                "actual_points": [2] * len(seasons),
                "season": seasons,
                "gameweek": [1, 1, 1, 1, 1, 1, 2],
                "player_id": list(range(1, len(seasons) + 1)),
            }
        )
        tuned_seasons = []

        def tune(frame_to_tune, **_):
            tuned_seasons.extend(frame_to_tune["season"])
            return DEFAULT_HYPERPARAMS

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_local.py",
                "--data-path",
                "training.parquet",
                "--output-path",
                str(tmp_path),
                "--quality-gate",
                "--tune",
            ],
        )
        monkeypatch.setattr(train_local, "load_training_data", lambda **_: frame)
        monkeypatch.setattr(train_local, "tune_hyperparameters", tune)
        monkeypatch.setattr(
            train_local, "run_quality_gate", lambda *_: self.passing_report()
        )

        train_local.main()

        assert set(tuned_seasons) == {
            "2021-22",
            "2022-23",
            "2023-24",
            "2024-25",
        }


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
        assert best_params["objective"] == "reg:squarederror"

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
