"""
Local XGBoost training script for FPL player point predictions.

This script trains an XGBoost regression model locally for development
and validation before deploying to SageMaker. Running locally costs $0.

Usage:
    python sagemaker/train_local.py --data-path path/to/features.parquet
    python sagemaker/train_local.py --data-dir path/to/parquet/dir --output-path models/
    python sagemaker/train_local.py --data-dir data/ --tune --n-trials 50
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

import boto3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    ndcg_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Add project root and lambdas to path for imports
# lambdas is needed because feature_config uses 'from common...' for Lambda compat
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "lambdas"))

from lambdas.common.feature_config import (  # noqa: E402
    FEATURE_COLS,
    FEATURE_VERSION,
    MODEL_V5_FEATURES,
    TARGET_COL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default hyperparameters for regression model
DEFAULT_HYPERPARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "random_state": 42,
}

QUALITY_EARLY_STOPPING_ROUNDS = 20

# Default hyperparameters for haul classifier
HAUL_CLASSIFIER_HYPERPARAMS = {
    "objective": "binary:logistic",
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "random_state": 42,
    "scale_pos_weight": 5,  # Handle class imbalance (hauls are rare)
}

# Haul threshold (10+ points is considered a haul)
HAUL_THRESHOLD = 10

QUALITY_TRAIN_SEASONS = {"2021-22", "2022-23", "2023-24"}
QUALITY_VALIDATION_SEASON = "2024-25"
QUALITY_COMPLETED_TEST_SEASON = "2025-26"
QUALITY_OUTER_TEST_SEASON = "2026-27"


def split_quality_gate_seasons(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split training, validation, completed-test, and outer-test seasons."""
    train = df[df["season"].isin(QUALITY_TRAIN_SEASONS)].copy()
    validation = df[df["season"] == QUALITY_VALIDATION_SEASON].copy()
    completed_test = df[df["season"] == QUALITY_COMPLETED_TEST_SEASON].copy()
    outer_test = df[df["season"] == QUALITY_OUTER_TEST_SEASON].copy()
    if set(train["season"]) != QUALITY_TRAIN_SEASONS:
        raise ValueError(
            "Quality gate requires every season from 2021-22 through 2023-24"
        )
    if validation.empty or completed_test.empty or outer_test.empty:
        raise ValueError("Quality gate requires 2021-22 through 2026-27 data")
    if not {1, 2}.issubset(set(outer_test["gameweek"])):
        raise ValueError("2026-27 outer holdout must contain both GW1 and GW2")
    return train, validation, completed_test, outer_test


def evaluate_quality_gates(
    baseline: dict[str, float], candidate: dict[str, float]
) -> dict:
    """Evaluate the approved balanced early-season release thresholds."""
    tolerance = 1e-12
    checks = {
        "early_ndcg_improved": candidate["early_ndcg_at_15"]
        > baseline["early_ndcg_at_15"],
        "early_haul_recall_improved": candidate["early_haul_recall_at_15"]
        > baseline["early_haul_recall_at_15"],
        "early_fwd_haul_recall_non_regression": (
            candidate["early_fwd_haul_recall_at_15"]
            >= baseline["early_fwd_haul_recall_at_15"] - tolerance
        ),
        "early_fwd_haul_ndcg_improved": candidate["early_fwd_haul_ndcg"]
        > baseline["early_fwd_haul_ndcg"],
        "early_mae_within_tolerance": candidate["early_mae"]
        <= baseline["early_mae"] + 0.05 + tolerance,
        "full_mae_within_tolerance": candidate["mae"]
        <= baseline["mae"] + 0.02 + tolerance,
        "brier_within_tolerance": candidate["brier_score"]
        <= baseline["brier_score"] + 0.002 + tolerance,
    }
    return {"passed": all(checks.values()), "checks": checks}


def fit_point_calibration(predictions, actual_points) -> dict[str, float]:
    """Fit a linear correction for systematic point-prediction bias."""
    predictions = np.asarray(predictions, dtype=float)
    actual_points = np.asarray(actual_points, dtype=float)
    if len(predictions) < 2 or np.ptp(predictions) == 0:
        return {"point_slope": 1.0, "point_intercept": 0.0}
    slope, intercept = np.polyfit(predictions, actual_points, 1)
    return {
        "point_slope": max(0.0, float(slope)),
        "point_intercept": float(intercept),
    }


def fit_haul_calibration(probabilities, outcomes) -> dict[str, float]:
    """Fit Platt scaling parameters to raw haul probabilities."""
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    outcomes = np.asarray(outcomes, dtype=int)
    if len(np.unique(outcomes)) < 2:
        return {"haul_slope": 1.0, "haul_intercept": 0.0}
    logits = np.log(probabilities / (1 - probabilities)).reshape(-1, 1)
    calibrator = LogisticRegression().fit(logits, outcomes)
    return {
        "haul_slope": float(calibrator.coef_[0, 0]),
        "haul_intercept": float(calibrator.intercept_[0]),
    }


def mean_gameweek_ndcg(actual_points, scores, gameweeks, k: int | None = 15) -> float:
    """Return mean NDCG@k across gameweeks with positive relevance."""
    frame = pd.DataFrame(
        {"actual": actual_points, "score": scores, "gameweek": gameweeks}
    )
    values = []
    for _, group in frame.groupby("gameweek", sort=False):
        relevance = np.clip(group["actual"].to_numpy(dtype=float), 0, None)
        if relevance.sum() == 0:
            continue
        values.append(
            ndcg_score(
                [relevance],
                [group["score"].to_numpy(dtype=float)],
                k=min(k, len(group)) if k is not None else None,
            )
        )
    return float(np.mean(values)) if values else 0.0


def mean_gameweek_forward_haul_ndcg(
    actual_points, scores, gameweeks, positions
) -> float:
    """Return full-list mean NDCG for forward hauls by gameweek."""
    actual = np.asarray(actual_points, dtype=float)
    relevance = np.where(
        (np.asarray(positions) == 4) & (actual >= HAUL_THRESHOLD), actual, 0
    )
    return mean_gameweek_ndcg(relevance, scores, gameweeks, k=None)


def haul_recall_at_k(
    actual_points,
    scores,
    gameweeks,
    k: int = 15,
    haul_threshold: int = HAUL_THRESHOLD,
    cohort=None,
) -> float:
    """Return mean per-gameweek recall of actual hauls in the top k scores."""
    frame = pd.DataFrame(
        {"actual": actual_points, "score": scores, "gameweek": gameweeks}
    )
    if cohort is not None:
        frame["cohort"] = cohort
    recalls = []
    for _, group in frame.groupby("gameweek", sort=False):
        hauls = group["actual"] >= haul_threshold
        if cohort is not None:
            hauls &= group["cohort"]
        haul_count = int(hauls.sum())
        if haul_count == 0:
            continue
        top = group.nlargest(k, "score")
        top_hauls = top["actual"] >= haul_threshold
        if cohort is not None:
            top_hauls &= top["cohort"]
        recalls.append(float(top_hauls.sum() / haul_count))
    return float(np.mean(recalls)) if recalls else 0.0


def inference_rank_scores(point_predictions, haul_probabilities, weight):
    """Apply the same probability and score quantisation used by inference."""
    points = np.asarray(point_predictions, dtype=float)
    haul_percentages = (np.asarray(haul_probabilities, dtype=float) * 100).round(1)
    return (points + haul_percentages * weight / 100).round(2)


def select_rank_weight(
    point_predictions,
    haul_probabilities,
    actual_points,
    gameweeks=None,
    candidates=(0.0, 0.5, 1.0, 1.5, 2.0),
) -> float:
    """Select the haul weight with the best validation mean NDCG@15."""
    points = np.asarray(point_predictions, dtype=float)
    haul = np.asarray(haul_probabilities, dtype=float)
    if gameweeks is None:
        gameweeks = np.zeros(len(points), dtype=int)
    best_weight = float(candidates[0])
    best_score = float("-inf")
    for weight in candidates:
        score = mean_gameweek_ndcg(
            actual_points,
            inference_rank_scores(points, haul, weight),
            gameweeks,
        )
        if score > best_score:
            best_score = score
            best_weight = float(weight)
    return best_weight


def apply_haul_calibration(probabilities, metadata: dict[str, float]):
    """Apply stored Platt scaling parameters to haul probabilities."""
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(probabilities / (1 - probabilities))
    return 1 / (
        1 + np.exp(-(metadata["haul_slope"] * logits + metadata["haul_intercept"]))
    )


def score_backtest_predictions(
    regressor,
    classifier,
    frame: pd.DataFrame,
    feature_cols: list[str],
    metadata: dict[str, float],
) -> dict[str, float]:
    """Score one out-of-time frame with point, haul, and ranking metrics."""
    raw_points = regressor.predict(frame[feature_cols])
    points = raw_points * metadata["point_slope"] + metadata["point_intercept"]
    raw_haul = classifier.predict_proba(frame[feature_cols])[:, 1]
    haul = apply_haul_calibration(raw_haul, metadata)
    rank_scores = inference_rank_scores(points, haul, metadata["rank_haul_weight"])
    actual = frame[TARGET_COL].to_numpy(dtype=float)
    gameweeks = frame["gameweek"].to_numpy()
    positions = frame["position"].to_numpy()
    haul_outcomes = (actual >= HAUL_THRESHOLD).astype(int)
    early = gameweeks <= 3

    return {
        "mae": float(mean_absolute_error(actual, points)),
        "brier_score": float(brier_score_loss(haul_outcomes, haul)),
        "ndcg_at_15": mean_gameweek_ndcg(actual, rank_scores, gameweeks),
        "haul_recall_at_15": haul_recall_at_k(actual, rank_scores, gameweeks),
        "early_mae": float(mean_absolute_error(actual[early], points[early])),
        "early_ndcg_at_15": mean_gameweek_ndcg(
            actual[early], rank_scores[early], gameweeks[early]
        ),
        "early_haul_recall_at_15": haul_recall_at_k(
            actual[early], rank_scores[early], gameweeks[early]
        ),
        "early_fwd_haul_recall_at_15": haul_recall_at_k(
            actual[early],
            rank_scores[early],
            gameweeks[early],
            cohort=positions[early] == 4,
        ),
        "early_fwd_haul_ndcg": mean_gameweek_forward_haul_ndcg(
            actual[early], rank_scores[early], gameweeks[early], positions[early]
        ),
    }


def _train_quality_candidate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    completed_test: pd.DataFrame,
    outer_test: pd.DataFrame,
    feature_cols: list[str],
    hyperparams: dict | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Train, calibrate, and score one out-of-time model candidate."""
    regression_params = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
    regression_params["early_stopping_rounds"] = QUALITY_EARLY_STOPPING_ROUNDS
    classifier_params = HAUL_CLASSIFIER_HYPERPARAMS.copy()
    classifier_params["early_stopping_rounds"] = QUALITY_EARLY_STOPPING_ROUNDS
    if hyperparams:
        classifier_params.update(
            {
                key: value
                for key, value in hyperparams.items()
                if key in {"n_estimators", "max_depth", "learning_rate"}
            }
        )

    regressor = xgb.XGBRegressor(**regression_params)
    regressor.fit(
        train[feature_cols],
        train[TARGET_COL],
        eval_set=[(validation[feature_cols], validation[TARGET_COL])],
        verbose=False,
    )

    classifier = xgb.XGBClassifier(**classifier_params)
    classifier.fit(
        train[feature_cols],
        (train[TARGET_COL] >= HAUL_THRESHOLD).astype(int),
        eval_set=[
            (
                validation[feature_cols],
                (validation[TARGET_COL] >= HAUL_THRESHOLD).astype(int),
            )
        ],
        verbose=False,
    )

    validation_points = regressor.predict(validation[feature_cols])
    validation_haul = classifier.predict_proba(validation[feature_cols])[:, 1]
    metadata = {
        **fit_point_calibration(validation_points, validation[TARGET_COL]),
        **fit_haul_calibration(
            validation_haul,
            (validation[TARGET_COL] >= HAUL_THRESHOLD).astype(int),
        ),
        "regression_n_estimators": int(regressor.best_iteration) + 1,
        "classifier_n_estimators": int(classifier.best_iteration) + 1,
    }
    calibrated_points = (
        validation_points * metadata["point_slope"] + metadata["point_intercept"]
    )
    calibrated_haul = apply_haul_calibration(validation_haul, metadata)
    metadata["rank_haul_weight"] = select_rank_weight(
        calibrated_points,
        calibrated_haul,
        validation[TARGET_COL],
        validation["gameweek"],
    )
    metrics = score_backtest_predictions(
        regressor, classifier, completed_test, feature_cols, metadata
    )
    outer_metrics = score_backtest_predictions(
        regressor, classifier, outer_test, feature_cols, metadata
    )
    for key in (
        "early_mae",
        "early_ndcg_at_15",
        "early_haul_recall_at_15",
        "early_fwd_haul_recall_at_15",
        "early_fwd_haul_ndcg",
    ):
        metrics[key] = outer_metrics[key]
    return metrics, metadata


def run_quality_gate(
    df: pd.DataFrame,
    hyperparams: dict | None = None,
) -> dict:
    """Compare fresh v4 and v5 models on identical fixed-season splits."""
    validate_features(df)
    train, validation, completed_test, outer_test = split_quality_gate_seasons(df)
    baseline_features = [
        feature for feature in FEATURE_COLS if feature not in MODEL_V5_FEATURES
    ]
    baseline_metrics, _ = _train_quality_candidate(
        train,
        validation,
        completed_test,
        outer_test,
        baseline_features,
        hyperparams,
    )
    candidate_metrics, metadata = _train_quality_candidate(
        train,
        validation,
        completed_test,
        outer_test,
        FEATURE_COLS,
        hyperparams,
    )
    gate = evaluate_quality_gates(baseline_metrics, candidate_metrics)
    return {
        **gate,
        "baseline": baseline_metrics,
        "candidate": {**candidate_metrics, "metadata": metadata},
    }


def save_model_metadata(metadata: dict, output_path: str) -> str:
    """Save calibration and ranking parameters beside the model."""
    path = Path(output_path)
    output_dir = path if path.suffix == "" else path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    return str(metadata_path)


def load_training_data(
    data_path: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load training data from Parquet file(s).

    Args:
        data_path: Path to a single Parquet file.
        data_dir: Path to directory containing multiple Parquet files.

    Returns:
        DataFrame with features and target column.

    Raises:
        ValueError: If neither data_path nor data_dir is provided.
        FileNotFoundError: If the specified path does not exist.
    """
    if data_path:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        logger.info(f"Loading data from {data_path}")
        return pd.read_parquet(data_path)

    if data_dir:
        dir_path = Path(data_dir)
        if not dir_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        parquet_files = list(dir_path.rglob("*_features_training.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No training parquet files found in {data_dir}")

        logger.info(f"Loading {len(parquet_files)} parquet files from {data_dir}")
        dfs = []
        for f in parquet_files:
            df = pd.read_parquet(f)
            # Infer season from directory path if not present in data
            # e.g. data/historical/season_2023_24/gw20_features_training.parquet
            if "season" not in df.columns:
                season_match = re.search(r"season_(\d{4}_\d{2})", str(f))
                if season_match:
                    raw = season_match.group(1)  # e.g. "2023_24"
                    df["season"] = raw.replace("_", "-")  # "2023-24"
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    raise ValueError("Either data_path or data_dir must be provided")


def validate_features(df: pd.DataFrame) -> None:
    """
    Validate that the DataFrame contains required columns.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If required columns are missing.
    """
    missing_features = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Missing target column '{TARGET_COL}'. "
            "Ensure data was generated in 'historical' mode."
        )

    missing_values = df[FEATURE_COLS].columns[df[FEATURE_COLS].isna().any()].tolist()
    if missing_values:
        raise ValueError(f"Missing feature values: {missing_values}")

    key_columns = [
        column for column in ("season", "gameweek", "player_id") if column in df.columns
    ]
    if {"gameweek", "player_id"}.issubset(key_columns) and df.duplicated(
        key_columns
    ).any():
        raise ValueError(f"Duplicate player-week rows for key {key_columns}")


def train_model(
    df: pd.DataFrame,
    hyperparams: Optional[dict] = None,
    test_size: float = 0.2,
) -> tuple[xgb.XGBRegressor, pd.DataFrame, pd.DataFrame]:
    """
    Train an XGBoost regression model.

    Args:
        df: DataFrame with features and target.
        hyperparams: XGBoost hyperparameters (uses defaults if None).
        test_size: Fraction of data to use for testing.

    Returns:
        Tuple of (trained model, test features DataFrame, test target Series).
    """
    validate_features(df)

    params = DEFAULT_HYPERPARAMS.copy()
    if hyperparams:
        params.update(hyperparams)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    logger.info(f"Training data shape: {X.shape}")
    logger.info(f"Target distribution: mean={y.mean():.2f}, std={y.std():.2f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=params.get("random_state", 42)
    )

    logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    logger.info("Model training complete")
    return model, X_test, y_test


def _has_temporal_columns(df: pd.DataFrame) -> bool:
    """Check whether the DataFrame contains gameweek and season columns."""
    return "gameweek" in df.columns and "season" in df.columns


def _temporal_train_test_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically using season and gameweek columns.

    Earlier gameweeks form the training set, later gameweeks form the test set.
    This mirrors real-world usage where predictions are always forward-looking.

    Args:
        df: DataFrame with 'season' and 'gameweek' columns.
        test_fraction: Fraction of data to hold out for testing.

    Returns:
        Tuple of (train_df, test_df).
    """
    # Create sort key from (season, gameweek) for chronological ordering
    df = df.copy()
    df["_sort_key"] = (
        df["season"].astype(str) + "_" + df["gameweek"].astype(str).str.zfill(2)
    )
    df = df.sort_values("_sort_key").reset_index(drop=True)

    # Split on gameweek boundaries to avoid leaking same-gameweek data
    unique_keys = df["_sort_key"].unique()
    n_test_keys = max(1, int(len(unique_keys) * test_fraction))
    test_keys = set(unique_keys[-n_test_keys:])

    train_df = df[~df["_sort_key"].isin(test_keys)].drop(columns=["_sort_key"])
    test_df = df[df["_sort_key"].isin(test_keys)].drop(columns=["_sort_key"])

    return train_df, test_df


def train_model_temporal(
    df: pd.DataFrame,
    hyperparams: Optional[dict] = None,
    test_fraction: float = 0.2,
) -> tuple[xgb.XGBRegressor, pd.DataFrame, pd.Series, dict]:
    """
    Train an XGBoost regression model using temporal train/test split.

    Splits data chronologically so earlier gameweeks train the model and
    later gameweeks are used for evaluation. Falls back to random split
    if gameweek/season columns are missing.

    Args:
        df: DataFrame with features, target, and optionally gameweek/season.
        hyperparams: XGBoost hyperparameters (uses defaults if None).
        test_fraction: Fraction of data to hold out for testing.

    Returns:
        Tuple of (trained model, test features, test target, split_info dict).
    """
    validate_features(df)

    params = DEFAULT_HYPERPARAMS.copy()
    if hyperparams:
        params.update(hyperparams)

    if not _has_temporal_columns(df):
        logger.warning(
            "Columns 'gameweek' and/or 'season' not found. "
            "Falling back to random split."
        )
        model, X_test, y_test = train_model(
            df, hyperparams=hyperparams, test_size=test_fraction
        )
        split_info = {"split_type": "random", "test_fraction": test_fraction}
        return model, X_test, y_test, split_info

    train_df, test_df = _temporal_train_test_split(df, test_fraction)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    # Log split details
    train_seasons = sorted(train_df["season"].unique())
    test_seasons = sorted(test_df["season"].unique())
    train_gws = sorted(train_df["gameweek"].unique())
    test_gws = sorted(test_df["gameweek"].unique())

    logger.info(f"Temporal split: {len(X_train)} train / {len(X_test)} test samples")
    logger.info(
        f"  Train seasons: {train_seasons}, GWs: {train_gws[0]}-{train_gws[-1]}"
    )
    logger.info(f"  Test seasons: {test_seasons}, GWs: {test_gws[0]}-{test_gws[-1]}")
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(
        f"Target distribution: mean={y_train.mean():.2f}, std={y_train.std():.2f}"
    )

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    split_info = {
        "split_type": "temporal",
        "test_fraction": test_fraction,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_seasons": train_seasons,
        "test_seasons": test_seasons,
    }

    logger.info("Model training complete (temporal split)")
    return model, X_test, y_test, split_info


def tune_hyperparameters(
    df: pd.DataFrame,
    n_trials: int = 50,
    test_fraction: float = 0.2,
    temporal: bool = True,
) -> dict:
    """
    Use Optuna to search for optimal XGBoost hyperparameters.

    Minimises MAE on the validation set using either temporal or random split.

    Args:
        df: DataFrame with features and target.
        n_trials: Number of Optuna trials to run.
        test_fraction: Fraction of data to hold out for validation.
        temporal: Use temporal split if True and columns are available.

    Returns:
        Dictionary of best hyperparameters.
    """
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "Optuna is required for hyperparameter tuning. "
            "Install with: pip install optuna"
        )

    validate_features(df)

    # Prepare train/test split
    use_temporal = temporal and _has_temporal_columns(df)

    if use_temporal:
        train_df, test_df = _temporal_train_test_split(df, test_fraction)
        X_train = train_df[FEATURE_COLS]
        y_train = train_df[TARGET_COL]
        X_test = test_df[FEATURE_COLS]
        y_test = test_df[TARGET_COL]
        logger.info(
            f"Tuning with temporal split: {len(X_train)} train / "
            f"{len(X_test)} test samples"
        )
    else:
        if temporal:
            logger.warning(
                "Columns 'gameweek' and/or 'season' not found. "
                "Falling back to random split for tuning."
            )
        X = df[FEATURE_COLS]
        y = df[TARGET_COL]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_fraction, random_state=42
        )
        logger.info(
            f"Tuning with random split: {len(X_train)} train / "
            f"{len(X_test)} test samples"
        )

    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "random_state": 42,
        }

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        return mae

    # Suppress Optuna's default logging to reduce noise
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="minimize")
    logger.info(f"Starting hyperparameter tuning ({n_trials} trials)...")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params["objective"] = "reg:squarederror"
    best_params["random_state"] = 42

    logger.info(f"Best trial MAE: {study.best_value:.3f}")
    logger.info("Best hyperparameters:")
    for key, value in sorted(best_params.items()):
        logger.info(f"  {key}: {value}")

    return best_params


def evaluate_model(
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate model performance on test set.

    Args:
        model: Trained XGBoost model.
        X_test: Test features.
        y_test: Test target values.

    Returns:
        Dictionary with evaluation metrics.
    """
    predictions = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions, squared=False),
        "r2": r2_score(y_test, predictions),
        "mean_prediction": float(predictions.mean()),
        "mean_actual": float(y_test.mean()),
    }

    logger.info("Model Evaluation Metrics:")
    logger.info(f"  MAE:  {metrics['mae']:.3f}")
    logger.info(f"  RMSE: {metrics['rmse']:.3f}")
    logger.info(f"  R²:   {metrics['r2']:.3f}")

    return metrics


def get_feature_importance(model: xgb.XGBRegressor) -> dict:
    """
    Get feature importance scores from the trained model.

    Args:
        model: Trained XGBoost model.

    Returns:
        Dictionary mapping feature names to importance scores.
    """
    importance = model.feature_importances_
    feature_importance = dict(zip(FEATURE_COLS, importance))

    logger.info("Feature Importance:")
    for feature, score in sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=True
    ):
        logger.info(f"  {feature}: {score:.4f}")

    return feature_importance


def save_model(model: xgb.XGBRegressor, output_path: str) -> str:
    """
    Save the trained model to disk.

    Args:
        model: Trained XGBoost model.
        output_path: Directory or file path for the model.

    Returns:
        Full path to the saved model file.
    """
    path = Path(output_path)

    if path.suffix == "":
        # It's a directory, create it and use default filename
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / "model.xgb"
    else:
        # It's a file path
        path.parent.mkdir(parents=True, exist_ok=True)
        model_path = path

    model.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")

    return str(model_path)


def load_model(model_path: str) -> xgb.XGBRegressor:
    """
    Load a trained model from disk.

    Args:
        model_path: Path to the saved model file.

    Returns:
        Loaded XGBoost model.
    """
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model


def train_haul_classifier(
    df: pd.DataFrame,
    hyperparams: Optional[dict] = None,
    test_size: float = 0.2,
    haul_threshold: int = HAUL_THRESHOLD,
    temporal: bool = True,
) -> tuple[xgb.XGBClassifier, pd.DataFrame, pd.Series]:
    """
    Train an XGBoost classifier to predict haul probability (10+ points).

    Uses temporal train/test split when gameweek/season columns are available
    and temporal=True, otherwise falls back to random split.

    Args:
        df: DataFrame with features and target.
        hyperparams: XGBoost hyperparameters (uses defaults if None).
        test_size: Fraction of data to use for testing.
        haul_threshold: Points threshold for a "haul" (default: 10).
        temporal: Use temporal split if True and columns are available.

    Returns:
        Tuple of (trained classifier, test features DataFrame, test target Series).
    """
    validate_features(df)

    params = HAUL_CLASSIFIER_HYPERPARAMS.copy()
    if hyperparams:
        params.update(hyperparams)

    use_temporal = temporal and _has_temporal_columns(df)

    # Create binary haul target
    haul_target = (df[TARGET_COL] >= haul_threshold).astype(int)

    haul_rate = haul_target.mean() * 100
    logger.info(f"Haul rate (>={haul_threshold} pts): {haul_rate:.2f}%")
    logger.info(f"Training data shape: {df[FEATURE_COLS].shape}")

    if use_temporal:
        train_df, test_df = _temporal_train_test_split(df, test_size)
        X_train = train_df[FEATURE_COLS]
        y_train = (train_df[TARGET_COL] >= haul_threshold).astype(int)
        X_test = test_df[FEATURE_COLS]
        y_test = (test_df[TARGET_COL] >= haul_threshold).astype(int)
        logger.info(
            f"Temporal split: {len(X_train)} train / {len(X_test)} test samples"
        )
    else:
        if temporal:
            logger.warning(
                "Columns 'gameweek' and/or 'season' not found. "
                "Falling back to random split for haul classifier."
            )
        X = df[FEATURE_COLS]
        y = haul_target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=params.get("random_state", 42)
        )
        logger.info(f"Random split: {len(X_train)} train / {len(X_test)} test samples")

    logger.info(
        f"Train haul rate: {y_train.mean() * 100:.2f}%, "
        f"Test haul rate: {y_test.mean() * 100:.2f}%"
    )

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    logger.info("Haul classifier training complete")
    return model, X_test, y_test


def evaluate_haul_classifier(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate haul classifier performance on test set.

    Args:
        model: Trained XGBoost classifier.
        X_test: Test features.
        y_test: Test target values (binary: haul or not).

    Returns:
        Dictionary with evaluation metrics.
    """
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "brier_score": brier_score_loss(y_test, probabilities),
    }

    logger.info("Haul Classifier Evaluation Metrics:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.3f}")
    logger.info(f"  Precision: {metrics['precision']:.3f}")
    logger.info(f"  Recall:    {metrics['recall']:.3f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.3f}")
    logger.info(f"  ROC AUC:   {metrics['roc_auc']:.3f}")
    logger.info(f"  Brier:     {metrics['brier_score']:.3f}")

    return metrics


def upload_model_to_s3(
    model_path: str,
    bucket: str,
    model_key: str = "models/model.xgb",
) -> str:
    """
    Upload a trained model file to S3.

    Args:
        model_path: Local path to the model file.
        bucket: S3 bucket name.
        model_key: S3 key for the model (default: models/model.xgb).

    Returns:
        S3 URI where the model was uploaded.
    """
    s3_client = boto3.client(
        "s3",
        region_name=os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2"),
        endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
    )

    logger.info(f"Uploading {model_path} to s3://{bucket}/{model_key}")
    s3_client.upload_file(model_path, bucket, model_key)

    s3_uri = f"s3://{bucket}/{model_key}"
    logger.info(f"Model uploaded to {s3_uri}")
    return s3_uri


def main():
    """Main entry point for local training."""
    parser = argparse.ArgumentParser(
        description="Train XGBoost model for FPL predictions"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to a single training Parquet file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory containing training Parquet files",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="models/",
        help="Output path for the trained model (default: models/)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for testing (default: 0.2)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of boosting rounds (default: 100)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum tree depth (default: 6)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--upload-s3",
        action="store_true",
        help="Upload trained model to S3 after training",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="fpl-ml-data-dev",
        help="S3 bucket for model upload (default: fpl-ml-data-dev)",
    )
    parser.add_argument(
        "--model-key",
        type=str,
        default="models/model.xgb",
        help="S3 key for model upload (default: models/model.xgb)",
    )
    parser.add_argument(
        "--train-haul-classifier",
        action="store_true",
        help="Also train a haul probability classifier",
    )
    parser.add_argument(
        "--haul-threshold",
        type=int,
        default=10,
        help="Points threshold for haul classification (default: 10)",
    )
    parser.add_argument(
        "--quality-gate",
        action="store_true",
        help="Compare the candidate with the baseline on the current outer holdout",
    )
    parser.add_argument(
        "--refit-all",
        action="store_true",
        help="Refit release models on five completed seasons after the gate passes",
    )

    # Split strategy
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        "--temporal-split",
        action="store_true",
        default=True,
        help="Use temporal train/test split (default)",
    )
    split_group.add_argument(
        "--random-split",
        action="store_true",
        help="Use random train/test split instead of temporal",
    )

    # Hyperparameter tuning
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter tuning before training",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna tuning trials (default: 50)",
    )

    args = parser.parse_args()

    if not args.data_path and not args.data_dir:
        parser.error("Either --data-path or --data-dir is required")
    if args.quality_gate and args.haul_threshold != HAUL_THRESHOLD:
        parser.error(f"--quality-gate requires --haul-threshold {HAUL_THRESHOLD}")

    use_temporal = not args.random_split

    # Build hyperparameters from CLI args
    hyperparams = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
    }

    # Load data
    df = load_training_data(data_path=args.data_path, data_dir=args.data_dir)
    logger.info(f"Loaded {len(df)} training samples")

    quality_report = None
    if args.refit_all and not args.quality_gate:
        parser.error("--refit-all requires --quality-gate")

    # Run hyperparameter tuning if requested
    if args.tune:
        logger.info("=" * 50)
        logger.info("Hyperparameter Tuning")
        logger.info("=" * 50)
        tuning_df = df
        if args.quality_gate:
            quality_train, quality_validation, _, _ = split_quality_gate_seasons(df)
            tuning_df = pd.concat(
                [quality_train, quality_validation], ignore_index=True
            )
        best_params = tune_hyperparameters(
            tuning_df,
            n_trials=args.n_trials,
            test_fraction=args.test_size,
            temporal=use_temporal,
        )
        # Override CLI hyperparams with tuned values
        hyperparams = {
            k: v
            for k, v in best_params.items()
            if k not in ("objective", "random_state")
        }

        # Save best params to output directory
        output_dir = Path(args.output_path)
        if output_dir.suffix != "":
            output_dir = output_dir.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        params_path = output_dir / "best_params.json"
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        logger.info(f"Best parameters saved to {params_path}")

        logger.info("=" * 50)
        logger.info("Training Final Model with Best Parameters")
        logger.info("=" * 50)

    if args.quality_gate:
        quality_report = run_quality_gate(df, hyperparams)
        output_dir = Path(args.output_path)
        if output_dir.suffix != "":
            output_dir = output_dir.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        quality_path = output_dir / "quality_report.json"
        with open(quality_path, "w") as quality_file:
            json.dump(quality_report, quality_file, indent=2)
        logger.info(f"Quality report saved to {quality_path}")
        if not quality_report["passed"]:
            failed = [
                name for name, passed in quality_report["checks"].items() if not passed
            ]
            raise SystemExit(f"Model quality gate failed: {', '.join(failed)}")
        if not args.refit_all:
            return

    # Train model
    if args.refit_all:
        validate_features(df)
        quality_train, quality_validation, completed_test, _ = (
            split_quality_gate_seasons(df)
        )
        refit_df = pd.concat(
            [quality_train, quality_validation, completed_test], ignore_index=True
        )
        refit_params = {**DEFAULT_HYPERPARAMS, **hyperparams}
        refit_params["n_estimators"] = int(
            quality_report["candidate"]["metadata"]["regression_n_estimators"]
        )
        model = xgb.XGBRegressor(**refit_params)
        model.fit(refit_df[FEATURE_COLS], refit_df[TARGET_COL], verbose=False)
        X_test = refit_df[FEATURE_COLS]
        y_test = refit_df[TARGET_COL]
    elif use_temporal:
        model, X_test, y_test, split_info = train_model_temporal(
            df, hyperparams=hyperparams, test_fraction=args.test_size
        )
    else:
        model, X_test, y_test = train_model(
            df, hyperparams=hyperparams, test_size=args.test_size
        )

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    get_feature_importance(model)

    # Save model
    model_path = save_model(model, args.output_path)
    model_metadata = (
        quality_report["candidate"]["metadata"].copy()
        if quality_report
        else {
            **fit_point_calibration(model.predict(X_test), y_test),
            "haul_slope": 1.0,
            "haul_intercept": 0.0,
            "rank_haul_weight": 1.5,
        }
    )
    model_metadata.update(
        {
            "feature_version": FEATURE_VERSION,
            "feature_count": len(FEATURE_COLS),
            "trained_through": (
                QUALITY_COMPLETED_TEST_SEASON if args.refit_all else None
            ),
        }
    )
    haul_model_path = None

    logger.info("Regression model training complete!")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Final MAE: {metrics['mae']:.3f} points")

    # Train haul classifier if requested
    if args.train_haul_classifier:
        logger.info("\n" + "=" * 50)
        logger.info("Training Haul Probability Classifier")
        logger.info("=" * 50)

        if args.refit_all:
            haul_params = HAUL_CLASSIFIER_HYPERPARAMS.copy()
            haul_params.update(
                {
                    key: value
                    for key, value in hyperparams.items()
                    if key in {"n_estimators", "max_depth", "learning_rate"}
                }
            )
            haul_params["n_estimators"] = int(
                quality_report["candidate"]["metadata"]["classifier_n_estimators"]
            )
            haul_model = xgb.XGBClassifier(**haul_params)
            X_test_haul = refit_df[FEATURE_COLS]
            y_test_haul = (refit_df[TARGET_COL] >= args.haul_threshold).astype(int)
            haul_model.fit(X_test_haul, y_test_haul, verbose=False)
        else:
            haul_model, X_test_haul, y_test_haul = train_haul_classifier(
                df,
                test_size=args.test_size,
                haul_threshold=args.haul_threshold,
                temporal=use_temporal,
            )

        haul_metrics = evaluate_haul_classifier(haul_model, X_test_haul, y_test_haul)

        # Save haul classifier
        haul_output_path = Path(args.output_path)
        if haul_output_path.suffix == "":
            haul_model_path = haul_output_path / "model_haul.xgb"
        else:
            haul_model_path = haul_output_path.parent / "model_haul.xgb"
        haul_model_path.parent.mkdir(parents=True, exist_ok=True)
        haul_model.save_model(str(haul_model_path))
        logger.info(f"Haul classifier saved to {haul_model_path}")

        raw_haul_probabilities = haul_model.predict_proba(X_test_haul)[:, 1]
        if not quality_report:
            model_metadata.update(
                fit_haul_calibration(raw_haul_probabilities, y_test_haul)
            )
            calibrated_haul_probabilities = apply_haul_calibration(
                raw_haul_probabilities, model_metadata
            )
            calibrated_points = (
                model.predict(X_test_haul) * model_metadata["point_slope"]
                + model_metadata["point_intercept"]
            )
            actual_points = df.loc[X_test_haul.index, TARGET_COL]
            model_metadata["rank_haul_weight"] = select_rank_weight(
                calibrated_points,
                calibrated_haul_probabilities,
                actual_points,
                df.loc[X_test_haul.index, "gameweek"],
            )

        logger.info(f"Haul classifier ROC AUC: {haul_metrics['roc_auc']:.3f}")

    metadata_path = save_model_metadata(model_metadata, args.output_path)
    logger.info(f"Model metadata saved to: {metadata_path}")

    if args.upload_s3:
        upload_model_to_s3(model_path, args.bucket, args.model_key)
        metadata_key = str(Path(args.model_key).with_name("model_metadata.json"))
        upload_model_to_s3(metadata_path, args.bucket, metadata_key)
        if haul_model_path:
            haul_model_key = str(Path(args.model_key).with_name("model_haul.xgb"))
            upload_model_to_s3(str(haul_model_path), args.bucket, haul_model_key)


if __name__ == "__main__":
    main()
