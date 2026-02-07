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
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
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

from lambdas.common.feature_config import FEATURE_COLS, TARGET_COL  # noqa: E402

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
    "colsample_bytree": 0.7,
    "random_state": 42,
}

# Default hyperparameters for haul classifier
HAUL_CLASSIFIER_HYPERPARAMS = {
    "objective": "binary:logistic",
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "random_state": 42,
    "scale_pos_weight": 5,  # Handle class imbalance (hauls are rare)
}

# Haul threshold (10+ points is considered a haul)
HAUL_THRESHOLD = 10


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
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
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
    }

    logger.info("Haul Classifier Evaluation Metrics:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.3f}")
    logger.info(f"  Precision: {metrics['precision']:.3f}")
    logger.info(f"  Recall:    {metrics['recall']:.3f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.3f}")
    logger.info(f"  ROC AUC:   {metrics['roc_auc']:.3f}")

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

    # Run hyperparameter tuning if requested
    if args.tune:
        logger.info("=" * 50)
        logger.info("Hyperparameter Tuning")
        logger.info("=" * 50)
        best_params = tune_hyperparameters(
            df,
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

    # Train model
    if use_temporal:
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

    # Upload to S3 if requested
    if args.upload_s3:
        s3_uri = upload_model_to_s3(model_path, args.bucket, args.model_key)
        logger.info(f"Model uploaded to: {s3_uri}")

    logger.info("Regression model training complete!")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Final MAE: {metrics['mae']:.3f} points")

    # Train haul classifier if requested
    if args.train_haul_classifier:
        logger.info("\n" + "=" * 50)
        logger.info("Training Haul Probability Classifier")
        logger.info("=" * 50)

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

        # Upload haul classifier to S3 if requested
        if args.upload_s3:
            haul_model_key = args.model_key.replace("model.xgb", "model_haul.xgb")
            s3_uri = upload_model_to_s3(
                str(haul_model_path), args.bucket, haul_model_key
            )
            logger.info(f"Haul classifier uploaded to: {s3_uri}")

        logger.info(f"Haul classifier ROC AUC: {haul_metrics['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
