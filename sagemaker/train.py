"""
SageMaker training entry point for FPL player point predictions.

This script is the entry point for AWS SageMaker training jobs.
It follows SageMaker conventions for input/output paths.

SageMaker Paths:
    Input data: /opt/ml/input/data/training/
    Model output: /opt/ml/model/
    Hyperparameters: /opt/ml/input/config/hyperparameters.json
"""

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# SageMaker environment paths
SM_CHANNEL_TRAINING = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
SM_OUTPUT_DATA_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")

# Feature columns (must match feature processor output)
FEATURE_COLS = [
    "points_last_3",
    "points_last_5",
    "minutes_pct",
    "form_score",
    "opponent_strength",
    "home_away",
    "chance_of_playing",
    "form_x_difficulty",
]

TARGET_COL = "actual_points"


def parse_args():
    """Parse hyperparameters passed by SageMaker."""
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    # SageMaker specific
    parser.add_argument("--model-dir", type=str, default=SM_MODEL_DIR)
    parser.add_argument("--train", type=str, default=SM_CHANNEL_TRAINING)
    parser.add_argument("--output-data-dir", type=str, default=SM_OUTPUT_DATA_DIR)

    return parser.parse_args()


def load_training_data(data_dir: str) -> pd.DataFrame:
    """
    Load all Parquet files from the training directory.

    Args:
        data_dir: Directory containing training Parquet files.

    Returns:
        Combined DataFrame from all Parquet files.
    """
    data_path = Path(data_dir)
    parquet_files = list(data_path.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    logger.info(f"Found {len(parquet_files)} parquet files in {data_dir}")

    dfs = []
    for f in parquet_files:
        df = pd.read_parquet(f)
        logger.info(f"  Loaded {len(df)} rows from {f.name}")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total training samples: {len(combined)}")

    return combined


def validate_data(df: pd.DataFrame) -> None:
    """Validate that required columns are present."""
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")


def train(args):
    """Main training function."""
    logger.info("Starting SageMaker training job")
    logger.info(f"Hyperparameters: n_estimators={args.n_estimators}, "
                f"max_depth={args.max_depth}, learning_rate={args.learning_rate}")

    # Load data
    df = load_training_data(args.train)
    validate_data(df)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target stats: mean={y.mean():.2f}, std={y.std():.2f}, "
                f"min={y.min()}, max={y.max()}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Train model
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=True,
    )

    # Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)

    logger.info("Evaluation Metrics:")
    logger.info(f"  MAE:  {mae:.3f}")
    logger.info(f"  RMSE: {rmse:.3f}")
    logger.info(f"  R²:   {r2:.3f}")

    # Log feature importance
    importance = dict(zip(FEATURE_COLS, model.feature_importances_))
    logger.info("Feature Importance:")
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {feature}: {score:.4f}")

    # Save model
    model_path = Path(args.model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    model_file = model_path / "model.xgb"
    model.save_model(str(model_file))
    logger.info(f"Model saved to {model_file}")

    # Save metrics for SageMaker
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    metrics_file = model_path / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")

    logger.info("Training complete!")


def model_fn(model_dir: str) -> xgb.XGBRegressor:
    """
    Load model for SageMaker inference.

    This function is called by SageMaker when loading the model
    for real-time inference endpoints.

    Args:
        model_dir: Directory containing the saved model.

    Returns:
        Loaded XGBoost model.
    """
    model_path = Path(model_dir) / "model.xgb"
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))
    return model


if __name__ == "__main__":
    args = parse_args()
    train(args)
