"""
Local XGBoost training script for FPL player point predictions.

This script trains an XGBoost regression model locally for development
and validation before deploying to SageMaker. Running locally costs $0.

Usage:
    python sagemaker/train_local.py --data-path path/to/features.parquet
    python sagemaker/train_local.py --data-dir path/to/parquet/dir --output-path models/
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Feature columns used for training (must match feature processor output)
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

# Default hyperparameters
DEFAULT_HYPERPARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": 42,
}


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

        parquet_files = list(dir_path.glob("*_features_training.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No training parquet files found in {data_dir}"
            )

        logger.info(f"Loading {len(parquet_files)} parquet files from {data_dir}")
        dfs = [pd.read_parquet(f) for f in parquet_files]
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

    args = parser.parse_args()

    if not args.data_path and not args.data_dir:
        parser.error("Either --data-path or --data-dir is required")

    # Build hyperparameters from CLI args
    hyperparams = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
    }

    # Load data
    df = load_training_data(data_path=args.data_path, data_dir=args.data_dir)
    logger.info(f"Loaded {len(df)} training samples")

    # Train model
    model, X_test, y_test = train_model(
        df, hyperparams=hyperparams, test_size=args.test_size
    )

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    get_feature_importance(model)

    # Save model
    model_path = save_model(model, args.output_path)

    logger.info("Training complete!")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Final MAE: {metrics['mae']:.3f} points")


if __name__ == "__main__":
    main()
