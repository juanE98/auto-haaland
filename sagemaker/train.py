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
import re
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# SageMaker environment paths
SM_CHANNEL_TRAINING = os.environ.get(
    "SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"
)
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
SM_OUTPUT_DATA_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")

# Feature version - must match lambdas/common/feature_config.py FEATURE_VERSION
FEATURE_VERSION = "2.3.0"

# Feature columns (must match feature processor output)
# Keep in sync with lambdas/common/feature_config.py
# Total: 73 rolling + 9 static + 32 bootstrap + 28 team + 24 opponent
#        + 16 fixture + 8 position + 5 interaction + 5 derived = 200
FEATURE_COLS = [
    # Rolling features (73 total)
    # Core stats with standard windows (1, 3, 5)
    "points_last_1",
    "points_last_3",
    "points_last_5",
    "goals_last_1",
    "goals_last_3",
    "goals_last_5",
    "assists_last_1",
    "assists_last_3",
    "assists_last_5",
    "clean_sheets_last_1",
    "clean_sheets_last_3",
    "clean_sheets_last_5",
    "bps_last_1",
    "bps_last_3",
    "bps_last_5",
    "ict_index_last_1",
    "ict_index_last_3",
    "ict_index_last_5",
    "threat_last_1",
    "threat_last_3",
    "threat_last_5",
    "creativity_last_1",
    "creativity_last_3",
    "creativity_last_5",
    "influence_last_1",
    "influence_last_3",
    "influence_last_5",
    "bonus_last_1",
    "bonus_last_3",
    "bonus_last_5",
    # New xG/xA stats with standard windows (1, 3, 5)
    "expected_goals_last_1",
    "expected_goals_last_3",
    "expected_goals_last_5",
    "expected_assists_last_1",
    "expected_assists_last_3",
    "expected_assists_last_5",
    # New minutes/starts stats with standard windows (1, 3, 5)
    "minutes_last_1",
    "minutes_last_3",
    "minutes_last_5",
    "starts_last_1",
    "starts_last_3",
    "starts_last_5",
    # Stats with medium/long windows (3, 5)
    "yellow_cards_last_3",
    "yellow_cards_last_5",
    "saves_last_3",
    "saves_last_5",
    "transfers_balance_last_3",
    "transfers_balance_last_5",
    # New rarer event stats with longer windows (3, 5, 10)
    "red_cards_last_3",
    "red_cards_last_5",
    "red_cards_last_10",
    "own_goals_last_3",
    "own_goals_last_5",
    "own_goals_last_10",
    "penalties_saved_last_3",
    "penalties_saved_last_5",
    "penalties_saved_last_10",
    "penalties_missed_last_3",
    "penalties_missed_last_5",
    "penalties_missed_last_10",
    # Extended window (10) for key existing stats
    "points_last_10",
    "goals_last_10",
    "assists_last_10",
    "clean_sheets_last_10",
    "bps_last_10",
    "ict_index_last_10",
    "threat_last_10",
    "creativity_last_10",
    "influence_last_10",
    "bonus_last_10",
    "yellow_cards_last_10",
    "saves_last_10",
    "transfers_balance_last_10",
    # Static features (9 total)
    "form_score",
    "opponent_strength",
    "home_away",
    "chance_of_playing",
    "position",
    "opponent_attack_strength",
    "opponent_defence_strength",
    "selected_by_percent",
    "now_cost",
    # Bootstrap features (32 total) - Phase 2
    # FPL Expected Points & Value (8)
    "ep_this",
    "ep_next",
    "points_per_game",
    "value_form",
    "value_season",
    "cost_change_start",
    "cost_change_event",
    "cost_change_event_fall",
    # Availability & Status (6)
    "status_available",
    "status_injured",
    "status_suspended",
    "status_doubtful",
    "has_news",
    "news_injury_flag",
    # Dream Team & Recognition (4)
    "dreamteam_count",
    "in_dreamteam",
    "dreamteam_rate",
    "bonus_rate",
    # Transfer Momentum (6)
    "transfers_in_event",
    "transfers_out_event",
    "net_transfers_event",
    "transfer_momentum",
    "transfers_in_rank",
    "ownership_change_rate",
    # Set Piece Responsibility (4)
    "corners_and_indirect_freekicks_order",
    "direct_freekicks_order",
    "penalties_order",
    "set_piece_taker",
    # Season Totals Normalised (4)
    "total_points_rank_pct",
    "goals_per_90_season",
    "assists_per_90_season",
    "ict_per_90_season",
    # Team context features (28 total) - Phase 3
    # Team form (10)
    "team_goals_scored_last_3",
    "team_goals_scored_last_5",
    "team_goals_conceded_last_3",
    "team_goals_conceded_last_5",
    "team_clean_sheets_last_3",
    "team_clean_sheets_last_5",
    "team_wins_last_3",
    "team_wins_last_5",
    "team_form_score",
    "team_form_trend",
    # Team strength (8)
    "team_strength_overall",
    "team_strength_attack_home",
    "team_strength_attack_away",
    "team_strength_defence_home",
    "team_strength_defence_away",
    "team_attack_vs_opp_defence",
    "team_defence_vs_opp_attack",
    "strength_differential",
    # League position (4)
    "team_league_position",
    "team_points",
    "team_goal_difference",
    "team_position_change_last_5",
    # Player context (6)
    "team_total_points_avg",
    "player_share_of_team_points",
    "player_share_of_team_goals",
    "team_avg_ict",
    "team_players_available",
    "squad_depth_at_position",
    # Opponent analysis features (24 total) - Phase 3
    # Defensive vulnerability (12)
    "opp_goals_conceded_last_3",
    "opp_goals_conceded_last_5",
    "opp_clean_sheets_last_3",
    "opp_clean_sheets_last_5",
    "opp_clean_sheets_rate",
    "opp_goals_conceded_home",
    "opp_goals_conceded_away",
    "opp_xgc_per_90",
    "opp_defensive_errors_last_5",
    "opp_saves_rate",
    "opp_big_chances_conceded_last_5",
    "opp_defensive_rating",
    # Attacking threat (8)
    "opp_goals_scored_last_3",
    "opp_goals_scored_last_5",
    "opp_xg_per_90",
    "opp_shots_on_target_last_5",
    "opp_big_chances_last_5",
    "opp_goals_scored_home",
    "opp_goals_scored_away",
    "opp_attacking_rating",
    # Context (4)
    "opp_league_position",
    "opp_form_score",
    "opp_days_rest",
    "opp_fixture_congestion",
    # Fixture context features (16 total) - Phase 4
    # Difficulty (6)
    "fdr_current",
    "fdr_next_3_avg",
    "fdr_next_5_avg",
    "fixture_swing",
    "is_tough_fixture",
    "is_easy_fixture",
    # DGW/BGW (4)
    "is_double_gameweek",
    "is_blank_gameweek",
    "dgw_fixture_count",
    "next_is_dgw",
    # Timing (6)
    "days_since_last_game",
    "fixture_congestion_7d",
    "fixture_congestion_14d",
    "kickoff_hour",
    "is_weekend_game",
    "is_evening_kickoff",
    # Position-specific features (8 total) - Phase 4
    # GK (2)
    "gk_saves_per_90",
    "gk_penalty_save_rate",
    # DEF (2)
    "def_clean_sheet_rate",
    "def_goal_involvement",
    # MID (2)
    "mid_goal_involvement_rate",
    "mid_creativity_threat_ratio",
    # FWD (2)
    "fwd_shots_per_90",
    "fwd_conversion_rate",
    # Interaction features (5 total) - Phase 4
    "form_x_fixture_difficulty",
    "ict_x_minutes",
    "ownership_x_form",
    "value_x_form",
    "momentum_score",
    # Derived features (5 total)
    "minutes_pct",
    "form_x_difficulty",
    "points_per_90",
    "goal_contributions_last_3",
    "points_volatility",
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
        # Infer season from directory path if not present in data
        if "season" not in df.columns:
            season_match = re.search(r"season_(\d{4}_\d{2})", str(f))
            if season_match:
                raw = season_match.group(1)
                df["season"] = raw.replace("_", "-")
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


def _temporal_train_test_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple:
    """
    Split data chronologically using season and gameweek columns.

    Args:
        df: DataFrame with 'season' and 'gameweek' columns.
        test_fraction: Fraction of data to hold out for testing.

    Returns:
        Tuple of (train_df, test_df).
    """
    df = df.copy()
    df["_sort_key"] = (
        df["season"].astype(str) + "_" + df["gameweek"].astype(str).str.zfill(2)
    )
    df = df.sort_values("_sort_key").reset_index(drop=True)

    split_idx = int(len(df) * (1 - test_fraction))
    train_df = df.iloc[:split_idx].drop(columns=["_sort_key"])
    test_df = df.iloc[split_idx:].drop(columns=["_sort_key"])

    return train_df, test_df


def train(args):
    """Main training function."""
    logger.info("Starting SageMaker training job")
    logger.info(
        f"Hyperparameters: n_estimators={args.n_estimators}, "
        f"max_depth={args.max_depth}, learning_rate={args.learning_rate}"
    )

    use_temporal = not args.random_split

    # Load data
    df = load_training_data(args.train)
    validate_data(df)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    logger.info(f"Features shape: {X.shape}")
    logger.info(
        f"Target stats: mean={y.mean():.2f}, std={y.std():.2f}, "
        f"min={y.min()}, max={y.max()}"
    )

    # Train-test split
    has_temporal = "gameweek" in df.columns and "season" in df.columns

    if use_temporal and has_temporal:
        train_df, test_df = _temporal_train_test_split(df, args.test_size)
        X_train = train_df[FEATURE_COLS]
        y_train = train_df[TARGET_COL]
        X_test = test_df[FEATURE_COLS]
        y_test = test_df[TARGET_COL]

        train_gws = sorted(train_df["gameweek"].unique())
        test_gws = sorted(test_df["gameweek"].unique())
        logger.info("Using temporal train/test split")
        logger.info(f"  Train GWs: {train_gws[0]}-{train_gws[-1]}")
        logger.info(f"  Test GWs: {test_gws[0]}-{test_gws[-1]}")
    else:
        if use_temporal and not has_temporal:
            logger.warning(
                "Columns 'gameweek' and/or 'season' not found. "
                "Falling back to random split."
            )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state
        )
        logger.info("Using random train/test split")

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
