"""
Historical Data Import Script

Downloads per-gameweek CSVs from the vaastav/Fantasy-Premier-League GitHub
repository and engineers training features matching the feature processor output.

Usage:
    python scripts/import_historical.py --seasons 2021-22,2022-23,2023-24 \
        --output-dir data/historical/
    python scripts/import_historical.py --seasons 2023-24 --output-dir data/historical/ \
        --upload-s3 --bucket fpl-ml-data-dev
"""

import argparse
import io
import logging
import os
import time
from pathlib import Path

import httpx
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Base URL for vaastav/Fantasy-Premier-League GitHub repo raw files
GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
)

# Feature columns matching the feature processor output
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

# Mapping from position string to numeric element_type
POSITION_MAP = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}


def convert_season_format(vaastav_season: str) -> str:
    """
    Convert vaastav season format to internal format.

    Args:
        vaastav_season: Season string like "2023-24"

    Returns:
        Internal format like "2023_24"
    """
    return vaastav_season.replace("-", "_")


def fetch_csv(url: str, timeout: float = 30.0) -> pd.DataFrame | None:
    """
    Fetch a CSV file from a URL.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        DataFrame or None if the URL returns 404
    """
    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        if response.status_code == 404:
            logger.debug(f"404 for {url}")
            return None
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text), encoding="utf-8")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None
        logger.warning(f"HTTP error fetching {url}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error fetching {url}: {e}")
        return None


def fetch_gameweek_csv(season: str, gameweek: int) -> pd.DataFrame | None:
    """
    Fetch a gameweek CSV from the vaastav repo.

    Args:
        season: Vaastav season format (e.g. "2023-24")
        gameweek: Gameweek number

    Returns:
        DataFrame with gameweek data or None if not found
    """
    url = f"{GITHUB_RAW_BASE}/{season}/gws/gw{gameweek}.csv"
    return fetch_csv(url)


def fetch_teams_csv(season: str) -> pd.DataFrame | None:
    """
    Fetch teams.csv from the vaastav repo.

    Args:
        season: Vaastav season format (e.g. "2023-24")

    Returns:
        DataFrame with team data or None if not found
    """
    url = f"{GITHUB_RAW_BASE}/{season}/teams.csv"
    return fetch_csv(url)


def build_team_strength_map(teams_df: pd.DataFrame) -> dict[int, int]:
    """
    Build a mapping from team ID to strength rating.

    The vaastav teams.csv has a 'strength' column (1-5 scale).

    Args:
        teams_df: DataFrame from teams.csv

    Returns:
        Dict mapping team ID to strength (1-5)
    """
    strength_map = {}
    for _, row in teams_df.iterrows():
        team_id = int(row.get("id", row.get("team_id", 0)))
        strength = int(row.get("strength", 3))
        strength_map[team_id] = strength
    return strength_map


def build_team_name_map(teams_df: pd.DataFrame) -> dict[str, int]:
    """
    Build a mapping from team name/short_name to team ID.

    The vaastav gameweek CSVs use team names (e.g. "Man Utd") rather than
    numeric IDs, so this mapping is needed to convert them.

    Args:
        teams_df: DataFrame from teams.csv

    Returns:
        Dict mapping team name to team ID
    """
    name_map = {}
    for _, row in teams_df.iterrows():
        team_id = int(row["id"])
        name_map[row["name"]] = team_id
        name_map[row["short_name"]] = team_id
    return name_map


def calculate_rolling_average(values: list[float], window: int) -> float:
    """
    Calculate rolling average over the last `window` values.

    Args:
        values: List of values (most recent last)
        window: Number of values to average

    Returns:
        Rolling average, or average of available data if fewer values exist
    """
    if not values:
        return 0.0
    recent = values[-window:] if len(values) >= window else values
    return sum(recent) / len(recent)


def calculate_minutes_pct(minutes_list: list[int], window: int = 5) -> float:
    """
    Calculate minutes played percentage over recent games.

    Args:
        minutes_list: List of minutes played per game (most recent last)
        window: Number of games to consider

    Returns:
        Average minutes percentage (0-1 scale)
    """
    if not minutes_list:
        return 0.0
    recent = minutes_list[-window:] if len(minutes_list) >= window else minutes_list
    total_minutes = sum(recent)
    max_minutes = 90 * len(recent)
    return total_minutes / max_minutes if max_minutes > 0 else 0.0


def engineer_historical_features(
    gw_df: pd.DataFrame,
    gameweek: int,
    player_history: dict[int, list[dict]],
    team_strength_map: dict[int, int],
    team_name_to_id: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Engineer training features from a single gameweek's data.

    Uses rolling history from prior gameweeks to calculate features,
    then uses the current gameweek's total_points as the target.

    Args:
        gw_df: DataFrame for the current gameweek
        gameweek: Current gameweek number
        player_history: Dict mapping player name to list of prior GW dicts
        team_strength_map: Dict mapping team ID to strength (1-5)
        team_name_to_id: Optional mapping from team name to numeric ID

    Returns:
        DataFrame with engineered features and actual_points target
    """
    features = []

    for _, row in gw_df.iterrows():
        player_name = str(row.get("name", ""))
        # Use 'element' as player_id if available, otherwise hash the name
        player_id = int(row["element"]) if "element" in row else hash(player_name)

        # Get prior history for this player
        history = player_history.get(player_id, [])

        # Calculate rolling features from prior gameweeks only
        if history:
            points_list = [h["total_points"] for h in history]
            minutes_list = [h["minutes"] for h in history]
            points_last_3 = calculate_rolling_average(points_list, 3)
            points_last_5 = calculate_rolling_average(points_list, 5)
            minutes_pct = calculate_minutes_pct(minutes_list, 5)
            # form_score: use points_last_5 as proxy (FPL form unavailable)
            form_score = points_last_5
        else:
            points_last_3 = 0.0
            points_last_5 = 0.0
            minutes_pct = 0.0
            form_score = 0.0

        # Opponent strength from team strength map
        opponent_team = int(row.get("opponent_team", 0))
        opponent_strength = team_strength_map.get(opponent_team, 3)

        # Home/away
        was_home = row.get("was_home", False)
        if isinstance(was_home, str):
            home_away = 1 if was_home.lower() in ("true", "1", "yes") else 0
        else:
            home_away = 1 if was_home else 0

        # Chance of playing: not available historically, default 100
        chance_of_playing = 100

        # Interaction feature
        form_x_difficulty = form_score * opponent_strength

        # Actual points (target)
        actual_points = int(row.get("total_points", 0))

        # Team ID (vaastav GW CSVs use team name strings, not numeric IDs)
        team_raw = row.get("team", row.get("team_id", 0))
        if isinstance(team_raw, str) and team_name_to_id:
            team_id = team_name_to_id.get(team_raw, 0)
        else:
            team_id = int(team_raw)

        # Position (vaastav GW CSVs use strings like "DEF", "FWD", etc.)
        pos_raw = row.get("element_type", row.get("position", 0))
        if isinstance(pos_raw, str):
            position = POSITION_MAP.get(pos_raw, 0)
        else:
            position = int(pos_raw)

        feature_row = {
            "player_id": player_id,
            "player_name": player_name,
            "team_id": team_id,
            "position": position,
            "gameweek": gameweek,
            "points_last_3": round(points_last_3, 2),
            "points_last_5": round(points_last_5, 2),
            "minutes_pct": round(minutes_pct, 3),
            "form_score": round(form_score, 2),
            "opponent_strength": opponent_strength,
            "home_away": home_away,
            "chance_of_playing": chance_of_playing,
            "form_x_difficulty": round(form_x_difficulty, 2),
            "actual_points": actual_points,
        }

        features.append(feature_row)

    return pd.DataFrame(features)


def process_season(
    season: str,
    output_dir: str,
    min_gameweek: int = 4,
    max_gameweek: int = 38,
) -> list[str]:
    """
    Process a full season of historical data.

    Downloads GW CSVs and teams.csv, engineers features from GW `min_gameweek`
    onwards (earlier GWs used only for building rolling history).

    Args:
        season: Vaastav season format (e.g. "2023-24")
        output_dir: Directory to write Parquet files
        min_gameweek: First gameweek to generate training rows for
        max_gameweek: Last gameweek to process

    Returns:
        List of output file paths
    """
    internal_season = convert_season_format(season)
    season_dir = Path(output_dir) / f"season_{internal_season}"
    season_dir.mkdir(parents=True, exist_ok=True)

    # Fetch teams data
    logger.info(f"Fetching teams data for {season}...")
    teams_df = fetch_teams_csv(season)
    if teams_df is None:
        logger.error(f"Could not fetch teams.csv for {season}")
        return []

    team_strength_map = build_team_strength_map(teams_df)
    team_name_to_id = build_team_name_map(teams_df)
    logger.info(f"Built team strength map with {len(team_strength_map)} teams")

    # Fetch all gameweek CSVs
    logger.info(f"Fetching gameweek CSVs for {season}...")
    gw_data: dict[int, pd.DataFrame] = {}

    for gw in range(1, max_gameweek + 1):
        df = fetch_gameweek_csv(season, gw)
        if df is not None:
            gw_data[gw] = df
            logger.info(f"  GW{gw}: {len(df)} rows")
        else:
            logger.info(f"  GW{gw}: not found (skipping)")
        # Brief pause to avoid rate limiting
        time.sleep(0.2)

    if not gw_data:
        logger.error(f"No gameweek data found for {season}")
        return []

    logger.info(f"Downloaded {len(gw_data)} gameweeks for {season}")

    # Build rolling history and generate features
    # player_history: maps player_id -> list of dicts with total_points, minutes
    player_history: dict[int, list[dict]] = {}
    output_files = []

    for gw in sorted(gw_data.keys()):
        gw_df = gw_data[gw]

        if gw >= min_gameweek:
            # Engineer features using accumulated history
            features_df = engineer_historical_features(
                gw_df, gw, player_history, team_strength_map, team_name_to_id
            )

            if len(features_df) > 0:
                output_path = season_dir / f"gw{gw}_features_training.parquet"
                features_df.to_parquet(output_path, engine="pyarrow", index=False)
                output_files.append(str(output_path))
                logger.info(
                    f"  Saved GW{gw} features: {len(features_df)} rows "
                    f"-> {output_path}"
                )

        # Update player history with this gameweek's data (for future GWs)
        for _, row in gw_df.iterrows():
            player_id = (
                int(row["element"]) if "element" in row else hash(str(row.get("name")))
            )
            if player_id not in player_history:
                player_history[player_id] = []

            player_history[player_id].append(
                {
                    "total_points": int(row.get("total_points", 0)),
                    "minutes": int(row.get("minutes", 0)),
                    "round": gw,
                }
            )

    logger.info(
        f"Season {season} complete: {len(output_files)} feature files generated"
    )
    return output_files


def upload_to_s3(
    output_dir: str,
    bucket: str,
    s3_prefix: str = "processed",
) -> int:
    """
    Upload local Parquet files to S3.

    Args:
        output_dir: Local directory containing Parquet files
        bucket: S3 bucket name
        s3_prefix: Prefix for S3 keys

    Returns:
        Number of files uploaded
    """
    import boto3

    s3_client = boto3.client(
        "s3",
        region_name=os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2"),
        endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
    )

    output_path = Path(output_dir)
    parquet_files = list(output_path.rglob("*.parquet"))

    uploaded = 0
    for file_path in parquet_files:
        # Build S3 key: processed/season_2023_24/gw5_features_training.parquet
        relative = file_path.relative_to(output_path)
        s3_key = f"{s3_prefix}/{relative}"

        logger.info(f"Uploading {file_path} -> s3://{bucket}/{s3_key}")
        s3_client.upload_file(str(file_path), bucket, s3_key)
        uploaded += 1

    logger.info(f"Uploaded {uploaded} files to s3://{bucket}/{s3_prefix}/")
    return uploaded


def main():
    """Main entry point for historical data import."""
    parser = argparse.ArgumentParser(
        description="Import historical FPL data from vaastav/Fantasy-Premier-League"
    )
    parser.add_argument(
        "--seasons",
        type=str,
        required=True,
        help="Comma-separated seasons (e.g. 2021-22,2022-23,2023-24)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/historical/",
        help="Output directory for Parquet files (default: data/historical/)",
    )
    parser.add_argument(
        "--min-gameweek",
        type=int,
        default=4,
        help="First gameweek to generate training rows for (default: 4)",
    )
    parser.add_argument(
        "--max-gameweek",
        type=int,
        default=38,
        help="Last gameweek to process (default: 38)",
    )
    parser.add_argument(
        "--upload-s3",
        action="store_true",
        help="Upload output files to S3",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="fpl-ml-data-dev",
        help="S3 bucket name (default: fpl-ml-data-dev)",
    )

    args = parser.parse_args()

    seasons = [s.strip() for s in args.seasons.split(",")]
    all_files = []

    for season in seasons:
        logger.info(f"Processing season {season}...")
        files = process_season(
            season=season,
            output_dir=args.output_dir,
            min_gameweek=args.min_gameweek,
            max_gameweek=args.max_gameweek,
        )
        all_files.extend(files)

    logger.info(f"Total: {len(all_files)} feature files generated")

    if args.upload_s3:
        upload_to_s3(args.output_dir, args.bucket)

    return all_files


if __name__ == "__main__":
    main()
