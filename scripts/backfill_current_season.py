"""
Current Season Backfill Script

Uses the FPL API's element-summary endpoint to reconstruct historical features
for the current season. This produces training data from completed gameweeks
using the same feature engineering as the feature processor Lambda.

Usage:
    python scripts/backfill_current_season.py --output-dir data/current/
    python scripts/backfill_current_season.py --output-dir data/current/ \
        --upload-s3 --bucket fpl-ml-data-dev --start-gw 4
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lambdas.common.fpl_api import FPLApiClient  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

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

# Batch size for player summary API calls
BATCH_SIZE = 50
BATCH_DELAY = 1.0  # Seconds between batches


def get_finished_gameweeks(events: list[dict]) -> list[int]:
    """
    Get list of finished gameweek numbers from events data.

    Args:
        events: List of event dicts from bootstrap-static

    Returns:
        Sorted list of finished gameweek numbers
    """
    return sorted(e["id"] for e in events if e.get("finished", False))


def get_team_strength_map(teams: list[dict]) -> dict[int, int]:
    """
    Build team ID to strength mapping from bootstrap teams data.

    Args:
        teams: List of team dicts from bootstrap-static

    Returns:
        Dict mapping team ID to strength (1-5)
    """
    return {t["id"]: t.get("strength", 3) for t in teams}


def get_team_attack_defence_map(
    teams: list[dict],
) -> dict[int, dict[str, int]]:
    """
    Build team ID to attack/defence strength mapping from bootstrap teams.

    Args:
        teams: List of team dicts from bootstrap-static

    Returns:
        Dict mapping team ID to {"attack_home", "attack_away",
        "defence_home", "defence_away"}
    """
    return {
        t["id"]: {
            "attack_home": t.get("strength_attack_home", 1200),
            "attack_away": t.get("strength_attack_away", 1200),
            "defence_home": t.get("strength_defence_home", 1200),
            "defence_away": t.get("strength_defence_away", 1200),
        }
        for t in teams
    }


def fetch_all_player_histories(
    api_client: FPLApiClient,
    player_ids: list[int],
    batch_size: int = BATCH_SIZE,
    batch_delay: float = BATCH_DELAY,
) -> dict[int, list[dict]]:
    """
    Fetch element-summary for all players in batches.

    Args:
        api_client: FPL API client
        player_ids: List of player IDs to fetch
        batch_size: Number of players per batch
        batch_delay: Delay between batches in seconds

    Returns:
        Dict mapping player_id to list of history entries
    """
    all_histories = {}
    total = len(player_ids)

    for i in range(0, total, batch_size):
        batch = player_ids[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size

        logger.info(
            f"Fetching player histories: batch {batch_num}/{total_batches} "
            f"({len(batch)} players)"
        )

        for player_id in batch:
            try:
                summary = api_client.get_player_summary(player_id)
                history = summary.get("history", [])
                all_histories[player_id] = history
            except Exception as e:
                logger.warning(f"Failed to fetch history for player {player_id}: {e}")
                all_histories[player_id] = []

        # Rate limiting pause between batches
        if i + batch_size < total:
            time.sleep(batch_delay)

    logger.info(f"Fetched histories for {len(all_histories)} players")
    return all_histories


def filter_history_before_gameweek(
    history: list[dict],
    gameweek: int,
) -> list[dict]:
    """
    Filter player history to only include entries before the target gameweek.

    Prevents data leakage by excluding the current and future GW data.

    Args:
        history: Full player history for the season
        gameweek: Target gameweek (entries with round < gameweek are kept)

    Returns:
        Filtered history list
    """
    return [h for h in history if h.get("round", 0) < gameweek]


def get_gameweek_entry(history: list[dict], gameweek: int) -> dict | None:
    """
    Get a player's entry for a specific gameweek.

    Args:
        history: Player history list
        gameweek: Target gameweek

    Returns:
        History dict for the gameweek, or None if not found
    """
    entries = [h for h in history if h.get("round") == gameweek]
    return entries[0] if entries else None


def calculate_rolling_average(values: list[float], window: int) -> float:
    """Calculate rolling average over the last `window` values."""
    if not values:
        return 0.0
    recent = values[-window:] if len(values) >= window else values
    return sum(recent) / len(recent)


def calculate_minutes_pct(history: list[dict], window: int = 5) -> float:
    """Calculate minutes played percentage over recent games."""
    if not history:
        return 0.0
    recent = history[-window:] if len(history) >= window else history
    total_minutes = sum(h.get("minutes", 0) for h in recent)
    max_minutes = 90 * len(recent)
    return total_minutes / max_minutes if max_minutes > 0 else 0.0


def get_fixture_info(
    player_team_id: int,
    fixtures: list[dict],
    team_strength_map: dict[int, int],
    team_attack_defence_map: dict[int, dict[str, int]] | None = None,
) -> tuple[int, int, int, int]:
    """
    Get opponent strength and home/away status from fixtures.

    Args:
        player_team_id: Player's team ID
        fixtures: List of fixtures for the gameweek
        team_strength_map: Team ID to strength mapping
        team_attack_defence_map: Optional team ID to attack/defence mapping

    Returns:
        Tuple of (opponent_strength, is_home, opp_attack_strength,
        opp_defence_strength)
    """
    ad_map = team_attack_defence_map or {}
    for fixture in fixtures:
        if fixture.get("team_h") == player_team_id:
            # Player is home, opponent is away
            opponent_id = fixture.get("team_a")
            ad = ad_map.get(opponent_id, {})
            return (
                team_strength_map.get(opponent_id, 3),
                1,
                ad.get("attack_away", 1200),
                ad.get("defence_away", 1200),
            )
        elif fixture.get("team_a") == player_team_id:
            # Player is away, opponent is home
            opponent_id = fixture.get("team_h")
            ad = ad_map.get(opponent_id, {})
            return (
                team_strength_map.get(opponent_id, 3),
                0,
                ad.get("attack_home", 1200),
                ad.get("defence_home", 1200),
            )

    return 3, 0, 1200, 1200


def engineer_backfill_features(
    players: list[dict],
    all_histories: dict[int, list[dict]],
    fixtures: list[dict],
    team_strength_map: dict[int, int],
    gameweek: int,
    team_attack_defence_map: dict[int, dict[str, int]] | None = None,
) -> pd.DataFrame:
    """
    Engineer training features for a historical gameweek using API data.

    Only uses history from prior gameweeks to avoid data leakage.

    Args:
        players: List of player dicts from bootstrap-static
        all_histories: Dict mapping player_id to full season history
        fixtures: Fixtures for the target gameweek
        team_strength_map: Team ID to strength mapping
        gameweek: Target gameweek number
        team_attack_defence_map: Optional team ID to attack/defence mapping

    Returns:
        DataFrame with features and actual_points target
    """
    features = []

    for player in players:
        player_id = player["id"]
        team_id = player["team"]
        full_history = all_histories.get(player_id, [])

        # Get actual points for this gameweek (target variable)
        gw_entry = get_gameweek_entry(full_history, gameweek)
        if gw_entry is None:
            # Player did not play in this gameweek
            continue

        actual_points = gw_entry.get("total_points", 0)

        # Filter history to ONLY prior gameweeks (prevent data leakage)
        prior_history = filter_history_before_gameweek(full_history, gameweek)

        # Calculate rolling features from prior history
        if prior_history:
            points_list = [h.get("total_points", 0) for h in prior_history]
            points_last_3 = calculate_rolling_average(points_list, 3)
            points_last_5 = calculate_rolling_average(points_list, 5)
            minutes_pct = calculate_minutes_pct(prior_history, 5)
            form_score = points_last_5
            goals_list = [h.get("goals_scored", 0) for h in prior_history]
            assists_list = [h.get("assists", 0) for h in prior_history]
            cs_list = [h.get("clean_sheets", 0) for h in prior_history]
            bps_list = [h.get("bps", 0) for h in prior_history]
            goals_last_3 = calculate_rolling_average(goals_list, 3)
            assists_last_3 = calculate_rolling_average(assists_list, 3)
            clean_sheets_last_3 = calculate_rolling_average(cs_list, 3)
            bps_last_3 = calculate_rolling_average(bps_list, 3)
            ict_list = [float(h.get("ict_index", 0) or 0) for h in prior_history]
            threat_list = [float(h.get("threat", 0) or 0) for h in prior_history]
            creativity_list = [
                float(h.get("creativity", 0) or 0) for h in prior_history
            ]
            ict_index_last_3 = calculate_rolling_average(ict_list, 3)
            threat_last_3 = calculate_rolling_average(threat_list, 3)
            creativity_last_3 = calculate_rolling_average(creativity_list, 3)
        else:
            points_last_3 = 0.0
            points_last_5 = 0.0
            minutes_pct = 0.0
            form_score = 0.0
            goals_last_3 = 0.0
            assists_last_3 = 0.0
            clean_sheets_last_3 = 0.0
            bps_last_3 = 0.0
            ict_index_last_3 = 0.0
            threat_last_3 = 0.0
            creativity_last_3 = 0.0

        # Opponent info from fixtures
        opponent_strength, home_away, opp_attack_strength, opp_defence_strength = (
            get_fixture_info(
                team_id, fixtures, team_strength_map, team_attack_defence_map
            )
        )

        # Selected by percent (ownership from bootstrap)
        selected_by_percent = float(player.get("selected_by_percent", 0) or 0)

        # Chance of playing: not reliably available historically
        chance_of_playing = 100

        # Interaction feature
        form_x_difficulty = form_score * opponent_strength

        feature_row = {
            "player_id": player_id,
            "player_name": player.get("web_name", ""),
            "team_id": team_id,
            "position": player.get("element_type", 0),
            "gameweek": gameweek,
            "points_last_3": round(points_last_3, 2),
            "points_last_5": round(points_last_5, 2),
            "minutes_pct": round(minutes_pct, 3),
            "form_score": round(form_score, 2),
            "opponent_strength": opponent_strength,
            "home_away": home_away,
            "chance_of_playing": chance_of_playing,
            "form_x_difficulty": round(form_x_difficulty, 2),
            "goals_last_3": round(goals_last_3, 2),
            "assists_last_3": round(assists_last_3, 2),
            "clean_sheets_last_3": round(clean_sheets_last_3, 2),
            "bps_last_3": round(bps_last_3, 2),
            "ict_index_last_3": round(ict_index_last_3, 2),
            "threat_last_3": round(threat_last_3, 2),
            "creativity_last_3": round(creativity_last_3, 2),
            "opponent_attack_strength": opp_attack_strength,
            "opponent_defence_strength": opp_defence_strength,
            "selected_by_percent": selected_by_percent,
            "actual_points": actual_points,
        }

        features.append(feature_row)

    return pd.DataFrame(features)


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
        relative = file_path.relative_to(output_path)
        s3_key = f"{s3_prefix}/{relative}"

        logger.info(f"Uploading {file_path} -> s3://{bucket}/{s3_key}")
        s3_client.upload_file(str(file_path), bucket, s3_key)
        uploaded += 1

    logger.info(f"Uploaded {uploaded} files to s3://{bucket}/{s3_prefix}/")
    return uploaded


def main():
    """Main entry point for current season backfill."""
    parser = argparse.ArgumentParser(
        description="Backfill current season FPL data using the FPL API"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/current/",
        help="Output directory for Parquet files (default: data/current/)",
    )
    parser.add_argument(
        "--start-gw",
        type=int,
        default=4,
        help="First gameweek to generate training rows for (default: 4)",
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
    parser.add_argument(
        "--end-gw",
        type=int,
        default=None,
        help="Last gameweek to include (default: all finished gameweeks)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Players per API batch (default: {BATCH_SIZE})",
    )

    args = parser.parse_args()

    with FPLApiClient() as api_client:
        # Step 1: Fetch bootstrap data
        logger.info("Fetching bootstrap-static data...")
        bootstrap = api_client.get_bootstrap_static()

        players = bootstrap.get("elements", [])
        teams = bootstrap.get("teams", [])
        events = bootstrap.get("events", [])

        season = api_client.get_season_string()
        logger.info(f"Season: {season}, Players: {len(players)}, Teams: {len(teams)}")

        team_strength_map = get_team_strength_map(teams)
        team_attack_defence_map = get_team_attack_defence_map(teams)

        # Step 2: Get finished gameweeks
        finished_gws = get_finished_gameweeks(events)
        target_gws = [gw for gw in finished_gws if gw >= args.start_gw]
        if args.end_gw is not None:
            target_gws = [gw for gw in target_gws if gw <= args.end_gw]
        logger.info(
            f"Finished gameweeks: {len(finished_gws)}, "
            f"Target gameweeks (GW{args.start_gw}"
            f"{f'-{args.end_gw}' if args.end_gw else '+'}): {len(target_gws)}"
        )

        if not target_gws:
            logger.warning("No target gameweeks to process")
            return []

        # Step 3: Fetch all player histories
        player_ids = [p["id"] for p in players]
        logger.info(f"Fetching histories for {len(player_ids)} players...")
        all_histories = fetch_all_player_histories(
            api_client, player_ids, batch_size=args.batch_size
        )

        # Step 4: Process each target gameweek
        season_dir = Path(args.output_dir) / f"season_{season}"
        season_dir.mkdir(parents=True, exist_ok=True)

        output_files = []

        for gw in target_gws:
            logger.info(f"Processing GW{gw}...")

            # Fetch fixtures for this gameweek
            fixtures = api_client.get_fixtures(gameweek=gw)

            # Engineer features
            features_df = engineer_backfill_features(
                players,
                all_histories,
                fixtures,
                team_strength_map,
                gw,
                team_attack_defence_map,
            )

            if len(features_df) > 0:
                output_path = season_dir / f"gw{gw}_features_training.parquet"
                features_df.to_parquet(output_path, engine="pyarrow", index=False)
                output_files.append(str(output_path))
                logger.info(f"  GW{gw}: {len(features_df)} rows -> {output_path}")
            else:
                logger.warning(f"  GW{gw}: no data generated")

    logger.info(f"Backfill complete: {len(output_files)} files generated")

    if args.upload_s3:
        upload_to_s3(args.output_dir, args.bucket)

    return output_files


if __name__ == "__main__":
    main()
