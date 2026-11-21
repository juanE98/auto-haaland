"""
Current Season Backfill Script

Uses the FPL API's element-summary endpoint to reconstruct historical features
for the current season. This produces training data from completed gameweeks
using the same feature engineering as the feature processor Lambda.

Usage:
    python scripts/backfill_current_season.py --output-dir data/current/
    python scripts/backfill_current_season.py --output-dir data/current/ \
        --upload-s3 --bucket fpl-ml-data-dev --start-gw 1
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Add project root and lambdas to path for imports
# lambdas is needed because feature_config uses 'from common...' for Lambda compat
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "lambdas"))

from lambdas.common.feature_categories.fixture_features import (  # noqa: E402
    FIXTURE_FEATURES,
)
from lambdas.common.feature_categories.interaction_features import (  # noqa: E402
    INTERACTION_FEATURES,
)
from lambdas.common.feature_categories.opponent_features import (  # noqa: E402
    OPPONENT_FEATURES,
)
from lambdas.common.feature_categories.position_features import (  # noqa: E402
    POSITION_FEATURES,
)
from lambdas.common.feature_categories.team_features import (  # noqa: E402
    TEAM_FEATURES,
    _count_games_at_current_team,
)
from lambdas.common.feature_config import (  # noqa: E402
    BOOTSTRAP_FEATURES,
    FEATURE_COLS,
    aggregate_history_by_round,
    compute_derived_features,
    compute_prior_features,
    compute_prior_fwd_ict_per_90,
    compute_prior_team_context,
    compute_rolling_features,
    get_prior_team_features,
)
from lambdas.common.fpl_api import FPLApiClient  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Batch size for player summary API calls
BATCH_SIZE = 50
BATCH_DELAY = 1.0  # Seconds between batches

ADDITIVE_HISTORY_FIELDS = {
    "total_points",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "bps",
    "ict_index",
    "threat",
    "creativity",
    "influence",
    "bonus",
    "yellow_cards",
    "saves",
    "expected_goals",
    "expected_assists",
    "starts",
    "red_cards",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "defensive_contribution",
    "clearances_blocks_interceptions",
    "tackles",
    "recoveries",
}


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
) -> dict[int, dict[str, list[dict]]]:
    """
    Fetch element-summary for all players in batches.

    Args:
        api_client: FPL API client
        player_ids: List of player IDs to fetch
        batch_size: Number of players per batch
        batch_delay: Delay between batches in seconds

    Returns:
        Dict mapping player_id to current and previous-season history
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
                all_histories[player_id] = {
                    "history": summary.get("history", []),
                    "history_past": summary.get("history_past", []),
                }
            except Exception as e:
                logger.warning(f"Failed to fetch history for player {player_id}: {e}")
                all_histories[player_id] = {"history": [], "history_past": []}

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
    if not entries:
        return None

    combined = entries[-1].copy()
    for field in ADDITIVE_HISTORY_FIELDS:
        combined[field] = sum(float(entry.get(field, 0) or 0) for entry in entries)
    combined["fixture_count"] = len(entries)
    return combined


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
                fixture.get("team_h_difficulty")
                or team_strength_map.get(opponent_id, 3),
                1,
                ad.get("attack_away", 1200),
                ad.get("defence_away", 1200),
            )
        elif fixture.get("team_a") == player_team_id:
            # Player is away, opponent is home
            opponent_id = fixture.get("team_h")
            ad = ad_map.get(opponent_id, {})
            return (
                fixture.get("team_a_difficulty")
                or team_strength_map.get(opponent_id, 3),
                0,
                ad.get("attack_home", 1200),
                ad.get("defence_home", 1200),
            )

    return 3, 0, 1200, 1200


def _team_from_history_entry(
    entry: dict,
    fixtures: list[dict],
    gameweek: int,
) -> int | None:
    """Reconstruct the player's team from one history entry and its fixture."""
    fixture_id = entry.get("fixture")
    if fixture_id is not None:
        fixture = next((f for f in fixtures if f.get("id") == fixture_id), None)
        if fixture is not None:
            candidates = [fixture]
        else:
            candidates = []
    else:
        candidates = []

    was_home = entry.get("was_home")
    opponent_id = entry.get("opponent_team")
    if not candidates and opponent_id is not None and isinstance(was_home, bool):
        candidates = [
            fixture
            for fixture in fixtures
            if fixture.get("event") in (None, gameweek)
            and (
                (was_home and fixture.get("team_a") == opponent_id)
                or (not was_home and fixture.get("team_h") == opponent_id)
            )
        ]
        if len(candidates) != 1:
            return None

    if len(candidates) != 1 or not isinstance(was_home, bool):
        return None

    fixture = candidates[0]
    expected_opponent = fixture.get("team_a" if was_home else "team_h")
    if opponent_id is not None and opponent_id != expected_opponent:
        return None
    return fixture.get("team_h" if was_home else "team_a")


def get_gameweek_player_team_ids(
    all_histories: dict[int, object],
    fixtures: list[dict],
    gameweek: int,
) -> dict[int, int]:
    """Build a fail-closed target-gameweek player roster from raw history."""
    player_team_ids = {}
    for player_id, player_summary in all_histories.items():
        full_history = (
            player_summary
            if isinstance(player_summary, list)
            else player_summary.get("history", [])
        )
        entries = [entry for entry in full_history if entry.get("round") == gameweek]
        if not entries:
            continue
        derived_team_ids = [
            _team_from_history_entry(entry, fixtures, gameweek) for entry in entries
        ]
        team_ids = set(derived_team_ids)
        if None in team_ids or len(team_ids) != 1:
            logger.warning(
                "Could not reconstruct GW%s team for player %s", gameweek, player_id
            )
            continue
        player_team_ids[player_id] = team_ids.pop()
    return player_team_ids


def engineer_backfill_features(
    players: list[dict],
    all_histories: dict[int, object],
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
    player_team_ids = get_gameweek_player_team_ids(all_histories, fixtures, gameweek)
    prior_team_context = compute_prior_team_context(player_team_ids, all_histories)

    for player in players:
        player_id = player["id"]
        team_id = player_team_ids.get(player_id)
        player_summary = all_histories.get(player_id, {})
        if isinstance(player_summary, list):
            full_history = player_summary
            history_past = []
        else:
            full_history = player_summary.get("history", [])
            history_past = player_summary.get("history_past", [])

        # Get actual points for this gameweek (target variable)
        gw_entry = get_gameweek_entry(full_history, gameweek)
        if gw_entry is None:
            # Player did not play in this gameweek
            continue
        if team_id is None:
            continue

        actual_points = gw_entry.get("total_points", 0)

        # Filter history to ONLY prior gameweeks (prevent data leakage)
        prior_history = aggregate_history_by_round(
            filter_history_before_gameweek(full_history, gameweek)
        )

        # Calculate rolling features from prior history
        if prior_history:
            rolling = compute_rolling_features(prior_history)
            form_score = rolling.get("points_last_5", 0.0)
        else:
            rolling = {name: 0.0 for name in FEATURE_COLS if "_last_" in name}
            form_score = 0.0

        # Opponent info from fixtures
        opponent_strength, home_away, opp_attack_strength, opp_defence_strength = (
            get_fixture_info(
                team_id, fixtures, team_strength_map, team_attack_defence_map
            )
        )

        # Selected by percent (ownership from bootstrap)
        selected_by_percent = 0.0

        # Chance of playing: not reliably available historically
        chance_of_playing = 100

        # Static features
        static = {
            "form_score": round(form_score, 2),
            "opponent_strength": opponent_strength,
            "home_away": home_away,
            "chance_of_playing": chance_of_playing,
            "position": player.get("element_type", 0),
            "opponent_attack_strength": opp_attack_strength,
            "opponent_defence_strength": opp_defence_strength,
            "selected_by_percent": selected_by_percent,
            "now_cost": gw_entry.get("value", 0),
        }

        # Derived features
        derived = compute_derived_features(prior_history, rolling, static)
        early_season = {
            "history_game_count": float(min(len(prior_history), 10)),
            **compute_prior_features(history_past),
        }
        early_season["prior_fwd_ict_per_90"] = compute_prior_fwd_ict_per_90(
            early_season["prior_ict_per_90"], static["position"]
        )
        opponent_team_ids = [
            fixture["team_a"] if fixture.get("team_h") == team_id else fixture["team_h"]
            for fixture in fixtures
            if fixture.get("event") in (None, gameweek)
            and team_id in (fixture.get("team_h"), fixture.get("team_a"))
        ]
        prior_team_features = get_prior_team_features(
            prior_team_context, team_id, opponent_team_ids
        )

        # Current bootstrap totals include the target and later gameweeks, so
        # they cannot be reconstructed safely for historical rows.
        bootstrap = {feat: 0.0 for feat in BOOTSTRAP_FEATURES}
        bootstrap["status_available"] = 1.0

        # Team and opponent features (default values for backfill)
        # Full computation would require team fixtures data not passed here
        team_feats = {feat: 0.0 for feat in TEAM_FEATURES}

        # Compute games_at_current_team from player history and fixtures
        fixtures_by_id = {f.get("id"): f for f in fixtures if f.get("id")}
        team_feats["games_at_current_team"] = float(
            _count_games_at_current_team(prior_history, team_id, fixtures_by_id)
        )

        opponent_feats = {feat: 0.0 for feat in OPPONENT_FEATURES}

        # Fixture, position, and interaction features (default values)
        fixture_feats = {feat: 0.0 for feat in FIXTURE_FEATURES}
        fixture_feats["fdr_current"] = 3.0  # Default medium difficulty
        fixture_feats["fdr_next_3_avg"] = 3.0
        fixture_feats["fdr_next_5_avg"] = 3.0
        fixture_count = int(gw_entry.get("fixture_count", 1) or 1)
        fixture_feats["dgw_fixture_count"] = float(fixture_count)
        fixture_feats["is_double_gameweek"] = 1.0 if fixture_count > 1 else 0.0
        fixture_feats["days_since_last_game"] = 7.0
        fixture_feats["kickoff_hour"] = 15.0  # Default 3pm
        fixture_feats["is_weekend_game"] = 1.0

        position_feats = {feat: 0.0 for feat in POSITION_FEATURES}
        interaction_feats = {feat: 0.0 for feat in INTERACTION_FEATURES}

        # Build feature row
        feature_row = {
            "player_id": player_id,
            "player_name": player.get("web_name", ""),
            "team_id": team_id,
            "gameweek": gameweek,
        }
        feature_row.update(rolling)
        feature_row.update(static)
        feature_row.update(bootstrap)
        feature_row.update(team_feats)
        feature_row.update(opponent_feats)
        feature_row.update(fixture_feats)
        feature_row.update(position_feats)
        feature_row.update(interaction_feats)
        feature_row.update(derived)
        feature_row.update(early_season)
        feature_row.update(prior_team_features)
        feature_row["actual_points"] = actual_points

        features.append(feature_row)

    metadata = [
        "player_id",
        "player_name",
        "team_id",
        "gameweek",
        "chance_of_playing",
    ]
    return pd.DataFrame(features).reindex(
        columns=metadata + FEATURE_COLS + ["actual_points"]
    )


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
        default=1,
        help="First gameweek to generate training rows for (default: 1)",
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
