"""FPL Prediction CLI - Query predictions, trigger pipeline, and manage data."""

import json
import subprocess
import sys

import boto3
import click
import httpx
from tabulate import tabulate


class FPLClient:
    """HTTP client for FPL Prediction API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def get_top(
        self,
        gameweek: int,
        position: str | None = None,
        limit: int = 10,
        sort_by: str = "points",
    ) -> dict:
        """Get top predicted scorers for a gameweek."""
        params = {"gameweek": gameweek, "limit": limit, "sort_by": sort_by}
        if position:
            params["position"] = position
        return self._get("/top", params)

    def get_player(self, player_id: int, gameweek: int | None = None) -> dict:
        """Get predictions for a specific player."""
        params = {"gameweek": gameweek} if gameweek else {}
        return self._get(f"/predictions/{player_id}", params)

    def compare(self, player_ids: list[int], gameweek: int) -> dict:
        """Compare multiple players for a gameweek."""
        params = {"players": ",".join(map(str, player_ids)), "gameweek": gameweek}
        return self._get("/compare", params)

    def get_predictions(self, gameweek: int, limit: int = 100) -> dict:
        """Get all predictions for a gameweek."""
        params = {"gameweek": gameweek, "limit": limit}
        return self._get("/predictions", params)

    def _get(self, path: str, params: dict) -> dict:
        """Make GET request to API."""
        url = f"{self.base_url}{path}"
        response = httpx.get(url, params=params, timeout=30.0)
        response.raise_for_status()
        return response.json()


def format_points(points: float) -> str:
    """Format predicted points for display."""
    return f"{points:.1f}"


@click.group()
@click.option(
    "--endpoint",
    envvar="API_ENDPOINT",
    help="API Gateway URL (or set API_ENDPOINT env var)",
)
@click.option(
    "--local",
    is_flag=True,
    help="Use local API at http://localhost:3000",
)
@click.pass_context
def cli(ctx, endpoint: str | None, local: bool):
    """FPL Prediction CLI - Query predictions and trigger pipeline."""
    ctx.ensure_object(dict)

    if local:
        base_url = "http://localhost:3000"
    elif endpoint:
        base_url = endpoint
    else:
        base_url = None

    ctx.obj["base_url"] = base_url


def get_client(ctx) -> FPLClient:
    """Get FPLClient from context, validating endpoint is set."""
    base_url = ctx.obj.get("base_url")
    if not base_url:
        click.echo("Error: --endpoint or --local required", err=True)
        click.echo(
            "Set API_ENDPOINT env var or use --local for local testing", err=True
        )
        ctx.exit(1)
    return FPLClient(base_url)


@cli.command()
@click.option(
    "--gameweek",
    "-g",
    required=True,
    type=int,
    help="Gameweek number",
)
@click.option(
    "--position",
    "-p",
    type=click.Choice(["GKP", "DEF", "MID", "FWD"], case_sensitive=False),
    help="Filter by position",
)
@click.option(
    "--limit",
    "-n",
    default=10,
    type=int,
    help="Number of results (default: 10)",
)
@click.option(
    "--sort",
    "-s",
    type=click.Choice(["points", "haul"], case_sensitive=False),
    default="points",
    help="Sort by predicted points or haul probability (default: points)",
)
@click.pass_context
def top(ctx, gameweek: int, position: str | None, limit: int, sort: str):
    """Get top predicted scorers for a gameweek."""
    client = get_client(ctx)

    try:
        data = client.get_top(gameweek, position, limit, sort_by=sort)
    except httpx.HTTPStatusError as e:
        click.echo(f"Error: {e.response.status_code} - {e.response.text}", err=True)
        ctx.exit(1)

    predictions = data.get("predictions", [])
    if not predictions:
        click.echo(f"No predictions found for gameweek {gameweek}")
        return

    sort_label = "Haul %" if sort == "haul" else "Points"
    title = f"Top {len(predictions)} Predictions - Gameweek {gameweek}"
    if position:
        title += f" ({position})"
    title += f" [sorted by {sort_label}]"
    click.echo(title)
    click.echo()

    # Include haul probability if available
    has_haul = any(p.get("haul_probability") for p in predictions)

    if has_haul:
        table_data = [
            [
                i + 1,
                p.get("player_name", f"ID: {p['player_id']}"),
                p.get("position", "-"),
                format_points(p["predicted_points"]),
                f"{p.get('haul_probability', 0):.0f}%",
            ]
            for i, p in enumerate(predictions)
        ]
        headers = ["Rank", "Player", "Pos", "Points", "Haul %"]
    else:
        table_data = [
            [
                i + 1,
                p.get("player_name", f"ID: {p['player_id']}"),
                p.get("position", "-"),
                format_points(p["predicted_points"]),
            ]
            for i, p in enumerate(predictions)
        ]
        headers = ["Rank", "Player", "Pos", "Points"]

    click.echo(tabulate(table_data, headers=headers))


@cli.command()
@click.argument("player_id", type=int)
@click.option(
    "--gameweek",
    "-g",
    type=int,
    help="Specific gameweek (omit for all)",
)
@click.pass_context
def player(ctx, player_id: int, gameweek: int | None):
    """Get predictions for a specific player."""
    client = get_client(ctx)

    try:
        data = client.get_player(player_id, gameweek)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            click.echo(f"Player {player_id} not found", err=True)
        else:
            click.echo(f"Error: {e.response.status_code} - {e.response.text}", err=True)
        ctx.exit(1)

    player_name = data.get("player_name", f"ID: {player_id}")
    position = data.get("position", "-")

    click.echo(f"Player: {player_name} (ID: {player_id}) - Position: {position}")
    click.echo()

    if "predictions" in data:
        table_data = [
            [p["gameweek"], format_points(p["predicted_points"])]
            for p in data["predictions"]
        ]
        click.echo(tabulate(table_data, headers=["Gameweek", "Points"]))
    else:
        click.echo(
            f"Gameweek {data['gameweek']}: {format_points(data['predicted_points'])} pts"
        )


@cli.command()
@click.argument("players")
@click.option(
    "--gameweek",
    "-g",
    required=True,
    type=int,
    help="Gameweek number",
)
@click.pass_context
def compare(ctx, players: str, gameweek: int):
    """Compare multiple players for a gameweek.

    PLAYERS: Comma-separated player IDs (e.g., 328,350,233)
    """
    client = get_client(ctx)

    try:
        player_ids = [int(p.strip()) for p in players.split(",")]
    except ValueError:
        click.echo("Error: Invalid player IDs. Use comma-separated numbers.", err=True)
        ctx.exit(1)

    if len(player_ids) > 15:
        click.echo("Error: Maximum 15 players allowed", err=True)
        ctx.exit(1)

    try:
        data = client.compare(player_ids, gameweek)
    except httpx.HTTPStatusError as e:
        click.echo(f"Error: {e.response.status_code} - {e.response.text}", err=True)
        ctx.exit(1)

    predictions = data.get("predictions", [])
    missing = data.get("missing", [])

    click.echo(f"Player Comparison - Gameweek {gameweek}")
    click.echo()

    if predictions:
        table_data = [
            [
                p.get("player_name", f"ID: {p['player_id']}"),
                p.get("position", "-"),
                format_points(p["predicted_points"]),
            ]
            for p in predictions
        ]
        click.echo(tabulate(table_data, headers=["Player", "Pos", "Points"]))

    if missing:
        click.echo()
        click.echo(f"Not found: {', '.join(map(str, missing))}")


@cli.command()
@click.option(
    "--gameweek",
    "-g",
    required=True,
    type=int,
    help="Gameweek number",
)
@click.option(
    "--limit",
    "-n",
    default=100,
    type=int,
    help="Number of results (default: 100)",
)
@click.pass_context
def predictions(ctx, gameweek: int, limit: int):
    """List all predictions for a gameweek."""
    client = get_client(ctx)

    try:
        data = client.get_predictions(gameweek, limit)
    except httpx.HTTPStatusError as e:
        click.echo(f"Error: {e.response.status_code} - {e.response.text}", err=True)
        ctx.exit(1)

    preds = data.get("predictions", [])
    if not preds:
        click.echo(f"No predictions found for gameweek {gameweek}")
        return

    click.echo(f"Predictions - Gameweek {gameweek} ({len(preds)} players)")
    click.echo()

    table_data = [
        [
            p["player_id"],
            p.get("player_name", "-"),
            p.get("position", "-"),
            format_points(p["predicted_points"]),
        ]
        for p in preds
    ]
    click.echo(tabulate(table_data, headers=["ID", "Player", "Pos", "Points"]))


@cli.command()
@click.option(
    "--state-machine",
    envvar="STATE_MACHINE_ARN",
    help="Step Functions state machine ARN (or set STATE_MACHINE_ARN env var)",
)
@click.option(
    "--region",
    default="ap-southeast-2",
    help="AWS region (default: ap-southeast-2)",
)
def run(state_machine: str | None, region: str):
    """Trigger the FPL prediction pipeline."""
    if not state_machine:
        click.echo(
            "Error: --state-machine or STATE_MACHINE_ARN env var required", err=True
        )
        sys.exit(1)

    client = boto3.client("stepfunctions", region_name=region)

    try:
        response = client.start_execution(
            stateMachineArn=state_machine,
            input=json.dumps({"fetch_player_details": True}),
        )
        execution_arn = response["executionArn"]
        click.echo("Pipeline started successfully!")
        click.echo()
        click.echo(f"Execution ARN: {execution_arn}")
        click.echo()
        click.echo("Check status with:")
        click.echo(
            f"  aws stepfunctions describe-execution --execution-arn {execution_arn}"
        )
    except client.exceptions.StateMachineDoesNotExist:
        click.echo(f"Error: State machine not found: {state_machine}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error starting pipeline: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--data-dir",
    type=str,
    default="data/",
    help="Directory containing training Parquet files (default: data/)",
)
@click.option(
    "--output-path",
    type=str,
    default="models/",
    help="Output path for the trained model (default: models/)",
)
@click.option(
    "--upload-s3",
    is_flag=True,
    help="Upload trained model to S3",
)
@click.option(
    "--bucket",
    type=str,
    default="fpl-ml-data-dev",
    help="S3 bucket for model upload (default: fpl-ml-data-dev)",
)
@click.option(
    "--temporal-split/--random-split",
    default=True,
    help="Use temporal or random train/test split (default: temporal)",
)
@click.option(
    "--tune",
    is_flag=True,
    help="Run Optuna hyperparameter tuning before training",
)
@click.option(
    "--n-trials",
    type=int,
    default=50,
    help="Number of Optuna tuning trials (default: 50)",
)
def train(
    data_dir: str,
    output_path: str,
    upload_s3: bool,
    bucket: str,
    temporal_split: bool,
    tune: bool,
    n_trials: int,
):
    """Train XGBoost model locally and optionally upload to S3."""
    cmd = [
        sys.executable,
        "sagemaker/train_local.py",
        "--data-dir",
        data_dir,
        "--output-path",
        output_path,
    ]
    if temporal_split:
        cmd.append("--temporal-split")
    else:
        cmd.append("--random-split")
    if tune:
        cmd.extend(["--tune", "--n-trials", str(n_trials)])
    if upload_s3:
        cmd.extend(["--upload-s3", "--bucket", bucket])

    click.echo(f"Training model with data from {data_dir}...")
    result = subprocess.run(cmd, capture_output=False)
    sys.exit(result.returncode)


@cli.command("import-historical")
@click.option(
    "--seasons",
    type=str,
    required=True,
    help="Comma-separated seasons (e.g. 2021-22,2022-23,2023-24)",
)
@click.option(
    "--output-dir",
    type=str,
    default="data/historical/",
    help="Output directory (default: data/historical/)",
)
@click.option(
    "--upload-s3",
    is_flag=True,
    help="Upload output files to S3",
)
@click.option(
    "--bucket",
    type=str,
    default="fpl-ml-data-dev",
    help="S3 bucket name (default: fpl-ml-data-dev)",
)
def import_historical(seasons: str, output_dir: str, upload_s3: bool, bucket: str):
    """Import historical FPL data from vaastav/Fantasy-Premier-League."""
    cmd = [
        sys.executable,
        "scripts/import_historical.py",
        "--seasons",
        seasons,
        "--output-dir",
        output_dir,
    ]
    if upload_s3:
        cmd.extend(["--upload-s3", "--bucket", bucket])

    click.echo(f"Importing historical data for seasons: {seasons}...")
    result = subprocess.run(cmd, capture_output=False)
    sys.exit(result.returncode)


@cli.command()
@click.option(
    "--output-dir",
    type=str,
    default="data/current/",
    help="Output directory (default: data/current/)",
)
@click.option(
    "--start-gw",
    type=int,
    default=4,
    help="First gameweek to generate training data for (default: 4)",
)
@click.option(
    "--upload-s3",
    is_flag=True,
    help="Upload output files to S3",
)
@click.option(
    "--bucket",
    type=str,
    default="fpl-ml-data-dev",
    help="S3 bucket name (default: fpl-ml-data-dev)",
)
def backfill(output_dir: str, start_gw: int, upload_s3: bool, bucket: str):
    """Backfill current season data using the FPL API."""
    cmd = [
        sys.executable,
        "scripts/backfill_current_season.py",
        "--output-dir",
        output_dir,
        "--start-gw",
        str(start_gw),
    ]
    if upload_s3:
        cmd.extend(["--upload-s3", "--bucket", bucket])

    click.echo(f"Backfilling current season data from GW{start_gw}...")
    result = subprocess.run(cmd, capture_output=False)
    sys.exit(result.returncode)


def main():
    """Entry point for CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
