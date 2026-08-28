"""Archive the prediction table to S3 and optionally clear it for a new season."""

import argparse
import gzip
import json
import logging
import os
from datetime import datetime, timezone

import boto3
from boto3.dynamodb.types import TypeSerializer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def archive_predictions(table, s3_client, bucket: str, key: str) -> list[dict]:
    """Scan the table, upload a lossless gzip archive, and verify it exists."""
    response = table.scan()
    items = list(response.get("Items", []))
    while response.get("LastEvaluatedKey"):
        response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
        items.extend(response.get("Items", []))

    serializer = TypeSerializer()
    payload = {
        "table": table.name,
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "item_count": len(items),
        "items": [
            {name: serializer.serialize(value) for name, value in item.items()}
            for item in items
        ],
    }
    body = gzip.compress(json.dumps(payload).encode("utf-8"))
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
        ContentEncoding="gzip",
    )
    uploaded = s3_client.head_object(Bucket=bucket, Key=key)
    if uploaded["ContentLength"] != len(body):
        raise RuntimeError("Prediction archive upload size verification failed")
    return items


def clear_predictions(table, archived_items: list[dict]) -> int:
    """Delete only the primary keys captured in the verified archive."""
    with table.batch_writer() as writer:
        for item in archived_items:
            writer.delete_item(
                Key={
                    "player_id": item["player_id"],
                    "gameweek": item["gameweek"],
                }
            )
    return len(archived_items)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--season", required=True, help="Season being archived")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument(
        "--confirm-table",
        help="Required with --clear; must exactly match --table",
    )
    args = parser.parse_args()

    if args.clear and args.confirm_table != args.table:
        parser.error("--clear requires --confirm-table to exactly match --table")

    region = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2")
    dynamodb = boto3.resource("dynamodb", region_name=region)
    s3_client = boto3.client("s3", region_name=region)
    table = dynamodb.Table(args.table)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    key = f"archives/dynamodb/season_{args.season}/predictions_{timestamp}.json.gz"

    items = archive_predictions(table, s3_client, args.bucket, key)
    logger.info("Archived %s items to s3://%s/%s", len(items), args.bucket, key)

    if args.clear:
        deleted = clear_predictions(table, items)
        logger.info("Deleted %s archived items from %s", deleted, args.table)
    else:
        logger.info("Archive only; table was not changed")


if __name__ == "__main__":
    main()
