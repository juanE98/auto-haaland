"""Tests for the season prediction reset utility."""

import gzip
import json
from decimal import Decimal
from unittest.mock import MagicMock

from scripts.reset_predictions import archive_predictions, clear_predictions


def test_archive_predictions_paginates_and_verifies_upload():
    table = MagicMock(name="fpl-predictions-dev")
    table.name = "fpl-predictions-dev"
    table.scan.side_effect = [
        {
            "Items": [{"player_id": 1, "gameweek": 38, "score": Decimal("5.5")}],
            "LastEvaluatedKey": {"player_id": 1, "gameweek": 38},
        },
        {"Items": [{"player_id": 2, "gameweek": 38}]},
    ]
    s3 = MagicMock()
    s3.head_object.return_value = {"ContentLength": 0}

    def report_uploaded_size(**_kwargs):
        return {"ContentLength": len(s3.put_object.call_args.kwargs["Body"])}

    s3.head_object.side_effect = report_uploaded_size

    items = archive_predictions(table, s3, "archive-bucket", "archive.json.gz")

    assert len(items) == 2
    uploaded = s3.put_object.call_args.kwargs["Body"]
    payload = json.loads(gzip.decompress(uploaded))
    assert payload["item_count"] == 2
    assert payload["items"][0]["score"] == {"N": "5.5"}
    s3.head_object.assert_called_once_with(
        Bucket="archive-bucket", Key="archive.json.gz"
    )


def test_archive_predictions_rejects_incomplete_upload():
    table = MagicMock(name="fpl-predictions-dev")
    table.name = "fpl-predictions-dev"
    table.scan.return_value = {"Items": [{"player_id": 1, "gameweek": 38}]}
    s3 = MagicMock()
    s3.head_object.return_value = {"ContentLength": 1}

    try:
        archive_predictions(table, s3, "archive-bucket", "archive.json.gz")
    except RuntimeError as error:
        assert "size verification failed" in str(error)
    else:
        raise AssertionError("Expected an incomplete archive upload to fail")


def test_clear_predictions_deletes_only_archived_keys():
    table = MagicMock()
    writer = MagicMock()
    table.batch_writer.return_value.__enter__.return_value = writer
    items = [
        {"player_id": 1, "gameweek": 38, "other": "ignored"},
        {"player_id": 2, "gameweek": 1},
    ]

    deleted = clear_predictions(table, items)

    assert deleted == 2
    assert writer.delete_item.call_count == 2
    writer.delete_item.assert_any_call(Key={"player_id": 1, "gameweek": 38})
    writer.delete_item.assert_any_call(Key={"player_id": 2, "gameweek": 1})
