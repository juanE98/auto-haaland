"""
Integration test fixtures for LocalStack.

These fixtures handle the case where AWS_ENDPOINT_URL is set (LocalStack)
vs when it's not (moto mocks).
"""

import os

import boto3
import pytest
from botocore.exceptions import ClientError


def is_localstack():
    """Check if tests are running against LocalStack."""
    return os.environ.get("AWS_ENDPOINT_URL") is not None


@pytest.fixture(scope="session")
def localstack_s3_client():
    """
    Create an S3 client for LocalStack.
    Session-scoped to reuse across all integration tests.
    """
    if not is_localstack():
        pytest.skip("LocalStack not available (AWS_ENDPOINT_URL not set)")

    client = boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
        region_name="ap-southeast-2",
    )
    return client


@pytest.fixture(scope="session")
def localstack_dynamodb_resource():
    """
    Create a DynamoDB resource for LocalStack.
    Session-scoped to reuse across all integration tests.
    """
    if not is_localstack():
        pytest.skip("LocalStack not available (AWS_ENDPOINT_URL not set)")

    resource = boto3.resource(
        "dynamodb",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
        region_name="ap-southeast-2",
    )
    return resource


@pytest.fixture(scope="session")
def s3_bucket(localstack_s3_client):
    """
    Ensure the fpl-ml-data bucket exists in LocalStack.
    Creates it if it doesn't exist, handles already-exists gracefully.
    """
    bucket_name = "fpl-ml-data"

    try:
        localstack_s3_client.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={"LocationConstraint": "ap-southeast-2"},
        )
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            pass  # Bucket exists, continue
        else:
            raise

    return bucket_name


@pytest.fixture(scope="session")
def dynamodb_predictions_table(localstack_dynamodb_resource):
    """
    Ensure the fpl-predictions table exists in LocalStack.
    Creates it if it doesn't exist, handles already-exists gracefully.
    """
    table_name = "fpl-predictions"

    try:
        table = localstack_dynamodb_resource.create_table(
            TableName=table_name,
            KeySchema=[
                {"AttributeName": "player_id", "KeyType": "HASH"},
                {"AttributeName": "gameweek", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "player_id", "AttributeType": "N"},
                {"AttributeName": "gameweek", "AttributeType": "N"},
                {"AttributeName": "predicted_points", "AttributeType": "N"},
                {"AttributeName": "position", "AttributeType": "S"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "gameweek-points-index",
                    "KeySchema": [
                        {"AttributeName": "gameweek", "KeyType": "HASH"},
                        {"AttributeName": "predicted_points", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                },
                {
                    "IndexName": "position-points-index",
                    "KeySchema": [
                        {"AttributeName": "position", "KeyType": "HASH"},
                        {"AttributeName": "predicted_points", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                },
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        table.wait_until_exists()
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            table = localstack_dynamodb_resource.Table(table_name)
        else:
            raise

    return table


@pytest.fixture
def clean_s3_bucket(localstack_s3_client, s3_bucket):
    """
    Provide a clean S3 bucket by deleting all objects before each test.
    Use this fixture when tests need isolation.
    """
    # Clean up before test
    paginator = localstack_s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=s3_bucket):
        if "Contents" in page:
            objects = [{"Key": obj["Key"]} for obj in page["Contents"]]
            localstack_s3_client.delete_objects(
                Bucket=s3_bucket, Delete={"Objects": objects}
            )

    yield s3_bucket

    # Optionally clean up after test too (commented out to allow inspection)
    # for page in paginator.paginate(Bucket=s3_bucket):
    #     if "Contents" in page:
    #         objects = [{"Key": obj["Key"]} for obj in page["Contents"]]
    #         localstack_s3_client.delete_objects(
    #             Bucket=s3_bucket, Delete={"Objects": objects}
    #         )


@pytest.fixture
def clean_dynamodb_table(dynamodb_predictions_table):
    """
    Provide a clean DynamoDB table by deleting all items before each test.
    Use this fixture when tests need isolation.
    """
    # Clean up before test - scan and delete all items
    response = dynamodb_predictions_table.scan()
    with dynamodb_predictions_table.batch_writer() as writer:
        for item in response.get("Items", []):
            writer.delete_item(
                Key={
                    "player_id": item["player_id"],
                    "gameweek": item["gameweek"],
                }
            )

    # Handle pagination
    while "LastEvaluatedKey" in response:
        response = dynamodb_predictions_table.scan(
            ExclusiveStartKey=response["LastEvaluatedKey"]
        )
        with dynamodb_predictions_table.batch_writer() as writer:
            for item in response.get("Items", []):
                writer.delete_item(
                    Key={
                        "player_id": item["player_id"],
                        "gameweek": item["gameweek"],
                    }
                )

    yield dynamodb_predictions_table
