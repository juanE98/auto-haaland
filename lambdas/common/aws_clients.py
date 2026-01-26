"""
AWS client factory utilities.

Provides boto3 clients with proper configuration for both:
- Local development (LocalStack)
- Production (Real AWS)
"""

import logging
import os
from typing import Optional

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)


def get_s3_client(endpoint_url: Optional[str] = None):
    """
    Get S3 client configured for local or production use.

    Args:
        endpoint_url: Override endpoint (for LocalStack). If None, uses env var or defaults to AWS.

    Returns:
        boto3 S3 client
    """
    # Check for LocalStack endpoint
    if endpoint_url is None:
        endpoint_url = os.getenv("AWS_ENDPOINT_URL")
        if endpoint_url:
            logger.info(f"Using LocalStack S3 at {endpoint_url}")

    config = Config(
        region_name=os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2"),
        signature_version="s3v4",
        retries={"max_attempts": 3, "mode": "adaptive"},
    )

    return boto3.client("s3", endpoint_url=endpoint_url, config=config)


def get_dynamodb_resource(endpoint_url: Optional[str] = None):
    """
    Get DynamoDB resource configured for local or production use.

    Args:
        endpoint_url: Override endpoint (for LocalStack). If None, uses env var or defaults to AWS.

    Returns:
        boto3 DynamoDB resource
    """
    # Check for LocalStack endpoint
    if endpoint_url is None:
        endpoint_url = os.getenv("AWS_ENDPOINT_URL")
        if endpoint_url:
            logger.info(f"Using LocalStack DynamoDB at {endpoint_url}")

    config = Config(
        region_name=os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2"),
        retries={"max_attempts": 3, "mode": "adaptive"},
    )

    return boto3.resource("dynamodb", endpoint_url=endpoint_url, config=config)


def get_stepfunctions_client(endpoint_url: Optional[str] = None):
    """
    Get Step Functions client configured for local or production use.

    Args:
        endpoint_url: Override endpoint (for LocalStack). If None, uses env var or defaults to AWS.

    Returns:
        boto3 Step Functions client
    """
    if endpoint_url is None:
        endpoint_url = os.getenv("AWS_ENDPOINT_URL")
        if endpoint_url:
            logger.info(f"Using LocalStack Step Functions at {endpoint_url}")

    config = Config(
        region_name=os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2"),
        retries={"max_attempts": 3, "mode": "adaptive"},
    )

    return boto3.client("stepfunctions", endpoint_url=endpoint_url, config=config)
