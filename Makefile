.PHONY: setup install test test-unit test-integration local-up local-down local-api local-logs clean help import-historical backfill train-and-upload

# Default target
help:
	@echo "Auto-Haaland FPL ML System - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Install development dependencies"
	@echo "  make install        - Install production dependencies only"
	@echo ""
	@echo "Local Infrastructure:"
	@echo "  make local-up       - Start LocalStack (S3, DynamoDB, etc.)"
	@echo "  make local-down     - Stop LocalStack"
	@echo "  make local-logs     - Show LocalStack logs"
	@echo "  make local-api      - Start local API with SAM CLI"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run all tests"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-integration - Run integration tests (requires LocalStack)"
	@echo ""
	@echo "Data & Training:"
	@echo "  make import-historical - Import historical data from GitHub"
	@echo "  make backfill          - Backfill current season from FPL API"
	@echo "  make train-local       - Train XGBoost model locally"
	@echo "  make train-and-upload  - Train locally and upload model to S3"
	@echo ""
	@echo "AWS Deployment:"
	@echo "  make build          - Build SAM application"
	@echo "  make deploy-dev     - Deploy to dev environment"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove build artifacts and caches"

setup:
	@echo "Installing development dependencies..."
	venv/bin/pip install -r requirements-dev.txt

install:
	@echo "Installing production dependencies..."
	venv/bin/pip install -r requirements.txt

test: test-unit test-integration

test-unit:
	@echo "Running unit tests..."
	venv/bin/pytest tests/unit -v --cov=lambdas --cov-report=term-missing

test-integration: local-up
	@echo "Running integration tests..."
	sleep 5  # Wait for LocalStack to be ready
	venv/bin/pytest tests/integration -v
	$(MAKE) local-down

local-up:
	@echo "Starting LocalStack..."
	docker-compose up -d
	@echo "Waiting for LocalStack to be ready..."
	@sleep 10
	@echo "LocalStack is ready at http://localhost:4566"
	@echo "DynamoDB Admin is ready at http://localhost:8001"

local-down:
	@echo "Stopping LocalStack..."
	docker-compose down

local-logs:
	docker-compose logs -f localstack

local-api:
	@echo "Starting local API Gateway..."
	sam local start-api --host 0.0.0.0 --port 3000

import-historical:
	@echo "Importing historical data..."
	venv/bin/python scripts/import_historical.py --seasons 2021-22,2022-23,2023-24 --output-dir data/historical/

backfill:
	@echo "Backfilling current season data..."
	venv/bin/python scripts/backfill_current_season.py --output-dir data/current/

train-local:
	@echo "Training model locally..."
	venv/bin/python sagemaker/train_local.py --data-dir data/ --output-path models/

train-and-upload:
	@echo "Training model and uploading to S3..."
	venv/bin/python sagemaker/train_local.py --data-dir data/ --output-path models/ --upload-s3 --bucket fpl-ml-data-dev

build:
	@echo "Building SAM application..."
	sam build

deploy-dev:
	@echo "Deploying to dev environment..."
	sam deploy --config-env dev

clean:
	@echo "Cleaning build artifacts..."
	rm -rf .aws-sam
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete!"
