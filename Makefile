.PHONY: setup install test test-unit test-integration local-up local-down local-api local-logs clean help import-historical backfill train-local train-and-upload top haul player compare predictions run-pipeline lint format

# Helper for comma in $(if ...) expansions
comma := ,

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
	@echo "CLI Queries (GW is required):"
	@echo "  make top GW=22                 - Top 10 predicted scorers for gameweek"
	@echo "  make top GW=22 LIMIT=20        - Top N predicted scorers for gameweek"
	@echo "  make player ID=328 GW=22       - Predictions for a specific player"
	@echo "  make compare IDS=328,350 GW=22 - Compare players for a gameweek"
	@echo "  make predictions GW=22         - All predictions for a gameweek"
	@echo ""
	@echo "Pipeline:"
	@echo "  make run-pipeline              - Trigger pipeline (auto-detect gameweek)"
	@echo "  make run-pipeline GW=23        - Trigger pipeline for specific gameweek"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           - Run lint checks (black, isort, flake8)"
	@echo "  make format         - Auto-format code (black, isort)"
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
	venv/bin/python scripts/backfill_current_season.py --output-dir data/current/ $(if $(END_GW),--end-gw $(END_GW),)

train-local:
	@echo "Training models locally..."
	venv/bin/python sagemaker/train_local.py --data-dir data/ --output-path models/ --train-haul-classifier

train-and-upload:
	@echo "Training models and uploading to S3..."
	venv/bin/python sagemaker/train_local.py --data-dir data/ --output-path models/ --upload-s3 --bucket fpl-ml-data-dev --train-haul-classifier

API_ENDPOINT ?= $(shell aws ssm get-parameter --name /auto-haaland/dev/api-endpoint --query Parameter.Value --output text --region ap-southeast-2 2>/dev/null)
STATE_MACHINE_ARN ?= $(shell aws ssm get-parameter --name /auto-haaland/dev/state-machine-arn --query Parameter.Value --output text --region ap-southeast-2 2>/dev/null)
top:
ifndef GW
	$(error GW is required. Usage: make top GW=22)
endif
	venv/bin/python -m cli.fpl --endpoint $(API_ENDPOINT) top -g $(GW) $(if $(LIMIT),-n $(LIMIT),) $(if $(SORT),-s $(SORT),)

haul:
ifndef GW
	$(error GW is required. Usage: make haul GW=24)
endif
	venv/bin/python -m cli.fpl --endpoint $(API_ENDPOINT) top -g $(GW) $(if $(LIMIT),-n $(LIMIT),) -s haul

player:
ifndef ID
	$(error ID is required. Usage: make player ID=328 GW=22)
endif
ifndef GW
	$(error GW is required. Usage: make player ID=328 GW=22)
endif
	venv/bin/python -m cli.fpl --endpoint $(API_ENDPOINT) player $(ID) -g $(GW)

compare:
ifndef IDS
	$(error IDS is required. Usage: make compare IDS=328,350 GW=22)
endif
ifndef GW
	$(error GW is required. Usage: make compare IDS=328,350 GW=22)
endif
	venv/bin/python -m cli.fpl --endpoint $(API_ENDPOINT) compare $(IDS) -g $(GW)

predictions:
ifndef GW
	$(error GW is required. Usage: make predictions GW=22)
endif
	venv/bin/python -m cli.fpl --endpoint $(API_ENDPOINT) predictions -g $(GW)

run-pipeline:
	@echo "Triggering FPL prediction pipeline$(if $(GW), for GW$(GW),)..."
	@aws stepfunctions start-execution \
		--state-machine-arn $(STATE_MACHINE_ARN) \
		--input '{"fetch_player_details": true$(if $(GW), $(comma) "gameweek": $(GW),)}' \
		--region ap-southeast-2 \
		--no-cli-pager \
		--query 'executionArn' \
		--output text > /dev/null && echo "Pipeline started successfully." || echo "Failed to start pipeline."

lint:
	@echo "Running lint checks..."
	venv/bin/black --check lambdas cli tests sagemaker scripts
	venv/bin/isort --check-only lambdas cli tests scripts
	venv/bin/flake8 lambdas cli scripts --max-line-length=120 --ignore=E501,W503,E203
	@echo "All lint checks passed!"

format:
	@echo "Formatting code..."
	venv/bin/black lambdas cli tests sagemaker scripts
	venv/bin/isort lambdas cli tests scripts
	@echo "Formatting complete!"

build:
	@echo "Building SAM application..."
	sam build

deploy-dev:
	@echo "Deploying to dev environment..."
	sam deploy --config-env dev --resolve-image-repos

clean:
	@echo "Cleaning build artifacts..."
	rm -rf .aws-sam
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete!"
