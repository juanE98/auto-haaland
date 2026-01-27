# Auto-Haaland: FPL ML Prediction System

![CI](https://github.com/juanE98/auto-haaland/actions/workflows/ci.yml/badge.svg)
![Deploy](https://github.com/juanE98/auto-haaland/actions/workflows/deploy.yml/badge.svg)

![Python](https://img.shields.io/badge/python-3.12-%233776AB.svg?style=for-the-badge&logo=python&logoColor=white)
![AWS Lambda](https://img.shields.io/badge/aws--lambda-%23FF9900.svg?style=for-the-badge&logo=aws-lambda&logoColor=white)
![AWS S3](https://img.shields.io/badge/aws--s3-%23569A31.svg?style=for-the-badge&logo=amazon-s3&logoColor=white)
![AWS DynamoDB](https://img.shields.io/badge/aws--dynamodb-%234053D6.svg?style=for-the-badge&logo=amazon-dynamodb&logoColor=white)
![AWS SageMaker](https://img.shields.io/badge/aws--sagemaker-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)
![XGBoost](https://img.shields.io/badge/xgboost-2.0.3-%23337AB7.svg?style=for-the-badge&logo=xgboost&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-8.0.0-%230A9EDC.svg?style=for-the-badge&logo=pytest&logoColor=white)
![LocalStack](https://img.shields.io/badge/localstack-%234D4D4D.svg?style=for-the-badge&logo=localstack&logoColor=white)

An automated Fantasy Premier League prediction system using XGBoost and AWS.

## Purpose
 - Play around with AWS and its ML services

## Features

- Fetches live FPL data via API with rate limiting
- Lambda-based data pipeline for processing player statistics
- XGBoost model training on SageMaker
- Batch predictions stored in DynamoDB
- REST API for retrieving predictions
- Full local development environment with LocalStack

## Quick Start

### Setup

```bash
# Automated setup (requires sudo)
./setup.sh

# Or manual setup
sudo apt install python3.12-venv docker-compose
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

### Commands

Run `make help` for all available commands. Key ones below:

```bash
# Setup
make setup                      # Install dev dependencies
make install                    # Install production dependencies only

# Local Infrastructure
make local-up                   # Start LocalStack (S3, DynamoDB)
make local-down                 # Stop LocalStack
make local-logs                 # View LocalStack logs
make local-api                  # Start local API with SAM CLI

# Testing
make test                       # Run all tests (unit + integration)
make test-unit                  # Unit tests only (fast, no dependencies)
make test-integration           # Integration tests (starts/stops LocalStack)

# Data & Training
make import-historical          # Import historical data from GitHub
make backfill                   # Backfill current season from FPL API
make backfill END_GW=22         # Backfill up to GW22 only
make train-local                # Train XGBoost model locally
make train-and-upload           # Train locally and upload model to S3

# Pipeline
make run-pipeline               # Trigger pipeline (auto-detect gameweek)
make run-pipeline GW=23         # Trigger pipeline for specific gameweek

# CLI Queries
make top GW=22                  # Top 10 predicted scorers for gameweek
make top GW=22 LIMIT=20         # Top N predicted scorers for gameweek
make player ID=328 GW=22        # Predictions for a specific player
make compare IDS=328,350 GW=22  # Compare players for a gameweek
make predictions GW=22          # All predictions for a gameweek

# Code Quality
make lint                       # Run lint checks (black, isort, flake8)
make format                     # Auto-format code (black, isort)

# AWS Deployment
make build                      # Build SAM application
make deploy-dev                 # Deploy to dev environment

# Cleanup
make clean                      # Remove build artifacts and caches
```

## Architecture

![AWS Architecture Diagram](./aws-architecture-diagram.png)

See [fpl-ml-aws-architecture.md](./fpl-ml-aws-architecture.md) for details.

## Project Structure

```
auto-haaland/
├── lambdas/
│   ├── common/              # Shared: FPL API client, AWS utilities, feature config
│   ├── data_fetcher/        # Fetches bootstrap, fixtures, player histories to S3
│   ├── feature_processor/   # Feature engineering from raw data to Parquet
│   ├── inference/           # Loads XGBoost model, runs predictions
│   ├── prediction_loader/   # Batch writes predictions to DynamoDB
│   └── api_handler/         # REST API (top, player, compare, predictions)
├── sagemaker/
│   ├── train.py             # SageMaker training script
│   └── train_local.py       # Local training with optional S3 upload
├── scripts/
│   ├── import_historical.py # Import historical data from vaastav GitHub repo
│   └── backfill_current_season.py  # Backfill current season via FPL API
├── cli/
│   └── fpl.py               # CLI tool for queries, pipeline, training
├── statemachine/
│   └── pipeline.asl.json    # Step Functions state machine definition
├── tests/
│   ├── unit/                # Unit tests (moto, httpx mock)
│   └── integration/         # Integration tests (LocalStack)
├── template.yaml            # SAM template (Lambdas, DynamoDB, S3, API GW, Step Functions)
├── docker-compose.yml       # LocalStack configuration
├── Makefile                 # Development commands
└── setup.sh                 # Automated setup
```

## Data Storage

```
s3://fpl-ml-data/raw/season_YYYY_YY/
├── gwN_bootstrap.json       # Players, teams, gameweeks
├── gwN_fixtures.json        # Fixtures for gameweek N
└── gwN_players/
    └── player_*.json        # Individual player histories
```

## Technology Stack

- Python 3.12
- AWS: Lambda, S3, DynamoDB, SageMaker, Step Functions
- XGBoost for predictions
- LocalStack for local AWS emulation
- pytest + moto for testing

## Testing

- 60% Unit Tests (fast, no dependencies)
- 30% Integration Tests (LocalStack/moto)
- 10% E2E Tests (real AWS, minimal)


## Cost Estimate

- **Development**: ~$3-6
- **Production**: (~$0.60/gameweek)
