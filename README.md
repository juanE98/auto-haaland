# Auto-Haaland: FPL ML Prediction System

An automated Fantasy Premier League prediction system using XGBoost and AWS. Fetches live FPL data, engineers features, trains ML models, and delivers predictions via API.

## Features

- **FPL API Integration**: Rate-limited client with exponential backoff
- **Data Pipeline**: Lambda functions for fetching and processing player data
- **ML Training**: XGBoost models on SageMaker with hyperparameter tuning
- **Predictions**: Daily batch predictions stored in DynamoDB
- **API**: REST endpoints for retrieving predictions
- **Local Development**: LocalStack for AWS service emulation

## Quick Start

### 1. Environment Setup

Run the automated setup script:

```bash
./setup.sh
```

This installs system packages (python3-venv, docker-compose), creates a virtual environment, and installs dependencies.

**Manual setup:**
```bash
sudo apt install python3.12-venv docker-compose
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

### 2. Testing

```bash
# Test with real FPL API
source venv/bin/activate
python scripts/test_real_fpl_api.py

# Run unit tests (fast)
make test-unit

# Run integration tests (starts/stops LocalStack automatically)
make test-integration

# Run all tests
make test
```

### 3. Local Development

```bash
# Start LocalStack (AWS emulation)
make local-up

# Stop LocalStack
make local-down

# View logs
make local-logs
```

LocalStack provides:
- S3 at `http://localhost:4566`
- DynamoDB Admin UI at `http://localhost:8001`

## Project Structure

```
auto-haaland/
├── lambdas/
│   ├── common/
│   │   ├── fpl_api.py       # FPL API client with rate limiting
│   │   └── aws_clients.py   # AWS client factories
│   └── data_fetcher/
│       └── handler.py        # Fetch bootstrap, fixtures, player data
├── tests/
│   ├── unit/                 # 18 unit tests
│   └── integration/          # 5 integration tests with mocked S3
├── scripts/
│   └── test_real_fpl_api.py  # Real API verification
├── docker-compose.yml        # LocalStack configuration
├── Makefile                  # Development commands
└── setup.sh                  # Automated environment setup
```

## Architecture

```
EventBridge (6pm AEST Brisbane)
    ↓
Step Functions (Orchestration)
    ↓
Lambda: Fetch FPL Data → Lambda: Feature Engineering
    ↓
SageMaker: Train Model → Batch Predictions
    ↓
DynamoDB (Predictions Store)
    ↓
API Gateway + Lambda (REST API)
```

See [fpl-ml-aws-architecture.md](./fpl-ml-aws-architecture.md) for detailed architecture.

## Data Storage

Raw JSON stored in S3:
```
s3://fpl-ml-data/raw/season_2025_26/
├── gw20_bootstrap.json       # All players, teams, gameweeks
├── gw20_fixtures.json        # Fixtures for GW20
└── gw20_players/
    ├── player_381.json       # Salah's history
    └── player_*.json         # Other players
```

## Available Commands

```bash
make help                # Show all commands
make setup              # Install dependencies
make test               # Run all tests
make test-unit          # Run unit tests
make test-integration   # Run integration tests (with LocalStack)
make local-up           # Start LocalStack
make local-down         # Stop LocalStack
make local-logs         # View LocalStack logs
make clean              # Remove build artifacts
```

## Technology Stack

**Backend:**
- Python 3.12
- AWS Lambda, S3, DynamoDB, SageMaker, Step Functions
- XGBoost for ML predictions

**Development:**
- LocalStack for AWS emulation
- pytest with moto for S3 mocking
- Docker Compose for local infrastructure

**Dependencies:**
- boto3, httpx, pandas, pyarrow, numpy
- xgboost, scikit-learn
- pytest, black, flake8, mypy

## Testing Philosophy

- **60% Unit Tests**: Fast, no external dependencies
- **30% Integration Tests**: LocalStack/moto for AWS services
- **10% E2E Tests**: Real AWS (minimal, cost-effective)

**Current coverage:** 23/23 tests passing

## Cost Estimate

- **Development**: ~$3-6 (local testing + SageMaker validation)
- **Production**: ~$40-60/year (~$0.60 per gameweek)

## Documentation

- [Implementation Plan](./implementation-plan.md) - Development roadmap and testing strategy
- [Architecture](./fpl-ml-aws-architecture.md) - AWS infrastructure design

## License

MIT
