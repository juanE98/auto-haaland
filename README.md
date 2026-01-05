# Auto-Haaland: FPL ML Prediction System

An automated Fantasy Premier League prediction system using XGBoost and AWS.

## Quick Start

### 1. Run Setup Script

The easiest way to get started:

```bash
# Run the automated setup script (requires sudo for system packages)
./setup.sh
```

This script will:
- Install system packages (python3-venv, docker-compose)
- Create a virtual environment
- Install all Python dependencies

**Alternative manual setup:**

```bash
# Install system package
sudo apt install python3.12-venv docker-compose

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt
```

### 2. Start Local Infrastructure

```bash
# Start LocalStack (S3, DynamoDB, Step Functions, etc.)
make local-up
```

This starts:
- **LocalStack** at `http://localhost:4566` (AWS services emulation)
- **DynamoDB Admin** at `http://localhost:8001` (GUI for DynamoDB)

### 3. Run Tests

```bash
# Run all tests
make test

# Or run specific test suites
make test-unit          # Unit tests only (fast)
make test-integration   # Integration tests (requires LocalStack)
```

### 4. Stop Local Infrastructure

```bash
make local-down
```

## Available Commands

Run `make help` to see all available commands:

```bash
make help
```

## Project Structure

```
auto-haaland/
├── setup.sh                 # Automated setup script
├── docker-compose.yml       # LocalStack setup
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Dev/test dependencies
├── pytest.ini              # Test configuration
├── Makefile                # Common commands
├── lambdas/
│   ├── common/             # Shared utilities
│   │   ├── fpl_api.py      # FPL API client
│   │   └── aws_clients.py  # AWS client factories
│   └── data_fetcher/       # Data fetcher Lambda
│       └── handler.py
├── tests/
│   ├── conftest.py         # Shared test fixtures
│   ├── unit/               # Unit tests (25+)
│   └── integration/        # Integration tests (5+)
├── scripts/
│   └── test_real_fpl_api.py  # Real API test
└── README.md
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
    ↓
CLI Tool (Local)
```

## Development Workflow

1. **Phase 1:** Build data fetcher and feature engineering locally
2. **Phase 2:** Train XGBoost model on your laptop
3. **Phase 3:** Test with LocalStack (S3 + DynamoDB)
4. **Phase 4:** Deploy to AWS dev environment
5. **Phase 5:** Production deployment

## Cost Estimate

- **Development:** ~$3-6 (mostly local testing, 1x SageMaker validation)
- **Production:** ~$40-60/year (~$0.60 per gameweek)

## Documentation

- [Implementation Plan](./IMPLEMENTATION_PLAN.md) - Detailed development plan
- [Architecture](./fpl-ml-aws-architecture.md) - System architecture details

## Testing Philosophy

- **60% Unit Tests** - Fast, no dependencies
- **30% Integration Tests** - LocalStack/moto
- **10% E2E Tests** - Real AWS (expensive, use sparingly)

## Next Steps

See the [Implementation Plan](./IMPLEMENTATION_PLAN.md) for the detailed development roadmap.

Phase 1 starts with building the data fetcher to pull real FPL data!
