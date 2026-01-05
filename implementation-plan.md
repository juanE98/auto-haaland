# Auto-Haaland: Implementation Plan

## Summary

**Project:** FPL Player Prediction System
**Stack:** Python 3.11 + AWS SAM + XGBoost
**Cost:** ~$40-60/year (with Free Tier)
**Development Cost:** ~$3-6 (mostly local testing)

---

## Architecture

```
EventBridge (Every Sunday 6 AM)
    |
Lambda: Gameweek Checker (checks if GW finished)
    | (if new GW completed)
Step Functions Pipeline
    |
Lambda: Fetch FPL Data -> Lambda: Feature Engineering
    |
SageMaker: Train/Predict (Spot instances)
    |
DynamoDB (Predictions)
    |
API Gateway + Lambda (REST API)
    |
CLI Tool (Local)
```

**Trigger:** Weekly auto-check + manual runs during festive period

---

## Testing Strategy

### Recommendation: Test Locally First

Testing locally before deploying to AWS is critical because:
- **Cost control** - SageMaker training jobs are the primary cost driver (~$0.60/run)
- **Development velocity** - Local iteration is 10-100x faster than cloud deployments
- **Debugging capability** - Local environments provide better debugging tools
- **Confidence building** - Catch bugs before they cost money or corrupt data

### Testing Progression

```
Phase 1: Pure Local (No AWS)           <- Start here
    |
Phase 2: LocalStack/Moto (Emulated AWS)
    |
Phase 3: SAM CLI Local (Docker-based Lambda)
    |
Phase 4: AWS Dev Environment (Real services, test data)
    |
Phase 5: Production
```

### What to Test Where

| Component | Local Testing Method | AWS Required? |
|-----------|---------------------|---------------|
| Lambda functions | SAM CLI local invoke, pytest | No |
| FPL API fetching | Real API calls (free, no auth) | No |
| Feature engineering | pytest with sample data | No |
| XGBoost training | Local Python with xgboost | No |
| DynamoDB operations | LocalStack or moto | No |
| S3 operations | LocalStack or moto | No |
| Step Functions | Step Functions Local (Docker) | No |
| API Gateway + Lambda | SAM CLI local start-api | No |
| SageMaker training jobs | Local XGBoost, then 1x AWS validation | Yes |
| EventBridge schedules | Manual trigger for testing | Yes |
| Cross-service IAM | N/A | Yes |

### Test Pyramid

```
                /\
               /  \
              / E2E \        <- 10% (AWS, expensive)
             /--------\
            /Integration\    <- 30% (LocalStack/SAM)
           /--------------\
          /   Unit Tests   \ <- 60% (pytest/moto)
         /------------------\
```

---

## Local Development Setup

### Required Tools

```bash
# Install these tools
pip install pytest pytest-cov pytest-asyncio moto httpx boto3 xgboost pandas pyarrow
pip install aws-sam-cli
brew install docker docker-compose  # or apt-get on Linux
```

### Docker Compose (LocalStack)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  localstack:
    image: localstack/localstack:3.0
    container_name: fpl-localstack
    ports:
      - "4566:4566"
    environment:
      - SERVICES=s3,dynamodb,stepfunctions,events,sns,lambda
      - DEBUG=0
      - PERSISTENCE=1
    volumes:
      - "./localstack-data:/var/lib/localstack"
      - "/var/run/docker.sock:/var/run/docker.sock"

  dynamodb-admin:
    image: aaronshaf/dynamodb-admin
    container_name: fpl-dynamodb-admin
    ports:
      - "8001:8001"
    environment:
      - DYNAMO_ENDPOINT=http://localstack:4566
    depends_on:
      - localstack
```

### AWS Service Emulation

| Service | Emulation Option | Recommendation |
|---------|-----------------|----------------|
| S3 | LocalStack / moto | LocalStack for integration, moto for unit |
| DynamoDB | LocalStack / moto | LocalStack (better GSI support) |
| Lambda | SAM CLI Local | SAM CLI (real Docker containers) |
| Step Functions | Step Functions Local | Official Docker image |
| SageMaker | **None - must mock** | Local XGBoost + moto for S3 |
| EventBridge | LocalStack | Partial - manual trigger for testing |

---

## Files to Create

```
auto-haaland/
├── template.yaml                    # AWS SAM infrastructure
├── samconfig.toml                   # Deployment config
├── requirements.txt                 # Production dependencies
├── requirements-dev.txt             # Dev/test dependencies
├── docker-compose.yml               # LocalStack setup
├── pytest.ini                       # Test configuration
├── Makefile                         # Common commands
|
├── lambdas/
|   ├── __init__.py
|   ├── common/                      # Shared utilities
|   |   ├── __init__.py
|   |   ├── aws_clients.py           # Boto3 client factories
|   |   └── fpl_api.py               # FPL API wrapper
|   ├── gameweek_checker/           # Check if GW finished
|   |   ├── __init__.py
|   |   ├── handler.py
|   |   └── requirements.txt
|   ├── data_fetcher/               # Fetch FPL API data
|   |   ├── __init__.py
|   |   ├── handler.py
|   |   └── requirements.txt
|   ├── feature_processor/          # ML feature engineering
|   |   ├── __init__.py
|   |   ├── handler.py
|   |   └── requirements.txt
|   ├── prediction_loader/          # Load to DynamoDB
|   |   ├── __init__.py
|   |   ├── handler.py
|   |   └── requirements.txt
|   ├── api_handler/                # REST API endpoints
|   |   ├── __init__.py
|   |   ├── handler.py
|   |   └── requirements.txt
|   └── team_analyzer/              # Team-specific analysis
|       ├── __init__.py
|       ├── handler.py
|       └── requirements.txt
|
├── sagemaker/
|   ├── train.py                    # SageMaker entry point
|   ├── train_local.py              # Local training script
|   └── requirements.txt
|
├── statemachine/
|   └── pipeline.asl.json           # Step Functions workflow
|
├── cli/
|   ├── __init__.py
|   └── fpl.py                      # Local CLI tool
|
├── tests/
|   ├── __init__.py
|   ├── conftest.py                 # Shared fixtures
|   ├── unit/
|   |   ├── test_gameweek_checker.py
|   |   ├── test_data_fetcher.py
|   |   ├── test_feature_processor.py
|   |   ├── test_prediction_loader.py
|   |   ├── test_api_handler.py
|   |   └── test_team_analyzer.py
|   ├── integration/
|   |   ├── test_s3_operations.py
|   |   ├── test_dynamodb_operations.py
|   |   └── test_step_functions.py
|   └── fixtures/
|       ├── bootstrap_sample.json
|       └── features_sample.parquet
|
└── scripts/
    ├── setup_localstack.sh         # Initialize LocalStack resources
    └── fetch_test_data.py          # Download FPL data for tests
```

---

## Implementation Order (Optimized for Fast Feedback)

### Phase 1: Foundation and Data Fetcher (Week 1)

**Day 1-2: Environment Setup**
- [ ] Create `requirements.txt` and `requirements-dev.txt`
- [ ] Set up `docker-compose.yml` with LocalStack
- [ ] Create `Makefile` with common commands
- [ ] Initialize `pytest.ini` and `tests/conftest.py`

**Day 3-5: Data Fetcher Lambda**
`lambdas/data_fetcher/handler.py`
- [ ] Implement FPL API fetching (bootstrap, fixtures, player history)
- [ ] Handle rate limiting with exponential backoff
- [ ] Store raw JSON in S3
- [ ] Write unit tests with mocked HTTP responses
- [ ] Write integration tests with moto S3
- [ ] Test with real FPL API (free, no cost)

### Phase 2: Feature Processing (Week 2, Day 1-3)

`lambdas/feature_processor/handler.py`
- [ ] Calculate rolling averages (3/5 games)
- [ ] Engineer features (form, difficulty, injury)
- [ ] Output Parquet to S3
- [ ] Write unit tests for each feature calculation
- [ ] Integration test: S3 read/write with Parquet

### Phase 3: ML Pipeline (Week 2, Day 4-5)

`sagemaker/train_local.py` (Local development script)
- [ ] Train XGBoost locally with sample data
- [ ] Validate model accuracy locally
- [ ] Document hyperparameters for SageMaker

`sagemaker/train.py` (SageMaker entry point)
- [ ] Load 2 seasons historical data
- [ ] Train XGBoost regression
- [ ] Save model to S3
- [ ] Generate predictions

### Phase 4: Storage Layer (Week 3, Day 1-2)

`lambdas/prediction_loader/handler.py`
- [ ] Read predictions from S3
- [ ] Batch write to DynamoDB
- [ ] Integration test with LocalStack DynamoDB
- [ ] Verify GSI queries work correctly

### Phase 5: API Layer (Week 3, Day 3-5)

`lambdas/api_handler/handler.py`
- [ ] GET /predictions
- [ ] GET /predictions/{player_id}
- [ ] GET /top
- [ ] GET /compare
- [ ] Unit tests for all endpoints
- [ ] SAM local start-api testing

`lambdas/team_analyzer/handler.py`
- [ ] GET /team/{team_id}
- [ ] Transfer recommendations
- [ ] Captain suggestions
- [ ] Chip strategy

### Phase 6: Orchestration (Week 4, Day 1-2)

`statemachine/pipeline.asl.json`
- [ ] Chain: Fetch -> Process -> Train -> Predict -> Load
- [ ] Error handling with SNS alerts
- [ ] Test with Step Functions Local
- [ ] Verify error handling paths

`lambdas/gameweek_checker/handler.py`
- [ ] Fetch FPL API bootstrap
- [ ] Check if gameweek `finished: true`
- [ ] Compare against last processed GW (stored in S3/DynamoDB)
- [ ] Trigger Step Functions if new GW completed

### Phase 7: AWS Deployment (Week 4, Day 3-4)

`template.yaml`
- [ ] S3 bucket (fpl-ml-data)
- [ ] DynamoDB table with GSIs
- [ ] IAM roles
- [ ] Lambda functions (6)
- [ ] Step Functions state machine
- [ ] EventBridge rule (weekly)
- [ ] API Gateway
- [ ] SNS alerts
- [ ] Deploy to dev environment
- [ ] **One SageMaker validation run** (~$0.60)
- [ ] Verify EventBridge schedule

### Phase 8: CLI Tool (Week 4, Day 5)

`cli/fpl.py`
- [ ] `fpl top` - Top predicted scorers
- [ ] `fpl player <id>` - Player details
- [ ] `fpl compare <ids>` - Compare players
- [ ] `fpl my-team --team-id <id>` - Your team predictions
- [ ] `fpl captain --team-id <id>` - Captain recommendation
- [ ] `fpl transfers --team-id <id>` - Transfer suggestions
- [ ] `fpl run --gameweek <n>` - Manual pipeline trigger
- [ ] Test against deployed API

---

## SageMaker Cost Optimization

### Development Progression (Zero to Minimal Cost)

```python
# Stage 1: Pure Local (cost: $0)
# - Use scikit-learn XGBoost locally
# - Train on subset of data (100-1000 samples)
# - Iterate on features and hyperparameters

# Stage 2: Local with Full Data (cost: $0)
# - Train on 2 seasons of data locally
# - Validate model performance
# - Finalize hyperparameters

# Stage 3: SageMaker Validation (cost: ~$0.60)
# - Single training job to validate SageMaker script
# - Use Spot instances (70% savings)
# - ml.m5.large instance (avoid over-provisioning)

# Stage 4: Production (cost: ~$0.60/gameweek)
# - Weekly training only when new GW data available
```

### Local Training Script

```python
# sagemaker/train_local.py
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

def train_local(data_path: str, output_path: str):
    """Train XGBoost model locally - no SageMaker costs."""
    df = pd.read_parquet(data_path)

    X = df[['points_last_3', 'points_last_5', 'minutes_pct',
            'form_score', 'opponent_strength', 'home_away',
            'chance_of_playing', 'form_x_difficulty']]
    y = df['actual_points']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    model.save_model(f"{output_path}/model.xgb")
    return model
```

### Spot Instance Configuration

```python
# Use managed spot training for 70% savings
estimator = XGBoost(
    instance_type='ml.m5.large',
    instance_count=1,
    use_spot_instances=True,
    max_wait=3600,  # 1 hour max wait
    max_run=1800,   # 30 min max runtime
)
```

---

## Unit Testing Examples

### Feature Processor Tests

```python
# tests/unit/test_feature_processor.py
import pytest
from lambdas.feature_processor.handler import calculate_rolling_average

def test_rolling_average_3_games():
    points = [5, 8, 3, 10, 6]
    result = calculate_rolling_average(points, window=3)
    assert result == pytest.approx(6.33, rel=0.01)

def test_rolling_average_insufficient_data():
    points = [5, 8]
    result = calculate_rolling_average(points, window=3)
    assert result == pytest.approx(6.5)  # Use available data
```

### Data Fetcher Tests

```python
# tests/unit/test_data_fetcher.py
from unittest.mock import patch, MagicMock
from lambdas.data_fetcher.handler import fetch_bootstrap_data

@patch('lambdas.data_fetcher.handler.requests.get')
def test_fetch_bootstrap_handles_rate_limit(mock_get):
    mock_get.side_effect = [
        MagicMock(status_code=429, headers={'Retry-After': '1'}),
        MagicMock(status_code=200, json=lambda: {'elements': []})
    ]
    result = fetch_bootstrap_data()
    assert result == {'elements': []}
    assert mock_get.call_count == 2
```

---

## Integration Testing with Moto

### Test Fixtures

```python
# tests/conftest.py
import pytest
import boto3
from moto import mock_aws

@pytest.fixture
def aws_credentials():
    import os
    os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
    os.environ['AWS_DEFAULT_REGION'] = 'ap-southeast-2'

@pytest.fixture
def s3_client(aws_credentials):
    with mock_aws():
        conn = boto3.client('s3', region_name='ap-southeast-2')
        conn.create_bucket(
            Bucket='fpl-ml-data',
            CreateBucketConfiguration={'LocationConstraint': 'ap-southeast-2'}
        )
        yield conn

@pytest.fixture
def dynamodb_table(aws_credentials):
    with mock_aws():
        dynamodb = boto3.resource('dynamodb', region_name='ap-southeast-2')
        table = dynamodb.create_table(
            TableName='fpl-predictions',
            KeySchema=[
                {'AttributeName': 'player_id', 'KeyType': 'HASH'},
                {'AttributeName': 'gameweek', 'KeyType': 'RANGE'}
            ],
            AttributeDefinitions=[
                {'AttributeName': 'player_id', 'AttributeType': 'N'},
                {'AttributeName': 'gameweek', 'AttributeType': 'N'},
            ],
            BillingMode='PAY_PER_REQUEST'
        )
        yield table
```

### S3 Integration Test

```python
# tests/integration/test_s3_operations.py
def test_save_raw_data_to_s3(s3_client):
    from lambdas.data_fetcher.handler import save_to_s3
    import json

    data = {'elements': [{'id': 1, 'web_name': 'Salah'}]}
    save_to_s3(s3_client, 'fpl-ml-data', 'raw/gw20.json', data)

    response = s3_client.get_object(Bucket='fpl-ml-data', Key='raw/gw20.json')
    assert json.loads(response['Body'].read()) == data
```

---

## Step Functions Local Testing

```bash
# Start Step Functions Local
docker run -d -p 8083:8083 \
  --name stepfunctions-local \
  -e "LAMBDA_ENDPOINT=http://host.docker.internal:3001" \
  amazon/aws-stepfunctions-local

# Create state machine
aws stepfunctions create-state-machine \
  --endpoint-url http://localhost:8083 \
  --definition file://statemachine/pipeline.asl.json \
  --name fpl-pipeline \
  --role-arn arn:aws:iam::012345678901:role/DummyRole

# Execute
aws stepfunctions start-execution \
  --endpoint-url http://localhost:8083 \
  --state-machine-arn arn:aws:states:us-east-1:123456789012:stateMachine:fpl-pipeline \
  --input '{"gameweek": 20}'
```

---

## Makefile

```makefile
.PHONY: setup test test-unit test-integration local-up local-down deploy

setup:
	pip install -r requirements-dev.txt

test: test-unit test-integration

test-unit:
	pytest tests/unit -v --cov=lambdas

test-integration: local-up
	pytest tests/integration -v
	$(MAKE) local-down

local-up:
	docker-compose up -d
	sleep 5  # Wait for LocalStack

local-down:
	docker-compose down

local-api:
	sam local start-api

train-local:
	python sagemaker/train_local.py

build:
	sam build

deploy-dev:
	sam deploy --config-env dev
```

---

## AWS Resources

| Resource | Name | Purpose |
|----------|------|---------|
| S3 | fpl-ml-data | Raw data, features, models, predictions |
| DynamoDB | fpl-predictions | Query predictions |
| Lambda (x6) | fpl-* | Processing + API |
| Step Functions | fpl-pipeline | Orchestration |
| EventBridge | fpl-weekly-check | Sunday 6 AM trigger |
| API Gateway | fpl-api | REST endpoints |
| SNS | fpl-alerts | Failure notifications |

---

## Trigger Logic

**Automatic (weekly):**
```
Every Sunday 6 AM -> Check FPL API -> If GW finished -> Run pipeline
```

**Manual (festive/anytime):**
```bash
fpl run --gameweek 20
# or
aws stepfunctions start-execution --state-machine-arn <arn> --input '{"gameweek": 20}'
```

---

## Cost Summary

| Phase | Activity | Cost |
|-------|----------|------|
| Development (Week 1-3) | Local development | $0 |
| Development (Week 4) | AWS dev deployment | ~$2-5 |
| Development (Week 4) | SageMaker validation (1 run) | ~$0.60 |
| **Total Development** | | **~$3-6** |

| When | Production Cost |
|------|-----------------|
| Per gameweek run | ~$0.60 |
| Normal month (4 GWs) | ~$3-5 |
| Off-season (idle) | ~$0.70 |
| **Annual estimate** | **~$40-60** |

Only SageMaker costs money - Lambda, DynamoDB, API Gateway are Free Tier.

---

## Commands

```bash
# Local development
docker-compose up -d          # Start LocalStack
make test-unit                # Run unit tests
make test-integration         # Run integration tests
make local-api                # Start local API
make train-local              # Train model locally

# Deploy
sam build && sam deploy --guided

# Test locally with SAM
sam local invoke GameweekCheckerFunction

# Manual run
fpl run --gameweek 20

# Disable auto (off-season)
aws events disable-rule --name fpl-weekly-check
```

---

## CI/CD Pipeline

```yaml
# .github/workflows/test.yml
name: Test Pipeline

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements-dev.txt
      - run: pytest tests/unit -v --cov=lambdas

  integration-tests:
    runs-on: ubuntu-latest
    services:
      localstack:
        image: localstack/localstack:3.0
        ports:
          - 4566:4566
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install -r requirements-dev.txt
      - run: pytest tests/integration -v

  deploy-dev:
    needs: [unit-tests, integration-tests]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/setup-sam@v2
      - run: sam build && sam deploy --no-confirm-changeset
```
