# Quick Start Guide

## Phase 1 Complete! 🎉

You've successfully implemented the **Data Fetcher** component. Here's how to test it.

## What We Built

```
✓ FPL API Client (lambdas/common/fpl_api.py)
  - Fetches data from official FPL API
  - Rate limiting with exponential backoff
  - Auto-detects current gameweek and season

✓ Data Fetcher Lambda (lambdas/data_fetcher/handler.py)
  - Fetches bootstrap, fixtures, and player data
  - Stores raw JSON to S3
  - Handles errors gracefully

✓ Comprehensive Tests
  - 15+ unit tests for FPL API client
  - 10+ unit tests for data fetcher
  - 5+ integration tests with mocked S3
```

## Step 1: Setup Environment

### Quick Setup

Run the automated setup script (requires sudo):

```bash
./setup.sh
```

This will install system packages, create a virtual environment, and install all dependencies.

### Manual Setup

If you prefer manual setup:

```bash
# Install system packages
sudo apt install python3.12-venv docker-compose

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt
```

**Important:** Always activate the virtual environment before running commands:

```bash
source venv/bin/activate
```

## Step 2: Test with Real FPL API

The FPL API is free and requires no authentication!

```bash
# Test the FPL API client
python scripts/test_real_fpl_api.py
```

Expected output:
```
============================================================
Testing FPL API Client with Real API
============================================================

Test 1: Fetching bootstrap-static...
✓ Success! Found 38 gameweeks
✓ Found 20 teams
✓ Found 600+ players

Test 2: Getting current gameweek...
✓ Current gameweek: 20

... (more tests)

All tests passed! ✓
```

## Step 3: Run Unit Tests

```bash
# Run all unit tests
make test-unit

# Or manually:
pytest tests/unit -v
```

Expected: **All tests pass** (20+ tests)

## Step 4: Run Integration Tests

These use `moto` to mock AWS S3 locally:

```bash
# Start LocalStack first
make local-up

# Run integration tests
make test-integration

# Or manually:
pytest tests/integration -v
```

Expected: **All tests pass** (5+ tests)

## Step 5: Test End-to-End with LocalStack

```bash
# 1. Start LocalStack
make local-up

# 2. Run the data fetcher locally
cd lambdas/data_fetcher
python handler.py

# 3. Check LocalStack to see saved files
# Visit: http://localhost:8001 (DynamoDB Admin)
# Or use AWS CLI:
aws --endpoint-url=http://localhost:4566 s3 ls s3://fpl-ml-data/raw/ --recursive
```

## What Gets Stored in S3

```
s3://fpl-ml-data/
└── raw/
    └── season_2024_25/
        ├── gw20_bootstrap.json      <- All players, teams, gameweeks
        ├── gw20_fixtures.json        <- Fixtures for GW20
        └── gw20_players/
            ├── player_350.json       <- Salah's history
            ├── player_328.json       <- Haaland's history
            └── ...
```

## Test Data Structure

Here's what the data looks like:

### Bootstrap Static (`gw20_bootstrap.json`)
```json
{
  "events": [
    {"id": 20, "name": "Gameweek 20", "finished": false}
  ],
  "teams": [
    {"id": 1, "name": "Arsenal", "strength": 4}
  ],
  "elements": [
    {
      "id": 350,
      "web_name": "Salah",
      "team": 10,
      "element_type": 3,
      "now_cost": 130,
      "form": "8.5"
    }
  ]
}
```

### Player Summary (`player_350.json`)
```json
{
  "history": [
    {
      "element": 350,
      "fixture": 380,
      "total_points": 12,
      "minutes": 90,
      "goals_scored": 2,
      "assists": 0
    }
  ]
}
```

## Next Steps

Once all tests pass, you're ready for **Phase 2: Feature Processing**!

The feature processor will:
1. Read raw JSON from S3
2. Calculate rolling averages (3/5 games)
3. Engineer ML features (form, difficulty, etc.)
4. Save processed Parquet files to S3

## Troubleshooting

### ImportError: No module named 'lambdas'
```bash
# Run from project root, not inside lambdas/
cd /home/juan/dev/auto-haaland
python scripts/test_real_fpl_api.py
```

### LocalStack not starting
```bash
# Check Docker is running
docker ps

# Restart LocalStack
make local-down
make local-up
```

### Tests failing
```bash
# Make sure you're in project root
cd /home/juan/dev/auto-haaland

# Install dependencies
make setup

# Run tests
make test-unit
```

## Summary

✅ **Phase 1 Complete!**

You now have:
- Working FPL API client
- Data fetcher that stores to S3
- Comprehensive test suite
- Local development environment

Ready to move to Phase 2? Let's build the feature processor! 🚀
