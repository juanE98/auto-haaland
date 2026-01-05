# Auto-Haaland: FPL ML Prediction System

An automated Fantasy Premier League prediction system using XGBoost and AWS.

## Architecture

```
EventBridge (Weekly Trigger)
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

## Cost Summary

| When | Cost |
|------|------|
| Per gameweek run | ~$0.60 |
| Normal month (4 GWs) | ~$3-5 |
| Off-season (idle) | ~$0.70 |
| **Annual estimate** | **~$40-60** |

Only SageMaker costs money - Lambda, DynamoDB, API Gateway are free tier.

---

## Project Structure

```
auto-haaland/
├── template.yaml                 # AWS SAM infrastructure
├── samconfig.toml               # Deployment config
├── lambdas/
│   ├── data_fetcher/            # Fetch FPL API data
│   ├── feature_processor/       # ML feature engineering
│   ├── prediction_loader/       # Load to DynamoDB
│   ├── api_handler/             # REST API endpoints
│   └── team_analyzer/           # Team-specific analysis
├── sagemaker/
│   └── train.py                 # XGBoost training
├── statemachine/
│   └── pipeline.asl.json        # Step Functions workflow
└── cli/
    └── fpl.py                   # Local CLI tool
```

---

## Features

### ML Model
- **Algorithm:** XGBoost regression
- **Training:** SageMaker with Spot instances (70% savings)
- **Data:** 2 seasons historical (FPL API)

### Prediction Features
| Feature | Description |
|---------|-------------|
| `points_last_3/5` | Rolling average points |
| `minutes_percentage` | Playing time consistency |
| `form_score` | Weighted recent performance |
| `opponent_strength` | Fixture difficulty (FDR 1-5) |
| `home_away` | Home advantage |
| `chance_of_playing` | Injury probability (0-100%) |
| `form_x_difficulty` | Form × fixture interaction |

### CLI Commands

**General:**
```bash
fpl top                          # Top predicted scorers
fpl player <id>                  # Single player analysis
fpl compare <id1,id2,id3>        # Compare players
```

**Team-specific:**
```bash
fpl my-team --team-id <id>       # Your team's predictions
fpl captain --team-id <id>       # Captain recommendation
fpl transfers --team-id <id> --budget 1.5 --ft 2
fpl plan --team-id <id> --weeks 5
fpl chips --team-id <id>         # BB, TC, FH, WC strategy
```

---

## AWS Resources

| Resource | Name | Purpose |
|----------|------|---------|
| S3 | fpl-ml-data | Data, models, predictions |
| DynamoDB | fpl-predictions | Query predictions |
| Lambda (x5) | fpl-* | Processing + API |
| Step Functions | fpl-weekly-pipeline | Orchestration |
| EventBridge | fpl-weekly-trigger | Tuesday 1 AM schedule |
| API Gateway | fpl-predictions-api | REST endpoints |
| SNS | fpl-pipeline-alerts | Failure alerts |

---

## Deployment

```bash
# Build and deploy
sam build
sam deploy --guided

# Test locally
sam local invoke FplDataFetcherFunction
```

## Manual Pipeline Trigger

```bash
# Run for specific gameweek
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:REGION:ACCOUNT:stateMachine:fpl-weekly-pipeline \
  --input '{"gameweek": 20, "season": "2024_25", "retrain": false}'

# Off-season: disable schedule
aws events disable-rule --name fpl-weekly-trigger
```

---

## S3 Structure

```
fpl-ml-data/
├── raw/season_2024_25/          # Raw FPL API JSON
├── processed/                    # Parquet features
├── models/                       # Trained XGBoost models
└── predictions/                  # Weekly predictions
```

---

## DynamoDB Schema

**Table:** `fpl-predictions`

| Key | Type |
|-----|------|
| `player_id` (PK) | Number |
| `gameweek` (SK) | Number |

**GSIs:**
- `gameweek → predicted_points` (top scorers)
- `position → predicted_points` (by position)

**Sample item:**
```json
{
  "player_id": 350,
  "gameweek": 20,
  "player_name": "Mohamed Salah",
  "team": "Liverpool",
  "position": "MID",
  "predicted_points": 8.2,
  "confidence_lower": 5.8,
  "confidence_upper": 10.6,
  "fixtures_next_5": ["SHU (A)", "CRY (H)", "MUN (A)"],
  "fixture_difficulty": [2, 2, 4]
}
```

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /predictions?gameweek=20` | All predictions |
| `GET /predictions/{player_id}` | Single player |
| `GET /top?gameweek=20&position=MID` | Top scorers |
| `GET /compare?players=350,328&gameweek=20` | Compare players |
| `GET /team/{team_id}?gameweek=20` | Team analysis |

---

## Limitations

- Injury data from FPL API only (updated weekly)
- No real-time news, press conferences, or tactics
- Predictions only as good as historical patterns
