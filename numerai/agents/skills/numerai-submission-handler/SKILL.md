---
name: numerai-submission-handler
description: Submit predictions for the current Numerai round. Use when generating predictions from a trained model and submitting them to the tournament. Handles prediction generation, validation, and submission via MCP or API.
---

# Numerai Submission Handler

## Overview

This skill handles the end-to-end workflow for submitting predictions to a Numerai tournament round. Use it when you have a trained model and need to generate predictions on live data and submit them.

## Trigger Conditions

Use this skill when:
- A new round has opened and submissions are needed
- User requests "submit predictions", "make submission", "submit to round", or "generate live predictions"
- Model is trained and ready for deployment
- Checking submission status or resubmitting after failure

## Required Tools/Capabilities

- **Numerai MCP Server**: For submission and status queries
- **Trained Model**: Checkpoint or pkl file ready for inference
- **Live Data**: Current round's live features
- **Python Environment**: With `numerapi`, `pandas`, and model dependencies

## MCP Configuration

The Numerai MCP server is configured in `~/.claude/mcp.json`:

```json
{
  "servers": {
    "numerai": {
      "transport": "sse",
      "url": "https://api-tournament.numer.ai/mcp/sse",
      "headers": {
        "Authorization": "Token ${NUMERAI_MCP_AUTH}"
      }
    }
  }
}
```

**Required Environment Variable**: `NUMERAI_MCP_AUTH` must be set to your Numerai API token.

**Python Wrapper**: For programmatic access, use the `NumeraiMCP` class from `numerai_2026_pipeline/numerai_mcp.py`:

```python
from numerai_mcp import NumeraiMCP

client = NumeraiMCP()  # Reads NUMERAI_MCP_AUTH from environment
success = client.submit_predictions("my_model", predictions_dict)
```

## Workflow

### 0) Verify MCP Configuration

Before starting, verify MCP access is properly configured:

```python
import os

# Check environment variable is set
assert os.environ.get("NUMERAI_MCP_AUTH"), (
    "NUMERAI_MCP_AUTH environment variable not set. "
    "Export your Numerai API token: export NUMERAI_MCP_AUTH='your-token'"
)

# Test MCP connection (optional - via Python wrapper)
from numerai_mcp import NumeraiMCP, NumeraiMCPAuthError

try:
    client = NumeraiMCP()
    print("MCP client initialized successfully")
except NumeraiMCPAuthError as e:
    print(f"MCP authentication failed: {e}")
```

### 1) Check Round Status (MCP Required)

First, verify a round is open for submissions. Send GraphQL queries to the MCP endpoint at `https://api-tournament.numer.ai/mcp/sse`:

```graphql
query {
  rounds(tournament: 8) {
    number
    openTime
    closeTime
    resolveTime
  }
}
```

Also check your model's current submission status:

```graphql
query {
  account {
    models {
      id
      username
      submissions {
        round
        insertedAt
        filename
      }
    }
  }
}
```

**Via Python wrapper**:

```python
from numerai_mcp import NumeraiMCP

client = NumeraiMCP()
round_num = client.get_current_round()
round_details = client.get_round_details(round_num)
print(f"Round {round_num} - Closes: {round_details['close_time']}")
```

### 2) Load Live Data

Download and load the current round's live features:

```python
from numerapi import NumerAPI
import pandas as pd

napi = NumerAPI()
current_round = napi.get_current_round()

# Download live data
napi.download_dataset("v5.2/live.parquet", dest_path=f"data/live_{current_round}.parquet")
napi.download_dataset("v5.2/live_benchmark_models.parquet", dest_path=f"data/live_benchmark_{current_round}.parquet")

# Load data
live_features = pd.read_parquet(f"data/live_{current_round}.parquet")
live_benchmark = pd.read_parquet(f"data/live_benchmark_{current_round}.parquet")

print(f"Round {current_round}: {len(live_features):,} live samples")
```

### 3) Load Trained Model

Load the model checkpoint for inference:

```python
import joblib

# Load from checkpoint
checkpoint = joblib.load("checkpoints/final_model.joblib")
model = checkpoint["model"]
features = checkpoint["features"]
config = checkpoint.get("config", {})

print(f"Loaded model trained on {len(features)} features")
```

Or for pkl-based models:

```python
import cloudpickle

with open("model.pkl", "rb") as f:
    predict_fn = cloudpickle.load(f)
```

### 4) Generate Predictions

Generate predictions on live data:

```python
import pandas as pd
import numpy as np

# Ensure feature alignment
live_X = live_features[features]

# Generate raw predictions
predictions = model.predict(live_X)

# Create submission dataframe
submission = pd.DataFrame({
    "id": live_features.index,
    "prediction": predictions
})

# Validate predictions
assert len(submission) == len(live_features), "Prediction count mismatch"
assert not submission["prediction"].isna().any(), "NaN predictions found"
print(f"Generated {len(submission):,} predictions")
print(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
```

### 5) Apply Post-Processing (Optional)

For models that require ranking or normalization:

```python
# Per-era ranking to [0, 1]
def rank_per_era(df, features_df):
    """Rank predictions within each era."""
    df = df.copy()
    df["era"] = features_df["era"].values
    df["prediction"] = df.groupby("era")["prediction"].rank(pct=True)
    return df.drop(columns=["era"])

submission = rank_per_era(submission, live_features)

# Or simple global ranking
submission["prediction"] = submission["prediction"].rank(pct=True)
```

### 6) Validate Predictions

Before submitting, validate the predictions:

```python
# Check prediction statistics
print("Prediction Statistics:")
print(f"  Min: {submission['prediction'].min():.6f}")
print(f"  Max: {submission['prediction'].max():.6f}")
print(f"  Mean: {submission['prediction'].mean():.6f}")
print(f"  Std: {submission['prediction'].std():.6f}")

# Validation checks
assert submission["prediction"].min() >= 0, "Predictions below 0"
assert submission["prediction"].max() <= 1, "Predictions above 1"
assert submission["prediction"].std() > 0.01, "Predictions have low variance"
assert len(submission) > 1000, "Too few predictions"

print("\nValidation passed!")
```

### 7) Save Predictions to File

Save predictions in the required format:

```python
from pathlib import Path

predictions_dir = Path("predictions")
predictions_dir.mkdir(exist_ok=True)

filename = f"predictions_round_{current_round}.csv"
filepath = predictions_dir / filename

submission.to_csv(filepath, index=False)
print(f"Saved predictions to: {filepath}")
```

### 8) Submit via Numerai API

Submit the predictions:

```python
from numerapi import NumerAPI

napi = NumerAPI(public_id="YOUR_PUBLIC_ID", secret_key="YOUR_SECRET_KEY")

# Submit predictions
model_id = "your_model_name"  # or UUID
submission_id = napi.upload_predictions(
    file_path=str(filepath),
    model_id=model_id,
    version=2  # Use v2 API
)

print(f"Submitted! Submission ID: {submission_id}")
```

### 9) Submit via MCP (Alternative)

For automated pipelines using MCP, send mutations to the MCP endpoint at `https://api-tournament.numer.ai/mcp/sse`:

```graphql
mutation {
  uploadPredictions(
    modelId: "your-model-uuid"
    predictions: "base64-encoded-csv-content"
  ) {
    id
    insertedAt
  }
}
```

**Via Python wrapper**:

```python
from numerai_mcp import NumeraiMCP

client = NumeraiMCP()

# predictions_dict is a dict mapping stock IDs to prediction values
predictions_dict = {"stock_1": 0.52, "stock_2": 0.48, ...}
success = client.submit_predictions("my_model_name", predictions_dict)

if success:
    print("Submission successful!")
```

### 10) Verify Submission Status

After submission, verify it was accepted:

```python
# Check submission status
submissions = napi.get_submissions(model_id)
latest = submissions[0] if submissions else None

if latest:
    print(f"Latest submission:")
    print(f"  Round: {latest['round']}")
    print(f"  Submitted: {latest['insertedAt']}")
    print(f"  Filename: {latest['filename']}")
```

Or via MCP:

```graphql
query {
  account {
    models {
      username
      submissions(limit: 1) {
        round
        insertedAt
        filename
      }
    }
  }
}
```

## Automated Pickle Submissions

For models deployed as pickles, submissions are automatic. Monitor via:

```graphql
query {
  account {
    models {
      username
      computePickleUpload {
        validationStatus
        triggerStatus
        triggers {
          id
          status
          round
          insertedAt
        }
      }
    }
  }
}
```

## Success Criteria

The skill is complete when:
- Predictions generated for all live samples
- Predictions pass validation checks (range, variance, count)
- Submission uploaded successfully to Numerai
- Submission confirmed in account/model status
- User informed of submission ID and status

## Common Issues

- **Round not open**: Check round timing before attempting submission
- **Model mismatch**: Ensure live data version matches model training data
- **Feature alignment**: Feature order must match training exactly
- **Authentication**: Verify API keys have submission scope
- **Duplicate submission**: Only one submission per model per round is scored

## Submission Timing

- **Early submission**: Recommended to avoid last-minute issues
- **Late submission**: Allowed until round close, but risky
- **Resubmission**: Allowed; latest submission is used for scoring

## Next Steps

After submission:
1. Use `numerai-performance-tracker` skill to monitor round results
2. Check diagnostics portal for submission quality metrics
3. Plan next round's experiments based on feedback
