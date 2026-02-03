---
name: numerai-data-fetcher
description: Fetch fresh Numerai data for training or inference. Use when starting a new round, updating datasets, or preparing for live submissions. Handles data download, validation, and version management.
---

# Numerai Data Fetcher

## Overview

This skill handles downloading and preparing Numerai tournament data. Use it when you need fresh data for training new models or generating live predictions for submission.

## Trigger Conditions

Use this skill when:
- Starting work on a new round
- Training data is outdated or missing
- Live features are needed for submission
- User requests "download data", "get new data", "update dataset", or "fetch round data"
- Preparing for inference on the current round

## Required Tools/Capabilities

- **Numerai MCP Server**: For querying round status and data versions
- **File System Access**: For saving downloaded files
- **Network Access**: For downloading parquet files from Numerai
- **Python Environment**: With `numerapi`, `pandas`, and `pyarrow` installed

## Workflow

### 1) Check Current Round Status (MCP Required)

Query the current round to understand what data is available:

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

Also check available data versions:

```graphql
query {
  listDatasets
}
```

### 2) Determine Data Requirements

Based on the user's goal:
- **Training**: Need `train.parquet`, `validation.parquet`, and `benchmark_models.parquet`
- **Inference**: Need `live.parquet` and `live_benchmark_models.parquet`
- **Full refresh**: Need all files for the current data version

### 3) Download Training Data

For training workflows, download the core datasets:

```python
from numerapi import NumerAPI

napi = NumerAPI()

# Download training data
napi.download_dataset("v5.2/train.parquet", dest_path="data/train.parquet")
napi.download_dataset("v5.2/validation.parquet", dest_path="data/validation.parquet")

# Download benchmark models for BMC calculation
napi.download_dataset("v5.2/train_benchmark_models.parquet", dest_path="data/train_benchmark_models.parquet")
napi.download_dataset("v5.2/validation_benchmark_models.parquet", dest_path="data/validation_benchmark_models.parquet")

# Download feature metadata
napi.download_dataset("v5.2/features.json", dest_path="data/features.json")
```

### 4) Download Live Data for Inference

For submission workflows, download live round data:

```python
from numerapi import NumerAPI

napi = NumerAPI()

# Get current round number
current_round = napi.get_current_round()
print(f"Current round: {current_round}")

# Download live features
napi.download_dataset("v5.2/live.parquet", dest_path=f"data/live_{current_round}.parquet")

# Download live benchmark models (for models that use benchmark as input)
napi.download_dataset("v5.2/live_benchmark_models.parquet", dest_path=f"data/live_benchmark_models_{current_round}.parquet")
```

### 5) Validate Downloaded Data

After downloading, verify data integrity:

```python
import pandas as pd

# Check training data
train = pd.read_parquet("data/train.parquet")
print(f"Training samples: {len(train):,}")
print(f"Features: {train.filter(like='feature_').shape[1]}")
print(f"Era range: {train['era'].min()} to {train['era'].max()}")

# Check live data
live = pd.read_parquet(f"data/live_{current_round}.parquet")
print(f"Live samples: {len(live):,}")
print(f"Live era: {live['era'].unique()}")

# Verify no missing values in features
assert not train.filter(like='feature_').isna().any().any(), "Training data has NaN values"
assert not live.filter(like='feature_').isna().any().any(), "Live data has NaN values"
```

### 6) Build Combined Datasets (Optional)

For the agents pipeline, combine train and validation into full datasets:

```bash
PYTHONPATH=numerai python3 -m agents.code.data.build_full_datasets
```

This creates:
- `numerai/v5.2/full.parquet` - Combined train + validation
- `numerai/v5.2/full_benchmark_models.parquet` - Combined benchmark models
- `numerai/v5.2/downsampled_full.parquet` - Every 4th era for quick iteration
- `numerai/v5.2/downsampled_full_benchmark_models.parquet` - Downsampled benchmarks

## Data Version Management

### Check Available Versions

```python
from numerapi import NumerAPI
napi = NumerAPI()

# List all available datasets
datasets = napi.list_datasets()
for ds in datasets:
    print(ds)
```

### Version Compatibility

- Current version: **v5.2** (as of 2025)
- Keep all files from the same version together
- Store version in config files for reproducibility

## Success Criteria

The skill is complete when:
- All requested data files are downloaded to the specified location
- Data files pass validation checks (correct shape, no unexpected NaN values)
- Era ranges match expected values for the data version
- Live data corresponds to the current open round
- User is informed of data statistics (sample count, era range, feature count)

## Common Issues

- **Rate limiting**: Space out downloads or use retries for large files
- **Disk space**: Full datasets are large (10+ GB); verify space before download
- **Version mismatch**: Ensure all files use the same data version
- **Round timing**: Live data only available during open submission window

## Next Steps

After data is fetched:
1. Use `numerai-model-trainer` skill to train models on the new data
2. Use `numerai-submission-handler` skill to generate predictions on live data
