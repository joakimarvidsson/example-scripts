---
name: numerai-model-trainer
description: Train Numerai models for a new round. Use when you have a finalized model configuration and need to train on full data, run HPO, or prepare models for deployment. Handles training, checkpointing, and model validation.
---

# Numerai Model Trainer

## Overview

This skill handles the training pipeline for Numerai models. Use it when you have a validated model configuration and need to train on production data, run hyperparameter optimization, or prepare model checkpoints for deployment.

Note: For experiment iteration and model development, use the `numerai-experiment-design` skill instead. This skill is for production training runs.

## Trigger Conditions

Use this skill when:
- A model configuration has been finalized through experimentation
- Training models for the current round's submission
- Running hyperparameter optimization on a proven architecture
- User requests "train model", "run training", "prepare for production", or "train on full data"
- Checkpoints need to be created for deployment

## Required Tools/Capabilities

- **Pipeline Scripts**: `agents.code.modeling` module
- **GPU Access**: For deep learning models (optional but recommended)
- **File System Access**: For saving checkpoints and logs
- **Python Environment**: With model dependencies installed (lightgbm, torch, etc.)

## Workflow

### 1) Verify Data Availability

Before training, confirm data is available and up-to-date:

```python
import os
from pathlib import Path

data_dir = Path("numerai/v5.2")
required_files = [
    "full.parquet",
    "full_benchmark_models.parquet",
]

for f in required_files:
    path = data_dir / f
    if not path.exists():
        print(f"Missing: {path}")
        print("Run: PYTHONPATH=numerai python3 -m agents.code.data.build_full_datasets")
    else:
        print(f"Found: {path} ({path.stat().st_size / 1e9:.2f} GB)")
```

### 2) Prepare Training Configuration

Create or verify the training config:

```python
CONFIG = {
    "model": {
        "type": "LGBMRegressor",
        "params": {
            "n_estimators": 2000,
            "learning_rate": 0.01,
            "num_leaves": 64,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbose": -1,
        }
    },
    "training": {
        "cv": {
            "n_splits": 5,
            "purge_gap": 4,
        },
        "early_stopping_rounds": 100,
    },
    "data": {
        "data_version": "v5.2",
        "feature_set": "all",
        "target_col": "target",
        "era_col": "era",
    },
    "output": {
        "save_checkpoints": True,
        "checkpoint_dir": "checkpoints",
    },
    "preprocessing": {},
}
```

### 3) Run Cross-Validation Training

For production models, run full CV to get reliable OOF predictions:

```bash
PYTHONPATH=numerai python3 -m agents.code.modeling \
    --config numerai/agents/experiments/production/configs/final_model.py \
    --output-dir numerai/agents/experiments/production
```

This will:
- Train on each CV fold
- Generate out-of-fold predictions
- Calculate metrics (corr, BMC, sharpe)
- Save fold checkpoints

### 4) Validate Training Results

After training, verify model quality:

```python
import json
from pathlib import Path

results_dir = Path("numerai/agents/experiments/production/results")
results_file = list(results_dir.glob("*.json"))[0]

with open(results_file) as f:
    results = json.load(f)

print("Training Results:")
print(f"  corr_mean: {results['corr_mean']:.6f}")
print(f"  bmc_mean: {results['bmc_mean']:.6f}")
print(f"  bmc_last_200_eras: {results['bmc_last_200_eras']['mean']:.6f}")
print(f"  sharpe: {results.get('sharpe', 'N/A')}")

# Quality gates
assert results['corr_mean'] > 0.01, "corr_mean too low"
assert results['bmc_mean'] > 0, "bmc_mean should be positive"
print("\nQuality gates passed!")
```

### 5) Train Final Model on Full Data

For deployment, retrain on all available data (no holdout):

```python
# Option 1: Use the pipeline with full data flag
PYTHONPATH=numerai python3 -m agents.code.modeling \
    --config numerai/agents/experiments/production/configs/final_model.py \
    --output-dir numerai/agents/experiments/production \
    --full-data  # Train on train + validation combined
```

Or manually:

```python
import pandas as pd
from agents.code.modeling.utils.model_factory import create_model

# Load all data
train = pd.read_parquet("numerai/v5.2/train.parquet")
val = pd.read_parquet("numerai/v5.2/validation.parquet")
full = pd.concat([train, val], ignore_index=True)

# Get features and target
features = [c for c in full.columns if c.startswith("feature_")]
X = full[features]
y = full["target"]

# Create and train model
model = create_model("LGBMRegressor", model_params)
model.fit(X, y)

# Save checkpoint
import joblib
joblib.dump(model, "checkpoints/final_model.joblib")
```

### 6) Save Model Checkpoint

Ensure checkpoints are properly saved for deployment:

```python
import joblib
from pathlib import Path
from datetime import datetime

checkpoint_dir = Path("numerai/agents/experiments/production/checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Save with timestamp and metadata
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_path = checkpoint_dir / f"model_{timestamp}.joblib"

checkpoint = {
    "model": model,
    "features": features,
    "config": CONFIG,
    "metrics": results,
    "trained_at": timestamp,
}
joblib.dump(checkpoint, checkpoint_path)
print(f"Checkpoint saved: {checkpoint_path}")
```

## Hyperparameter Optimization (Optional)

For HPO, use the multi-objective optimization workflow:

```bash
PYTHONPATH=numerai python3 -m agents.code.hpo.run_hpo \
    --config numerai/agents/experiments/hpo/hpo_config.py \
    --n-trials 100 \
    --output-dir numerai/agents/experiments/hpo
```

HPO config structure:

```python
HPO_CONFIG = {
    "search_space": {
        "n_estimators": {"type": "int", "low": 500, "high": 3000},
        "learning_rate": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
        "num_leaves": {"type": "int", "low": 16, "high": 256},
        "feature_fraction": {"type": "float", "low": 0.5, "high": 1.0},
    },
    "objectives": ["bmc_mean", "sharpe"],
    "n_startup_trials": 10,
}
```

## GPU Training (Deep Learning Models)

For neural network models:

```python
import torch

# Check GPU availability
if torch.cuda.is_available():
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("Warning: No GPU available, training will be slow")

# In config
CONFIG["model"]["params"]["device"] = device
```

## Success Criteria

The skill is complete when:
- Model trained successfully on specified data
- Metrics are within expected ranges (corr_mean > 0.01, positive BMC)
- Checkpoints saved and loadable
- No training errors or warnings
- Training logs captured for debugging

## Common Issues

- **Memory errors**: Use downsampled data first, then scale up
- **Slow training**: Enable GPU for deep models, use feature subsampling
- **Poor metrics**: Review preprocessing, check for data leakage
- **Checkpoint size**: Large models may need compression or weight pruning

## Next Steps

After training is complete:
1. Use `numerai-model-upload` skill to create deployment pickle
2. Use `numerai-submission-handler` skill to generate predictions
3. Use `numerai-performance-tracker` skill to monitor deployed model
