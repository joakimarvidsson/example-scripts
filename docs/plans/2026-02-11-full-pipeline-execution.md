# Full Pipeline Execution Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Commit all pending work, create missing configs, add CatBoost HPO support, run HPO for XGB/CatBoost, train all models on full data, validate TorchMLP, experiment with target transforms, and evaluate the final ensemble.

**Architecture:** Build on existing HPO v3/v4 pipeline. Extend search spaces and CLI to support XGBoost and CatBoost. Create configs for TorchMLP and target transform variants. Train and evaluate all models, then combine via rank-averaging ensemble.

**Tech Stack:** LightGBM, XGBoost, CatBoost, PyTorch, Optuna, numerai-tools, cloudpickle

---

## Phase 0: Git Cleanup

### Task 1: Commit modified infrastructure files

These 3 files are already modified but unstaged. They add pickle safety, TorchMLP registration, and advanced target transforms.

**Files:**
- Stage: `numerai/agents/code/modeling/models/lgbm_regressor.py`
- Stage: `numerai/agents/code/modeling/utils/model_factory.py`
- Stage: `numerai/agents/code/modeling/utils/target_transforms.py`

**Step 1: Stage and commit**
```bash
cd /Users/joakim/Documents/Projects/Numerai/example-scripts
git add numerai/agents/code/modeling/models/lgbm_regressor.py \
        numerai/agents/code/modeling/utils/model_factory.py \
        numerai/agents/code/modeling/utils/target_transforms.py
git commit -m "feat: add pickle safety, TorchMLP registration, and advanced target transforms

- lgbm_regressor: add recursion guard in __getattr__ during unpickling
- model_factory: register TorchMLPRegressor model type
- target_transforms: add inverse-normal transform, benchmark subtraction variants, drop_na support"
```

### Task 2: Commit new model wrappers

**Files:**
- Stage: `numerai/agents/code/modeling/models/mlp_regressor.py`
- Stage: `numerai/agents/code/modeling/models/torch_mlp_regressor.py`

**Step 1: Stage and commit**
```bash
git add numerai/agents/code/modeling/models/mlp_regressor.py \
        numerai/agents/code/modeling/models/torch_mlp_regressor.py
git commit -m "feat: add MLP model wrappers (sklearn and PyTorch)

- mlp_regressor: sklearn MLPRegressor wrapper with StandardScaler
- torch_mlp_regressor: custom PyTorch MLP with early stopping, era-aware splits, GPU support"
```

### Task 3: Commit HPO pipeline and utilities

**Files:**
- Stage: `numerai/agents/code/modeling/hpo_cli.py`
- Stage: `numerai/agents/code/modeling/hpo_v2_cli.py`
- Stage: `numerai/agents/code/modeling/hpo_v3_cli.py`
- Stage: `numerai/agents/code/modeling/hpo_v4_cli.py`
- Stage: `numerai/agents/code/modeling/utils/hpo.py`
- Stage: `numerai/agents/code/modeling/utils/hpo_v2.py`
- Stage: `numerai/agents/code/modeling/utils/hpo_v3.py`
- Stage: `numerai/agents/code/modeling/utils/hpo_v4.py`
- Stage: `numerai/agents/code/modeling/utils/era_stride_cv.py`

**Step 1: Stage and commit**
```bash
git add numerai/agents/code/modeling/hpo_cli.py \
        numerai/agents/code/modeling/hpo_v2_cli.py \
        numerai/agents/code/modeling/hpo_v3_cli.py \
        numerai/agents/code/modeling/hpo_v4_cli.py \
        numerai/agents/code/modeling/utils/hpo.py \
        numerai/agents/code/modeling/utils/hpo_v2.py \
        numerai/agents/code/modeling/utils/hpo_v3.py \
        numerai/agents/code/modeling/utils/hpo_v4.py \
        numerai/agents/code/modeling/utils/era_stride_cv.py
git commit -m "feat: add HPO pipeline v1-v4 with era-stride CV

- hpo v1-v2: initial Optuna-based HPO with BMC Sharpe optimization
- hpo v3: three-phase pipeline (HPO, neutralization, holdout) with proper data splits
- hpo v4: enhanced with payout optimization, two-stage neutralization, and ensemble training
- era_stride_cv: cross-validation with era stride splits for better generalization"
```

### Task 4: Commit export and upload tools

**Files:**
- Stage: `numerai/agents/code/modeling/export_cli.py`
- Stage: `numerai/agents/code/modeling/utils/model_export.py`

**Step 1: Stage and commit**
```bash
git add numerai/agents/code/modeling/export_cli.py \
        numerai/agents/code/modeling/utils/model_export.py
git commit -m "feat: add model export CLI for Numerai pickle submissions"
```

### Task 5: Commit baseline configs

**Files:**
- Stage: `numerai/agents/baselines/configs/lgbm_ender20_downsampled_hpo_v2.py`
- Stage: `numerai/agents/baselines/configs/lgbm_quick_test.py`

**Step 1: Stage and commit**
```bash
git add numerai/agents/baselines/configs/lgbm_ender20_downsampled_hpo_v2.py \
        numerai/agents/baselines/configs/lgbm_quick_test.py
git commit -m "feat: add HPO-optimized LGBM config and quick test config"
```

### Task 6: Commit strategic documentation

**Files:**
- Stage: `docs/plans/2026-02-05-classic-grandmaster-strategy.md`
- Stage: `docs/plans/2026-02-11-full-pipeline-execution.md`

**Step 1: Stage and commit**
```bash
git add docs/
git commit -m "docs: add grandmaster strategy plan and full pipeline execution plan"
```

---

## Phase 1: Infrastructure - Add CatBoost HPO Support

### Task 7: Add CatBoost search space to HPO v3

The HPO v3 has search spaces for LGBMRegressor and XGBRegressor but not CatBoostRegressor.

**Files:**
- Modify: `numerai/agents/code/modeling/utils/hpo_v3.py:33-53` (SEARCH_SPACES_V3)

**Step 1: Add CatBoost search space**

Add to `SEARCH_SPACES_V3` dict after the XGBRegressor entry:

```python
    "CatBoostRegressor": {
        "learning_rate": ("log_float", 0.005, 0.1),
        "depth": ("int", 4, 10),
        "l2_leaf_reg": ("log_float", 1e-2, 10.0),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bylevel": ("float", 0.3, 1.0),
        "min_data_in_leaf": ("int", 1, 100),
        "random_strength": ("log_float", 1e-3, 10.0),
    },
```

**Step 2: Verify the CatBoost model factory integration**

The `catboost_regressor.py` wrapper maps `iterations` as the n_estimators equivalent. The HPO phase passes `n_estimators` which gets added to params. We need to ensure CatBoost gets `iterations` instead:

Check `run_hpo_phase` to see how `n_estimators` is injected. If it's passed as `model_params["n_estimators"]`, we need a mapping for CatBoost (which uses `iterations`).

In `hpo_v3.py`, find where `n_estimators` is set on model_params and add:
```python
if model_type == "CatBoostRegressor":
    model_params["iterations"] = n_estimators
else:
    model_params["n_estimators"] = n_estimators
```

Similarly in `hpo_v4.py` `train_ensemble_models()` which also sets `n_estimators`.

**Step 3: Commit**
```bash
git add numerai/agents/code/modeling/utils/hpo_v3.py \
        numerai/agents/code/modeling/utils/hpo_v4.py
git commit -m "feat: add CatBoost search space to HPO v3 and fix n_estimators mapping"
```

### Task 8: Add --model-type flag to HPO v3 CLI

The `hpo_v3_cli.py` doesn't expose the `model_type` parameter.

**Files:**
- Modify: `numerai/agents/code/modeling/hpo_v3_cli.py:28-36` (hpo_parser arguments)

**Step 1: Add --model-type argument to hpo_parser**

After the existing arguments, add:
```python
hpo_parser.add_argument("--model-type", default="LGBMRegressor",
                        choices=["LGBMRegressor", "XGBRegressor", "CatBoostRegressor"],
                        help="Model type to optimize")
```

**Step 2: Pass model_type to run_hpo_phase**

In the `if args.command == "hpo":` block, add `model_type=args.model_type` to the `run_hpo_phase()` call.

**Step 3: Commit**
```bash
git add numerai/agents/code/modeling/hpo_v3_cli.py
git commit -m "feat: add --model-type flag to HPO v3 CLI for XGB and CatBoost support"
```

### Task 9: Add --model-type flag to HPO v4 ensemble training

The `hpo_v4_cli.py` and `train_ensemble_models()` default to LGBMRegressor.

**Files:**
- Modify: `numerai/agents/code/modeling/hpo_v4_cli.py:33-39` (ensemble_parser)

**Step 1: Add --model-type to ensemble and holdout parsers**

Add to each subparser:
```python
parser.add_argument("--model-type", default="LGBMRegressor",
                    choices=["LGBMRegressor", "XGBRegressor", "CatBoostRegressor"],
                    help="Model type to train")
```

**Step 2: Pass model_type through to train_ensemble_models and evaluate_on_holdout_v4**

**Step 3: Commit**
```bash
git add numerai/agents/code/modeling/hpo_v4_cli.py
git commit -m "feat: add --model-type flag to HPO v4 CLI"
```

---

## Phase 2: Create Missing Configs

### Task 10: Create CatBoost downsampled config

**Files:**
- Create: `numerai/agents/baselines/configs/catboost_ender20_downsampled.py`

**Step 1: Create the config**

Follow the pattern of `xgb_ender20_downsampled.py` but with CatBoost params:

```python
CONFIG = {
    "data": {
        "data_version": "v5.2",
        "feature_set": "all",
        "target_col": "target",
        "era_col": "era",
        "embargo_eras": 13,
        "full_data_path": "numerai/v5.2/downsampled_full.parquet",
        "benchmark_data_path": "numerai/v5.2/downsampled_full_benchmark_models.parquet",
    },
    "model": {
        "type": "CatBoostRegressor",
        "x_groups": ["features", "era", "benchmark_models"],
        "params": {
            "iterations": 2000,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
            "subsample": 0.8,
            "colsample_bylevel": 0.1,
            "random_seed": 1337,
            "verbose": 0,
        },
    },
    "training": {
        "cv": {
            "enabled": True,
            "n_splits": 5,
            "embargo": 13,
            "mode": "expanding",
            "min_train_size": 0,
        },
    },
    "preprocessing": {"missing_value": 2.0, "nan_missing_all_twos": False},
    "output": {
        "output_dir": "baselines",
        "results_name": "catboost_ender20_downsampled",
    },
}
```

**Step 2: Commit**
```bash
git add numerai/agents/baselines/configs/catboost_ender20_downsampled.py
git commit -m "feat: add CatBoost downsampled baseline config"
```

### Task 11: Create CatBoost full-data config

**Files:**
- Create: `numerai/agents/baselines/configs/catboost_ender20_baseline.py`

**Step 1: Create the config** (same as downsampled but without data path overrides)

```python
CONFIG = {
    "data": {
        "data_version": "v5.2",
        "feature_set": "all",
        "target_col": "target",
        "era_col": "era",
        "embargo_eras": 13,
    },
    "model": {
        "type": "CatBoostRegressor",
        "x_groups": ["features", "era", "benchmark_models"],
        "params": {
            "iterations": 2000,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
            "subsample": 0.8,
            "colsample_bylevel": 0.1,
            "random_seed": 1337,
            "verbose": 0,
        },
    },
    "training": {
        "cv": {
            "enabled": True,
            "n_splits": 5,
            "embargo": 13,
            "mode": "expanding",
            "min_train_size": 0,
        },
    },
    "preprocessing": {"missing_value": 2.0, "nan_missing_all_twos": False},
    "output": {
        "output_dir": "baselines",
        "results_name": "catboost_ender20_baseline",
    },
}
```

**Step 2: Commit**
```bash
git add numerai/agents/baselines/configs/catboost_ender20_baseline.py
git commit -m "feat: add CatBoost full-data baseline config"
```

### Task 12: Create TorchMLP downsampled config

**Files:**
- Create: `numerai/agents/baselines/configs/torch_mlp_ender20_downsampled.py`

**Step 1: Create the config**

```python
CONFIG = {
    "data": {
        "data_version": "v5.2",
        "feature_set": "all",
        "target_col": "target",
        "era_col": "era",
        "embargo_eras": 13,
        "full_data_path": "numerai/v5.2/downsampled_full.parquet",
        "benchmark_data_path": "numerai/v5.2/downsampled_full_benchmark_models.parquet",
    },
    "model": {
        "type": "TorchMLPRegressor",
        "x_groups": ["features", "era", "benchmark_models"],
        "params": {
            "hidden_layer_sizes": [512, 256, 128],
            "activation": "gelu",
            "learning_rate_init": 0.0003,
            "batch_size": 4096,
            "max_iter": 120,
            "dropout": 0.1,
            "weight_decay": 0.0,
            "early_stopping": True,
            "patience": 10,
            "validation_fraction": 0.1,
            "validation_split_mode": "era",
            "scale": True,
            "random_state": 1337,
        },
    },
    "training": {
        "cv": {
            "enabled": True,
            "n_splits": 5,
            "embargo": 13,
            "mode": "expanding",
            "min_train_size": 0,
        },
    },
    "preprocessing": {"missing_value": 2.0, "nan_missing_all_twos": False},
    "output": {
        "output_dir": "baselines",
        "results_name": "torch_mlp_ender20_downsampled",
    },
}
```

**Step 2: Commit**
```bash
git add numerai/agents/baselines/configs/torch_mlp_ender20_downsampled.py
git commit -m "feat: add TorchMLP downsampled baseline config"
```

### Task 13: Create target transform variant configs

Create LGBM configs with different target transforms to increase ensemble diversity.

**Files:**
- Create: `numerai/agents/baselines/configs/lgbm_ender20_downsampled_invnorm.py`
- Create: `numerai/agents/baselines/configs/lgbm_ender20_downsampled_residualized.py`

**Step 1: Create inverse-normal transform config**

```python
# LGBM with inverse-normal benchmark subtraction
# Different target transform increases ensemble diversity
CONFIG = {
    "data": {
        "data_version": "v5.2",
        "feature_set": "all",
        "target_col": "target",
        "era_col": "era",
        "embargo_eras": 13,
        "full_data_path": "numerai/v5.2/downsampled_full.parquet",
        "benchmark_data_path": "numerai/v5.2/downsampled_full_benchmark_models.parquet",
        "benchmark_model": "v52_lgbm_ender20",
    },
    "model": {
        "type": "LGBMRegressor",
        "x_groups": ["features", "era", "benchmark_models"],
        "params": {
            "learning_rate": 0.0899,
            "max_depth": 6,
            "num_leaves": 173,
            "colsample_bytree": 0.772,
            "min_data_in_leaf": 5929,
            "reg_alpha": 0.0188,
            "reg_lambda": 0.297,
            "subsample": 0.711,
            "n_estimators": 2000,
            "n_jobs": -1,
            "random_state": 1337,
        },
        "target_transform": {
            "type": "subtract_benchmark_inverse_normal",
            "benchmark_col": "v52_lgbm_ender20",
            "era_col": "era",
            "per_era": True,
            "scale": 0.07,
        },
    },
    "training": {
        "cv": {
            "enabled": True,
            "n_splits": 5,
            "embargo": 13,
            "mode": "expanding",
            "min_train_size": 0,
        },
    },
    "preprocessing": {"missing_value": 2.0, "nan_missing_all_twos": False},
    "output": {
        "output_dir": "baselines",
        "results_name": "lgbm_ender20_downsampled_invnorm",
    },
}
```

**Step 2: Create residualized transform config**

```python
# LGBM trained on residualized target (orthogonal to benchmark)
CONFIG = {
    "data": {
        "data_version": "v5.2",
        "feature_set": "all",
        "target_col": "target",
        "era_col": "era",
        "embargo_eras": 13,
        "full_data_path": "numerai/v5.2/downsampled_full.parquet",
        "benchmark_data_path": "numerai/v5.2/downsampled_full_benchmark_models.parquet",
        "benchmark_model": "v52_lgbm_ender20",
    },
    "model": {
        "type": "LGBMRegressor",
        "x_groups": ["features", "era", "benchmark_models"],
        "params": {
            "learning_rate": 0.0899,
            "max_depth": 6,
            "num_leaves": 173,
            "colsample_bytree": 0.772,
            "min_data_in_leaf": 5929,
            "reg_alpha": 0.0188,
            "reg_lambda": 0.297,
            "subsample": 0.711,
            "n_estimators": 2000,
            "n_jobs": -1,
            "random_state": 1337,
        },
        "target_transform": {
            "type": "residual_to_benchmark",
            "benchmark_col": "v52_lgbm_ender20",
            "era_col": "era",
            "per_era": True,
            "fit_intercept": True,
            "proportion": 1.0,
            "drop_na": True,
        },
    },
    "training": {
        "cv": {
            "enabled": True,
            "n_splits": 5,
            "embargo": 13,
            "mode": "expanding",
            "min_train_size": 0,
        },
    },
    "preprocessing": {"missing_value": 2.0, "nan_missing_all_twos": False},
    "output": {
        "output_dir": "baselines",
        "results_name": "lgbm_ender20_downsampled_residualized",
    },
}
```

**Step 3: Commit**
```bash
git add numerai/agents/baselines/configs/lgbm_ender20_downsampled_invnorm.py \
        numerai/agents/baselines/configs/lgbm_ender20_downsampled_residualized.py
git commit -m "feat: add LGBM target transform variant configs for ensemble diversity

- invnorm: inverse-normal benchmark subtraction
- residualized: fully orthogonalized to benchmark"
```

---

## Phase 3: Run HPO Sweeps

All HPO commands run from `numerai/` directory.

### Task 14: Run XGBoost HPO

**Step 1: Run HPO phase**
```bash
cd /Users/joakim/Documents/Projects/Numerai/example-scripts/numerai
python -m agents.code.modeling.hpo_v3_cli hpo \
    --model-type XGBRegressor \
    --n-trials 20 \
    --n-estimators 2000 \
    --study-name xgb_hpo_v3
```
Expected: Creates SQLite DB and best params JSON in `agents/baselines/hpo/`

**Step 2: Verify results**

Check the best params JSON file created. Expected BMC Sharpe > 0.1.

### Task 15: Run CatBoost HPO

**Step 1: Run HPO phase**
```bash
cd /Users/joakim/Documents/Projects/Numerai/example-scripts/numerai
python -m agents.code.modeling.hpo_v3_cli hpo \
    --model-type CatBoostRegressor \
    --n-trials 20 \
    --n-estimators 2000 \
    --study-name catboost_hpo_v3
```
Expected: Creates SQLite DB and best params JSON in `agents/baselines/hpo/`

---

## Phase 4: Train on Full Data

### Task 16: Train LGBM ensemble on full data

Uses existing HPO v3 best params.

**Step 1: Train ensemble**
```bash
cd /Users/joakim/Documents/Projects/Numerai/example-scripts/numerai
python -m agents.code.modeling.hpo_v4_cli ensemble \
    --best-params-file agents/baselines/hpo/lgbmregressor_hpo_v3_20260206_130238_best.json \
    --model-type LGBMRegressor \
    --n-estimators 2000 \
    --output-dir agents/baselines/models_lgbm
```

### Task 17: Train XGBoost ensemble on full data

**Step 1: Train ensemble** (using HPO results from Task 14)
```bash
cd /Users/joakim/Documents/Projects/Numerai/example-scripts/numerai
python -m agents.code.modeling.hpo_v4_cli ensemble \
    --best-params-file agents/baselines/hpo/<xgb_best_params_file>.json \
    --model-type XGBRegressor \
    --n-estimators 2000 \
    --output-dir agents/baselines/models_xgb
```

### Task 18: Train CatBoost ensemble on full data

**Step 1: Train ensemble** (using HPO results from Task 15)
```bash
cd /Users/joakim/Documents/Projects/Numerai/example-scripts/numerai
python -m agents.code.modeling.hpo_v4_cli ensemble \
    --best-params-file agents/baselines/hpo/<catboost_best_params_file>.json \
    --model-type CatBoostRegressor \
    --n-estimators 2000 \
    --output-dir agents/baselines/models_catboost
```

---

## Phase 5: TorchMLP Validation

### Task 19: Run TorchMLP on downsampled data

**Step 1: Run training pipeline**
```bash
cd /Users/joakim/Documents/Projects/Numerai/example-scripts/numerai
python -m agents.code.modeling --config agents/baselines/configs/torch_mlp_ender20_downsampled.py
```

**Step 2: Check results**

Read the results JSON. Compare CORR and BMC metrics to LGBM baseline.
Key question: Does the TorchMLP produce predictions with low correlation to tree models? (diversity check)

---

## Phase 6: Target Transform Experiments

### Task 20: Run LGBM with inverse-normal transform

**Step 1: Run training**
```bash
cd /Users/joakim/Documents/Projects/Numerai/example-scripts/numerai
python -m agents.code.modeling --config agents/baselines/configs/lgbm_ender20_downsampled_invnorm.py
```

**Step 2: Check results** - Compare to standard LGBM baseline

### Task 21: Run LGBM with residualized transform

**Step 1: Run training**
```bash
cd /Users/joakim/Documents/Projects/Numerai/example-scripts/numerai
python -m agents.code.modeling --config agents/baselines/configs/lgbm_ender20_downsampled_residualized.py
```

**Step 2: Check results** - Should have lower CORR but positive BMC (orthogonal signal)

---

## Phase 7: Ensemble Evaluation

### Task 22: Build cross-model ensemble from OOF predictions

Combine predictions from all model types using rank averaging.

**Step 1: Gather all OOF prediction parquet files**

Identify all predictions files from Tasks 16-21 outputs.

**Step 2: Build ensemble**
```bash
cd /Users/joakim/Documents/Projects/Numerai/example-scripts/numerai
python -m agents.code.modeling.ensemble \
    --predictions \
        agents/baselines/predictions/lgbm_*.parquet \
        agents/baselines/predictions/xgb_*.parquet \
        agents/baselines/predictions/catboost_*.parquet \
    --output agents/baselines/predictions/cross_model_ensemble.parquet \
    --method rank_average
```

**Step 3: Evaluate ensemble metrics**

Compute CORR, BMC, and Payout metrics on the ensemble. Compare to individual model metrics.
The ensemble should have:
- Similar or better CORR than the best individual model
- Better BMC than most individual models (diversification benefit)
- Lower max drawdown than individual models

### Task 23: Run full holdout evaluation

If ensemble looks good on OOF, run the final holdout evaluation:

```bash
cd /Users/joakim/Documents/Projects/Numerai/example-scripts/numerai
python -m agents.code.modeling.hpo_v4_cli holdout \
    --best-params-file agents/baselines/hpo/lgbmregressor_hpo_v3_20260206_130238_best.json \
    --neut-config-file agents/baselines/hpo/lgbmregressor_hpo_v3_20260206_130238_best_neutralization.json \
    --ensemble-dir agents/baselines/models_lgbm
```

---

## Execution Notes

- **Tasks 1-13**: Code/config changes that can be done immediately
- **Tasks 14-15**: HPO sweeps (~30-60 min each with 20 trials on downsampled data)
- **Tasks 16-18**: Full-data training (~1-3 hours each depending on model type)
- **Tasks 19-21**: Downsampled training (~10-30 min each)
- **Tasks 22-23**: Ensemble evaluation (~15-30 min)

Tasks 14+15 can run in parallel. Tasks 16-18 can run in parallel after their HPO completes. Tasks 19-21 can run in parallel.
