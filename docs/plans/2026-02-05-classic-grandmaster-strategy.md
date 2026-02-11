# Classic Tournament Grandmaster 2026 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a winning ensemble strategy for Numerai Classic Tournament Grandmaster 2026

**Architecture:** Train diverse models (LGBM, XGBoost, CatBoost) on full v5.2 data, combine via rank averaging, upload to multiple model slots for maximum coverage and diversification.

**Tech Stack:** Python 3.12, LightGBM, XGBoost, CatBoost, numerapi, pandas, numpy

---

## Phase 1: Fix Infrastructure & Add Models

### Task 1: Add XGBoost Model Wrapper

**Files:**
- Create: `numerai/agents/code/modeling/models/xgboost_regressor.py`
- Modify: `numerai/agents/code/modeling/utils/model_factory.py`

**Step 1: Create XGBoost wrapper**

```python
# numerai/agents/code/modeling/models/xgboost_regressor.py
from __future__ import annotations


class XGBRegressor:
    """Minimal wrapper exposing fit/predict for the training pipeline."""

    def __init__(self, feature_cols: list[str] | None = None, **params):
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for XGBRegressor. Install with `.venv/bin/pip install xgboost`."
            ) from exc
        self._xgb = xgb
        self._params = dict(params)
        self._model = xgb.XGBRegressor(**params)
        self._feature_cols = feature_cols

    def fit(self, X, y, **kwargs):
        X = self._filter_features(X, self._feature_cols)
        self._model.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        X = self._filter_features(X, self._feature_cols)
        return self._model.predict(X)

    @staticmethod
    def _filter_features(X, feature_cols):
        if not feature_cols or not hasattr(X, "columns"):
            return X
        missing = [col for col in feature_cols if col not in X.columns]
        if missing:
            raise ValueError(
                f"Missing feature columns for XGBRegressor: {missing[:5]}"
                + ("..." if len(missing) > 5 else "")
            )
        return X[feature_cols]

    def __getattr__(self, name: str):
        return getattr(self._model, name)
```

**Step 2: Register XGBoost in model factory**

Add to `model_factory.py` after line 16:
```python
    elif model_type == "XGBRegressor":
        from agents.code.modeling.models.xgboost_regressor import XGBRegressor
        model = XGBRegressor(feature_cols=feature_cols, **model_params)
```

**Step 3: Verify import works**

Run: `cd /Users/joakim/Documents/Projects/Numerai/example-scripts && python -c "from agents.code.modeling.utils.model_factory import build_model; print('OK')"`

**Step 4: Commit**

```bash
git add numerai/agents/code/modeling/models/xgboost_regressor.py numerai/agents/code/modeling/utils/model_factory.py
git commit -m "feat: add XGBoost model wrapper"
```

---

### Task 2: Add CatBoost Model Wrapper

**Files:**
- Create: `numerai/agents/code/modeling/models/catboost_regressor.py`
- Modify: `numerai/agents/code/modeling/utils/model_factory.py`

**Step 1: Create CatBoost wrapper**

```python
# numerai/agents/code/modeling/models/catboost_regressor.py
from __future__ import annotations


class CatBoostRegressor:
    """Minimal wrapper exposing fit/predict for the training pipeline."""

    def __init__(self, feature_cols: list[str] | None = None, **params):
        try:
            from catboost import CatBoostRegressor as _CatBoostRegressor
        except ImportError as exc:
            raise ImportError(
                "catboost is required for CatBoostRegressor. Install with `.venv/bin/pip install catboost`."
            ) from exc
        self._CatBoostRegressor = _CatBoostRegressor
        # Set silent mode by default
        params.setdefault("verbose", 0)
        self._params = dict(params)
        self._model = _CatBoostRegressor(**params)
        self._feature_cols = feature_cols

    def fit(self, X, y, **kwargs):
        X = self._filter_features(X, self._feature_cols)
        self._model.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        X = self._filter_features(X, self._feature_cols)
        return self._model.predict(X)

    @staticmethod
    def _filter_features(X, feature_cols):
        if not feature_cols or not hasattr(X, "columns"):
            return X
        missing = [col for col in feature_cols if col not in X.columns]
        if missing:
            raise ValueError(
                f"Missing feature columns for CatBoostRegressor: {missing[:5]}"
                + ("..." if len(missing) > 5 else "")
            )
        return X[feature_cols]

    def __getattr__(self, name: str):
        return getattr(self._model, name)
```

**Step 2: Register CatBoost in model factory**

Add to `model_factory.py`:
```python
    elif model_type == "CatBoostRegressor":
        from agents.code.modeling.models.catboost_regressor import CatBoostRegressor
        model = CatBoostRegressor(feature_cols=feature_cols, **model_params)
```

**Step 3: Commit**

```bash
git add numerai/agents/code/modeling/models/catboost_regressor.py numerai/agents/code/modeling/utils/model_factory.py
git commit -m "feat: add CatBoost model wrapper"
```

---

### Task 3: Install Dependencies

**Step 1: Install XGBoost and CatBoost**

Run:
```bash
cd /Users/joakim/Documents/Projects/Numerai/example-scripts
.venv/bin/pip install xgboost catboost
```

---

## Phase 2: Create Baseline Configs

### Task 4: Create XGBoost Baseline Config

**Files:**
- Create: `numerai/agents/baselines/configs/xgb_ender20_baseline.py`

**Step 1: Create config**

```python
# numerai/agents/baselines/configs/xgb_ender20_baseline.py
CONFIG = {
    'data': {
        'data_version': 'v5.2',
        'feature_set': 'all',
        'target_col': 'target',
        'era_col': 'era',
        'embargo_eras': 13,
    },
    'model': {
        'type': 'XGBRegressor',
        'x_groups': ['features', 'era', 'benchmark_models'],
        'params': {
            'n_estimators': 2000,
            'max_depth': 6,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 1337,
            'n_jobs': -1,
        },
    },
    'training': {
        'cv': {
            'enabled': True,
            'n_splits': 5,
            'embargo': 13,
            'mode': 'expanding',
            'min_train_size': 0,
        },
    },
    'preprocessing': {
        'missing_value': 2.0,
        'nan_missing_all_twos': False,
    },
    'output': {
        'output_dir': 'baselines',
        'results_name': 'xgb_ender20_baseline',
    },
}
```

**Step 2: Commit**

```bash
git add numerai/agents/baselines/configs/xgb_ender20_baseline.py
git commit -m "feat: add XGBoost baseline config"
```

---

### Task 5: Create CatBoost Baseline Config

**Files:**
- Create: `numerai/agents/baselines/configs/catboost_ender20_baseline.py`

**Step 1: Create config**

```python
# numerai/agents/baselines/configs/catboost_ender20_baseline.py
CONFIG = {
    'data': {
        'data_version': 'v5.2',
        'feature_set': 'all',
        'target_col': 'target',
        'era_col': 'era',
        'embargo_eras': 13,
    },
    'model': {
        'type': 'CatBoostRegressor',
        'x_groups': ['features', 'era', 'benchmark_models'],
        'params': {
            'iterations': 2000,
            'depth': 6,
            'learning_rate': 0.03,
            'l2_leaf_reg': 3.0,
            'random_seed': 1337,
            'thread_count': -1,
            'verbose': 0,
        },
    },
    'training': {
        'cv': {
            'enabled': True,
            'n_splits': 5,
            'embargo': 13,
            'mode': 'expanding',
            'min_train_size': 0,
        },
    },
    'preprocessing': {
        'missing_value': 2.0,
        'nan_missing_all_twos': False,
    },
    'output': {
        'output_dir': 'baselines',
        'results_name': 'catboost_ender20_baseline',
    },
}
```

**Step 2: Commit**

```bash
git add numerai/agents/baselines/configs/catboost_ender20_baseline.py
git commit -m "feat: add CatBoost baseline config"
```

---

## Phase 3: Create Ensemble Infrastructure

### Task 6: Create Ensemble Builder Script

**Files:**
- Create: `numerai/agents/code/modeling/ensemble.py`

**Step 1: Create ensemble script**

```python
# numerai/agents/code/modeling/ensemble.py
"""
Ensemble builder for combining multiple model predictions.
Usage: python -m agents.code.modeling.ensemble --predictions pred1.parquet pred2.parquet --output ensemble.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def rank_average(predictions_list: list[pd.DataFrame], era_col: str = "era") -> pd.DataFrame:
    """Combine predictions via per-era rank averaging."""
    # Align all predictions on same index
    base = predictions_list[0].copy()
    all_preds = []

    for i, df in enumerate(predictions_list):
        pred_col = f"pred_{i}"
        base[pred_col] = df["prediction"].values
        all_preds.append(pred_col)

    # Rank within each era, then average ranks
    def era_rank_avg(group):
        ranks = []
        for col in all_preds:
            ranks.append(group[col].rank(pct=True))
        return pd.Series(np.mean(ranks, axis=0), index=group.index)

    base["prediction"] = base.groupby(era_col, group_keys=False).apply(era_rank_avg)

    # Clean up temp columns
    base = base.drop(columns=all_preds)

    return base


def simple_average(predictions_list: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine predictions via simple averaging."""
    base = predictions_list[0].copy()
    pred_sum = base["prediction"].copy()

    for df in predictions_list[1:]:
        pred_sum += df["prediction"].values

    base["prediction"] = pred_sum / len(predictions_list)
    return base


def main():
    parser = argparse.ArgumentParser(description="Ensemble model predictions")
    parser.add_argument("--predictions", nargs="+", required=True, help="Prediction parquet files")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument("--method", default="rank_average", choices=["rank_average", "simple_average"])
    parser.add_argument("--era-col", default="era", help="Era column name")
    args = parser.parse_args()

    predictions_list = [pd.read_parquet(p) for p in args.predictions]

    if args.method == "rank_average":
        result = rank_average(predictions_list, args.era_col)
    else:
        result = simple_average(predictions_list)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    print(f"Saved ensemble predictions to {args.output}")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add numerai/agents/code/modeling/ensemble.py
git commit -m "feat: add ensemble builder for rank averaging predictions"
```

---

## Phase 4: Train Models (Run These Manually)

### Task 7: Train LGBM Baseline

**Step 1: Run training**

```bash
cd /Users/joakim/Documents/Projects/Numerai/example-scripts
python -m agents.code.modeling numerai/agents/baselines/configs/deep_lgbm_ender20_baseline.py
```

Expected output: Predictions saved to `baselines/predictions/deep_lgbm_ender20_baseline.parquet`

---

### Task 8: Train XGBoost Baseline

**Step 1: Run training**

```bash
cd /Users/joakim/Documents/Projects/Numerai/example-scripts
python -m agents.code.modeling numerai/agents/baselines/configs/xgb_ender20_baseline.py
```

---

### Task 9: Train CatBoost Baseline

**Step 1: Run training**

```bash
cd /Users/joakim/Documents/Projects/Numerai/example-scripts
python -m agents.code.modeling numerai/agents/baselines/configs/catboost_ender20_baseline.py
```

---

### Task 10: Create Ensemble

**Step 1: Build ensemble from all three models**

```bash
python -m agents.code.modeling.ensemble \
  --predictions \
    baselines/predictions/deep_lgbm_ender20_baseline.parquet \
    baselines/predictions/xgb_ender20_baseline.parquet \
    baselines/predictions/catboost_ender20_baseline.parquet \
  --output baselines/predictions/ensemble_rank_avg.parquet \
  --method rank_average
```

---

## Phase 5: Upload Models

### Task 11: Create Model Upload Script

**Files:**
- Create: `numerai/agents/code/upload_model.py`

**Step 1: Create upload script**

```python
# numerai/agents/code/upload_model.py
"""
Upload trained model predictions to Numerai.
Usage: python -m agents.code.upload_model --predictions pred.parquet --model-name mymodel
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd
import numerapi


def upload_predictions(predictions_path: str, model_name: str, public_id: str, secret_key: str):
    """Upload predictions to Numerai tournament."""
    napi = numerapi.NumerAPI(public_id=public_id, secret_key=secret_key)

    # Get model ID
    models = napi.get_models()
    if model_name not in models:
        print(f"Model '{model_name}' not found. Available models: {list(models.keys())[:10]}...")
        return False

    model_id = models[model_name]

    # Load predictions and format for submission
    df = pd.read_parquet(predictions_path)

    # Numerai expects 'id' and 'prediction' columns
    if 'id' not in df.columns:
        print("Warning: 'id' column not found, using index")
        df = df.reset_index()
        df = df.rename(columns={'index': 'id'})

    submission = df[['id', 'prediction']].copy()

    # Save to temp CSV and upload
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        submission.to_csv(f.name, index=False)
        print(f"Uploading to model '{model_name}' (id: {model_id})...")
        napi.upload_predictions(f.name, model_id=model_id)
        print(f"Successfully uploaded predictions to {model_name}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Upload predictions to Numerai")
    parser.add_argument("--predictions", required=True, help="Prediction parquet file")
    parser.add_argument("--model-name", required=True, help="Numerai model name")
    parser.add_argument("--public-id", default="W4CDC5RCN4NKFMM5TGKEIRFOFBWBYG7Z")
    parser.add_argument("--secret-key", default="3HB7LJS3NQOEUTVDGS3M7OLJMNKXDPE6XRLUSB2X2XIMN76YUZEKKTJXFKAMKZAI")
    args = parser.parse_args()

    upload_predictions(args.predictions, args.model_name, args.public_id, args.secret_key)


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add numerai/agents/code/upload_model.py
git commit -m "feat: add model upload script"
```

---

### Task 12: Upload to Model Slots

**Recommended model slots for diversification:**

| Model | Target Slot | Rationale |
|-------|-------------|-----------|
| LGBM ensemble | `alpha_ensemble` | Primary ensemble slot |
| XGBoost | `alphasamurai` | XGBoost dedicated |
| CatBoost | `alphaninja` | CatBoost dedicated |
| Rank-avg ensemble | `alphaensemble` | Best combined model |

**Step 1: Upload ensemble to alpha_ensemble**

```bash
python -m agents.code.upload_model \
  --predictions baselines/predictions/ensemble_rank_avg.parquet \
  --model-name alpha_ensemble
```

**Step 2: Upload to additional slots for diversification**

```bash
python -m agents.code.upload_model --predictions baselines/predictions/deep_lgbm_ender20_baseline.parquet --model-name alphasamurai
python -m agents.code.upload_model --predictions baselines/predictions/xgb_ender20_baseline.parquet --model-name alphaninja
python -m agents.code.upload_model --predictions baselines/predictions/catboost_ender20_baseline.parquet --model-name alphaensemble
```

---

## Summary

**Models to train:**
1. LGBM (deep_lgbm_ender20_baseline) - Gradient boosting baseline
2. XGBoost (xgb_ender20_baseline) - Alternative gradient boosting
3. CatBoost (catboost_ender20_baseline) - Handles categoricals well

**Ensemble strategy:**
- Rank averaging across all three models
- Per-era ranking ensures stable combination

**Upload targets:**
- `alpha_ensemble` - Main ensemble
- `alphasamurai`, `alphaninja`, `alphaensemble` - Individual models for diversification

**Expected improvement:**
- Single model BMC: ~0.0001 to 0.0005
- Ensemble BMC: ~0.0005 to 0.001 (diversification benefit)
