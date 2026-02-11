"""Era-stride CV for better hyperparameter generalization.

Strategy: Split eras into k strides (e.g., stride 0 = eras 0,3,6,9...,
stride 1 = eras 1,4,7,10...), then do expanding walk-forward within each stride.

This creates more diverse validation sets that test generalization across
different "time slices" of the data.

Example with 3 strides and 2 folds per stride = 6 total CV folds:
  Stride 0 (eras 0,3,6,...): Fold 1: train=[0,3,6...], val=[9,12,15...]
  Stride 0 (eras 0,3,6,...): Fold 2: train=[0,3,6,9...], val=[18,21,24...]
  Stride 1 (eras 1,4,7,...): Fold 1: train=[1,4,7...], val=[10,13,16...]
  ... etc.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def _era_sort_key(era):
    try:
        return int(era)
    except (TypeError, ValueError):
        return str(era)


def _sorted_unique_eras(eras) -> List:
    return sorted(set(eras), key=_era_sort_key)


def era_stride_cv_splits(
    eras: Sequence,
    n_strides: int = 3,
    folds_per_stride: int = 2,
    embargo: int = 16,
    min_train_eras: int = 20,
) -> List[Tuple[List, List]]:
    """Create era-stride CV splits for better generalization.

    Args:
        eras: Sequence of era values
        n_strides: Number of era strides (e.g., 3 means every 3rd era per stride)
        folds_per_stride: Number of walk-forward folds per stride
        embargo: Number of eras to embargo before validation
        min_train_eras: Minimum training eras required

    Returns:
        List of (train_eras, val_eras) tuples

    Example:
        With 300 eras, n_strides=3, folds_per_stride=2:
        - Stride 0: eras [0, 3, 6, 9, ...] (100 eras)
        - Stride 1: eras [1, 4, 7, 10, ...] (100 eras)
        - Stride 2: eras [2, 5, 8, 11, ...] (100 eras)
        Each stride gets 2 expanding walk-forward folds.
        Total: 3 × 2 = 6 folds
    """
    all_eras = _sorted_unique_eras(eras)
    n_eras = len(all_eras)

    if n_strides < 1:
        raise ValueError("n_strides must be >= 1")
    if folds_per_stride < 1:
        raise ValueError("folds_per_stride must be >= 1")

    splits = []

    for stride_idx in range(n_strides):
        # Get eras for this stride (every n_strides-th era starting at stride_idx)
        stride_eras = [all_eras[i] for i in range(stride_idx, n_eras, n_strides)]

        if len(stride_eras) < min_train_eras + embargo + 1:
            continue

        # Calculate fold boundaries for walk-forward within this stride
        # Reserve some eras for validation at the end
        n_stride_eras = len(stride_eras)
        val_eras_per_fold = max(1, (n_stride_eras - min_train_eras) // (folds_per_stride + 1))

        for fold_idx in range(folds_per_stride):
            # Expanding window: train on earlier eras, validate on later
            # Each fold adds more training data and validates on next chunk
            train_end_idx = min_train_eras + fold_idx * val_eras_per_fold
            val_start_idx = train_end_idx + embargo
            val_end_idx = val_start_idx + val_eras_per_fold

            if val_end_idx > n_stride_eras:
                break

            train_eras = stride_eras[:train_end_idx]
            val_eras = stride_eras[val_start_idx:val_end_idx]

            if len(train_eras) >= min_train_eras and len(val_eras) > 0:
                splits.append((train_eras, val_eras))

    return splits


def describe_splits(splits: List[Tuple[List, List]], n_strides: int = 3) -> dict:
    """Describe the CV splits for logging."""
    info = {
        "n_folds": len(splits),
        "n_strides": n_strides,
        "folds": [],
    }

    for i, (train_eras, val_eras) in enumerate(splits):
        fold_info = {
            "fold": i,
            "stride": i // (len(splits) // n_strides) if n_strides > 0 else 0,
            "train_eras": len(train_eras),
            "val_eras": len(val_eras),
            "train_era_range": f"{train_eras[0]}-{train_eras[-1]}" if train_eras else "",
            "val_era_range": f"{val_eras[0]}-{val_eras[-1]}" if val_eras else "",
        }
        info["folds"].append(fold_info)

    return info


def build_oof_predictions_strided(
    eras: Sequence,
    data_loader,
    model_type: str,
    model_params: dict,
    model_config: dict,
    n_strides: int = 3,
    folds_per_stride: int = 2,
    embargo: int = 16,
    min_train_eras: int = 20,
    max_train_samples: int | None = None,
    sample_seed: int = 1337,
    id_col: str | None = "id",
    era_col: str = "era",
    target_col: str = "target",
    feature_cols: list[str] | None = None,
    early_stopping_rounds: int | None = None,
    eval_metric: str = "rmse",
):
    """Build OOF predictions using era-stride CV.

    This is similar to build_oof_predictions but uses era-stride splits
    for better hyperparameter generalization.
    """
    import pandas as pd
    import numpy as np
    from agents.code.modeling.utils.model_factory import build_model
    from agents.code.modeling.utils.model_data import ModelDataBatch

    splits = era_stride_cv_splits(
        eras,
        n_strides=n_strides,
        folds_per_stride=folds_per_stride,
        embargo=embargo,
        min_train_eras=min_train_eras,
    )

    if not splits:
        raise ValueError("No valid CV splits could be created. Check n_strides, embargo, and data size.")

    predictions = []
    fold_info = []
    n_estimators_used = []

    for fold_idx, (train_eras, val_eras) in enumerate(splits):
        # Load data for this fold
        train_data = _load_data(data_loader, train_eras)
        val_data = _load_data(data_loader, val_eras)

        train_rows = len(train_data.X)
        val_rows = len(val_data.X)

        if train_rows == 0 or val_rows == 0:
            continue

        # Downsample if needed
        if max_train_samples and train_rows > max_train_samples:
            train_data = _subset_data(train_data, max_train_samples, sample_seed)
            train_rows = max_train_samples

        # Build model
        model = build_model(model_type, model_params, model_config, feature_cols=feature_cols)

        # Train with optional early stopping
        fit_params = {}
        actual_n_estimators = model_params.get("n_estimators", 2000)

        if early_stopping_rounds and model_type in ("LGBMRegressor", "XGBRegressor"):
            # Get the feature columns the model will use
            # The model wrapper filters to feature_cols, so we need to match that
            if feature_cols:
                # Use only feature_cols that exist in the data
                available_cols = [c for c in feature_cols if c in train_data.X.columns]
                train_X_for_fit = train_data.X[available_cols]
                val_X_for_eval = val_data.X[available_cols]
            else:
                # Filter to numeric columns only
                train_X_for_fit = train_data.X.select_dtypes(include=[np.number])
                val_X_for_eval = val_data.X.select_dtypes(include=[np.number])

            if model_type == "LGBMRegressor":
                fit_params["eval_set"] = [(val_X_for_eval, val_data.y)]
                fit_params["callbacks"] = [
                    _get_lgbm_early_stopping(early_stopping_rounds)
                ]
            elif model_type == "XGBRegressor":
                fit_params["eval_set"] = [(val_X_for_eval, val_data.y)]
                fit_params["verbose"] = False

            # Train with filtered features
            model.fit(train_X_for_fit, train_data.y, **fit_params)
        else:
            model.fit(train_data.X, train_data.y, **fit_params)

        # Track actual n_estimators used (for early stopping)
        # Note: best_iteration_ is 0-indexed, so add 1 for actual tree count
        if hasattr(model, "best_iteration_"):
            actual_n_estimators = model.best_iteration_ + 1
        elif hasattr(model, "_model") and hasattr(model._model, "best_iteration_"):
            actual_n_estimators = model._model.best_iteration_ + 1
        n_estimators_used.append(actual_n_estimators)

        # Generate predictions
        preds = model.predict(val_data.X)

        # Build predictions DataFrame
        fold_predictions = {}
        if id_col and val_data.id is not None:
            fold_predictions[id_col] = _as_array(val_data.id)
        fold_predictions[era_col] = _as_array(val_data.era)
        fold_predictions[target_col] = _as_array(val_data.y)
        fold_predictions["prediction"] = np.asarray(preds).ravel()
        fold_predictions["cv_fold"] = fold_idx
        predictions.append(pd.DataFrame(fold_predictions))

        fold_info.append({
            "fold": fold_idx,
            "stride": fold_idx // folds_per_stride,
            "train_eras": len(train_eras),
            "val_eras": len(val_eras),
            "train_rows": train_rows,
            "val_rows": val_rows,
            "n_estimators_used": actual_n_estimators,
        })

    if not predictions:
        raise ValueError("No CV folds produced predictions.")

    oof = pd.concat(predictions, ignore_index=True)

    cv_meta = {
        "cv_type": "era_stride",
        "n_strides": n_strides,
        "folds_per_stride": folds_per_stride,
        "embargo": embargo,
        "min_train_eras": min_train_eras,
        "folds_used": len(fold_info),
        "folds": fold_info,
        "avg_n_estimators": int(np.mean(n_estimators_used)) if n_estimators_used else None,
    }

    return oof, cv_meta


def _get_lgbm_early_stopping(stopping_rounds: int):
    """Get LightGBM early stopping callback."""
    try:
        from lightgbm import early_stopping
        return early_stopping(stopping_rounds=stopping_rounds, verbose=False)
    except ImportError:
        return None


def _load_data(data_loader, eras):
    """Load data for given eras."""
    if hasattr(data_loader, "load"):
        return data_loader.load(eras)
    return data_loader(eras)


def _subset_data(data, max_samples: int, seed: int):
    """Randomly subset data."""
    from agents.code.modeling.utils.model_data import ModelDataBatch
    import numpy as np

    total = len(data.X)
    if total <= max_samples:
        return data

    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=max_samples, replace=False)

    return ModelDataBatch(
        X=_subset_value(data.X, indices),
        y=_subset_value(data.y, indices),
        era=_subset_value(data.era, indices),
        id=_subset_value(data.id, indices) if data.id is not None else None,
    )


def _subset_value(value, indices):
    if value is None:
        return None
    if hasattr(value, "iloc"):
        return value.iloc[indices]
    return value[indices]


def _as_array(values):
    import numpy as np
    if hasattr(values, "to_numpy"):
        return values.to_numpy()
    return np.asarray(values)
