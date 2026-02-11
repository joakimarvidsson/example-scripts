"""HPO v3: Three-phase pipeline with proper data splits.

Phase 1 - HPO: Era-stride CV on downsampled data (eras ≤ 800)
Phase 2 - Neutralization: Tune feature neutralization on full data (eras 816-1116)
Phase 3 - Holdout: Final evaluation on full data (eras ≥ 1133)

16-era embargo between each phase.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from optuna.storages import RDBStorage

from agents.code.metrics import numerai_metrics


# Default era boundaries (with 16-era embargo)
DEFAULT_HPO_MAX_ERA = 800
DEFAULT_NEUT_MIN_ERA = 816
DEFAULT_NEUT_MAX_ERA = 1116
DEFAULT_HOLDOUT_MIN_ERA = 1133


# Search spaces (same as v2, no n_estimators)
SEARCH_SPACES_V3 = {
    "LGBMRegressor": {
        "learning_rate": ("log_float", 0.005, 0.1),
        "max_depth": ("int", 4, 12),
        "num_leaves": ("int", 31, 512),
        "colsample_bytree": ("float", 0.3, 1.0),
        "min_data_in_leaf": ("int", 1000, 10000),
        "reg_alpha": ("log_float", 1e-6, 10.0),
        "reg_lambda": ("log_float", 1e-6, 10.0),
        "subsample": ("float", 0.6, 1.0),
    },
    "XGBRegressor": {
        "learning_rate": ("log_float", 0.005, 0.1),
        "max_depth": ("int", 4, 12),
        "colsample_bytree": ("float", 0.3, 1.0),
        "subsample": ("float", 0.6, 1.0),
        "min_child_weight": ("int", 1, 100),
        "reg_alpha": ("log_float", 1e-6, 10.0),
        "reg_lambda": ("log_float", 1e-6, 10.0),
        "gamma": ("log_float", 1e-6, 1.0),
    },
    "CatBoostRegressor": {
        "learning_rate": ("log_float", 0.005, 0.1),
        "depth": ("int", 4, 10),
        "l2_leaf_reg": ("log_float", 1e-2, 10.0),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bylevel": ("float", 0.3, 1.0),
        "min_data_in_leaf": ("int", 1, 100),
        "random_strength": ("log_float", 1e-3, 10.0),
    },
}


def _sample_param(trial: optuna.Trial, name: str, spec: tuple) -> Any:
    """Sample a hyperparameter based on its specification."""
    param_type, *args = spec
    if param_type == "int":
        return trial.suggest_int(name, args[0], args[1])
    elif param_type == "float":
        return trial.suggest_float(name, args[0], args[1])
    elif param_type == "log_float":
        return trial.suggest_float(name, args[0], args[1], log=True)
    elif param_type == "categorical":
        return trial.suggest_categorical(name, args[0])
    else:
        raise ValueError(f"Unknown param type: {param_type}")


def sample_params(
    trial: optuna.Trial,
    model_type: str,
    search_space: dict | None = None,
) -> dict:
    """Sample hyperparameters for a given model type."""
    space = search_space or SEARCH_SPACES_V3.get(model_type, {})
    params = {}
    for name, spec in space.items():
        params[name] = _sample_param(trial, name, spec)
    return params


def load_data_for_phase(
    phase: str,
    data_version: str = "v5.2",
    feature_set: str = "all",
    downsampled_path: str | None = None,
    full_path: str | None = None,
    benchmark_path: str | None = None,
    hpo_max_era: int = DEFAULT_HPO_MAX_ERA,
    neut_min_era: int = DEFAULT_NEUT_MIN_ERA,
    neut_max_era: int = DEFAULT_NEUT_MAX_ERA,
    holdout_min_era: int = DEFAULT_HOLDOUT_MIN_ERA,
    era_col: str = "era",
    target_col: str = "target",
    id_col: str = "id",
) -> tuple[pd.DataFrame, list[str]]:
    """Load data for a specific phase.

    Args:
        phase: One of 'hpo', 'neutralization', 'holdout'
        data_version: Numerai data version
        feature_set: Feature set to use
        downsampled_path: Path to downsampled data (for HPO)
        full_path: Path to full data (for neutralization/holdout)
        benchmark_path: Path to benchmark models
        hpo_max_era: Max era for HPO phase
        neut_min_era: Min era for neutralization phase
        neut_max_era: Max era for neutralization phase
        holdout_min_era: Min era for holdout phase

    Returns:
        Tuple of (dataframe, feature_columns)
    """
    from numerapi import NumerAPI
    from agents.code.modeling.utils.data import (
        attach_benchmark_models,
        load_features,
    )

    napi = NumerAPI()
    features = load_features(napi, data_version, feature_set)

    # Determine which data file and era filter to use
    if phase == "hpo":
        # Use downsampled data for HPO (faster)
        data_path = downsampled_path or f"numerai/{data_version}/downsampled_full.parquet"
        era_filter = lambda era: int(era) <= hpo_max_era
        phase_desc = f"HPO (eras ≤ {hpo_max_era}, downsampled)"
    elif phase == "neutralization":
        # Use full data for neutralization tuning
        data_path = full_path or f"numerai/{data_version}/full.parquet"
        era_filter = lambda era: neut_min_era <= int(era) <= neut_max_era
        phase_desc = f"Neutralization (eras {neut_min_era}-{neut_max_era}, full)"
    elif phase == "holdout":
        # Use full data for holdout
        data_path = full_path or f"numerai/{data_version}/full.parquet"
        era_filter = lambda era: int(era) >= holdout_min_era
        phase_desc = f"Holdout (eras ≥ {holdout_min_era}, full)"
    else:
        raise ValueError(f"Unknown phase: {phase}")

    print(f"Loading {phase_desc}...")

    # Load data
    cols_to_load = [era_col, target_col] + features
    if id_col:
        cols_to_load = [id_col] + cols_to_load

    df = pd.read_parquet(data_path, columns=cols_to_load)

    # Filter to phase eras
    mask = df[era_col].apply(era_filter)
    df = df[mask].reset_index(drop=True)

    n_eras = df[era_col].nunique()
    print(f"  Loaded {len(df):,} rows, {len(features)} features, {n_eras} eras")

    # Attach benchmark models if path provided
    benchmark_cols = []
    if benchmark_path:
        try:
            bench_df = pd.read_parquet(benchmark_path)
            bench_df = bench_df[bench_df[era_col].apply(era_filter)].reset_index(drop=True)

            # Get benchmark columns (exclude id and era)
            benchmark_cols = [c for c in bench_df.columns if c not in [id_col, era_col, "id"]]

            if len(bench_df) == len(df):
                # Same length - can concat directly (row-aligned)
                for col in benchmark_cols:
                    df[col] = bench_df[col].values
                print(f"  Attached {len(benchmark_cols)} benchmark model columns (row-aligned)")
            else:
                # Different lengths - try to merge on era (for aggregated benchmarks)
                # or skip if not possible
                print(f"  Warning: Benchmark data length ({len(bench_df)}) != main data ({len(df)})")
                print(f"  Skipping benchmark models for this phase")
                benchmark_cols = []
        except Exception as e:
            print(f"  Warning: Could not load benchmark models: {e}")
            benchmark_cols = []

    return df, features, benchmark_cols


def create_hpo_objective(
    hpo_data: pd.DataFrame,
    features: list[str],
    benchmark_cols: list[str],
    model_type: str,
    era_col: str = "era",
    target_col: str = "target",
    id_col: str = "id",
    benchmark_model: str = "v52_lgbm_ender20",
    data_version: str = "v5.2",
    metric: str = "bmc_sharpe",
    search_space: dict | None = None,
    fixed_params: dict | None = None,
    n_strides: int = 3,
    folds_per_stride: int = 2,
    embargo: int = 16,
    min_train_eras: int = 20,
    n_estimators: int = 2000,
) -> Callable[[optuna.Trial], float]:
    """Create HPO objective function with era-stride CV on HPO data only."""
    from agents.code.modeling.utils.era_stride_cv import era_stride_cv_splits
    from agents.code.modeling.utils.model_factory import build_model
    from agents.code.modeling.utils.model_data import ModelDataBatch

    fixed = fixed_params or {}
    all_eras = hpo_data[era_col].unique()

    # Build x_cols
    x_cols = features + benchmark_cols

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        sampled_params = sample_params(trial, model_type, search_space)
        model_params = {**sampled_params, **fixed}

        # CatBoost uses 'iterations' instead of 'n_estimators'
        if model_type == "CatBoostRegressor":
            model_params["iterations"] = n_estimators
            model_params.setdefault("verbose", 0)
        else:
            model_params["n_estimators"] = n_estimators

        # Add standard params
        if model_type == "LGBMRegressor":
            model_params.setdefault("n_jobs", -1)
            model_params.setdefault("random_state", 1337)
            model_params.setdefault("verbose", -1)
        elif model_type == "XGBRegressor":
            model_params.setdefault("n_jobs", -1)
            model_params.setdefault("random_state", 1337)
            model_params.setdefault("verbosity", 0)

        try:
            # Get era-stride CV splits
            splits = era_stride_cv_splits(
                all_eras,
                n_strides=n_strides,
                folds_per_stride=folds_per_stride,
                embargo=embargo,
                min_train_eras=min_train_eras,
            )

            if not splits:
                return float("-inf")

            predictions = []

            for fold_idx, (train_eras, val_eras) in enumerate(splits):
                # Get train/val data
                train_mask = hpo_data[era_col].isin(train_eras)
                val_mask = hpo_data[era_col].isin(val_eras)

                train_X = hpo_data.loc[train_mask, x_cols]
                train_y = hpo_data.loc[train_mask, target_col]
                val_X = hpo_data.loc[val_mask, x_cols]
                val_y = hpo_data.loc[val_mask, target_col]

                # Build and train model
                model = build_model(model_type, model_params, {}, feature_cols=features)
                model.fit(train_X, train_y)

                # Predict
                preds = model.predict(val_X)

                # Build predictions DataFrame
                fold_pred = pd.DataFrame({
                    id_col: hpo_data.loc[val_mask, id_col].values if id_col else range(len(preds)),
                    era_col: hpo_data.loc[val_mask, era_col].values,
                    target_col: val_y.values,
                    "prediction": preds,
                })
                predictions.append(fold_pred)

            # Combine predictions
            oof = pd.concat(predictions, ignore_index=True)

            # Save to temp file for metrics
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                temp_path = Path(f.name)
            oof.to_parquet(temp_path, index=False)

            try:
                # Calculate metrics
                summaries = numerai_metrics.summarize_prediction_file_with_bmc(
                    temp_path,
                    ["prediction"],
                    target_col,
                    data_version,
                    benchmark_model=benchmark_model,
                    era_col=era_col,
                    id_col=id_col,
                )

                # Extract metric
                if metric == "bmc_sharpe":
                    value = summaries["bmc"].loc["prediction", "sharpe"]
                elif metric == "bmc_mean":
                    value = summaries["bmc"].loc["prediction", "mean"]
                elif metric == "corr_sharpe":
                    value = summaries["corr"].loc["prediction", "sharpe"]
                elif metric == "corr_mean":
                    value = summaries["corr"].loc["prediction", "mean"]
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                # Store metrics
                trial.set_user_attr("corr_mean", float(summaries["corr"].loc["prediction", "mean"]))
                trial.set_user_attr("corr_sharpe", float(summaries["corr"].loc["prediction", "sharpe"]))
                trial.set_user_attr("bmc_mean", float(summaries["bmc"].loc["prediction", "mean"]))
                trial.set_user_attr("bmc_sharpe", float(summaries["bmc"].loc["prediction", "sharpe"]))
                trial.set_user_attr("folds_used", len(splits))

            finally:
                temp_path.unlink(missing_ok=True)

            return float(value)

        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            import traceback
            traceback.print_exc()
            return float("-inf")

    return objective


def run_hpo_phase(
    config_path: Path | None = None,
    n_trials: int = 20,
    study_name: str | None = None,
    db_path: str | Path | None = None,
    metric: str = "bmc_sharpe",
    # Data settings
    data_version: str = "v5.2",
    feature_set: str = "all",
    downsampled_path: str | None = None,
    benchmark_path: str | None = None,
    model_type: str = "LGBMRegressor",
    # Era splits
    hpo_max_era: int = DEFAULT_HPO_MAX_ERA,
    # CV settings
    n_strides: int = 3,
    folds_per_stride: int = 2,
    embargo: int = 16,
    min_train_eras: int = 20,
    n_estimators: int = 2000,
    # WandB
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    timeout: float | None = None,
) -> tuple[optuna.Study, dict]:
    """Run HPO phase on downsampled data (eras ≤ hpo_max_era).

    Returns:
        Tuple of (study, best_params)
    """
    from agents.code.modeling.utils.config import load_config
    from agents.code.modeling.utils.pipeline import resolve_output_locations

    # Load config if provided
    if config_path:
        config = load_config(config_path)
        data_config = config.get("data", {})
        model_config = config.get("model", {})

        data_version = data_config.get("data_version", data_version)
        feature_set = data_config.get("feature_set", feature_set)
        downsampled_path = data_config.get("full_data_path", downsampled_path)
        benchmark_path = data_config.get("benchmark_data_path", benchmark_path)
        model_type = model_config.get("type", model_type)

        output_dir, _, _, _ = resolve_output_locations(config, None)
    else:
        output_dir = Path("agents/baselines")

    hpo_dir = output_dir / "hpo"
    hpo_dir.mkdir(parents=True, exist_ok=True)

    # Setup study
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if study_name is None:
        study_name = f"{model_type.lower()}_hpo_v3_{timestamp}"

    if db_path is None:
        db_path = hpo_dir / f"{study_name}.db"

    storage = RDBStorage(
        url=f"sqlite:///{db_path}",
        engine_kwargs={"connect_args": {"check_same_thread": False}},
    )

    print(f"\n{'='*60}")
    print(f"HPO Phase (v3)")
    print(f"{'='*60}")
    print(f"Study: {study_name}")
    print(f"Database: {db_path}")
    print(f"Metric: {metric}")
    print(f"Trials: {n_trials}")
    print(f"Era range: ≤ {hpo_max_era} (downsampled)")
    print(f"CV: {n_strides} strides × {folds_per_stride} folds")
    print(f"n_estimators: {n_estimators} (fixed)")
    print(f"{'='*60}\n")

    # Load HPO data
    hpo_data, features, benchmark_cols = load_data_for_phase(
        "hpo",
        data_version=data_version,
        feature_set=feature_set,
        downsampled_path=downsampled_path,
        benchmark_path=benchmark_path,
        hpo_max_era=hpo_max_era,
    )

    # Create objective
    objective = create_hpo_objective(
        hpo_data,
        features,
        benchmark_cols,
        model_type,
        metric=metric,
        n_strides=n_strides,
        folds_per_stride=folds_per_stride,
        embargo=embargo,
        min_train_eras=min_train_eras,
        n_estimators=n_estimators,
        data_version=data_version,
    )

    # Create study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    # Initialize WandB if requested
    wandb_run = None
    if wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=study_name,
                config={
                    "phase": "hpo",
                    "model_type": model_type,
                    "n_trials": n_trials,
                    "metric": metric,
                    "hpo_max_era": hpo_max_era,
                    "n_strides": n_strides,
                    "folds_per_stride": folds_per_stride,
                    "n_estimators": n_estimators,
                },
                reinit=True,
            )
            print(f"WandB: {wandb_run.url}")
        except Exception as e:
            print(f"WandB init failed: {e}")

    # Run optimization
    print(f"\nStarting HPO with {n_trials} trials...")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    # Results
    best_params = study.best_params
    best_value = study.best_value

    print(f"\n{'='*60}")
    print(f"HPO Phase Complete!")
    print(f"Best {metric}: {best_value:.6f}")
    print(f"Best params: {json.dumps(best_params, indent=2)}")
    print(f"{'='*60}")

    # Save results
    results = {
        "phase": "hpo",
        "study_name": study_name,
        "metric": metric,
        "best_value": best_value,
        "best_params": best_params,
        "n_trials": len(study.trials),
        "best_trial_number": study.best_trial.number,
        "best_trial_attrs": study.best_trial.user_attrs,
        "config": {
            "hpo_max_era": hpo_max_era,
            "n_strides": n_strides,
            "folds_per_stride": folds_per_stride,
            "n_estimators": n_estimators,
        },
    }

    results_file = hpo_dir / f"{study_name}_best.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_file}")

    if wandb_run:
        wandb_run.summary["best_value"] = best_value
        wandb_run.finish()

    return study, best_params


def evaluate_on_neutralization(
    best_params: dict,
    model_type: str = "LGBMRegressor",
    data_version: str = "v5.2",
    feature_set: str = "all",
    downsampled_path: str | None = None,
    full_path: str | None = None,
    benchmark_path: str | None = None,
    hpo_max_era: int = DEFAULT_HPO_MAX_ERA,
    neut_min_era: int = DEFAULT_NEUT_MIN_ERA,
    neut_max_era: int = DEFAULT_NEUT_MAX_ERA,
    n_estimators: int = 2000,
    neutralization_strengths: list[float] | None = None,
) -> dict:
    """Train on HPO data, evaluate different neutralization strengths on neutralization data.

    Returns:
        Dict with best neutralization strength and metrics
    """
    from agents.code.modeling.utils.model_factory import build_model

    if neutralization_strengths is None:
        neutralization_strengths = [0.0, 0.25, 0.5, 0.75, 1.0]

    print(f"\n{'='*60}")
    print(f"Neutralization Tuning Phase")
    print(f"{'='*60}")
    print(f"Training on eras ≤ {hpo_max_era}")
    print(f"Evaluating on eras {neut_min_era}-{neut_max_era}")
    print(f"Neutralization strengths: {neutralization_strengths}")
    print(f"{'='*60}\n")

    # Load training data (HPO era range, but could use full for final training)
    train_data, features, benchmark_cols = load_data_for_phase(
        "hpo",
        data_version=data_version,
        feature_set=feature_set,
        downsampled_path=downsampled_path,
        benchmark_path=benchmark_path,
        hpo_max_era=hpo_max_era,
    )

    # Load neutralization data
    neut_data, _, neut_bench_cols = load_data_for_phase(
        "neutralization",
        data_version=data_version,
        feature_set=feature_set,
        full_path=full_path,
        benchmark_path=benchmark_path,
        neut_min_era=neut_min_era,
        neut_max_era=neut_max_era,
    )

    x_cols = features + benchmark_cols

    # Build model params
    model_params = {**best_params}
    if model_type == "CatBoostRegressor":
        model_params["iterations"] = n_estimators
        model_params.setdefault("verbose", 0)
    else:
        model_params["n_estimators"] = n_estimators
    if model_type == "LGBMRegressor":
        model_params.setdefault("n_jobs", -1)
        model_params.setdefault("random_state", 1337)
        model_params.setdefault("verbose", -1)

    # Train model
    print("Training model on HPO data...")
    model = build_model(model_type, model_params, {}, feature_cols=features)
    model.fit(train_data[x_cols], train_data["target"])

    # Predict on neutralization data
    print("Predicting on neutralization data...")
    neut_preds = model.predict(neut_data[x_cols])
    neut_data = neut_data.copy()
    neut_data["raw_prediction"] = neut_preds

    # Evaluate different neutralization strengths
    results = []

    for strength in neutralization_strengths:
        if strength == 0.0:
            neut_data["prediction"] = neut_data["raw_prediction"]
        else:
            # Feature neutralization
            neut_data["prediction"] = _neutralize_predictions(
                neut_data["raw_prediction"].values,
                neut_data[features].values,
                strength,
            )

        # Save to temp file and compute metrics
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_path = Path(f.name)

        neut_data[["id", "era", "target", "prediction"]].to_parquet(temp_path, index=False)

        try:
            summaries = numerai_metrics.summarize_prediction_file_with_bmc(
                temp_path,
                ["prediction"],
                "target",
                data_version,
                era_col="era",
                id_col="id",
            )

            result = {
                "strength": strength,
                "corr_mean": float(summaries["corr"].loc["prediction", "mean"]),
                "corr_sharpe": float(summaries["corr"].loc["prediction", "sharpe"]),
                "bmc_mean": float(summaries["bmc"].loc["prediction", "mean"]),
                "bmc_sharpe": float(summaries["bmc"].loc["prediction", "sharpe"]),
            }
            results.append(result)
            print(f"  Strength {strength:.2f}: BMC Sharpe = {result['bmc_sharpe']:.4f}, Corr Sharpe = {result['corr_sharpe']:.4f}")

        finally:
            temp_path.unlink(missing_ok=True)

    # Find best neutralization strength
    best_result = max(results, key=lambda x: x["bmc_sharpe"])

    print(f"\nBest neutralization strength: {best_result['strength']}")
    print(f"  BMC Sharpe: {best_result['bmc_sharpe']:.4f}")
    print(f"  Corr Sharpe: {best_result['corr_sharpe']:.4f}")

    return {
        "best_strength": best_result["strength"],
        "best_metrics": best_result,
        "all_results": results,
    }


def _neutralize_predictions(
    predictions: np.ndarray,
    features: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Neutralize predictions to features."""
    if strength == 0.0:
        return predictions

    # Simple ridge regression neutralization
    from sklearn.linear_model import Ridge

    # Fit linear model: predictions ~ features
    model = Ridge(alpha=0.01)
    model.fit(features, predictions)

    # Get the component explained by features
    feature_component = model.predict(features)

    # Neutralize
    neutralized = predictions - strength * feature_component

    # Re-rank to [0, 1]
    from scipy.stats import rankdata
    neutralized = rankdata(neutralized) / len(neutralized)

    return neutralized


def evaluate_on_holdout(
    best_params: dict,
    best_neutralization_strength: float,
    model_type: str = "LGBMRegressor",
    data_version: str = "v5.2",
    feature_set: str = "all",
    downsampled_path: str | None = None,
    full_path: str | None = None,
    benchmark_path: str | None = None,
    hpo_max_era: int = DEFAULT_HPO_MAX_ERA,
    neut_max_era: int = DEFAULT_NEUT_MAX_ERA,
    holdout_min_era: int = DEFAULT_HOLDOUT_MIN_ERA,
    n_estimators: int = 2000,
) -> dict:
    """Final evaluation on holdout data. Only call this once!

    Trains on all data up to neut_max_era, evaluates on holdout.
    """
    from agents.code.modeling.utils.model_factory import build_model

    print(f"\n{'='*60}")
    print(f"HOLDOUT EVALUATION (FINAL)")
    print(f"{'='*60}")
    print(f"Training on eras ≤ {neut_max_era}")
    print(f"Evaluating on eras ≥ {holdout_min_era}")
    print(f"Neutralization strength: {best_neutralization_strength}")
    print(f"{'='*60}\n")

    # Load ALL training data (HPO + neutralization eras)
    # We need to load full data up to neut_max_era
    print("Loading training data (all eras up to neutralization)...")

    from numerapi import NumerAPI
    from agents.code.modeling.utils.data import load_features

    napi = NumerAPI()
    features = load_features(napi, data_version, feature_set)

    full_data_path = full_path or f"numerai/{data_version}/full.parquet"

    cols = ["id", "era", "target"] + features
    full_df = pd.read_parquet(full_data_path, columns=cols)
    full_df["era_int"] = full_df["era"].astype(int)

    train_df = full_df[full_df["era_int"] <= neut_max_era].copy()
    holdout_df = full_df[full_df["era_int"] >= holdout_min_era].copy()

    print(f"  Train: {len(train_df):,} rows, {train_df['era'].nunique()} eras")
    print(f"  Holdout: {len(holdout_df):,} rows, {holdout_df['era'].nunique()} eras")

    # Build model params
    model_params = {**best_params}
    if model_type == "CatBoostRegressor":
        model_params["iterations"] = n_estimators
        model_params.setdefault("verbose", 0)
    else:
        model_params["n_estimators"] = n_estimators
    if model_type == "LGBMRegressor":
        model_params.setdefault("n_jobs", -1)
        model_params.setdefault("random_state", 1337)
        model_params.setdefault("verbose", -1)

    # Train model
    print("\nTraining final model...")
    model = build_model(model_type, model_params, {}, feature_cols=features)
    model.fit(train_df[features], train_df["target"])

    # Predict on holdout
    print("Predicting on holdout...")
    holdout_preds = model.predict(holdout_df[features])
    holdout_df["raw_prediction"] = holdout_preds

    # Apply neutralization
    if best_neutralization_strength > 0:
        holdout_df["prediction"] = _neutralize_predictions(
            holdout_df["raw_prediction"].values,
            holdout_df[features].values,
            best_neutralization_strength,
        )
    else:
        holdout_df["prediction"] = holdout_df["raw_prediction"]

    # Compute metrics
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        temp_path = Path(f.name)

    holdout_df[["id", "era", "target", "prediction"]].to_parquet(temp_path, index=False)

    try:
        summaries = numerai_metrics.summarize_prediction_file_with_bmc(
            temp_path,
            ["prediction"],
            "target",
            data_version,
            era_col="era",
            id_col="id",
        )

        result = {
            "corr_mean": float(summaries["corr"].loc["prediction", "mean"]),
            "corr_sharpe": float(summaries["corr"].loc["prediction", "sharpe"]),
            "corr_max_dd": float(summaries["corr"].loc["prediction", "max_drawdown"]),
            "bmc_mean": float(summaries["bmc"].loc["prediction", "mean"]),
            "bmc_sharpe": float(summaries["bmc"].loc["prediction", "sharpe"]),
            "bmc_max_dd": float(summaries["bmc"].loc["prediction", "max_drawdown"]),
        }

    finally:
        temp_path.unlink(missing_ok=True)

    print(f"\n{'='*60}")
    print(f"HOLDOUT RESULTS")
    print(f"{'='*60}")
    print(f"Corr Mean:   {result['corr_mean']:.6f}")
    print(f"Corr Sharpe: {result['corr_sharpe']:.4f}")
    print(f"Corr Max DD: {result['corr_max_dd']:.4f}")
    print(f"BMC Mean:    {result['bmc_mean']:.6f}")
    print(f"BMC Sharpe:  {result['bmc_sharpe']:.4f}")
    print(f"BMC Max DD:  {result['bmc_max_dd']:.4f}")
    print(f"{'='*60}")

    return result
