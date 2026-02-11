"""Improved HPO with era-stride CV, early stopping, and WandB integration.

Key improvements over hpo.py:
1. Era-stride CV for better hyperparameter generalization
2. Fixed n_estimators with early stopping (avoids searching over this)
3. WandB integration for metric logging
4. Tracks average n_estimators used across folds for final model training
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from optuna.storages import RDBStorage

from agents.code.metrics import numerai_metrics


# Search spaces WITHOUT n_estimators (use early stopping instead)
SEARCH_SPACES_V2 = {
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
        "l2_leaf_reg": ("log_float", 1e-6, 10.0),
        "random_strength": ("float", 0.0, 5.0),
        "bagging_temperature": ("float", 0.0, 5.0),
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
    space = search_space or SEARCH_SPACES_V2.get(model_type, {})
    params = {}
    for name, spec in space.items():
        params[name] = _sample_param(trial, name, spec)
    return params


class WandBCallback:
    """Optuna callback for WandB logging."""

    def __init__(self, wandb_run, metric_name: str = "bmc_sharpe"):
        self.run = wandb_run
        self.metric_name = metric_name

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        # Log trial metrics
        log_dict = {
            "trial": trial.number,
            f"trial/{self.metric_name}": trial.value,
            "trial/best_value": study.best_value,
        }

        # Log all parameters
        for key, value in trial.params.items():
            log_dict[f"params/{key}"] = value

        # Log user attributes (additional metrics)
        for key, value in trial.user_attrs.items():
            log_dict[f"metrics/{key}"] = value

        self.run.log(log_dict)


def create_objective_v2(
    base_config: dict,
    data_loader: Callable,
    eras: pd.Series,
    features: list[str],
    model_type: str,
    era_col: str = "era",
    target_col: str = "target",
    id_col: str = "id",
    benchmark_model: str = "v52_lgbm_ender20",
    benchmark_data_path: str | None = None,
    data_version: str = "v5.2",
    metric: str = "bmc_sharpe",
    search_space: dict | None = None,
    fixed_params: dict | None = None,
    # Era-stride CV settings
    n_strides: int = 3,
    folds_per_stride: int = 2,
    embargo: int = 16,
    min_train_eras: int = 20,
    # Early stopping settings
    n_estimators: int = 5000,
    early_stopping_rounds: int = 100,
) -> Callable[[optuna.Trial], float]:
    """Create an Optuna objective function with era-stride CV and early stopping.

    Key differences from v1:
    - Uses era-stride CV for better generalization testing
    - Fixed n_estimators with early stopping (not searched)
    - Tracks average n_estimators used for final model training
    """
    from agents.code.modeling.utils.era_stride_cv import build_oof_predictions_strided

    training_config = base_config.get("training", {})
    model_config = base_config.get("model", {})
    fixed = fixed_params or {}

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters (excludes n_estimators)
        sampled_params = sample_params(trial, model_type, search_space)

        # Merge with fixed params
        model_params = {**sampled_params, **fixed}

        # Set n_estimators (fixed, will use early stopping)
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
            model_params.setdefault("early_stopping_rounds", early_stopping_rounds)
        elif model_type == "CatBoostRegressor":
            model_params.setdefault("random_state", 1337)
            model_params.setdefault("verbose", 0)

        try:
            # Build OOF predictions with era-stride CV
            predictions, cv_meta = build_oof_predictions_strided(
                eras,
                data_loader,
                model_type,
                model_params,
                model_config,
                n_strides=n_strides,
                folds_per_stride=folds_per_stride,
                embargo=embargo,
                min_train_eras=min_train_eras,
                max_train_samples=training_config.get("max_train_samples"),
                sample_seed=int(training_config.get("sample_seed", 1337)),
                id_col=id_col,
                era_col=era_col,
                target_col=target_col,
                feature_cols=features,
                early_stopping_rounds=early_stopping_rounds if model_type == "LGBMRegressor" else None,
            )

            # Save predictions to temp file for metrics calculation
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                temp_path = Path(f.name)
            predictions.to_parquet(temp_path, index=False)

            try:
                # Calculate metrics
                summaries = numerai_metrics.summarize_prediction_file_with_bmc(
                    temp_path,
                    ["prediction"],
                    target_col,
                    data_version,
                    benchmark_model=benchmark_model,
                    benchmark_data_path=benchmark_data_path,
                    era_col=era_col,
                    id_col=id_col,
                )

                # Extract the requested metric
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

                # Store metrics as user attributes
                trial.set_user_attr("corr_mean", float(summaries["corr"].loc["prediction", "mean"]))
                trial.set_user_attr("corr_sharpe", float(summaries["corr"].loc["prediction", "sharpe"]))
                trial.set_user_attr("corr_max_dd", float(summaries["corr"].loc["prediction", "max_drawdown"]))
                trial.set_user_attr("bmc_mean", float(summaries["bmc"].loc["prediction", "mean"]))
                trial.set_user_attr("bmc_sharpe", float(summaries["bmc"].loc["prediction", "sharpe"]))
                trial.set_user_attr("bmc_max_dd", float(summaries["bmc"].loc["prediction", "max_drawdown"]))
                trial.set_user_attr("folds_used", cv_meta["folds_used"])
                trial.set_user_attr("cv_type", "era_stride")
                trial.set_user_attr("n_strides", n_strides)

                # Track average n_estimators used (important for final training)
                if cv_meta.get("avg_n_estimators"):
                    trial.set_user_attr("avg_n_estimators", cv_meta["avg_n_estimators"])

            finally:
                temp_path.unlink(missing_ok=True)

            return float(value)

        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            import traceback
            traceback.print_exc()
            return float("-inf")

    return objective


def run_hpo_v2(
    config_path: Path,
    n_trials: int = 20,
    study_name: str | None = None,
    db_path: str | Path | None = None,
    metric: str = "bmc_sharpe",
    search_space: dict | None = None,
    fixed_params: dict | None = None,
    # Era-stride CV settings
    n_strides: int = 3,
    folds_per_stride: int = 2,
    embargo: int = 16,
    min_train_eras: int = 20,
    # Early stopping settings
    n_estimators: int = 5000,
    early_stopping_rounds: int = 100,
    # WandB settings
    wandb_project: str | None = "numerai-hpo",
    wandb_entity: str | None = None,
    timeout: float | None = None,
    n_jobs: int = 1,
) -> tuple[optuna.Study, dict]:
    """Run HPO with era-stride CV, early stopping, and WandB logging.

    Args:
        config_path: Path to config file
        n_trials: Number of trials to run
        study_name: Optuna study name
        db_path: SQLite database path
        metric: Metric to optimize
        search_space: Custom search space (defaults exclude n_estimators)
        fixed_params: Parameters to keep fixed
        n_strides: Number of era strides for CV
        folds_per_stride: Walk-forward folds per stride
        embargo: Era embargo for CV
        min_train_eras: Minimum training eras per fold
        n_estimators: Fixed n_estimators (uses early stopping)
        early_stopping_rounds: Early stopping patience
        wandb_project: WandB project name (None to disable)
        wandb_entity: WandB entity/team
        timeout: Optional timeout in seconds
        n_jobs: Number of parallel jobs

    Returns:
        Tuple of (study, best_params)
    """
    from numerapi import NumerAPI

    from agents.code.modeling.utils.config import load_config
    from agents.code.modeling.utils.constants import DEFAULT_BENCHMARK_MODEL
    from agents.code.modeling.utils.data import (
        apply_missing_all_twos_as_nan,
        attach_benchmark_models,
        load_features,
        load_full_data,
    )
    from agents.code.modeling.utils.model_data import (
        build_model_data_loader,
        build_x_cols,
        normalize_x_groups,
    )
    from agents.code.modeling.utils.pipeline import resolve_output_locations

    config = load_config(config_path)
    data_config = config.get("data", {})
    preprocessing_config = config.get("preprocessing", {})
    model_config = config.get("model", {})

    # Extract config values
    data_version = data_config.get("data_version", "v5.2")
    feature_set = data_config.get("feature_set", "small")
    target_col = data_config.get("target_col", "target")
    era_col = data_config.get("era_col", "era")
    id_col = data_config.get("id_col", "id")
    full_data_path = data_config.get("full_data_path")
    benchmark_data_path = data_config.get("benchmark_data_path")
    benchmark_model = data_config.get("benchmark_model", DEFAULT_BENCHMARK_MODEL)

    nan_missing_all_twos = preprocessing_config.get("nan_missing_all_twos", False)
    missing_value = preprocessing_config.get("missing_value", 2.0)

    model_type = model_config.get("type", "LGBMRegressor")

    # Resolve paths
    output_dir, _, _, _ = resolve_output_locations(config, None)
    hpo_dir = output_dir / "hpo"
    hpo_dir.mkdir(parents=True, exist_ok=True)

    # Setup study name and database
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if study_name is None:
        study_name = f"{model_type.lower()}_{config_path.stem}_v2_{timestamp}"

    if db_path is None:
        db_path = hpo_dir / f"{study_name}.db"

    storage = RDBStorage(
        url=f"sqlite:///{db_path}",
        engine_kwargs={"connect_args": {"check_same_thread": False}},
    )

    # Initialize WandB
    wandb_run = None
    wandb_callback = None
    if wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=study_name,
                config={
                    "model_type": model_type,
                    "config_path": str(config_path),
                    "n_trials": n_trials,
                    "metric": metric,
                    "n_strides": n_strides,
                    "folds_per_stride": folds_per_stride,
                    "embargo": embargo,
                    "n_estimators": n_estimators,
                    "early_stopping_rounds": early_stopping_rounds,
                    "data_version": data_version,
                    "feature_set": feature_set,
                },
                reinit=True,
            )
            wandb_callback = WandBCallback(wandb_run, metric)
            print(f"WandB initialized: {wandb_run.url}")
        except Exception as e:
            print(f"WandB initialization failed: {e}")
            wandb_run = None

    print(f"\n{'='*60}")
    print(f"HPO Study: {study_name}")
    print(f"Database: {db_path}")
    print(f"Optimizing: {metric}")
    print(f"Trials: {n_trials}")
    print(f"CV: {n_strides} strides × {folds_per_stride} folds = {n_strides * folds_per_stride} total folds")
    print(f"Embargo: {embargo} eras")
    print(f"n_estimators: {n_estimators} (with early stopping @ {early_stopping_rounds})")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    napi = NumerAPI()
    features = load_features(napi, data_version, feature_set)
    full = load_full_data(
        napi,
        data_version,
        features,
        era_col,
        target_col,
        id_col,
        full_data_path=full_data_path,
    )
    print(f"Loaded {len(full)} rows, {len(features)} features, {full[era_col].nunique()} eras")

    if nan_missing_all_twos:
        full = apply_missing_all_twos_as_nan(full, features, era_col, missing_value)

    # Handle x_groups for benchmark models
    raw_x_groups = model_config.get("x_groups") or model_config.get("data_needed")
    x_groups = normalize_x_groups(raw_x_groups)
    benchmark_cols: list[str] = []

    if "benchmark_models" in x_groups:
        if not id_col:
            raise ValueError("id_col is required to attach benchmark models.")
        full, benchmark_cols = attach_benchmark_models(
            full,
            napi,
            data_version,
            benchmark_data_path,
            era_col,
            id_col,
        )
        print(f"Attached {len(benchmark_cols)} benchmark model columns")

    x_cols = build_x_cols(
        x_groups=x_groups,
        features=features,
        benchmark_cols=benchmark_cols,
        era_col=era_col,
        id_col=id_col,
        baseline_col=None,
    )

    data_loader = build_model_data_loader(
        full=full,
        x_cols=x_cols,
        era_col=era_col,
        target_col=target_col,
        id_col=id_col,
    )

    # Create objective
    objective = create_objective_v2(
        base_config=config,
        data_loader=data_loader,
        eras=full[era_col],
        features=features,
        model_type=model_type,
        era_col=era_col,
        target_col=target_col,
        id_col=id_col,
        benchmark_model=benchmark_model,
        benchmark_data_path=benchmark_data_path,
        data_version=data_version,
        metric=metric,
        search_space=search_space,
        fixed_params=fixed_params,
        n_strides=n_strides,
        folds_per_stride=folds_per_stride,
        embargo=embargo,
        min_train_eras=min_train_eras,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
    )

    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    # Optuna callbacks
    callbacks = []
    if wandb_callback:
        callbacks.append(wandb_callback)

    # Run optimization
    print(f"\nStarting HPO with {n_trials} trials...")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=True,
        callbacks=callbacks,
    )

    # Get best results
    best_params = study.best_params
    best_value = study.best_value
    best_attrs = study.best_trial.user_attrs

    print(f"\n{'='*60}")
    print(f"HPO Complete!")
    print(f"Best {metric}: {best_value:.6f}")
    print(f"Best params: {json.dumps(best_params, indent=2)}")
    if best_attrs.get("avg_n_estimators"):
        print(f"Recommended n_estimators for final model: {best_attrs['avg_n_estimators']}")
    print(f"{'='*60}")

    # Save best params to file
    results = {
        "study_name": study_name,
        "metric": metric,
        "best_value": best_value,
        "best_params": best_params,
        "n_trials": len(study.trials),
        "best_trial_number": study.best_trial.number,
        "best_trial_attrs": best_attrs,
        "cv_config": {
            "type": "era_stride",
            "n_strides": n_strides,
            "folds_per_stride": folds_per_stride,
            "embargo": embargo,
            "min_train_eras": min_train_eras,
        },
        "early_stopping": {
            "n_estimators": n_estimators,
            "early_stopping_rounds": early_stopping_rounds,
            "recommended_n_estimators": best_attrs.get("avg_n_estimators"),
        },
    }

    results_file = hpo_dir / f"{study_name}_best.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_file}")

    # Log final results to WandB
    if wandb_run:
        wandb_run.summary["best_value"] = best_value
        wandb_run.summary["best_params"] = best_params
        wandb_run.summary["recommended_n_estimators"] = best_attrs.get("avg_n_estimators")
        wandb_run.finish()

    return study, best_params


def create_hpo_config_v2(
    base_config_path: Path,
    best_params: dict,
    recommended_n_estimators: int | None = None,
    output_path: Path | None = None,
) -> Path:
    """Create a new config file with the best HPO parameters.

    Args:
        base_config_path: Path to original config
        best_params: Best parameters from HPO
        recommended_n_estimators: Recommended n_estimators from early stopping
        output_path: Output path (auto-generated if None)

    Returns:
        Path to new config file
    """
    from agents.code.modeling.utils.config import load_config

    config = load_config(base_config_path)

    # Update model params with best params
    model_config = config.get("model", {})
    existing_params = model_config.get("params", {})

    # Merge: best_params override existing, but keep non-HPO params
    merged_params = {**existing_params, **best_params}

    # Set recommended n_estimators if provided
    if recommended_n_estimators:
        # Add 10% buffer to recommended n_estimators for final training
        merged_params["n_estimators"] = int(recommended_n_estimators * 1.1)

    model_config["params"] = merged_params
    config["model"] = model_config

    # Generate output path
    if output_path is None:
        stem = base_config_path.stem
        output_path = base_config_path.parent / f"{stem}_hpo_v2.py"

    # Write as Python config
    with open(output_path, "w") as f:
        f.write(f"CONFIG = {repr(config)}\n")

    print(f"Created HPO config: {output_path}")
    return output_path
