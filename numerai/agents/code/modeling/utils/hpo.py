"""Optuna HPO with SQLite storage for Numerai models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from optuna.storages import RDBStorage

from agents.code.metrics import numerai_metrics


# Default search spaces per model type
SEARCH_SPACES = {
    "LGBMRegressor": {
        "n_estimators": ("int", 100, 3000),
        "learning_rate": ("log_float", 0.001, 0.1),
        "max_depth": ("int", 3, 12),
        "num_leaves": ("int", 16, 512),
        "colsample_bytree": ("float", 0.1, 1.0),
        "min_data_in_leaf": ("int", 500, 10000),
        "reg_alpha": ("log_float", 1e-8, 10.0),
        "reg_lambda": ("log_float", 1e-8, 10.0),
    },
    "XGBRegressor": {
        "n_estimators": ("int", 100, 3000),
        "learning_rate": ("log_float", 0.001, 0.1),
        "max_depth": ("int", 3, 12),
        "colsample_bytree": ("float", 0.1, 1.0),
        "subsample": ("float", 0.5, 1.0),
        "min_child_weight": ("int", 1, 100),
        "reg_alpha": ("log_float", 1e-8, 10.0),
        "reg_lambda": ("log_float", 1e-8, 10.0),
    },
    "CatBoostRegressor": {
        "iterations": ("int", 100, 3000),
        "learning_rate": ("log_float", 0.001, 0.1),
        "depth": ("int", 4, 10),
        "l2_leaf_reg": ("log_float", 1e-8, 10.0),
        "random_strength": ("float", 0.0, 10.0),
        "bagging_temperature": ("float", 0.0, 10.0),
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
    space = search_space or SEARCH_SPACES.get(model_type, {})
    params = {}
    for name, spec in space.items():
        params[name] = _sample_param(trial, name, spec)
    return params


def create_objective(
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
) -> Callable[[optuna.Trial], float]:
    """Create an Optuna objective function for HPO.

    Args:
        base_config: Base config dict (used for CV settings, etc.)
        data_loader: Function to load data batches by era
        eras: Series of era values
        features: List of feature column names
        model_type: Model type string (e.g., "LGBMRegressor")
        era_col: Era column name
        target_col: Target column name
        id_col: ID column name
        benchmark_model: Benchmark model for BMC calculation
        benchmark_data_path: Path to benchmark data
        data_version: Data version string
        metric: Metric to optimize ("bmc_sharpe", "bmc_mean", "corr_sharpe", "corr_mean")
        search_space: Custom search space (uses defaults if None)
        fixed_params: Parameters that remain fixed during search

    Returns:
        Objective function for Optuna
    """
    from agents.code.modeling.utils.numerai_cv import build_oof_predictions, era_cv_splits

    training_config = base_config.get("training", {})
    cv_config = dict(training_config.get("cv", {}))
    cv_config.setdefault("embargo", 13)
    model_config = base_config.get("model", {})

    fixed = fixed_params or {}

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        sampled_params = sample_params(trial, model_type, search_space)

        # Merge with fixed params (fixed params take precedence)
        model_params = {**sampled_params, **fixed}

        # Add standard params
        if model_type == "LGBMRegressor":
            model_params.setdefault("n_jobs", -1)
            model_params.setdefault("random_state", 1337)
            model_params.setdefault("verbose", -1)
        elif model_type == "XGBRegressor":
            model_params.setdefault("n_jobs", -1)
            model_params.setdefault("random_state", 1337)
            model_params.setdefault("verbosity", 0)
        elif model_type == "CatBoostRegressor":
            model_params.setdefault("random_state", 1337)
            model_params.setdefault("verbose", 0)

        try:
            # Build OOF predictions
            predictions, cv_meta = build_oof_predictions(
                eras,
                data_loader,
                model_type,
                model_params,
                model_config,
                cv_config,
                max_train_samples=training_config.get("max_train_samples"),
                sample_seed=int(training_config.get("sample_seed", 1337)),
                id_col=id_col,
                era_col=era_col,
                target_col=target_col,
                feature_cols=features,
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

                # Store additional metrics as user attributes
                trial.set_user_attr("corr_mean", float(summaries["corr"].loc["prediction", "mean"]))
                trial.set_user_attr("corr_sharpe", float(summaries["corr"].loc["prediction", "sharpe"]))
                trial.set_user_attr("bmc_mean", float(summaries["bmc"].loc["prediction", "mean"]))
                trial.set_user_attr("bmc_sharpe", float(summaries["bmc"].loc["prediction", "sharpe"]))
                trial.set_user_attr("folds_used", cv_meta["folds_used"])

            finally:
                temp_path.unlink(missing_ok=True)

            return float(value)

        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return float("-inf")

    return objective


def run_hpo(
    config_path: Path,
    n_trials: int = 50,
    study_name: str | None = None,
    db_path: str | Path | None = None,
    metric: str = "bmc_sharpe",
    search_space: dict | None = None,
    fixed_params: dict | None = None,
    timeout: float | None = None,
    n_jobs: int = 1,
) -> tuple[optuna.Study, dict]:
    """Run HPO for a Numerai model configuration.

    Args:
        config_path: Path to config file
        n_trials: Number of trials to run
        study_name: Optuna study name (auto-generated if None)
        db_path: SQLite database path (auto-generated if None)
        metric: Metric to optimize
        search_space: Custom search space
        fixed_params: Parameters to keep fixed
        timeout: Optional timeout in seconds
        n_jobs: Number of parallel jobs (1 = sequential)

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
    embargo_eras = data_config.get("embargo_eras", 13)
    benchmark_model = data_config.get("benchmark_model", DEFAULT_BENCHMARK_MODEL)

    nan_missing_all_twos = preprocessing_config.get("nan_missing_all_twos", False)
    missing_value = preprocessing_config.get("missing_value", 2.0)

    model_type = model_config.get("type", "LGBMRegressor")

    # Resolve paths
    output_dir, _, _, _ = resolve_output_locations(config, None)
    hpo_dir = output_dir / "hpo"
    hpo_dir.mkdir(parents=True, exist_ok=True)

    # Setup study name and database
    if study_name is None:
        study_name = f"{model_type.lower()}_{config_path.stem}"

    if db_path is None:
        db_path = hpo_dir / f"{study_name}.db"

    storage = RDBStorage(
        url=f"sqlite:///{db_path}",
        engine_kwargs={"connect_args": {"check_same_thread": False}},
    )

    print(f"HPO Study: {study_name}")
    print(f"Database: {db_path}")
    print(f"Optimizing: {metric}")
    print(f"Trials: {n_trials}")

    # Load data
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
    objective = create_objective(
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
    )

    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    # Get best params
    best_params = study.best_params
    best_value = study.best_value

    print(f"\nBest {metric}: {best_value:.6f}")
    print(f"Best params: {json.dumps(best_params, indent=2)}")

    # Save best params to file
    results_file = hpo_dir / f"{study_name}_best.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "study_name": study_name,
                "metric": metric,
                "best_value": best_value,
                "best_params": best_params,
                "n_trials": len(study.trials),
                "best_trial_number": study.best_trial.number,
                "best_trial_attrs": study.best_trial.user_attrs,
            },
            f,
            indent=2,
        )
    print(f"Saved best params to {results_file}")

    return study, best_params


def create_hpo_config(
    base_config_path: Path,
    best_params: dict,
    output_path: Path | None = None,
) -> Path:
    """Create a new config file with the best HPO parameters.

    Args:
        base_config_path: Path to original config
        best_params: Best parameters from HPO
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
    model_config["params"] = merged_params
    config["model"] = model_config

    # Generate output path
    if output_path is None:
        stem = base_config_path.stem
        output_path = base_config_path.parent / f"{stem}_hpo.py"

    # Write as Python config
    with open(output_path, "w") as f:
        f.write(f"CONFIG = {repr(config)}\n")

    print(f"Created HPO config: {output_path}")
    return output_path
