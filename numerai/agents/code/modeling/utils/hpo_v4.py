"""HPO v4: Enhanced pipeline with payout optimization and ensembling.

Phase 1 - HPO: Era-stride CV on downsampled data (eras <= 800)
Phase 2 - Neutralization: Two-stage neutralization tuning (eras 816-1116)
         Stage A: Prediction neutralization (example/benchmark)
         Stage B: Feature group neutralization (selective)
Phase 3 - Ensemble: Train 5 models on different era offsets
Phase 4 - Holdout: Final evaluation on full data (eras >= 1133)

Payout function: clip(0.75 * CORR + 2.25 * BMC, -0.05, 0.05)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

# Default era boundaries (with 16-era embargo)
DEFAULT_HPO_MAX_ERA = 800
DEFAULT_NEUT_MIN_ERA = 816
DEFAULT_NEUT_MAX_ERA = 1116
DEFAULT_HOLDOUT_MIN_ERA = 1133

# Feature groups in Numerai v5.2
FEATURE_GROUPS = [
    "intelligence",
    "wisdom",
    "charisma",
    "dexterity",
    "strength",
    "constitution",
    "agility",
    "serenity",
]


def compute_payout_metric(
    corr_per_era: pd.Series,
    bmc_per_era: pd.Series,
    clip_value: float = 0.05,
) -> dict:
    """Compute payout-weighted metric.

    Payout = clip(0.75 * CORR + 2.25 * BMC, -clip_value, clip_value)

    Returns dict with payout_mean, payout_sharpe, etc.
    """
    # Compute per-era payout
    payout_per_era = 0.75 * corr_per_era + 2.25 * bmc_per_era
    payout_per_era = payout_per_era.clip(-clip_value, clip_value)

    payout_mean = payout_per_era.mean()
    payout_std = payout_per_era.std(ddof=0)
    payout_sharpe = payout_mean / payout_std if payout_std > 0 else 0.0

    cumsum = payout_per_era.cumsum()
    rolling_max = cumsum.expanding(min_periods=1).max()
    max_drawdown = (rolling_max - cumsum).max()

    return {
        "payout_mean": float(payout_mean),
        "payout_std": float(payout_std),
        "payout_sharpe": float(payout_sharpe),
        "payout_max_dd": float(max_drawdown),
        "payout_total": float(payout_per_era.sum()),
    }


def get_feature_groups(features: list[str], data_version: str = "v5.2") -> dict[str, list[str]]:
    """Get feature group membership for all features.

    Returns dict mapping group name -> list of feature names in that group.
    """
    from numerapi import NumerAPI
    import json as json_module

    napi = NumerAPI()
    napi.download_dataset(f"{data_version}/features.json")

    with open(f"{data_version}/features.json") as f:
        feature_metadata = json_module.load(f)

    feature_sets = feature_metadata["feature_sets"]

    groups = {}
    for group in FEATURE_GROUPS:
        if group in feature_sets:
            # Intersection with our features
            group_features = [f for f in features if f in feature_sets.get(group, [])]
            if group_features:
                groups[group] = group_features

    return groups


def neutralize_to_features(
    predictions: np.ndarray,
    features: np.ndarray,
    proportion: float = 1.0,
) -> np.ndarray:
    """Neutralize predictions to features using least squares.

    Based on numerai-tools neutralize function.
    """
    if proportion == 0.0:
        return predictions

    # Add intercept column
    neutralizers = np.hstack([
        features,
        np.ones((len(features), 1))
    ])

    # Compute adjustment using pseudo-inverse
    inverse = np.linalg.pinv(neutralizers, rcond=1e-6)
    adjustments = proportion * neutralizers.dot(inverse.dot(predictions))

    neutralized = predictions - adjustments
    return neutralized


def neutralize_to_predictions(
    predictions: np.ndarray,
    other_predictions: np.ndarray,
    proportion: float = 1.0,
) -> np.ndarray:
    """Neutralize predictions to other predictions (e.g., example or benchmark).

    This removes correlation with the other predictions.
    """
    if proportion == 0.0:
        return predictions

    # Reshape if needed
    if other_predictions.ndim == 1:
        other_predictions = other_predictions.reshape(-1, 1)

    return neutralize_to_features(predictions, other_predictions, proportion)


def apply_neutralization_config(
    raw_predictions: np.ndarray,
    features_df: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    config: dict,
    example_preds: np.ndarray | None = None,
    benchmark_preds: np.ndarray | None = None,
) -> np.ndarray:
    """Apply full neutralization config to predictions.

    Config format:
    {
        "example_prop": 0.5,      # Proportion to neutralize against example predictions
        "benchmark_prop": 0.25,   # Proportion to neutralize against benchmark predictions
        "group_props": {          # Per-group neutralization proportions
            "intelligence": 0.0,
            "wisdom": 0.5,
            ...
        }
    }

    Order: prediction neutralization first, then feature group neutralization.
    """
    predictions = raw_predictions.copy()

    # Stage A: Prediction neutralization (first)
    if example_preds is not None and config.get("example_prop", 0) > 0:
        predictions = neutralize_to_predictions(
            predictions, example_preds, config["example_prop"]
        )

    if benchmark_preds is not None and config.get("benchmark_prop", 0) > 0:
        predictions = neutralize_to_predictions(
            predictions, benchmark_preds, config["benchmark_prop"]
        )

    # Stage B: Feature group neutralization (second)
    group_props = config.get("group_props", {})
    for group_name, proportion in group_props.items():
        if proportion > 0 and group_name in feature_groups:
            group_features = feature_groups[group_name]
            group_data = features_df[group_features].values
            predictions = neutralize_to_features(predictions, group_data, proportion)

    # Re-rank to [0, 1]
    predictions = rankdata(predictions) / len(predictions)

    return predictions


def evaluate_neutralization_config(
    raw_predictions: np.ndarray,
    features_df: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    config: dict,
    targets: np.ndarray,
    eras: np.ndarray,
    example_preds: np.ndarray | None = None,
    benchmark_preds: np.ndarray | None = None,
    data_version: str = "v5.2",
) -> dict:
    """Evaluate a neutralization config and return metrics.

    Returns payout metrics (optimized for payout Sharpe).
    """
    from numerai_tools.scoring import correlation_contribution, numerai_corr

    # Apply neutralization
    predictions = apply_neutralization_config(
        raw_predictions,
        features_df,
        feature_groups,
        config,
        example_preds,
        benchmark_preds,
    )

    # Create DataFrame for per-era metrics
    df = pd.DataFrame({
        "era": eras,
        "target": targets,
        "prediction": predictions,
    })

    # Compute per-era CORR
    per_era_corr = df.groupby("era").apply(
        lambda d: numerai_corr(d[["prediction"]], d["target"]).iloc[0]
    )

    # Compute per-era BMC if benchmark predictions are available.
    if benchmark_preds is not None and len(benchmark_preds) == len(df):
        df["benchmark"] = benchmark_preds

        def _bmc_for_era(d: pd.DataFrame) -> float:
            return float(
                correlation_contribution(
                    d[["prediction"]],
                    d["benchmark"],
                    d["target"],
                ).iloc[0]
            )

        per_era_bmc = df.groupby("era").apply(_bmc_for_era)
    else:
        # Conservative fallback when no benchmark signal is available.
        per_era_bmc = pd.Series(0.0, index=per_era_corr.index)

    # Compute payout metric
    payout_metrics = compute_payout_metric(per_era_corr, per_era_bmc)

    # Also compute individual metrics
    corr_mean = per_era_corr.mean()
    corr_std = per_era_corr.std(ddof=0)
    corr_sharpe = corr_mean / corr_std if corr_std > 0 else 0.0

    return {
        **payout_metrics,
        "corr_mean": float(corr_mean),
        "corr_sharpe": float(corr_sharpe),
        "bmc_mean": float(per_era_bmc.mean()),
        "bmc_sharpe": float(per_era_bmc.mean() / per_era_bmc.std(ddof=0)) if per_era_bmc.std() > 0 else 0.0,
    }


def run_neutralization_tuning(
    best_params: dict,
    model_type: str = "LGBMRegressor",
    data_version: str = "v5.2",
    feature_set: str = "all",
    downsampled_path: str | None = None,
    full_path: str | None = None,
    benchmark_path: str | None = None,
    example_preds_path: str | None = None,
    hpo_max_era: int = DEFAULT_HPO_MAX_ERA,
    neut_min_era: int = DEFAULT_NEUT_MIN_ERA,
    neut_max_era: int = DEFAULT_NEUT_MAX_ERA,
    n_estimators: int = 2000,
    # Search grid
    pred_neut_strengths: list[float] | None = None,
    group_neut_strengths: list[float] | None = None,
) -> dict:
    """Run two-stage neutralization tuning optimized for payout function.

    Stage A: Grid search prediction neutralization (example + benchmark)
    Stage B: Grid search feature group neutralization (given best pred neut)

    Optimizes for: clip(0.75 * CORR + 2.25 * BMC, -0.05, 0.05) Sharpe
    """
    from agents.code.modeling.utils.model_factory import build_model
    from agents.code.modeling.utils.hpo_v3 import load_data_for_phase
    from numerai_tools.scoring import correlation_contribution, numerai_corr

    if pred_neut_strengths is None:
        pred_neut_strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
    if group_neut_strengths is None:
        group_neut_strengths = [0.0, 0.5, 1.0]

    print(f"\n{'='*60}")
    print(f"Neutralization Tuning Phase (v4)")
    print(f"{'='*60}")
    print(f"Training on eras <= {hpo_max_era}")
    print(f"Evaluating on eras {neut_min_era}-{neut_max_era}")
    print(f"Optimizing for: payout = clip(0.75*CORR + 2.25*BMC, -0.05, 0.05)")
    print(f"Prediction neut strengths: {pred_neut_strengths}")
    print(f"Group neut strengths: {group_neut_strengths}")
    print(f"{'='*60}\n")

    # Load training data
    train_data, features, benchmark_cols = load_data_for_phase(
        "hpo",
        data_version=data_version,
        feature_set=feature_set,
        downsampled_path=downsampled_path,
        benchmark_path=benchmark_path,
        hpo_max_era=hpo_max_era,
    )

    # Load neutralization data
    neut_data, _, neut_benchmark_cols = load_data_for_phase(
        "neutralization",
        data_version=data_version,
        feature_set=feature_set,
        full_path=full_path,
        benchmark_path=benchmark_path,
        neut_min_era=neut_min_era,
        neut_max_era=neut_max_era,
    )

    # Get feature groups
    feature_groups = get_feature_groups(features, data_version)
    print(f"Feature groups: {list(feature_groups.keys())}")
    print(f"Group sizes: {[(g, len(f)) for g, f in feature_groups.items()]}")

    # Build and train model
    x_cols = features + benchmark_cols
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

    print("\nTraining model on HPO data...")
    model = build_model(model_type, model_params, {}, feature_cols=features)
    model.fit(train_data[x_cols], train_data["target"])

    # Get raw predictions on neutralization data
    print("Predicting on neutralization data...")
    raw_preds = model.predict(neut_data[x_cols])

    # Load example predictions
    example_preds = None
    example_candidates = [
        c for c in neut_data.columns
        if "example" in c.lower() and ("pred" in c.lower() or "signal" in c.lower())
    ]
    if example_candidates:
        example_preds = neut_data[example_candidates[0]].to_numpy()
        print(f"  Using example predictions from neutralization data: {example_candidates[0]}")
    elif example_preds_path:
        try:
            if str(example_preds_path).endswith(".csv"):
                example_df = pd.read_csv(example_preds_path)
            else:
                example_df = pd.read_parquet(example_preds_path)

            if "prediction" in example_df.columns:
                if "id" in example_df.columns and "id" in neut_data.columns:
                    merged = neut_data[["id"]].merge(
                        example_df[["id", "prediction"]],
                        on="id",
                        how="left",
                    )
                    if merged["prediction"].notna().all():
                        example_preds = merged["prediction"].to_numpy()
                elif len(example_df) == len(neut_data):
                    example_preds = example_df["prediction"].to_numpy()

            if example_preds is not None:
                print(f"  Loaded example predictions from {example_preds_path}")
            else:
                print(
                    f"  Could not align example predictions from {example_preds_path}; "
                    "skipping example neutralization"
                )
        except Exception as e:
            print(f"  Could not load example predictions: {e}")

    # Load benchmark predictions (meta model proxy) from merged benchmark columns.
    benchmark_preds = None
    if neut_benchmark_cols:
        preferred = [c for c in neut_benchmark_cols if "ender20" in c.lower()]
        benchmark_col = preferred[0] if preferred else neut_benchmark_cols[0]
        benchmark_preds = neut_data[benchmark_col].to_numpy()
        print(f"  Using benchmark column: {benchmark_col}")

    # Prepare for evaluation
    targets = neut_data["target"].values
    eras = neut_data["era"].values
    features_df = neut_data[features]

    def compute_metrics_for_config(config: dict) -> dict:
        """Compute payout metrics for a neutralization config."""
        # Apply neutralization
        preds = apply_neutralization_config(
            raw_preds,
            features_df,
            feature_groups,
            config,
            example_preds,
            benchmark_preds,
        )

        df = pd.DataFrame({
            "id": neut_data["id"].values,
            "era": eras,
            "target": targets,
            "prediction": preds,
        })
        per_era_corr = df.groupby("era").apply(
            lambda d: numerai_corr(d[["prediction"]], d["target"]).iloc[0]
        )

        if benchmark_preds is not None and len(benchmark_preds) == len(df):
            df["benchmark"] = benchmark_preds
            per_era_bmc = df.groupby("era").apply(
                lambda d: float(
                    correlation_contribution(
                        d[["prediction"]],
                        d["benchmark"],
                        d["target"],
                    ).iloc[0]
                )
            )
        else:
            per_era_bmc = pd.Series(0.0, index=per_era_corr.index)

        payout_metrics = compute_payout_metric(per_era_corr, per_era_bmc)
        corr_mean = float(per_era_corr.mean())
        corr_std = float(per_era_corr.std(ddof=0))
        bmc_mean = float(per_era_bmc.mean())
        bmc_std = float(per_era_bmc.std(ddof=0))

        return {
            "corr_mean": corr_mean,
            "corr_sharpe": corr_mean / corr_std if corr_std > 0 else 0.0,
            "bmc_mean": bmc_mean,
            "bmc_sharpe": bmc_mean / bmc_std if bmc_std > 0 else 0.0,
            **payout_metrics,
        }

    # Stage A: Prediction neutralization grid search
    print("\n" + "="*40)
    print("Stage A: Prediction Neutralization")
    print("="*40)

    stage_a_results = []
    for ex_prop in pred_neut_strengths:
        for bm_prop in pred_neut_strengths:
            config = {
                "example_prop": ex_prop,
                "benchmark_prop": bm_prop,
                "group_props": {},
            }

            try:
                metrics = compute_metrics_for_config(config)
                result = {
                    "example_prop": ex_prop,
                    "benchmark_prop": bm_prop,
                    **metrics,
                }
                stage_a_results.append(result)

                print(f"  Ex={ex_prop:.2f}, Bm={bm_prop:.2f}: "
                      f"Payout Sharpe={metrics['payout_sharpe']:.4f}, "
                      f"BMC Sharpe={metrics['bmc_sharpe']:.4f}")
            except Exception as e:
                print(f"  Ex={ex_prop:.2f}, Bm={bm_prop:.2f}: Failed - {e}")

    # Find best Stage A config
    best_stage_a = max(stage_a_results, key=lambda x: x["payout_sharpe"])
    print(f"\nBest Stage A: Ex={best_stage_a['example_prop']:.2f}, "
          f"Bm={best_stage_a['benchmark_prop']:.2f}, "
          f"Payout Sharpe={best_stage_a['payout_sharpe']:.4f}")

    # Stage B: Feature group neutralization (given best pred neut)
    print("\n" + "="*40)
    print("Stage B: Feature Group Neutralization")
    print("="*40)

    # First, evaluate each group individually
    print("\nIndividual group effects:")
    group_individual_results = {}

    for group_name in feature_groups.keys():
        for strength in group_neut_strengths:
            if strength == 0:
                continue  # Skip baseline

            config = {
                "example_prop": best_stage_a["example_prop"],
                "benchmark_prop": best_stage_a["benchmark_prop"],
                "group_props": {group_name: strength},
            }

            try:
                metrics = compute_metrics_for_config(config)
                key = (group_name, strength)
                group_individual_results[key] = metrics

                print(f"  {group_name}@{strength:.1f}: "
                      f"Payout Sharpe={metrics['payout_sharpe']:.4f}, "
                      f"BMC Sharpe={metrics['bmc_sharpe']:.4f}")
            except Exception as e:
                print(f"  {group_name}@{strength:.1f}: Failed - {e}")

    # Find groups that improve payout Sharpe
    baseline_payout = best_stage_a["payout_sharpe"]
    beneficial_groups = {}

    for (group_name, strength), metrics in group_individual_results.items():
        if metrics["payout_sharpe"] > baseline_payout:
            if group_name not in beneficial_groups or metrics["payout_sharpe"] > beneficial_groups[group_name][1]:
                beneficial_groups[group_name] = (strength, metrics["payout_sharpe"])

    print(f"\nBeneficial groups (improve payout): {list(beneficial_groups.keys())}")

    # Combine beneficial groups
    best_group_config = {g: s for g, (s, _) in beneficial_groups.items()}

    if best_group_config:
        print(f"\nEvaluating combined beneficial groups: {best_group_config}")

        config = {
            "example_prop": best_stage_a["example_prop"],
            "benchmark_prop": best_stage_a["benchmark_prop"],
            "group_props": best_group_config,
        }

        combined_metrics = compute_metrics_for_config(config)
        print(f"Combined result: Payout Sharpe={combined_metrics['payout_sharpe']:.4f}, "
              f"BMC Sharpe={combined_metrics['bmc_sharpe']:.4f}")

        if combined_metrics["payout_sharpe"] > baseline_payout:
            best_neut_config = config
            best_neut_metrics = combined_metrics
        else:
            best_neut_config = {
                "example_prop": best_stage_a["example_prop"],
                "benchmark_prop": best_stage_a["benchmark_prop"],
                "group_props": {},
            }
            best_neut_metrics = best_stage_a
    else:
        best_neut_config = {
            "example_prop": best_stage_a["example_prop"],
            "benchmark_prop": best_stage_a["benchmark_prop"],
            "group_props": {},
        }
        best_neut_metrics = best_stage_a

    print(f"\n{'='*60}")
    print(f"Best Neutralization Config")
    print(f"{'='*60}")
    print(f"Example prop: {best_neut_config['example_prop']}")
    print(f"Benchmark prop: {best_neut_config['benchmark_prop']}")
    print(f"Group props: {best_neut_config['group_props']}")
    print(f"Payout Sharpe: {best_neut_metrics['payout_sharpe']:.4f}")
    print(f"BMC Sharpe: {best_neut_metrics['bmc_sharpe']:.4f}")
    print(f"Corr Sharpe: {best_neut_metrics['corr_sharpe']:.4f}")
    print(f"{'='*60}")

    return {
        "best_config": best_neut_config,
        "best_metrics": best_neut_metrics,
        "stage_a_results": stage_a_results,
        "stage_b_group_results": group_individual_results,
        "beneficial_groups": beneficial_groups,
    }


def train_ensemble_models(
    best_params: dict,
    model_type: str = "LGBMRegressor",
    data_version: str = "v5.2",
    feature_set: str = "all",
    data_dir: str = "/Users/joakim/Documents/Projects/Numerai/data",
    max_train_era: int = DEFAULT_NEUT_MAX_ERA,
    n_estimators: int = 2000,
    n_offsets: int = 4,
    memory_efficient: bool = True,
    save_dir: Path | str | None = None,
    full_model_subsample: float = 0.25,
) -> list:
    """Train ensemble of models on different era offsets.

    Trains n_offsets models on downsampled data (different offsets)
    plus 1 model on all data (subsampled for memory efficiency).

    Args:
        data_dir: Base directory containing v5.2/train.parquet and v5.2/validation.parquet
        memory_efficient: If True, load data per-model, use float32,
                         train sequentially with garbage collection,
                         and save each model to disk immediately.
        save_dir: Directory to save models (required if memory_efficient=True).
                 Each model saved as model_{offset}.pkl or model_full.pkl.
        full_model_subsample: Fraction of data to use for full model (0.0 to skip,
                             1.0 for all data). Default 0.25 to fit in 64GB RAM.

    Returns list of model info dicts. If memory_efficient=True, each dict has
    'model_path' instead of 'model'.
    """
    import gc
    import cloudpickle
    from agents.code.modeling.utils.model_factory import build_model
    from numerapi import NumerAPI
    from agents.code.modeling.utils.data import load_features

    if memory_efficient and save_dir is None:
        save_dir = Path("agents/baselines/models")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(data_dir) / data_version / "train.parquet"
    val_path = Path(data_dir) / data_version / "validation.parquet"

    n_full = 1 if full_model_subsample > 0 else 0
    n_total = n_offsets + n_full

    print(f"\n{'='*60}", flush=True)
    print(f"Ensemble Training Phase", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Training on eras <= {max_train_era}", flush=True)
    if full_model_subsample > 0:
        subsample_pct = int(full_model_subsample * 100)
        print(f"Ensemble: {n_offsets} downsampled + 1 full ({subsample_pct}% subsample) = {n_total} models", flush=True)
    else:
        print(f"Ensemble: {n_offsets} downsampled models (no full model)", flush=True)
    print(f"Data: {train_path} + {val_path}", flush=True)
    if memory_efficient:
        print(f"Memory-efficient mode: per-model data loading, float32, save to {save_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    napi = NumerAPI()
    features = load_features(napi, data_version, feature_set)

    # First pass: get era list from both files
    print(f"Scanning eras...", flush=True)
    train_eras_df = pd.read_parquet(train_path, columns=["era"])
    val_eras_df = pd.read_parquet(val_path, columns=["era"])

    train_eras_df["era_int"] = train_eras_df["era"].astype(int)
    val_eras_df["era_int"] = val_eras_df["era"].astype(int)

    # Get all eras up to max_train_era from both files
    train_eras = set(train_eras_df[train_eras_df["era_int"] <= max_train_era]["era_int"].unique())
    val_eras = set(val_eras_df[val_eras_df["era_int"] <= max_train_era]["era_int"].unique())
    all_eras = sorted(train_eras | val_eras)

    del train_eras_df, val_eras_df
    gc.collect()

    print(f"  Train file eras: {len(train_eras)} (up to era {max_train_era})", flush=True)
    print(f"  Val file eras: {len(val_eras)} (up to era {max_train_era})", flush=True)
    print(f"  Total training eras: {len(all_eras)}", flush=True)

    # Build model params
    model_params = {**best_params}
    if model_type == "CatBoostRegressor":
        model_params["iterations"] = n_estimators
        model_params.setdefault("verbose", 0)
    else:
        model_params["n_estimators"] = n_estimators
    if model_type == "LGBMRegressor":
        model_params.setdefault("n_jobs", -1)
        model_params.setdefault("verbose", -1)

    cols = ["era", "target"] + features
    models = []

    def load_eras_from_both(era_list: list[int]) -> pd.DataFrame:
        """Load specified eras from train and validation files."""
        era_set = set(era_list)
        # Era column uses zero-padded strings like "0001", "0574", etc.
        era_strs = [f"{e:04d}" for e in era_list]
        dfs = []

        # Load from train if any eras overlap
        if era_set & train_eras:
            train_df = pd.read_parquet(
                train_path, columns=cols,
                filters=[("era", "in", era_strs)]
            )
            if len(train_df) > 0:
                dfs.append(train_df)
            del train_df

        # Load from validation if any eras overlap
        if era_set & val_eras:
            val_df = pd.read_parquet(
                val_path, columns=cols,
                filters=[("era", "in", era_strs)]
            )
            if len(val_df) > 0:
                dfs.append(val_df)
            del val_df

        if not dfs:
            raise ValueError(f"No data found for eras {era_list[:5]}...")

        combined = pd.concat(dfs, ignore_index=True)
        del dfs
        gc.collect()
        return combined

    # Train downsampled models (different offsets)
    for offset in range(n_offsets):
        # Select every 4th era starting from offset
        subset_eras = [e for i, e in enumerate(all_eras) if i % 4 == offset]

        print(f"\nModel {offset + 1}/{n_offsets + 1} (offset={offset}):", flush=True)
        print(f"  Loading {len(subset_eras)} eras...", flush=True)

        subset_df = load_eras_from_both(subset_eras)

        # Convert to float32
        if memory_efficient:
            for col in features:
                subset_df[col] = subset_df[col].astype(np.float32)
            subset_df["target"] = subset_df["target"].astype(np.float32)

        print(f"  Training on {len(subset_df):,} rows...", flush=True)

        model = build_model(model_type, {**model_params, "random_state": 1337 + offset},
                          {}, feature_cols=features)
        model.fit(subset_df[features], subset_df["target"])

        model_info = {
            "type": "downsampled",
            "offset": offset,
            "n_eras": len(subset_eras),
            "n_rows": len(subset_df),
        }

        # Free data memory before saving
        del subset_df
        gc.collect()

        if memory_efficient:
            model_path = save_dir / f"model_offset_{offset}.pkl"
            with open(model_path, "wb") as f:
                cloudpickle.dump(model, f)
            print(f"  Saved to {model_path}", flush=True)
            model_info["model_path"] = str(model_path)
            del model
            gc.collect()
        else:
            model_info["model"] = model

        models.append(model_info)

    # Train full data model (with optional subsampling)
    if full_model_subsample > 0:
        n_total = n_offsets + 1
        subsample_pct = int(full_model_subsample * 100)
        print(f"\nModel {n_total}/{n_total} (full data, {subsample_pct}% subsample):", flush=True)
        print(f"  Loading all {len(all_eras)} training eras...", flush=True)

        full_df = load_eras_from_both(all_eras)

        # Subsample per era to maintain era distribution
        if full_model_subsample < 1.0:
            print(f"  Subsampling to {subsample_pct}% of rows...", flush=True)
            full_df = full_df.groupby("era", group_keys=False).apply(
                lambda x: x.sample(frac=full_model_subsample, random_state=1337)
            ).reset_index(drop=True)

        if memory_efficient:
            for col in features:
                full_df[col] = full_df[col].astype(np.float32)
            full_df["target"] = full_df["target"].astype(np.float32)

        print(f"  Training on {len(full_df):,} rows...", flush=True)

        model = build_model(model_type, {**model_params, "random_state": 1337},
                           {}, feature_cols=features)
        model.fit(full_df[features], full_df["target"])

        model_info = {
            "type": "full_subsampled",
            "offset": None,
            "n_eras": len(all_eras),
            "n_rows": len(full_df),
            "subsample": full_model_subsample,
        }
    else:
        print(f"\nSkipping full data model (full_model_subsample=0)", flush=True)
        model_info = None
        full_df = None

    # Free data memory before saving
    if full_df is not None:
        del full_df
        gc.collect()

    if model_info is not None:
        if memory_efficient:
            model_path = save_dir / "model_full.pkl"
            with open(model_path, "wb") as f:
                cloudpickle.dump(model, f)
            print(f"  Saved to {model_path}", flush=True)
            model_info["model_path"] = str(model_path)
            del model
            gc.collect()
        else:
            model_info["model"] = model

        models.append(model_info)

    print(f"\n{'='*60}", flush=True)
    print(f"Ensemble Training Complete!", flush=True)
    print(f"Trained {len(models)} models", flush=True)
    if memory_efficient:
        print(f"Models saved to {save_dir}", flush=True)
    print(f"{'='*60}", flush=True)

    return models


def ensemble_predict(
    models: list,
    features_df: pd.DataFrame,
    weights: list[float] | None = None,
    memory_efficient: bool = True,
) -> np.ndarray:
    """Generate ensemble predictions from multiple models.

    Args:
        models: List of model dicts from train_ensemble_models.
               Each dict has either 'model' or 'model_path'.
        features_df: DataFrame with feature columns
        weights: Optional weights for each model (default: equal weights)
        memory_efficient: If True, load models one at a time and delete after use.

    Returns:
        Ensemble predictions as numpy array
    """
    import gc
    import cloudpickle

    if weights is None:
        weights = [1.0] * len(models)

    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize

    predictions = np.zeros(len(features_df))

    for i, (model_info, weight) in enumerate(zip(models, weights)):
        # Load model from disk if needed
        if "model" in model_info:
            model = model_info["model"]
            loaded_from_disk = False
        elif "model_path" in model_info:
            with open(model_info["model_path"], "rb") as f:
                model = cloudpickle.load(f)
            loaded_from_disk = True
        else:
            raise ValueError(f"Model {i} has neither 'model' nor 'model_path'")

        preds = model.predict(features_df)
        predictions += weight * preds

        # Free memory if loaded from disk
        if loaded_from_disk and memory_efficient:
            del model
            gc.collect()

    return predictions


def evaluate_on_holdout_v4(
    models: list,
    neut_config: dict,
    model_type: str = "LGBMRegressor",
    data_version: str = "v5.2",
    feature_set: str = "all",
    data_dir: str = "/Users/joakim/Documents/Projects/Numerai/data",
    holdout_min_era: int = DEFAULT_HOLDOUT_MIN_ERA,
    memory_efficient: bool = True,
) -> dict:
    """Final holdout evaluation with ensemble and neutralization.

    Only call this once!

    Args:
        data_dir: Base directory containing v5.2/validation.parquet and benchmark files
        memory_efficient: If True, use float32 for features and aggressively
                         free memory after operations.
    """
    import gc
    from numerapi import NumerAPI
    from agents.code.modeling.utils.data import load_features
    from numerai_tools.scoring import numerai_corr, correlation_contribution

    val_path = Path(data_dir) / data_version / "validation.parquet"
    benchmark_path = Path(data_dir) / data_version / "validation_benchmark_models.parquet"
    example_path = Path(data_dir) / data_version / "validation_example_preds.parquet"

    print(f"\n{'='*70}", flush=True)
    print(f"HOLDOUT EVALUATION (FINAL)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Evaluating on eras >= {holdout_min_era}", flush=True)
    print(f"Ensemble: {len(models)} models", flush=True)
    print(f"Neutralization config: {neut_config}", flush=True)
    print(f"Data: {val_path}", flush=True)
    if memory_efficient:
        print(f"Memory-efficient mode: enabled", flush=True)
    print(f"{'='*70}\n", flush=True)

    napi = NumerAPI()
    features = load_features(napi, data_version, feature_set)

    # Load validation data for holdout eras
    print("Loading holdout data...", flush=True)
    cols = ["id", "era", "target"] + features
    val_df = pd.read_parquet(val_path, columns=cols)
    val_df["era_int"] = val_df["era"].astype(int)

    holdout_df = val_df[val_df["era_int"] >= holdout_min_era].copy()
    del val_df
    gc.collect()

    print(f"  Holdout: {len(holdout_df):,} rows, {holdout_df['era'].nunique()} eras", flush=True)
    print(f"  Era range: {holdout_df['era_int'].min()} - {holdout_df['era_int'].max()}", flush=True)

    # Load benchmark predictions for BMC calculation
    benchmark_preds = None
    if benchmark_path.exists():
        print(f"Loading benchmark predictions from {benchmark_path}...", flush=True)
        benchmark_df = pd.read_parquet(benchmark_path)

        # Handle case where 'id' is the index (reset it to column)
        if benchmark_df.index.name == "id":
            benchmark_df = benchmark_df.reset_index()

        # Find the meta model column (v52_lgbm_cyrusd20 is common in v5.2)
        meta_cols = [c for c in benchmark_df.columns if "cyrusd" in c.lower() or "lgbm" in c.lower()]
        if meta_cols:
            benchmark_col = meta_cols[0]
        else:
            # Use first prediction column
            benchmark_col = [c for c in benchmark_df.columns if c not in ["id", "era"]][0]

        print(f"  Using benchmark column: {benchmark_col}", flush=True)

        # Merge with holdout data
        holdout_df = holdout_df.merge(
            benchmark_df[["id", benchmark_col]].rename(columns={benchmark_col: "benchmark"}),
            on="id", how="left"
        )
        benchmark_preds = holdout_df["benchmark"].values
        print(f"  Benchmark coverage: {(~pd.isna(benchmark_preds)).sum():,}/{len(holdout_df):,}", flush=True)
        del benchmark_df
        gc.collect()

    # Load example predictions if available
    example_preds = None
    if example_path.exists():
        print(f"Loading example predictions from {example_path}...", flush=True)
        example_df = pd.read_parquet(example_path)

        # Handle case where 'id' is the index (reset it to column)
        if example_df.index.name == "id":
            example_df = example_df.reset_index()

        pred_col = [c for c in example_df.columns if "prediction" in c.lower()][0] if any("prediction" in c.lower() for c in example_df.columns) else example_df.columns[-1]
        holdout_df = holdout_df.merge(
            example_df[["id", pred_col]].rename(columns={pred_col: "example"}),
            on="id", how="left"
        )
        example_preds = holdout_df["example"].values
        del example_df
        gc.collect()

    # Convert features to float32 for memory efficiency
    if memory_efficient:
        print("Converting features to float32...", flush=True)
        for col in features:
            holdout_df[col] = holdout_df[col].astype(np.float32)
        gc.collect()

    # Get feature groups
    feature_groups = get_feature_groups(features, data_version)

    # Generate ensemble predictions
    print("\nGenerating ensemble predictions...", flush=True)
    raw_preds = ensemble_predict(models, holdout_df[features])

    # Apply neutralization
    print("Applying neutralization...", flush=True)
    predictions = apply_neutralization_config(
        raw_preds,
        holdout_df[features],
        feature_groups,
        neut_config,
        example_preds,
        benchmark_preds,
    )

    holdout_df = holdout_df[["id", "era", "era_int", "target"]].copy()
    holdout_df["prediction"] = predictions
    if benchmark_preds is not None:
        holdout_df["benchmark"] = benchmark_preds
    gc.collect()

    # Compute metrics directly from holdout frame.
    print("\nComputing metrics...", flush=True)

    def calc_corr(d):
        """Calculate Spearman correlation (rank-based like Numerai)."""
        pred = d["prediction"].values
        target = d["target"].values
        return spearmanr(pred, target)[0]

    def calc_bmc(d):
        """Calculate BMC (correlation contribution vs benchmark)."""
        pred = d["prediction"].values
        bench = d["benchmark"].values
        target = d["target"].values

        # BMC = corr(pred, target) - corr(pred_orthogonal_to_bench, target)
        # This is the contribution of our predictions beyond the benchmark
        corr_pred = spearmanr(pred, target)[0]
        corr_bench = spearmanr(bench, target)[0]

        # Residualize predictions with respect to benchmark
        bench_rank = rankdata(bench)
        pred_rank = rankdata(pred)
        # Simple linear regression residual
        beta = np.corrcoef(pred_rank, bench_rank)[0, 1] * np.std(pred_rank) / np.std(bench_rank)
        resid = pred_rank - beta * bench_rank

        corr_resid = spearmanr(resid, target)[0]
        # BMC is the portion of correlation from orthogonal component
        return corr_resid

    per_era_corr = holdout_df.groupby("era").apply(calc_corr)

    if benchmark_preds is not None and not pd.isna(benchmark_preds).all():
        per_era_bmc = holdout_df.groupby("era").apply(calc_bmc)
    else:
        per_era_bmc = pd.Series(0.0, index=per_era_corr.index)

    payout_metrics = compute_payout_metric(per_era_corr, per_era_bmc)

    corr_mean = float(per_era_corr.mean())
    corr_std = float(per_era_corr.std(ddof=0))
    corr_sharpe = corr_mean / corr_std if corr_std > 0 else 0.0
    corr_cumsum = per_era_corr.cumsum()
    corr_max_dd = float((corr_cumsum.expanding(min_periods=1).max() - corr_cumsum).max())

    bmc_mean = float(per_era_bmc.mean())
    bmc_std = float(per_era_bmc.std(ddof=0))
    bmc_sharpe = bmc_mean / bmc_std if bmc_std > 0 else 0.0
    bmc_cumsum = per_era_bmc.cumsum()
    bmc_max_dd = float((bmc_cumsum.expanding(min_periods=1).max() - bmc_cumsum).max())

    # Additional statistics
    corr_median = float(per_era_corr.median())
    corr_min = float(per_era_corr.min())
    corr_max = float(per_era_corr.max())
    corr_pos_rate = float((per_era_corr > 0).mean())

    bmc_median = float(per_era_bmc.median())
    bmc_min = float(per_era_bmc.min())
    bmc_max = float(per_era_bmc.max())
    bmc_pos_rate = float((per_era_bmc > 0).mean())

    payout_per_era = 0.75 * per_era_corr + 2.25 * per_era_bmc
    payout_per_era = payout_per_era.clip(-0.05, 0.05)
    payout_median = float(payout_per_era.median())
    payout_min = float(payout_per_era.min())
    payout_max = float(payout_per_era.max())
    payout_pos_rate = float((payout_per_era > 0).mean())

    result = {
        # CORR metrics
        "corr_mean": corr_mean,
        "corr_std": corr_std,
        "corr_median": corr_median,
        "corr_min": corr_min,
        "corr_max": corr_max,
        "corr_sharpe": corr_sharpe,
        "corr_max_dd": corr_max_dd,
        "corr_pos_rate": corr_pos_rate,
        # BMC metrics
        "bmc_mean": bmc_mean,
        "bmc_std": bmc_std,
        "bmc_median": bmc_median,
        "bmc_min": bmc_min,
        "bmc_max": bmc_max,
        "bmc_sharpe": bmc_sharpe,
        "bmc_max_dd": bmc_max_dd,
        "bmc_pos_rate": bmc_pos_rate,
        # Payout metrics
        "payout_median": payout_median,
        "payout_min": payout_min,
        "payout_max": payout_max,
        "payout_pos_rate": payout_pos_rate,
        **payout_metrics,
        # Meta
        "n_eras": len(per_era_corr),
        "n_rows": len(holdout_df),
    }

    print(f"\n{'='*70}", flush=True)
    print(f"HOLDOUT RESULTS ({result['n_eras']} eras, {result['n_rows']:,} rows)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"", flush=True)
    print(f"{'CORR Metrics':^35} | {'BMC Metrics':^35}", flush=True)
    print(f"{'-'*35} | {'-'*35}", flush=True)
    print(f"  Mean:     {corr_mean:>10.6f}           |   Mean:     {bmc_mean:>10.6f}", flush=True)
    print(f"  Median:   {corr_median:>10.6f}           |   Median:   {bmc_median:>10.6f}", flush=True)
    print(f"  Std:      {corr_std:>10.6f}           |   Std:      {bmc_std:>10.6f}", flush=True)
    print(f"  Min:      {corr_min:>10.6f}           |   Min:      {bmc_min:>10.6f}", flush=True)
    print(f"  Max:      {corr_max:>10.6f}           |   Max:      {bmc_max:>10.6f}", flush=True)
    print(f"  Sharpe:   {corr_sharpe:>10.4f}           |   Sharpe:   {bmc_sharpe:>10.4f}", flush=True)
    print(f"  Max DD:   {corr_max_dd:>10.4f}           |   Max DD:   {bmc_max_dd:>10.4f}", flush=True)
    print(f"  +ve Rate: {corr_pos_rate:>10.1%}           |   +ve Rate: {bmc_pos_rate:>10.1%}", flush=True)
    print(f"", flush=True)
    print(f"{'PAYOUT Metrics (0.75*CORR + 2.25*BMC, clipped)':^70}", flush=True)
    print(f"{'-'*70}", flush=True)
    print(f"  Mean:     {result['payout_mean']:>10.6f}    Sharpe:    {result['payout_sharpe']:>10.4f}", flush=True)
    print(f"  Median:   {payout_median:>10.6f}    Max DD:    {result['payout_max_dd']:>10.4f}", flush=True)
    print(f"  Min:      {payout_min:>10.6f}    +ve Rate:  {payout_pos_rate:>10.1%}", flush=True)
    print(f"  Max:      {payout_max:>10.6f}    Total:     {result['payout_total']:>10.4f}", flush=True)
    print(f"{'='*70}", flush=True)

    return result
