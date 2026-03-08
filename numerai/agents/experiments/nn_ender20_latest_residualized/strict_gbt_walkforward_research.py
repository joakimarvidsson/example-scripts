from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from agents.code.metrics import numerai_metrics
from agents.code.modeling.utils.target_transforms import (
    subtract_scaled_invnorm_column,
)

warnings.filterwarnings(
    "ignore",
    message=(
        "DataFrameGroupBy.apply operated on the grouping columns. "
        "This behavior is deprecated"
    ),
    category=FutureWarning,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_family: str
    feature_set: str
    offset: int | None
    residual_scale: float
    params: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict walk-forward GBT research on target_ender_20 with "
            "delta-CORR consistency penalties vs benchmark."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("numerai/agents/experiments/nn_ender20_latest_residualized"),
    )
    parser.add_argument(
        "--full-data-path",
        type=Path,
        default=Path("numerai/v5.2/full.parquet"),
    )
    parser.add_argument(
        "--benchmark-data-path",
        type=Path,
        default=Path("numerai/v5.2/full_benchmark_models.parquet"),
    )
    parser.add_argument(
        "--benchmark-model",
        default="v52_lgbm_ender20",
    )
    parser.add_argument("--target-col", default="target_ender_20")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--era-col", default="era")
    parser.add_argument("--min-eval-era", type=int, default=577)
    parser.add_argument("--max-eval-era", type=int, default=1197)
    parser.add_argument(
        "--eval-era-step",
        type=int,
        default=4,
        help="Evaluate every Nth era (default=4).",
    )
    parser.add_argument(
        "--early-era-max",
        type=int,
        default=889,
        help="Upper bound for early-window consistency stats.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=26,
        help="Number of eras per walk-forward validation block.",
    )
    parser.add_argument(
        "--max-rows-per-era",
        type=int,
        default=700,
        help="Cap rows per training era for memory/speed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
    )
    parser.add_argument(
        "--blend-lambdas",
        default="0.05,0.075,0.10,0.125,0.15,0.20,0.30,0.40,0.50,0.75,1.0",
        help=(
            "Benchmark anchor blend lambdas for strict selection. "
            "prediction = benchmark + lambda * (raw - benchmark)."
        ),
    )
    parser.add_argument(
        "--neutralize-benchmark-grid",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.2,1.5,2.0",
        help=(
            "Per-era benchmark neutralization strengths. "
            "prediction = prediction - p * proj(prediction, benchmark)."
        ),
    )
    parser.add_argument(
        "--neutralize-example-grid",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.2,1.5,2.0",
        help=(
            "Per-era example-prediction neutralization strengths. "
            "Only used when example predictions are available."
        ),
    )
    parser.add_argument(
        "--candidate-modes",
        default="neutralize,blend",
        help=(
            "Candidate generation modes for strict selection. "
            "Allowed: neutralize, blend, blend_neutralize."
        ),
    )
    parser.add_argument(
        "--example-preds-path",
        type=Path,
        default=Path("numerai/v5.2/validation_example_preds.parquet"),
        help=(
            "Validation example predictions for decorrelation checks. "
            "If missing, only benchmark decorrelation cap is enforced."
        ),
    )
    parser.add_argument(
        "--max-corr-with-benchmark",
        type=float,
        default=0.90,
        help="Hard cap on max(global corr, mean per-era corr) vs benchmark rank.",
    )
    parser.add_argument(
        "--max-corr-with-example",
        type=float,
        default=0.90,
        help="Hard cap on max(global corr, mean per-era corr) vs example rank.",
    )
    parser.add_argument(
        "--min-delta-mean",
        type=float,
        default=0.0,
        help="Minimum required average per-era CORR delta vs benchmark.",
    )
    parser.add_argument(
        "--min-delta-cumsum-end",
        type=float,
        default=0.0,
        help="Minimum required cumulative CORR delta at end of evaluation.",
    )
    parser.add_argument(
        "--selection-objective",
        choices=["strict_score", "delta_cumsum_end", "corr_sortino_vs_benchmark"],
        default="strict_score",
        help="Primary objective among feasible candidates.",
    )
    parser.add_argument(
        "--mlp-model",
        default="torch_resid_latest_ender20_std008_row_full_blend_noi09_live_ds575plus",
        help="Optional MLP predictions file stem for payout blend sweep.",
    )
    parser.add_argument(
        "--skip-mlp-blend",
        action="store_true",
        help="Skip GBT+MLP payout blend sweep.",
    )
    parser.add_argument(
        "--spec-names",
        default="",
        help=(
            "Optional comma-separated model spec names to run. "
            "Default runs all strict GBT specs."
        ),
    )
    parser.add_argument(
        "--reuse-raw-preds",
        action="store_true",
        help=(
            "Reuse cached raw walk-forward predictions if present "
            "(predictions/<spec_name>_raw_walkfwd.parquet)."
        ),
    )
    parser.add_argument(
        "--best-out-name",
        default="xgb_strict_best_ender20_walkfwd",
        help="Output stem for the selected best strict model.",
    )
    parser.add_argument(
        "--blend-out-name",
        default="blend_xgb_strict_best_ender20_walkfwd_with_mlp",
        help="Output stem for the best GBT+MLP blend artifact.",
    )
    parser.add_argument(
        "--summary-name",
        default="gbt_strict_walkforward_summary.json",
        help="Filename for the run summary JSON under results/.",
    )
    parser.add_argument(
        "--artifact-suffix",
        default="",
        help=(
            "Optional suffix appended to per-spec raw/strict artifact names, "
            "for example '_dense_e4_r700'."
        ),
    )
    return parser.parse_args()


def _load_feature_sets(features_json_path: Path) -> dict[str, list[str]]:
    data = json.loads(features_json_path.read_text())
    return data["feature_sets"]


def _resolve_feature_set_name(name: str) -> str:
    aliases = {"faith2": "faith"}
    return aliases.get(name, name)


def _artifact_name(base: str, suffix: str) -> str:
    suffix = suffix.strip()
    return f"{base}{suffix}" if suffix else base


def _feature_cols_for_spec(feature_sets: dict[str, list[str]], spec: str) -> list[str]:
    tokens = [t.strip() for t in str(spec).split("+") if t.strip()]
    out: list[str] = []
    seen: set[str] = set()

    for token in tokens:
        base = token
        cap: int | None = None
        if ":" in token:
            base, cap_str = token.split(":", 1)
            base = base.strip()
            cap = int(cap_str.strip())
        key = _resolve_feature_set_name(base)
        if key not in feature_sets:
            raise ValueError(f"Unknown feature set token '{base}' (resolved='{key}') in spec '{spec}'")
        cols = feature_sets[key]
        if cap is not None:
            cols = cols[:cap]
        for col in cols:
            if col not in seen:
                seen.add(col)
                out.append(col)
    if not out:
        raise ValueError(f"No feature columns resolved for spec '{spec}'")
    return out


def _rank01_per_era(values: pd.Series, eras: pd.Series) -> pd.Series:
    return values.groupby(eras, sort=False).rank(method="average", pct=True)


def _load_rows_for_eras(
    *,
    full_path: Path,
    bench_path: Path,
    eras: list[int],
    feature_cols: list[str],
    id_col: str,
    era_col: str,
    target_col: str,
    benchmark_model: str,
) -> pd.DataFrame:
    if not eras:
        return pd.DataFrame(columns=[id_col, era_col, target_col, benchmark_model] + feature_cols)

    era_strs = [f"{era:04d}" for era in eras]
    cols = [id_col, era_col, target_col] + feature_cols
    data_df = pd.read_parquet(full_path, columns=cols, filters=[(era_col, "in", era_strs)])

    benchmark_df = pd.read_parquet(
        bench_path,
        columns=[era_col, benchmark_model],
        filters=[(era_col, "in", era_strs)],
    ).reset_index()
    if id_col != benchmark_df.columns[0]:
        benchmark_df = benchmark_df.rename(columns={benchmark_df.columns[0]: id_col})

    merged = data_df.merge(
        benchmark_df[[id_col, era_col, benchmark_model]],
        on=[id_col, era_col],
        how="inner",
        validate="one_to_one",
    )
    merged[era_col] = merged[era_col].astype(str)
    return merged


def _sample_train_rows_per_era(
    train_df: pd.DataFrame,
    era_col: str,
    max_rows_per_era: int,
    seed: int,
) -> pd.DataFrame:
    if max_rows_per_era <= 0:
        return train_df
    if train_df.empty:
        return train_df
    rng = np.random.RandomState(seed)
    parts: list[pd.DataFrame] = []
    for _, g in train_df.groupby(era_col, sort=False):
        if len(g) <= max_rows_per_era:
            parts.append(g)
            continue
        parts.append(
            g.sample(
                n=max_rows_per_era,
                random_state=int(rng.randint(0, 2_000_000_000)),
                replace=False,
            )
        )
    return pd.concat(parts, ignore_index=True)


def _parse_lambdas(value: str) -> list[float]:
    out: list[float] = []
    for tok in value.split(","):
        tok = tok.strip()
        if not tok:
            continue
        x = float(tok)
        if x < 0.0:
            raise ValueError(f"Blend lambda must be >= 0, got {x}")
        out.append(x)
    if not out:
        raise ValueError("No blend lambdas provided.")
    return sorted(set(out))


def _parse_candidate_modes(value: str) -> list[str]:
    allowed = {"neutralize", "blend", "blend_neutralize"}
    modes = [tok.strip().lower() for tok in value.split(",") if tok.strip()]
    if not modes:
        raise ValueError("No --candidate-modes provided.")
    bad = [m for m in modes if m not in allowed]
    if bad:
        raise ValueError(
            f"Unsupported candidate mode(s): {bad}; allowed={sorted(allowed)}"
        )
    # Keep input order but dedupe.
    seen: set[str] = set()
    out: list[str] = []
    for m in modes:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _neutralize_per_era(
    pred: pd.Series,
    ref: pd.Series,
    eras: pd.Series,
    proportion: float,
) -> pd.Series:
    if proportion <= 0.0:
        return pred
    out = pred.copy().astype(np.float64)
    work = pd.DataFrame({"pred": pred, "ref": ref, "era": eras})
    for _, g in work.groupby("era", sort=False):
        idx = g.index
        p = g["pred"].to_numpy(dtype=np.float64, copy=False)
        r = g["ref"].to_numpy(dtype=np.float64, copy=False)
        mask = np.isfinite(p) & np.isfinite(r)
        if mask.sum() < 2:
            continue
        p_valid = p[mask]
        r_valid = r[mask]
        p_center = p_valid - p_valid.mean()
        r_center = r_valid - r_valid.mean()
        denom = float(np.dot(r_center, r_center))
        if denom <= 1e-12:
            continue
        beta = float(np.dot(p_center, r_center) / denom)
        p_valid_adj = p_center - float(proportion) * beta * r_center
        p_adj = p.copy()
        p_adj[mask] = p_valid_adj + p_valid.mean()
        out.loc[idx] = p_adj
    return out


def _compute_delta_metrics(
    df: pd.DataFrame,
    *,
    pred_col: str,
    benchmark_col: str,
    target_col: str,
    era_col: str,
    early_era_max: int,
    compute_bmc: bool = True,
) -> dict:
    corr = numerai_metrics.per_era_corr(
        df, [pred_col, benchmark_col], target_col, era_col=era_col
    )
    model_corr = corr[pred_col]
    bench_corr = corr[benchmark_col]
    delta_corr = (model_corr - bench_corr).sort_index(key=lambda s: s.astype(int))

    if compute_bmc:
        bmc = numerai_metrics.per_era_bmc(
            df, [pred_col], benchmark_col, target_col, era_col=era_col
        )[pred_col].sort_index(key=lambda s: s.astype(int))
        payout = (0.75 * model_corr + 2.25 * bmc).clip(-0.05, 0.05).sort_index(
            key=lambda s: s.astype(int)
        )
        bmc_mean = float(bmc.mean())
        bmc_pos_share = float((bmc > 0).mean())
        payout_mean = float(payout.mean())
    else:
        bmc_mean = np.nan
        bmc_pos_share = np.nan
        payout_mean = np.nan

    early_mask = delta_corr.index.astype(int) <= int(early_era_max)
    early_delta = delta_corr[early_mask]

    downside = np.minimum(delta_corr.to_numpy(dtype=np.float64), 0.0)
    downside_std = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
    delta_sortino = float(delta_corr.mean() / downside_std) if downside_std > 0 else 0.0

    roll20 = delta_corr.rolling(20, min_periods=20).mean()
    cumsum = delta_corr.cumsum()

    score = (
        40.0 * float(delta_corr.mean())
        + 18.0 * delta_sortino
        + (25.0 * bmc_mean if compute_bmc else 0.0)
        + (10.0 * payout_mean if compute_bmc else 0.0)
        + 60.0 * min(float(roll20.min(skipna=True)), 0.0)
        + 40.0 * min(float(cumsum.min()), 0.0)
        + 30.0 * min(float(early_delta.mean()) if len(early_delta) else 0.0, 0.0)
    )

    return {
        "delta_mean": float(delta_corr.mean()),
        "delta_pos_share": float((delta_corr > 0).mean()),
        "delta_cumsum_end": float(cumsum.iloc[-1]),
        "delta_cumsum_min": float(cumsum.min()),
        "delta_cumsum_max": float(cumsum.max()),
        "delta_roll20_min": float(roll20.min(skipna=True)),
        "delta_sortino": float(delta_sortino),
        "early_delta_mean": float(early_delta.mean()) if len(early_delta) else np.nan,
        "bmc_mean": bmc_mean,
        "bmc_pos_share": bmc_pos_share,
        "payout_mean": payout_mean,
        "strict_score": float(score),
    }


def _build_model(spec: ModelSpec):
    if spec.model_family == "xgb":
        return XGBRegressor(**spec.params)
    if spec.model_family == "lgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "lightgbm is required for lgbm specs. Install with `.venv/bin/pip install lightgbm`."
            ) from exc
        return lgb.LGBMRegressor(**spec.params)
    if spec.model_family == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError(
                "catboost is required for catboost specs. Install with `pip install catboost`."
            ) from exc
        return CatBoostRegressor(**spec.params)
    raise ValueError(f"Unsupported model_family: {spec.model_family}")


def _base_model_specs(seed: int) -> list[ModelSpec]:
    variants = [
        (
            "d4lr5e2",
            {
                "n_estimators": 140,
                "learning_rate": 0.05,
                "max_depth": 4,
                "subsample": 0.85,
                "colsample_bytree": 0.25,
                "reg_alpha": 0.1,
                "reg_lambda": 3.0,
                "min_child_weight": 25.0,
            },
        ),
        (
            "d5lr3e2",
            {
                "n_estimators": 240,
                "learning_rate": 0.03,
                "max_depth": 5,
                "subsample": 0.90,
                "colsample_bytree": 0.35,
                "reg_alpha": 0.0,
                "reg_lambda": 2.0,
                "min_child_weight": 20.0,
            },
        ),
        (
            "d3lr7e2",
            {
                "n_estimators": 180,
                "learning_rate": 0.07,
                "max_depth": 3,
                "subsample": 0.90,
                "colsample_bytree": 0.25,
                "reg_alpha": 0.2,
                "reg_lambda": 4.0,
                "min_child_weight": 40.0,
            },
        ),
        (
            "d6lr2e2",
            {
                "n_estimators": 320,
                "learning_rate": 0.02,
                "max_depth": 6,
                "subsample": 0.80,
                "colsample_bytree": 0.40,
                "reg_alpha": 0.0,
                "reg_lambda": 2.0,
                "min_child_weight": 35.0,
            },
        ),
    ]
    xgb_common = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_jobs": -1,
    }
    specs: list[ModelSpec] = []
    idx = 0

    def add_spec(
        name: str,
        residual_scale: float,
        offset: int | None,
        params: dict,
        feature_set: str = "medium",
        model_family: str = "xgb",
    ) -> None:
        nonlocal idx
        seed_value = seed + idx
        merged_params = dict(params)
        if model_family == "xgb":
            merged_params = {**xgb_common, **merged_params, "random_state": seed_value}
        elif model_family == "lgbm":
            merged_params = {**merged_params, "random_state": seed_value}
        elif model_family == "catboost":
            merged_params = {**merged_params, "random_seed": seed_value}
        else:
            merged_params = {**merged_params, "random_state": seed_value}
        specs.append(
            ModelSpec(
                name=name,
                model_family=model_family,
                feature_set=feature_set,
                offset=offset,
                residual_scale=residual_scale,
                params=merged_params,
            )
        )
        idx += 1

    for vname, params in variants[:3]:
        add_spec(
            name=f"xgb_strict_direct_medium_full_{vname}_walkfwd",
            residual_scale=0.0,
            offset=None,
            params=params,
            feature_set="medium",
        )
    for vname, params in variants:
        add_spec(
            name=f"xgb_strict_resid008_medium_full_{vname}_walkfwd",
            residual_scale=0.008,
            offset=None,
            params=params,
            feature_set="medium",
        )
    add_spec(
        name="xgb_strict_resid010_medium_full_d5lr3e2_walkfwd",
        residual_scale=0.010,
        offset=None,
        params=variants[1][1],
        feature_set="medium",
    )
    add_spec(
        name="xgb_strict_resid012_medium_full_d5lr3e2_walkfwd",
        residual_scale=0.012,
        offset=None,
        params=variants[1][1],
        feature_set="medium",
    )
    for off in (0, 1, 2, 3):
        add_spec(
            name=f"xgb_strict_resid008_medium_off{off}_d5lr3e2_walkfwd",
            residual_scale=0.008,
            offset=off,
            params=variants[1][1],
            feature_set="medium",
        )

    # Medium/small with explicit faith slices.
    add_spec(
        name="xgb_strict_direct_medfaith64_full_d5lr3e2_walkfwd",
        residual_scale=0.0,
        offset=None,
        params=variants[1][1],
        feature_set="medium+faith2:64",
    )
    add_spec(
        name="xgb_strict_resid008_medfaith64_full_d5lr3e2_walkfwd",
        residual_scale=0.008,
        offset=None,
        params=variants[1][1],
        feature_set="medium+faith2:64",
    )
    add_spec(
        name="xgb_strict_direct_smallfaith64_full_d5lr3e2_walkfwd",
        residual_scale=0.0,
        offset=None,
        params=variants[1][1],
        feature_set="small+faith2:64",
    )
    add_spec(
        name="xgb_strict_resid010_smallfaith64_full_d5lr3e2_walkfwd",
        residual_scale=0.010,
        offset=None,
        params=variants[1][1],
        feature_set="small+faith2:64",
    )
    add_spec(
        name="xgb_strict_direct_faith96_full_d5lr3e2_walkfwd",
        residual_scale=0.0,
        offset=None,
        params=variants[1][1],
        feature_set="faith2:96",
    )

    # Lower-dimensional feature variants can improve decorrelation feasibility.
    add_spec(
        name="xgb_strict_direct_small_full_d4lr5e2_walkfwd",
        residual_scale=0.0,
        offset=None,
        params=variants[0][1],
        feature_set="small",
    )
    add_spec(
        name="xgb_strict_direct_small_full_d5lr3e2_walkfwd",
        residual_scale=0.0,
        offset=None,
        params=variants[1][1],
        feature_set="small",
    )
    add_spec(
        name="xgb_strict_resid008_small_full_d4lr5e2_walkfwd",
        residual_scale=0.008,
        offset=None,
        params=variants[0][1],
        feature_set="small",
    )
    add_spec(
        name="xgb_strict_resid008_small_full_d5lr3e2_walkfwd",
        residual_scale=0.008,
        offset=None,
        params=variants[1][1],
        feature_set="small",
    )
    add_spec(
        name="xgb_strict_resid010_small_full_d5lr3e2_walkfwd",
        residual_scale=0.010,
        offset=None,
        params=variants[1][1],
        feature_set="small",
    )
    add_spec(
        name="xgb_strict_resid008_all_full_d5lr3e2_walkfwd",
        residual_scale=0.008,
        offset=None,
        params=variants[1][1],
        feature_set="all",
    )
    add_spec(
        name="xgb_strict_resid012_all_full_d5lr3e2_walkfwd",
        residual_scale=0.012,
        offset=None,
        params=variants[1][1],
        feature_set="all",
    )

    # LightGBM DART variants for constrained decorrelation sweeps.
    lgbm_common = {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.35,
        "reg_alpha": 0.0,
        "reg_lambda": 2.0,
        "min_child_samples": 80,
        "objective": "regression",
        "boosting_type": "dart",
        "drop_rate": 0.1,
        "skip_drop": 0.5,
        "n_jobs": -1,
        "verbose": -1,
    }
    add_spec(
        name="lgbm_dart_strict_resid008_medium_walkfwd",
        residual_scale=0.008,
        offset=None,
        params=lgbm_common,
        feature_set="medium",
        model_family="lgbm",
    )
    add_spec(
        name="lgbm_dart_strict_resid010_medium_walkfwd",
        residual_scale=0.010,
        offset=None,
        params=lgbm_common,
        feature_set="medium",
        model_family="lgbm",
    )
    add_spec(
        name="lgbm_dart_strict_resid012_medium_walkfwd",
        residual_scale=0.012,
        offset=None,
        params=lgbm_common,
        feature_set="medium",
        model_family="lgbm",
    )
    lgbm_small = {
        **lgbm_common,
        "num_leaves": 31,
        "colsample_bytree": 0.25,
    }
    add_spec(
        name="lgbm_dart_strict_resid010_small_walkfwd",
        residual_scale=0.010,
        offset=None,
        params=lgbm_small,
        feature_set="small",
        model_family="lgbm",
    )
    add_spec(
        name="lgbm_dart_strict_resid008_medfaith64_walkfwd",
        residual_scale=0.008,
        offset=None,
        params=lgbm_common,
        feature_set="medium+faith2:64",
        model_family="lgbm",
    )
    add_spec(
        name="lgbm_dart_strict_resid010_smallfaith64_walkfwd",
        residual_scale=0.010,
        offset=None,
        params=lgbm_small,
        feature_set="small+faith2:64",
        model_family="lgbm",
    )

    cat_common = {
        "iterations": 450,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 8.0,
        "subsample": 0.85,
        "loss_function": "RMSE",
        "allow_writing_files": False,
        "thread_count": -1,
        "verbose": 0,
    }
    add_spec(
        name="cat_strict_direct_medfaith64_walkfwd",
        residual_scale=0.0,
        offset=None,
        params=cat_common,
        feature_set="medium+faith2:64",
        model_family="catboost",
    )
    add_spec(
        name="cat_strict_resid008_medfaith64_walkfwd",
        residual_scale=0.008,
        offset=None,
        params=cat_common,
        feature_set="medium+faith2:64",
        model_family="catboost",
    )
    add_spec(
        name="cat_strict_resid006_medfaith64_walkfwd",
        residual_scale=0.006,
        offset=None,
        params=cat_common,
        feature_set="medium+faith2:64",
        model_family="catboost",
    )
    add_spec(
        name="cat_strict_resid010_medfaith64_walkfwd",
        residual_scale=0.010,
        offset=None,
        params=cat_common,
        feature_set="medium+faith2:64",
        model_family="catboost",
    )
    add_spec(
        name="cat_strict_resid008_medfaith64_d5_walkfwd",
        residual_scale=0.008,
        offset=None,
        params={**cat_common, "depth": 5},
        feature_set="medium+faith2:64",
        model_family="catboost",
    )
    add_spec(
        name="cat_strict_resid008_medfaith64_lr2p5e2_walkfwd",
        residual_scale=0.008,
        offset=None,
        params={
            **cat_common,
            "iterations": 600,
            "learning_rate": 0.025,
            "l2_leaf_reg": 10.0,
        },
        feature_set="medium+faith2:64",
        model_family="catboost",
    )
    add_spec(
        name="cat_strict_resid008_smallfaith64_walkfwd",
        residual_scale=0.008,
        offset=None,
        params={**cat_common, "depth": 5},
        feature_set="small+faith2:64",
        model_family="catboost",
    )
    add_spec(
        name="cat_strict_resid010_smallfaith64_walkfwd",
        residual_scale=0.010,
        offset=None,
        params={**cat_common, "depth": 5},
        feature_set="small+faith2:64",
        model_family="catboost",
    )
    return specs


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    m = pd.concat([a, b], axis=1).dropna()
    if m.shape[0] < 2:
        return np.nan
    return float(m.iloc[:, 0].corr(m.iloc[:, 1]))


def _corr_vs_reference_per_era_mean(
    df: pd.DataFrame,
    *,
    pred_col: str,
    ref_col: str,
    era_col: str,
) -> float:
    vals: list[float] = []
    for _, g in df.groupby(era_col, sort=False):
        c = _safe_corr(g[pred_col], g[ref_col])
        if not np.isnan(c):
            vals.append(float(c))
    if not vals:
        return np.nan
    return float(np.mean(vals))


def _load_example_preds_for_eval(
    *,
    example_preds_path: Path,
    full_path: Path,
    id_col: str,
    era_col: str,
    eval_eras: list[int],
) -> pd.DataFrame | None:
    if not example_preds_path.exists():
        print(
            f"Warning: missing {example_preds_path}; example correlation cap disabled.",
            flush=True,
        )
        return None

    ex = pd.read_parquet(example_preds_path)
    if id_col not in ex.columns:
        ex = ex.reset_index()
        if id_col not in ex.columns:
            ex = ex.rename(columns={ex.columns[0]: id_col})

    pred_col = "prediction"
    if pred_col not in ex.columns:
        numeric_cols = [c for c in ex.columns if c != id_col and pd.api.types.is_numeric_dtype(ex[c])]
        if not numeric_cols:
            raise ValueError(f"No numeric prediction column found in {example_preds_path}")
        pred_col = numeric_cols[0]

    ex = ex[[id_col, pred_col]].rename(columns={pred_col: "example_prediction"})

    era_strs = [f"{era:04d}" for era in eval_eras]
    id_era = pd.read_parquet(
        full_path,
        columns=[id_col, era_col],
        filters=[(era_col, "in", era_strs)],
    )
    id_era[era_col] = id_era[era_col].astype(str)
    merged = id_era.merge(ex, on=id_col, how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError(
            f"{example_preds_path} did not align to evaluation rows by id."
        )
    return merged


def _train_walkforward_model(
    spec: ModelSpec,
    *,
    full_path: Path,
    bench_path: Path,
    all_eras: list[int],
    eval_eras: list[int],
    block_size: int,
    max_rows_per_era: int,
    id_col: str,
    era_col: str,
    target_col: str,
    benchmark_model: str,
    feature_cols: list[str],
    seed: int,
) -> pd.DataFrame:
    blocks = [
        eval_eras[i : i + block_size] for i in range(0, len(eval_eras), block_size)
    ]
    preds: list[pd.DataFrame] = []

    for block_idx, val_eras in enumerate(blocks):
        train_end = min(val_eras)
        train_eras = [era for era in all_eras if era < train_end]
        if spec.offset is not None:
            train_eras = [
                era for idx, era in enumerate(train_eras) if idx % 4 == int(spec.offset)
            ]
        if not train_eras:
            raise ValueError(f"{spec.name}: no train eras for block {block_idx}.")

        train_df = _load_rows_for_eras(
            full_path=full_path,
            bench_path=bench_path,
            eras=train_eras,
            feature_cols=feature_cols,
            id_col=id_col,
            era_col=era_col,
            target_col=target_col,
            benchmark_model=benchmark_model,
        )
        train_df = _sample_train_rows_per_era(
            train_df, era_col, max_rows_per_era=max_rows_per_era, seed=seed + block_idx
        )
        val_df = _load_rows_for_eras(
            full_path=full_path,
            bench_path=bench_path,
            eras=val_eras,
            feature_cols=feature_cols,
            id_col=id_col,
            era_col=era_col,
            target_col=target_col,
            benchmark_model=benchmark_model,
        )

        y_train = train_df[target_col]
        if spec.residual_scale > 0:
            y_train = subtract_scaled_invnorm_column(
                y_train,
                train_df[[benchmark_model, era_col]],
                benchmark_col=benchmark_model,
                era_col=era_col,
                scale=spec.residual_scale,
                per_era=True,
                use_rank=False,
                clip_eps=1e-6,
                center=True,
                center_per_era=False,
            )

        model = _build_model(spec)
        model.fit(train_df[feature_cols], y_train)
        pred = model.predict(val_df[feature_cols]).astype(np.float64)

        out = val_df[[id_col, era_col, target_col, benchmark_model]].copy()
        out["prediction_raw"] = pred
        preds.append(out)

        print(
            f"{spec.name}: block {block_idx + 1}/{len(blocks)} "
            f"train_eras={len(train_eras)} train_rows={len(train_df):,} "
            f"val_eras={len(val_eras)} val_rows={len(val_df):,}"
        , flush=True)

    pred_df = pd.concat(preds, ignore_index=True)
    if pred_df[id_col].duplicated().any():
        raise ValueError(f"{spec.name}: duplicate ids in predictions.")
    return pred_df


def _select_strict_blend(
    pred_df: pd.DataFrame,
    *,
    id_col: str,
    example_df: pd.DataFrame | None,
    benchmark_col: str,
    target_col: str,
    era_col: str,
    early_era_max: int,
    blend_lambdas: list[float],
    neutralize_benchmark_grid: list[float],
    neutralize_example_grid: list[float],
    candidate_modes: list[str],
    max_corr_with_benchmark: float,
    max_corr_with_example: float,
    min_delta_mean: float,
    min_delta_cumsum_end: float,
    selection_objective: str,
) -> tuple[pd.DataFrame, dict]:
    ranked = pred_df.copy()
    ranked["prediction_rank"] = _rank01_per_era(ranked["prediction_raw"], ranked[era_col])
    ranked["benchmark_rank"] = _rank01_per_era(ranked[benchmark_col], ranked[era_col])
    if example_df is not None:
        ranked = ranked.merge(
            example_df[[id_col, era_col, "example_prediction"]],
            on=[id_col, era_col],
            how="left",
            validate="one_to_one",
        )
        ranked["example_rank"] = _rank01_per_era(ranked["example_prediction"], ranked[era_col])

    best_feasible_metrics: dict | None = None
    best_feasible_pred: pd.Series | None = None
    best_overall_metrics: dict | None = None
    best_overall_pred: pd.Series | None = None
    candidate_compute_bmc = selection_objective != "corr_sortino_vs_benchmark"

    if "example_rank" in ranked.columns:
        neutralize_example_grid_local = neutralize_example_grid
    else:
        neutralize_example_grid_local = [0.0]

    candidate_settings: list[dict[str, float | str]] = []
    for mode in candidate_modes:
        if mode == "blend":
            for lam in blend_lambdas:
                candidate_settings.append(
                    {"mode": "blend", "lambda": float(lam), "neutralize_bench": 0.0, "neutralize_example": 0.0}
                )
        elif mode == "neutralize":
            for nb in neutralize_benchmark_grid:
                for ne in neutralize_example_grid_local:
                    candidate_settings.append(
                        {
                            "mode": "neutralize",
                            "lambda": 1.0,
                            "neutralize_bench": float(nb),
                            "neutralize_example": float(ne),
                        }
                    )
        elif mode == "blend_neutralize":
            for lam in blend_lambdas:
                for nb in neutralize_benchmark_grid:
                    for ne in neutralize_example_grid_local:
                        candidate_settings.append(
                            {
                                "mode": "blend_neutralize",
                                "lambda": float(lam),
                                "neutralize_bench": float(nb),
                                "neutralize_example": float(ne),
                            }
                        )

    for setting in candidate_settings:
        cand = ranked.copy()
        mode = str(setting["mode"])
        if mode == "blend":
            lam = float(setting["lambda"])
            cand["prediction"] = cand["benchmark_rank"] + lam * (
                cand["prediction_rank"] - cand["benchmark_rank"]
            )
        elif mode == "neutralize":
            nb = float(setting["neutralize_bench"])
            ne = float(setting["neutralize_example"])
            pred = _neutralize_per_era(
                cand["prediction_rank"],
                cand["benchmark_rank"],
                cand[era_col],
                proportion=nb,
            )
            if "example_rank" in cand.columns and ne > 0.0:
                pred = _neutralize_per_era(
                    pred,
                    cand["example_rank"],
                    cand[era_col],
                    proportion=ne,
                )
            cand["prediction"] = pred
        elif mode == "blend_neutralize":
            lam = float(setting["lambda"])
            nb = float(setting["neutralize_bench"])
            ne = float(setting["neutralize_example"])
            pred = cand["benchmark_rank"] + lam * (
                cand["prediction_rank"] - cand["benchmark_rank"]
            )
            if nb > 0.0:
                pred = _neutralize_per_era(
                    pred,
                    cand["benchmark_rank"],
                    cand[era_col],
                    proportion=nb,
                )
            if "example_rank" in cand.columns and ne > 0.0:
                pred = _neutralize_per_era(
                    pred,
                    cand["example_rank"],
                    cand[era_col],
                    proportion=ne,
                )
            cand["prediction"] = pred
        else:
            raise ValueError(f"Unsupported candidate mode: {mode}")
        cand["prediction"] = _rank01_per_era(cand["prediction"], cand[era_col])
        metrics = _compute_delta_metrics(
            cand,
            pred_col="prediction",
            benchmark_col=benchmark_col,
            target_col=target_col,
            era_col=era_col,
            early_era_max=early_era_max,
            compute_bmc=candidate_compute_bmc,
        )
        metrics["mode"] = mode
        metrics["lambda"] = float(setting["lambda"])
        metrics["neutralize_bench"] = float(setting["neutralize_bench"])
        metrics["neutralize_example"] = float(setting["neutralize_example"])
        corr_bench_global = _safe_corr(cand["prediction"], cand["benchmark_rank"])
        corr_bench_era_mean = _corr_vs_reference_per_era_mean(
            cand, pred_col="prediction", ref_col="benchmark_rank", era_col=era_col
        )
        corr_bench_max = float(np.nanmax(np.abs([corr_bench_global, corr_bench_era_mean])))
        metrics["corr_with_benchmark_global"] = corr_bench_global
        metrics["corr_with_benchmark_era_mean"] = corr_bench_era_mean
        metrics["corr_with_benchmark_max_abs"] = corr_bench_max

        corr_example_global = np.nan
        corr_example_era_mean = np.nan
        corr_example_max = np.nan
        if "example_rank" in cand.columns:
            corr_example_global = _safe_corr(cand["prediction"], cand["example_rank"])
            corr_example_era_mean = _corr_vs_reference_per_era_mean(
                cand, pred_col="prediction", ref_col="example_rank", era_col=era_col
            )
            corr_example_max = float(
                np.nanmax(np.abs([corr_example_global, corr_example_era_mean]))
            )
        metrics["corr_with_example_global"] = corr_example_global
        metrics["corr_with_example_era_mean"] = corr_example_era_mean
        metrics["corr_with_example_max_abs"] = corr_example_max

        feasible = (
            corr_bench_max <= float(max_corr_with_benchmark)
            and metrics["delta_mean"] >= float(min_delta_mean)
            and metrics["delta_cumsum_end"] >= float(min_delta_cumsum_end)
        )
        if not np.isnan(corr_example_max):
            feasible = feasible and (corr_example_max <= float(max_corr_with_example))
        metrics["feasible"] = bool(feasible)
        if selection_objective == "delta_cumsum_end":
            objective = float(metrics["delta_cumsum_end"])
        elif selection_objective == "corr_sortino_vs_benchmark":
            objective = float(metrics["delta_sortino"])
        else:
            objective = float(metrics["strict_score"])
        metrics["selection_objective_value"] = objective

        if best_overall_metrics is None or objective > best_overall_metrics["selection_objective_value"]:
            best_overall_metrics = metrics
            best_overall_pred = cand["prediction"]
        if feasible and (
            best_feasible_metrics is None
            or objective > best_feasible_metrics["selection_objective_value"]
        ):
            best_feasible_metrics = metrics
            best_feasible_pred = cand["prediction"]

    selected_feasible = best_feasible_metrics is not None
    best_metrics = best_feasible_metrics if selected_feasible else best_overall_metrics
    best_pred = best_feasible_pred if selected_feasible else best_overall_pred

    assert best_metrics is not None
    assert best_pred is not None
    best_metrics = dict(best_metrics)
    best_metrics["selected_feasible"] = bool(selected_feasible)
    out = pred_df[[c for c in pred_df.columns if c != "prediction_raw"]].copy()
    out["prediction"] = best_pred.to_numpy(dtype=np.float64)

    # Compute full metrics (including BMC/payout) only for the selected candidate.
    final_metrics = _compute_delta_metrics(
        out,
        pred_col="prediction",
        benchmark_col=benchmark_col,
        target_col=target_col,
        era_col=era_col,
        early_era_max=early_era_max,
        compute_bmc=True,
    )
    for key in (
        "mode",
        "lambda",
        "neutralize_bench",
        "neutralize_example",
        "corr_with_benchmark_global",
        "corr_with_benchmark_era_mean",
        "corr_with_benchmark_max_abs",
        "corr_with_example_global",
        "corr_with_example_era_mean",
        "corr_with_example_max_abs",
        "feasible",
        "selection_objective_value",
        "selected_feasible",
    ):
        final_metrics[key] = best_metrics[key]
    return out, final_metrics


def _load_prediction_df(path: Path, *, id_col: str, era_col: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {id_col, era_col, "prediction"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def _sweep_gbt_mlp_blend(
    gbt_df: pd.DataFrame,
    mlp_df: pd.DataFrame,
    *,
    id_col: str,
    era_col: str,
    target_col: str,
    benchmark_col: str,
    early_era_max: int,
) -> tuple[pd.DataFrame, dict]:
    merged = gbt_df[[id_col, era_col, target_col, benchmark_col, "prediction"]].rename(
        columns={"prediction": "prediction_gbt"}
    ).merge(
        mlp_df[[id_col, "prediction"]].rename(columns={"prediction": "prediction_mlp"}),
        on=id_col,
        how="inner",
        validate="one_to_one",
    )

    ranked = merged.copy()
    ranked["gbt_rank"] = _rank01_per_era(ranked["prediction_gbt"], ranked[era_col])
    ranked["mlp_rank"] = _rank01_per_era(ranked["prediction_mlp"], ranked[era_col])

    best_metrics: dict | None = None
    best_pred: pd.Series | None = None
    for w_gbt in np.linspace(0.50, 1.0, 11):
        w_mlp = 1.0 - float(w_gbt)
        cand = ranked.copy()
        cand["prediction"] = w_gbt * cand["gbt_rank"] + w_mlp * cand["mlp_rank"]
        metrics = _compute_delta_metrics(
            cand,
            pred_col="prediction",
            benchmark_col=benchmark_col,
            target_col=target_col,
            era_col=era_col,
            early_era_max=early_era_max,
        )
        metrics["w_gbt"] = float(w_gbt)
        metrics["w_mlp"] = float(w_mlp)
        if best_metrics is None or metrics["payout_mean"] > best_metrics["payout_mean"]:
            best_metrics = metrics
            best_pred = cand["prediction"]

    assert best_metrics is not None
    assert best_pred is not None
    out = merged[[id_col, era_col, target_col, benchmark_col]].copy()
    out["prediction"] = best_pred.to_numpy(dtype=np.float64)
    return out, best_metrics


def _write_result_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _resolve_features_json() -> Path:
    candidates = [
        Path("numerai/v5.2/features.json"),
        Path("v5.2/features.json"),
        Path(__file__).resolve().parents[4] / "v5.2" / "features.json",
    ]
    for cand in candidates:
        c = cand.resolve()
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not locate features.json in expected locations: "
        + ", ".join(str(c) for c in candidates)
    )


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    predictions_dir = experiment_dir / "predictions"
    results_dir = experiment_dir / "results"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    feature_sets = _load_feature_sets(_resolve_features_json())
    specs = _base_model_specs(seed=args.seed)
    if args.spec_names.strip():
        wanted = {name.strip() for name in args.spec_names.split(",") if name.strip()}
        specs = [spec for spec in specs if spec.name in wanted]
        if not specs:
            raise ValueError(f"No matching specs in --spec-names: {sorted(wanted)}")

    era_only = pd.read_parquet(
        args.full_data_path.resolve(), columns=[args.era_col]
    )
    all_eras = sorted({int(e) for e in era_only[args.era_col].astype(str).tolist()})
    eval_eras = [
        era
        for era in all_eras
        if (
            int(args.min_eval_era) <= era <= int(args.max_eval_era)
            and ((era - int(args.min_eval_era)) % int(args.eval_era_step) == 0)
        )
    ]
    if not eval_eras:
        raise ValueError("No evaluation eras found.")

    example_df = _load_example_preds_for_eval(
        example_preds_path=args.example_preds_path.resolve(),
        full_path=args.full_data_path.resolve(),
        id_col=args.id_col,
        era_col=args.era_col,
        eval_eras=eval_eras,
    )

    run_rows: list[dict] = []
    strict_candidates: list[tuple[str, pd.DataFrame, dict]] = []
    lambdas = _parse_lambdas(args.blend_lambdas)
    neutralize_benchmark_grid = _parse_lambdas(args.neutralize_benchmark_grid)
    neutralize_example_grid = _parse_lambdas(args.neutralize_example_grid)
    candidate_modes = _parse_candidate_modes(args.candidate_modes)

    for spec in specs:
        feature_cols = _feature_cols_for_spec(feature_sets, spec.feature_set)
        raw_cache_name = _artifact_name(f"{spec.name}_raw_walkfwd", args.artifact_suffix)
        raw_cache_path = predictions_dir / f"{raw_cache_name}.parquet"
        raw_pred_df: pd.DataFrame | None = None
        if args.reuse_raw_preds and raw_cache_path.exists():
            cached = pd.read_parquet(raw_cache_path)
            required = {
                args.id_col,
                args.era_col,
                args.target_col,
                args.benchmark_model,
                "prediction_raw",
            }
            missing = required.difference(cached.columns)
            if missing:
                print(
                    f"Cache mismatch for {spec.name} (missing {sorted(missing)}); retraining.",
                    flush=True,
                )
            else:
                raw_pred_df = cached
                print(f"Reused cached raw predictions: {raw_cache_path}", flush=True)

        if raw_pred_df is None:
            raw_pred_df = _train_walkforward_model(
                spec,
                full_path=args.full_data_path.resolve(),
                bench_path=args.benchmark_data_path.resolve(),
                all_eras=all_eras,
                eval_eras=eval_eras,
                block_size=args.block_size,
                max_rows_per_era=args.max_rows_per_era,
                id_col=args.id_col,
                era_col=args.era_col,
                target_col=args.target_col,
                benchmark_model=args.benchmark_model,
                feature_cols=feature_cols,
                seed=args.seed,
            )
            raw_pred_df.to_parquet(raw_cache_path, index=False)
            print(f"Saved raw predictions cache: {raw_cache_path}", flush=True)

        strict_df, strict_metrics = _select_strict_blend(
            raw_pred_df,
            id_col=args.id_col,
            example_df=example_df,
            benchmark_col=args.benchmark_model,
            target_col=args.target_col,
            era_col=args.era_col,
            early_era_max=args.early_era_max,
            blend_lambdas=lambdas,
            neutralize_benchmark_grid=neutralize_benchmark_grid,
            neutralize_example_grid=neutralize_example_grid,
            candidate_modes=candidate_modes,
            max_corr_with_benchmark=float(args.max_corr_with_benchmark),
            max_corr_with_example=float(args.max_corr_with_example),
            min_delta_mean=float(args.min_delta_mean),
            min_delta_cumsum_end=float(args.min_delta_cumsum_end),
            selection_objective=str(args.selection_objective),
        )
        strict_name = _artifact_name(f"{spec.name}_strict", args.artifact_suffix)
        strict_path = predictions_dir / f"{strict_name}.parquet"
        strict_df.rename(columns={args.benchmark_model: "benchmark_prediction"}).to_parquet(
            strict_path, index=False
        )

        result_payload = {
            "data": {
                "data_version": "v5.2",
                "feature_set": spec.feature_set,
                "target": args.target_col,
                "oof_rows": int(strict_df.shape[0]),
                "oof_eras": int(len(eval_eras)),
                "walkforward_block_size_eras": int(args.block_size),
                "max_rows_per_era": int(args.max_rows_per_era),
            },
            "benchmark": {
                "model": args.benchmark_model,
                "file": str(args.benchmark_data_path),
            },
            "model": {
                "type": "xgb_walkforward_strict_delta",
                "base_model_name": spec.name,
                "offset": spec.offset,
                "residual_scale": spec.residual_scale,
                "params": spec.params,
            },
            "output": {"predictions_file": str(strict_path.relative_to(experiment_dir.parent))},
            "metrics": strict_metrics,
        }
        _write_result_json(results_dir / f"{strict_name}.json", result_payload)
        run_rows.append(
            {
                "model": strict_name,
                "feature_set": spec.feature_set,
                "offset": spec.offset if spec.offset is not None else "full",
                **strict_metrics,
            }
        )
        strict_candidates.append((strict_name, strict_df, strict_metrics))

    summary_df = pd.DataFrame(run_rows).sort_values("strict_score", ascending=False)
    feasible_count = int(summary_df["feasible"].sum()) if "feasible" in summary_df.columns else 0
    print("\nTop strict GBT candidates")
    print(
        summary_df[
            [
                "model",
                "offset",
                "mode",
                "lambda",
                "neutralize_bench",
                "neutralize_example",
                "feasible",
                "delta_mean",
                "early_delta_mean",
                "delta_roll20_min",
                "delta_cumsum_end",
                "delta_cumsum_min",
                "corr_with_benchmark_max_abs",
                "corr_with_example_max_abs",
                "bmc_mean",
                "payout_mean",
                "strict_score",
            ]
        ].to_string(index=False)
    )
    print(f"\nFeasible models: {feasible_count}/{len(summary_df)}", flush=True)

    feasible_df = summary_df[summary_df["feasible"] == True] if "feasible" in summary_df.columns else pd.DataFrame()
    if not feasible_df.empty:
        if args.selection_objective == "delta_cumsum_end":
            best_name = str(feasible_df.sort_values("delta_cumsum_end", ascending=False).iloc[0]["model"])
        elif args.selection_objective == "corr_sortino_vs_benchmark":
            best_name = str(feasible_df.sort_values("delta_sortino", ascending=False).iloc[0]["model"])
        else:
            best_name = str(feasible_df.sort_values("strict_score", ascending=False).iloc[0]["model"])
    else:
        best_name = str(summary_df.iloc[0]["model"])
    best_df = next(df for name, df, _ in strict_candidates if name == best_name)
    best_metrics = next(m for name, _, m in strict_candidates if name == best_name)

    best_out_name = args.best_out_name
    best_out_path = predictions_dir / f"{best_out_name}.parquet"
    best_df.rename(columns={args.benchmark_model: "benchmark_prediction"}).to_parquet(
        best_out_path, index=False
    )
    _write_result_json(
        results_dir / f"{best_out_name}.json",
        {
            "selection": {
                "best_base_model": best_name,
                "selection_objective": args.selection_objective,
                "required_constraints": {
                    "max_corr_with_benchmark": args.max_corr_with_benchmark,
                    "max_corr_with_example": args.max_corr_with_example,
                    "min_delta_mean": args.min_delta_mean,
                    "min_delta_cumsum_end": args.min_delta_cumsum_end,
                },
            },
            "metrics": best_metrics,
            "output": {
                "predictions_file": str(best_out_path.relative_to(experiment_dir.parent))
            },
        },
    )

    if not args.skip_mlp_blend:
        mlp_path = predictions_dir / f"{args.mlp_model}.parquet"
        if mlp_path.exists():
            mlp_df = _load_prediction_df(
                mlp_path, id_col=args.id_col, era_col=args.era_col
            )
            blend_df, blend_metrics = _sweep_gbt_mlp_blend(
                best_df,
                mlp_df,
                id_col=args.id_col,
                era_col=args.era_col,
                target_col=args.target_col,
                benchmark_col=args.benchmark_model,
                early_era_max=args.early_era_max,
            )
            blend_name = args.blend_out_name
            blend_path = predictions_dir / f"{blend_name}.parquet"
            blend_df.rename(columns={args.benchmark_model: "benchmark_prediction"}).to_parquet(
                blend_path, index=False
            )
            _write_result_json(
                results_dir / f"{blend_name}.json",
                {
                    "selection": {
                        "gbt_model": best_name,
                        "mlp_model": args.mlp_model,
                        "selection_objective": "payout_mean",
                    },
                    "metrics": blend_metrics,
                    "output": {
                        "predictions_file": str(
                            blend_path.relative_to(experiment_dir.parent)
                        )
                    },
                },
            )
            print("\nBest GBT+MLP blend by payout_mean")
            print(json.dumps(blend_metrics, indent=2))
        else:
            print(f"\nSkipped MLP blend: missing {mlp_path}")

    summary_path = results_dir / args.summary_name
    _write_result_json(
        summary_path,
        {
            "settings": {
                "min_eval_era": args.min_eval_era,
                "max_eval_era": args.max_eval_era,
                "eval_era_step": args.eval_era_step,
                "early_era_max": args.early_era_max,
                "block_size": args.block_size,
                "max_rows_per_era": args.max_rows_per_era,
                "blend_lambdas": lambdas,
                "neutralize_benchmark_grid": neutralize_benchmark_grid,
                "neutralize_example_grid": neutralize_example_grid,
                "candidate_modes": candidate_modes,
                "example_preds_path": str(args.example_preds_path),
                "max_corr_with_benchmark": args.max_corr_with_benchmark,
                "max_corr_with_example": args.max_corr_with_example,
                "min_delta_mean": args.min_delta_mean,
                "min_delta_cumsum_end": args.min_delta_cumsum_end,
                "selection_objective": args.selection_objective,
            },
            "top_models": summary_df.to_dict(orient="records"),
            "selected_best_model": best_name,
            "feasible_model_count": feasible_count,
        },
    )
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
