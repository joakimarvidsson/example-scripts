from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

@dataclass(frozen=True)
class EraSlice:
    start: int
    end: int
    era: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage correlation-cap tuning for cached raw walk-forward GBT predictions. "
            "Stage 1: coarse neutralization/blend grid. Stage 2: fine local grid."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("numerai/agents/experiments/nn_ender20_latest_residualized"),
    )
    parser.add_argument(
        "--model-files",
        default=(
            "xgb_strict_resid012_medium_full_d5lr3e2_walkfwd_raw_walkfwd.parquet,"
            "xgb_strict_resid008_medium_full_d6lr2e2_walkfwd_raw_walkfwd.parquet,"
            "lgbm_dart_strict_resid012_medium_walkfwd_raw_walkfwd.parquet"
        ),
        help=(
            "Comma-separated prediction files under <experiment-dir>/predictions, "
            "containing prediction_raw."
        ),
    )
    parser.add_argument(
        "--example-preds-path",
        type=Path,
        default=Path("numerai/v5.2/validation_example_preds.parquet"),
    )
    parser.add_argument(
        "--feature-data-path",
        type=Path,
        default=Path("numerai/v5.2/full.parquet"),
        help="Dataset used to load feature exposures for feature neutralization.",
    )
    parser.add_argument(
        "--features-json-path",
        type=Path,
        default=None,
        help="Optional explicit path to features.json. Defaults to alongside feature-data-path.",
    )
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--era-col", default="era")
    parser.add_argument("--target-col", default="target_ender_20")
    parser.add_argument("--benchmark-col", default="v52_lgbm_ender20")
    parser.add_argument("--min-era", type=int, default=577)
    parser.add_argument("--max-era", type=int, default=1197)
    parser.add_argument("--early-era-max", type=int, default=889)
    parser.add_argument("--max-corr-cap", type=float, default=0.99)
    parser.add_argument("--min-delta-mean", type=float, default=0.0)
    parser.add_argument("--min-delta-cumsum-end", type=float, default=0.0)
    parser.add_argument(
        "--selection-objective",
        choices=[
            "delta_cumsum_end",
            "payout_mean",
            "payout_sortino",
            "payout_cumsum_end",
        ],
        default="delta_cumsum_end",
        help=(
            "Candidate selection objective. Use payout_* objectives to optimize "
            "the proxy payout function directly."
        ),
    )
    parser.add_argument(
        "--payout-clip",
        type=float,
        default=0.05,
        help="Per-era payout clip for proxy payout = 0.75*corr + 2.25*bmc.",
    )
    parser.add_argument(
        "--coarse-lambda-grid",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50",
    )
    parser.add_argument(
        "--coarse-neutralize-grid",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )
    parser.add_argument(
        "--coarse-neutralize-example-grid",
        default="0.0",
        help="Per-era example-prediction neutralization proportions.",
    )
    parser.add_argument(
        "--feature-specs",
        default="",
        help=(
            "Comma-separated feature exposure specs for joint feature neutralization, "
            "for example 'small:64,faith2:64,medium:64'. Empty disables feature neutralization."
        ),
    )
    parser.add_argument(
        "--coarse-feature-neutralize-grid",
        default="0.0",
        help="Per-era feature-exposure neutralization proportions.",
    )
    parser.add_argument(
        "--coarse-ridge-alpha-grid",
        default="0.0,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1",
        help=(
            "Ridge alpha grid for per-era neutralization regression. "
            "0.0 reproduces OLS-style neutralization."
        ),
    )
    parser.add_argument(
        "--coarse-feature-ridge-alpha-grid",
        default="",
        help=(
            "Optional coarse ridge-alpha grid for feature neutralization. "
            "Defaults to --coarse-ridge-alpha-grid."
        ),
    )
    parser.add_argument("--fine-window-lambda", type=float, default=0.05)
    parser.add_argument("--fine-window-neutralize", type=float, default=0.05)
    parser.add_argument("--fine-window-neutralize-example", type=float, default=0.05)
    parser.add_argument("--fine-window-feature-neutralize", type=float, default=0.05)
    parser.add_argument("--fine-step-lambda", type=float, default=0.01)
    parser.add_argument("--fine-step-neutralize", type=float, default=0.01)
    parser.add_argument("--fine-step-neutralize-example", type=float, default=0.01)
    parser.add_argument("--fine-step-feature-neutralize", type=float, default=0.01)
    parser.add_argument(
        "--fine-ridge-alpha-grid",
        default="",
        help=(
            "Optional explicit fine ridge-alpha grid. "
            "If omitted, a local multiplier grid around coarse best is used."
        ),
    )
    parser.add_argument(
        "--fine-feature-ridge-alpha-grid",
        default="",
        help=(
            "Optional explicit fine ridge-alpha grid for feature neutralization. "
            "If omitted, a local multiplier grid around coarse best is used."
        ),
    )
    parser.add_argument(
        "--fine-ridge-alpha-multipliers",
        default="0.25,0.5,1.0,2.0,4.0",
        help=(
            "Local multipliers around coarse-best ridge_alpha for fine search "
            "when --fine-ridge-alpha-grid is omitted."
        ),
    )
    parser.add_argument(
        "--fine-feature-ridge-alpha-multipliers",
        default="0.25,0.5,1.0,2.0,4.0",
        help=(
            "Local multipliers around coarse-best feature ridge alpha for fine search "
            "when --fine-feature-ridge-alpha-grid is omitted."
        ),
    )
    parser.add_argument(
        "--output-name",
        default=f"corrcap_two_stage_tune_{date.today()}.json",
    )
    return parser.parse_args()


def _parse_float_grid(grid: str) -> list[float]:
    vals: list[float] = []
    for x in grid.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    if not vals:
        raise ValueError("Empty grid.")
    return sorted(set(vals))


def _resolve_feature_set_name(name: str) -> str:
    aliases = {"faith2": "faith"}
    return aliases.get(name, name)


def _load_feature_sets(features_json_path: Path) -> dict[str, list[str]]:
    data = json.loads(features_json_path.read_text())
    return data["feature_sets"]


def _feature_cols_for_spec(feature_sets: dict[str, list[str]], spec: str) -> list[str]:
    if str(spec).strip().lower() in {"", "none", "null"}:
        return []
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
            raise ValueError(f"Unknown feature set token '{base}' in spec '{spec}'")
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


def _fast_corr(x: np.ndarray, y: np.ndarray) -> float:
    x64 = x.astype(np.float64, copy=False)
    y64 = y.astype(np.float64, copy=False)
    xm = x64.mean()
    ym = y64.mean()
    xc = x64 - xm
    yc = y64 - ym
    denom = float(np.sqrt(np.dot(xc, xc) * np.dot(yc, yc)))
    if denom <= 1e-16:
        return 0.0
    return float(np.dot(xc, yc) / denom)


def _rank_pct_average(values: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    if n == 0:
        return values.astype(np.float32, copy=True)
    r = rankdata(values, method="average")
    out = (r - 0.5) / n
    return out.astype(np.float32, copy=False)


def _signed_pow15(arr: np.ndarray) -> np.ndarray:
    arr64 = arr.astype(np.float64, copy=False)
    return (np.sign(arr64) * (np.abs(arr64) ** 1.5)).astype(np.float64, copy=False)


def _build_era_slices(era_int: np.ndarray) -> list[EraSlice]:
    unique, starts = np.unique(era_int, return_index=True)
    ends = np.r_[starts[1:], era_int.shape[0]]
    return [
        EraSlice(start=int(s), end=int(e), era=int(er))
        for er, s, e in zip(unique, starts, ends)
    ]


def _rank_by_era(values: np.ndarray, era_slices: list[EraSlice]) -> np.ndarray:
    out = np.empty(values.shape[0], dtype=np.float32)
    for sl in era_slices:
        out[sl.start : sl.end] = _rank_pct_average(values[sl.start : sl.end])
    return out


def _neutralize_by_era(
    pred: np.ndarray,
    ref_centered: list[np.ndarray],
    ref_denoms: np.ndarray,
    era_slices: list[EraSlice],
    proportion: float,
    ridge_alpha: float,
) -> np.ndarray:
    if proportion <= 0.0:
        return pred.copy()
    ridge_alpha = float(max(ridge_alpha, 0.0))
    out = pred.astype(np.float64, copy=True)
    for i, sl in enumerate(era_slices):
        denom = float(ref_denoms[i])
        if denom <= 1e-16:
            continue
        p = out[sl.start : sl.end]
        p_mean = float(p.mean())
        p_center = p - p_mean
        beta = float(np.dot(p_center, ref_centered[i]) / (denom + ridge_alpha))
        out[sl.start : sl.end] = p_center - proportion * beta * ref_centered[i] + p_mean
    return out.astype(np.float32, copy=False)


def _neutralize_features_by_era(
    pred: np.ndarray,
    exposures: np.ndarray | None,
    era_slices: list[EraSlice],
    proportion: float,
    ridge_alpha: float,
) -> np.ndarray:
    if proportion <= 0.0 or exposures is None:
        return pred.copy()
    ridge_alpha = float(max(ridge_alpha, 0.0))
    out = pred.astype(np.float64, copy=True)
    x_all = np.asarray(exposures, dtype=np.float64)
    if x_all.ndim != 2:
        raise ValueError("Feature exposures must be a 2D array.")

    for sl in era_slices:
        x = x_all[sl.start : sl.end]
        y = out[sl.start : sl.end]
        if x.shape[0] < 5 or x.shape[1] == 0:
            continue

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        z = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
        try:
            if ridge_alpha > 0.0:
                a = z.T @ z
                reg = np.eye(a.shape[0], dtype=np.float64) * ridge_alpha
                reg[-1, -1] = 0.0
                beta = np.linalg.solve(a + reg, z.T @ y)
            else:
                beta = np.linalg.lstsq(z, y, rcond=1e-6)[0]
        except np.linalg.LinAlgError:
            continue
        adjust = z @ beta
        out[sl.start : sl.end] = y - proportion * adjust
    return out.astype(np.float32, copy=False)


def _rolling_mean_min(values: np.ndarray, window: int) -> float:
    if values.shape[0] < window:
        return np.nan
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    roll = np.convolve(values.astype(np.float64, copy=False), kernel, mode="valid")
    return float(np.min(roll))


def _target_transforms_per_era(
    target_raw: np.ndarray,
    era_slices: list[EraSlice],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    corr_targets: list[np.ndarray] = []
    bmc_targets: list[np.ndarray] = []
    for sl in era_slices:
        t = target_raw[sl.start : sl.end].astype(np.float64, copy=False)
        t_center = t - t.mean()
        corr_targets.append(_signed_pow15(t_center))

        t_bmc = t.copy()
        if bool(np.all((t_bmc >= 0.0) & (t_bmc <= 1.0))):
            t_bmc = t_bmc * 4.0
        t_bmc = t_bmc - t_bmc.mean()
        bmc_targets.append(t_bmc.astype(np.float64, copy=False))
    return corr_targets, bmc_targets


def _benchmark_gauss_per_era(
    benchmark_rank: np.ndarray,
    era_slices: list[EraSlice],
) -> tuple[list[np.ndarray], np.ndarray]:
    bench_gauss: list[np.ndarray] = []
    bench_denoms = np.zeros(len(era_slices), dtype=np.float64)
    for i, sl in enumerate(era_slices):
        m = norm.ppf(benchmark_rank[sl.start : sl.end].astype(np.float64, copy=False))
        bench_gauss.append(m)
        bench_denoms[i] = float(np.dot(m, m))
    return bench_gauss, bench_denoms


def _compute_benchmark_corr_per_era(
    benchmark_raw: np.ndarray,
    corr_targets: list[np.ndarray],
    era_slices: list[EraSlice],
) -> np.ndarray:
    out = np.zeros(len(era_slices), dtype=np.float64)
    for i, sl in enumerate(era_slices):
        b = benchmark_raw[sl.start : sl.end]
        b_rank = _rank_pct_average(b)
        b_gauss = norm.ppf(b_rank.astype(np.float64, copy=False))
        b_corr = _signed_pow15(b_gauss)
        out[i] = _fast_corr(b_corr, corr_targets[i])
    return out


def _cap_vs_reference(
    pred_rank: np.ndarray,
    ref_rank: np.ndarray,
    era_slices: list[EraSlice],
) -> tuple[float, float, float]:
    global_corr = _fast_corr(pred_rank, ref_rank)
    era_corrs = np.zeros(len(era_slices), dtype=np.float64)
    for i, sl in enumerate(era_slices):
        era_corrs[i] = _fast_corr(
            pred_rank[sl.start : sl.end], ref_rank[sl.start : sl.end]
        )
    era_mean = float(np.mean(era_corrs))
    cap = float(np.max(np.abs([global_corr, era_mean])))
    return global_corr, era_mean, cap


def _delta_metrics(
    pred_rank: np.ndarray,
    benchmark_corr_per_era: np.ndarray,
    corr_targets: list[np.ndarray],
    era_slices: list[EraSlice],
    early_era_max: int,
) -> dict:
    model_corr = np.zeros(len(era_slices), dtype=np.float64)
    eras = np.array([sl.era for sl in era_slices], dtype=np.int32)
    for i, sl in enumerate(era_slices):
        p = pred_rank[sl.start : sl.end].astype(np.float64, copy=False)
        p_gauss = norm.ppf(p)
        p_corr = _signed_pow15(p_gauss)
        model_corr[i] = _fast_corr(p_corr, corr_targets[i])

    delta = model_corr - benchmark_corr_per_era
    cumsum = np.cumsum(delta)
    early_mask = eras <= int(early_era_max)
    early_vals = delta[early_mask]
    early_mean = float(np.mean(early_vals)) if early_vals.size else np.nan

    downside = np.minimum(delta, 0.0)
    downside_std = float(np.sqrt(np.mean(np.square(downside)))) if delta.size else 0.0
    delta_sortino = float(np.mean(delta) / downside_std) if downside_std > 0.0 else 0.0

    return {
        "eras": eras,
        "model_corr_per_era": model_corr,
        "delta_per_era": delta,
        "delta_mean": float(np.mean(delta)),
        "delta_std": float(np.std(delta, ddof=0)),
        "delta_sortino": delta_sortino,
        "delta_cumsum_end": float(cumsum[-1]),
        "delta_cumsum_min": float(np.min(cumsum)),
        "delta_roll20_min": _rolling_mean_min(delta, window=20),
        "delta_pos_share": float(np.mean(delta > 0.0)),
        "early_delta_mean": early_mean,
    }


def _sortino(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    downside = np.minimum(values, 0.0)
    downside_std = float(np.sqrt(np.mean(np.square(downside))))
    mean = float(np.mean(values))
    if downside_std <= 1e-16:
        return float("inf") if mean > 0 else 0.0
    return float(mean / downside_std)


def _bmc_payout_metrics(
    *,
    pred_rank: np.ndarray,
    model_corr_per_era: np.ndarray,
    benchmark_gauss: list[np.ndarray],
    benchmark_gauss_denoms: np.ndarray,
    bmc_targets: list[np.ndarray],
    era_slices: list[EraSlice],
    payout_clip: float,
) -> dict:
    bmc = np.zeros(len(era_slices), dtype=np.float64)
    for i, sl in enumerate(era_slices):
        p = pred_rank[sl.start : sl.end].astype(np.float64, copy=False)
        p_gauss = norm.ppf(p)
        m = benchmark_gauss[i]
        denom = float(benchmark_gauss_denoms[i])
        if denom <= 1e-16:
            neutral = p_gauss
        else:
            beta = float(np.dot(p_gauss, m) / denom)
            neutral = p_gauss - beta * m
        bmc[i] = float(np.dot(bmc_targets[i], neutral) / len(neutral))

    payout = np.clip(
        0.75 * model_corr_per_era + 2.25 * bmc,
        -abs(float(payout_clip)),
        abs(float(payout_clip)),
    )
    bmc_cumsum = np.cumsum(bmc)
    payout_cumsum = np.cumsum(payout)
    return {
        "bmc_mean": float(np.mean(bmc)),
        "bmc_std": float(np.std(bmc, ddof=0)),
        "bmc_sortino": _sortino(bmc),
        "bmc_pos_share": float(np.mean(bmc > 0.0)),
        "bmc_cumsum_end": float(bmc_cumsum[-1]),
        "bmc_cumsum_min": float(np.min(bmc_cumsum)),
        "payout_mean": float(np.mean(payout)),
        "payout_std": float(np.std(payout, ddof=0)),
        "payout_sortino": _sortino(payout),
        "payout_pos_share": float(np.mean(payout > 0.0)),
        "payout_cumsum_end": float(payout_cumsum[-1]),
        "payout_cumsum_min": float(np.min(payout_cumsum)),
    }


def _objective_sort_spec(objective: str) -> tuple[list[str], list[bool]]:
    if objective == "delta_cumsum_end":
        return (
            ["delta_cumsum_end", "delta_mean", "delta_roll20_min"],
            [False, False, False],
        )
    if objective == "payout_mean":
        return (
            ["payout_mean", "payout_cumsum_end", "payout_sortino", "delta_cumsum_end"],
            [False, False, False, False],
        )
    if objective == "payout_sortino":
        return (
            ["payout_sortino", "payout_mean", "payout_cumsum_end", "delta_cumsum_end"],
            [False, False, False, False],
        )
    if objective == "payout_cumsum_end":
        return (
            ["payout_cumsum_end", "payout_mean", "payout_sortino", "delta_cumsum_end"],
            [False, False, False, False],
        )
    raise ValueError(f"Unsupported objective: {objective}")


def _evaluate_candidate(
    *,
    lam: float,
    neutralize_benchmark: float,
    neutralize_example: float,
    ridge_alpha: float,
    feature_spec: str,
    feature_neutralize: float,
    feature_ridge_alpha: float,
    raw_rank: np.ndarray,
    benchmark_rank: np.ndarray,
    example_rank: np.ndarray,
    benchmark_ref_centered: list[np.ndarray],
    benchmark_ref_denoms: np.ndarray,
    example_ref_centered: list[np.ndarray],
    example_ref_denoms: np.ndarray,
    feature_exposures: np.ndarray | None,
    era_slices: list[EraSlice],
    benchmark_corr_per_era: np.ndarray,
    corr_targets: list[np.ndarray],
    bmc_targets: list[np.ndarray],
    benchmark_gauss: list[np.ndarray],
    benchmark_gauss_denoms: np.ndarray,
    early_era_max: int,
    max_corr_cap: float,
    min_delta_mean: float,
    min_delta_cumsum_end: float,
    payout_clip: float,
) -> dict:
    blended = benchmark_rank + lam * (raw_rank - benchmark_rank)
    neutralized = _neutralize_by_era(
        blended,
        ref_centered=benchmark_ref_centered,
        ref_denoms=benchmark_ref_denoms,
        era_slices=era_slices,
        proportion=float(neutralize_benchmark),
        ridge_alpha=float(ridge_alpha),
    )
    neutralized = _neutralize_by_era(
        neutralized,
        ref_centered=example_ref_centered,
        ref_denoms=example_ref_denoms,
        era_slices=era_slices,
        proportion=float(neutralize_example),
        ridge_alpha=float(ridge_alpha),
    )
    neutralized = _neutralize_features_by_era(
        neutralized,
        exposures=feature_exposures,
        era_slices=era_slices,
        proportion=float(feature_neutralize),
        ridge_alpha=float(feature_ridge_alpha),
    )
    pred_rank = _rank_by_era(neutralized, era_slices=era_slices)

    c_bg, c_be, cap_b = _cap_vs_reference(
        pred_rank, benchmark_rank, era_slices=era_slices
    )
    c_eg, c_ee, cap_e = _cap_vs_reference(pred_rank, example_rank, era_slices=era_slices)
    cap_ok = bool((cap_b <= max_corr_cap) and (cap_e <= max_corr_cap))

    d = _delta_metrics(
        pred_rank=pred_rank,
        benchmark_corr_per_era=benchmark_corr_per_era,
        corr_targets=corr_targets,
        era_slices=era_slices,
        early_era_max=early_era_max,
    )
    b = _bmc_payout_metrics(
        pred_rank=pred_rank,
        model_corr_per_era=d["model_corr_per_era"],
        benchmark_gauss=benchmark_gauss,
        benchmark_gauss_denoms=benchmark_gauss_denoms,
        bmc_targets=bmc_targets,
        era_slices=era_slices,
        payout_clip=payout_clip,
    )

    feasible_positive = bool(
        cap_ok
        and (d["delta_mean"] >= min_delta_mean)
        and (d["delta_cumsum_end"] >= min_delta_cumsum_end)
    )
    payload = {
        "lambda": float(lam),
        "neutralize_benchmark": float(neutralize_benchmark),
        "neutralize_example": float(neutralize_example),
        "ridge_alpha": float(ridge_alpha),
        "feature_spec": str(feature_spec),
        "feature_neutralize": float(feature_neutralize),
        "feature_ridge_alpha": float(feature_ridge_alpha),
        "corr_with_benchmark_global": float(c_bg),
        "corr_with_benchmark_era_mean": float(c_be),
        "corr_with_benchmark_max_abs": float(cap_b),
        "corr_with_example_global": float(c_eg),
        "corr_with_example_era_mean": float(c_ee),
        "corr_with_example_max_abs": float(cap_e),
        "corr_cap_used": float(max(cap_b, cap_e)),
        "feasible_cap": cap_ok,
        "feasible_positive_delta": feasible_positive,
        "delta_mean": float(d["delta_mean"]),
        "delta_std": float(d["delta_std"]),
        "delta_sortino": float(d["delta_sortino"]),
        "delta_cumsum_end": float(d["delta_cumsum_end"]),
        "delta_cumsum_min": float(d["delta_cumsum_min"]),
        "delta_roll20_min": float(d["delta_roll20_min"]),
        "delta_pos_share": float(d["delta_pos_share"]),
        "early_delta_mean": float(d["early_delta_mean"]),
        "bmc_mean": float(b["bmc_mean"]),
        "bmc_std": float(b["bmc_std"]),
        "bmc_sortino": float(b["bmc_sortino"]),
        "bmc_pos_share": float(b["bmc_pos_share"]),
        "bmc_cumsum_end": float(b["bmc_cumsum_end"]),
        "bmc_cumsum_min": float(b["bmc_cumsum_min"]),
        "payout_mean": float(b["payout_mean"]),
        "payout_std": float(b["payout_std"]),
        "payout_sortino": float(b["payout_sortino"]),
        "payout_pos_share": float(b["payout_pos_share"]),
        "payout_cumsum_end": float(b["payout_cumsum_end"]),
        "payout_cumsum_min": float(b["payout_cumsum_min"]),
    }
    return payload


def _select_best(rows_df: pd.DataFrame, objective: str) -> tuple[dict, str]:
    sort_cols, sort_asc = _objective_sort_spec(objective)

    if objective == "delta_cumsum_end":
        positive = rows_df[rows_df["feasible_positive_delta"]]
        if not positive.empty:
            best = positive.sort_values(sort_cols, ascending=sort_asc).iloc[0]
            return best.to_dict(), "positive_delta_feasible"
        cap_only = rows_df[rows_df["feasible_cap"]]
        if cap_only.empty:
            best = rows_df.sort_values(
                ["corr_cap_used", *sort_cols], ascending=[True, *sort_asc]
            ).iloc[0]
            return best.to_dict(), "no_cap_feasible"
        best = cap_only.sort_values(sort_cols, ascending=sort_asc).iloc[0]
        return best.to_dict(), "no_positive_delta_feasible"

    cap_only = rows_df[rows_df["feasible_cap"]]
    if cap_only.empty:
        best = rows_df.sort_values(
            ["corr_cap_used", *sort_cols], ascending=[True, *sort_asc]
        ).iloc[0]
        return best.to_dict(), "no_cap_feasible"
    best = cap_only.sort_values(sort_cols, ascending=sort_asc).iloc[0]
    if bool(best["feasible_positive_delta"]):
        return best.to_dict(), "cap_feasible_positive_delta"
    return best.to_dict(), "cap_feasible"


def _load_example_preds(path: Path, id_col: str) -> pd.Series:
    ex = pd.read_parquet(path)
    if id_col not in ex.columns:
        ex = ex.reset_index()
        if id_col not in ex.columns:
            ex = ex.rename(columns={ex.columns[0]: id_col})
    pred_col = "prediction"
    if pred_col not in ex.columns:
        numeric_cols = [c for c in ex.columns if c != id_col and pd.api.types.is_numeric_dtype(ex[c])]
        if not numeric_cols:
            raise ValueError(f"No numeric prediction column found in {path}")
        pred_col = numeric_cols[0]
    ex = ex[[id_col, pred_col]].rename(columns={pred_col: "example_prediction"})
    return ex.set_index(id_col)["example_prediction"]


def _load_model_frame(
    *,
    model_path: Path,
    example_by_id: pd.Series,
    id_col: str,
    era_col: str,
    target_col: str,
    benchmark_col: str,
    min_era: int,
    max_era: int,
) -> pd.DataFrame:
    candidate_targets = [target_col]
    for fallback in ("target", "target_ender_20"):
        if fallback not in candidate_targets:
            candidate_targets.append(fallback)

    last_error: Exception | None = None
    df: pd.DataFrame | None = None
    chosen_target = target_col
    for candidate in candidate_targets:
        try:
            cols = [id_col, era_col, candidate, benchmark_col, "prediction_raw"]
            df = pd.read_parquet(model_path, columns=cols)
            chosen_target = candidate
            break
        except Exception as exc:
            last_error = exc
    if df is None:
        raise RuntimeError(f"Could not read target column from {model_path}: {last_error}")

    if chosen_target != target_col:
        df = df.rename(columns={chosen_target: target_col})

    df[era_col] = df[era_col].astype(str)
    era_int = df[era_col].astype(np.int32)
    mask = (era_int >= int(min_era)) & (era_int <= int(max_era))
    df = df.loc[mask].copy()
    era_int = era_int.loc[mask]

    df["example_prediction"] = df[id_col].map(example_by_id)
    missing = int(df["example_prediction"].isna().sum())
    if missing:
        raise ValueError(f"{model_path.name}: missing example predictions for {missing} rows.")

    order = np.argsort(era_int.to_numpy(dtype=np.int32), kind="mergesort")
    df = df.iloc[order].reset_index(drop=True)
    return df


def _load_model_arrays(
    *,
    model_path: Path,
    example_by_id: pd.Series,
    id_col: str,
    era_col: str,
    target_col: str,
    benchmark_col: str,
    min_era: int,
    max_era: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = _load_model_frame(
        model_path=model_path,
        example_by_id=example_by_id,
        id_col=id_col,
        era_col=era_col,
        target_col=target_col,
        benchmark_col=benchmark_col,
        min_era=min_era,
        max_era=max_era,
    )
    era_int = df[era_col].astype(np.int32).to_numpy(dtype=np.int32)
    target_raw = df[target_col].to_numpy(dtype=np.float32)
    benchmark_raw = df[benchmark_col].to_numpy(dtype=np.float32)
    prediction_raw = df["prediction_raw"].to_numpy(dtype=np.float32)
    example_raw = df["example_prediction"].to_numpy(dtype=np.float32)
    return df, era_int, target_raw, benchmark_raw, prediction_raw, example_raw


def _grid_from_center(center: float, window: float, step: float, lo: float, hi: float) -> list[float]:
    start = max(lo, center - window)
    end = min(hi, center + window)
    n = int(np.floor((end - start) / step)) + 1
    vals = [round(start + i * step, 10) for i in range(n)]
    if not vals:
        vals = [round(center, 10)]
    return sorted(set(vals))


def _ridge_fine_grid(
    *,
    center: float,
    coarse_grid: list[float],
    explicit_grid: str,
    multipliers: str,
) -> list[float]:
    if explicit_grid.strip():
        return _parse_float_grid(explicit_grid)

    mults = [m for m in _parse_float_grid(multipliers) if m > 0.0]
    if not mults:
        mults = [1.0]

    vals: set[float] = {0.0}
    center = float(max(center, 0.0))
    if center > 0.0:
        for m in mults:
            vals.add(center * float(m))
    else:
        for x in coarse_grid[: min(5, len(coarse_grid))]:
            vals.add(float(x))

    capped = [min(10.0, max(0.0, float(x))) for x in vals]
    return sorted(set(capped))


def _resolve_features_json_path(feature_data_path: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        path = explicit_path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing features.json: {path}")
        return path

    candidate = feature_data_path.resolve().parent / "features.json"
    if not candidate.exists():
        raise FileNotFoundError(f"Missing features.json alongside feature data: {candidate}")
    return candidate


def _parse_feature_specs(value: str) -> list[str]:
    specs: list[str] = []
    for token in value.split(","):
        spec = token.strip()
        if not spec:
            continue
        if spec.lower() in {"none", "null"}:
            continue
        if spec not in specs:
            specs.append(spec)
    return specs


def _load_feature_exposure_cache(
    *,
    feature_data_path: Path,
    features_json_path: Path,
    feature_specs: list[str],
    id_col: str,
    ordered_ids: pd.Series,
) -> dict[str, np.ndarray]:
    if not feature_specs:
        return {"none": None}

    feature_sets = _load_feature_sets(features_json_path)
    cache: dict[str, np.ndarray] = {"none": None}
    for spec in feature_specs:
        if spec == "none":
            continue
        cols = _feature_cols_for_spec(feature_sets, spec)
        if not cols:
            cache[spec] = None
            continue
        exposure_df = pd.read_parquet(feature_data_path, columns=[id_col, *cols])
        exposure_df = exposure_df.merge(
            ordered_ids.to_frame(),
            on=id_col,
            how="inner",
            validate="one_to_one",
        )
        exposure_df = exposure_df.set_index(id_col).loc[ordered_ids].reset_index()
        cache[spec] = exposure_df[cols].to_numpy(dtype=np.float32)
    return cache


def main() -> None:
    args = parse_args()
    started = time.time()

    experiment_dir = args.experiment_dir.resolve()
    predictions_dir = experiment_dir / "predictions"
    results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    model_files = [x.strip() for x in args.model_files.split(",") if x.strip()]
    if not model_files:
        raise ValueError("No model files provided.")

    coarse_lam = _parse_float_grid(args.coarse_lambda_grid)
    coarse_nb = _parse_float_grid(args.coarse_neutralize_grid)
    coarse_ne = _parse_float_grid(args.coarse_neutralize_example_grid)
    coarse_feature_p = _parse_float_grid(args.coarse_feature_neutralize_grid)
    coarse_ridge_alpha = _parse_float_grid(args.coarse_ridge_alpha_grid)
    coarse_feature_ridge_alpha = (
        _parse_float_grid(args.coarse_feature_ridge_alpha_grid)
        if str(args.coarse_feature_ridge_alpha_grid).strip()
        else list(coarse_ridge_alpha)
    )
    feature_specs = _parse_feature_specs(args.feature_specs)
    features_json_path = (
        _resolve_features_json_path(
            args.feature_data_path,
            args.features_json_path,
        )
        if feature_specs
        else None
    )
    example_by_id = _load_example_preds(args.example_preds_path.resolve(), args.id_col)
    print(f"Loaded example predictions: {len(example_by_id):,} rows", flush=True)

    payload: dict = {
        "date": str(date.today()),
        "settings": {
            "min_era": int(args.min_era),
            "max_era": int(args.max_era),
            "early_era_max": int(args.early_era_max),
            "max_corr_cap": float(args.max_corr_cap),
            "min_delta_mean": float(args.min_delta_mean),
            "min_delta_cumsum_end": float(args.min_delta_cumsum_end),
            "selection_objective": str(args.selection_objective),
            "payout_clip": float(args.payout_clip),
            "coarse_lambda_grid": coarse_lam,
            "coarse_neutralize_grid": coarse_nb,
            "coarse_neutralize_example_grid": coarse_ne,
            "feature_specs": feature_specs,
            "coarse_feature_neutralize_grid": coarse_feature_p,
            "coarse_ridge_alpha_grid": coarse_ridge_alpha,
            "coarse_feature_ridge_alpha_grid": coarse_feature_ridge_alpha,
            "fine_window_lambda": float(args.fine_window_lambda),
            "fine_window_neutralize": float(args.fine_window_neutralize),
            "fine_window_neutralize_example": float(args.fine_window_neutralize_example),
            "fine_window_feature_neutralize": float(args.fine_window_feature_neutralize),
            "fine_step_lambda": float(args.fine_step_lambda),
            "fine_step_neutralize": float(args.fine_step_neutralize),
            "fine_step_neutralize_example": float(args.fine_step_neutralize_example),
            "fine_step_feature_neutralize": float(args.fine_step_feature_neutralize),
            "fine_ridge_alpha_grid": str(args.fine_ridge_alpha_grid),
            "fine_feature_ridge_alpha_grid": str(args.fine_feature_ridge_alpha_grid),
            "fine_ridge_alpha_multipliers": str(args.fine_ridge_alpha_multipliers),
            "fine_feature_ridge_alpha_multipliers": str(
                args.fine_feature_ridge_alpha_multipliers
            ),
        },
        "models": [],
    }

    overall_rows: list[dict] = []

    for model_file in model_files:
        model_path = predictions_dir / model_file
        print(f"\n--- Tuning {model_file} ---", flush=True)
        load_t0 = time.time()
        (
            model_df,
            era_int,
            target_raw,
            benchmark_raw,
            prediction_raw,
            example_raw,
        ) = _load_model_arrays(
            model_path=model_path,
            example_by_id=example_by_id,
            id_col=args.id_col,
            era_col=args.era_col,
            target_col=args.target_col,
            benchmark_col=args.benchmark_col,
            min_era=args.min_era,
            max_era=args.max_era,
        )
        print(
            f"Loaded rows={len(era_int):,} eras={len(np.unique(era_int))} "
            f"in {time.time() - load_t0:.2f}s",
            flush=True,
        )

        prep_t0 = time.time()
        era_slices = _build_era_slices(era_int)

        raw_rank = _rank_by_era(prediction_raw, era_slices=era_slices)
        benchmark_rank = _rank_by_era(benchmark_raw, era_slices=era_slices)
        example_rank = _rank_by_era(example_raw, era_slices=era_slices)
        corr_ex_bench = _fast_corr(example_rank, benchmark_rank)

        benchmark_ref_centered: list[np.ndarray] = []
        benchmark_ref_denoms = np.zeros(len(era_slices), dtype=np.float64)
        for i, sl in enumerate(era_slices):
            r = benchmark_rank[sl.start : sl.end].astype(np.float64, copy=False)
            rc = r - r.mean()
            benchmark_ref_centered.append(rc)
            benchmark_ref_denoms[i] = float(np.dot(rc, rc))

        example_ref_centered: list[np.ndarray] = []
        example_ref_denoms = np.zeros(len(era_slices), dtype=np.float64)
        for i, sl in enumerate(era_slices):
            r = example_rank[sl.start : sl.end].astype(np.float64, copy=False)
            rc = r - r.mean()
            example_ref_centered.append(rc)
            example_ref_denoms[i] = float(np.dot(rc, rc))

        corr_targets, bmc_targets = _target_transforms_per_era(
            target_raw, era_slices=era_slices
        )
        benchmark_corr_per_era = _compute_benchmark_corr_per_era(
            benchmark_raw=benchmark_raw,
            corr_targets=corr_targets,
            era_slices=era_slices,
        )
        benchmark_gauss, benchmark_gauss_denoms = _benchmark_gauss_per_era(
            benchmark_rank=benchmark_rank,
            era_slices=era_slices,
        )
        print(
            f"Prepared ranks/transforms in {time.time() - prep_t0:.2f}s; "
            f"corr(example_rank, benchmark_rank)={corr_ex_bench:.6f}",
            flush=True,
        )

        feature_exposure_cache = _load_feature_exposure_cache(
            feature_data_path=args.feature_data_path.resolve(),
            features_json_path=features_json_path,
            feature_specs=feature_specs,
            id_col=args.id_col,
            ordered_ids=model_df[args.id_col],
        )

        coarse_t0 = time.time()
        coarse_rows: list[dict] = []
        for lam in coarse_lam:
            for nb in coarse_nb:
                for ne in coarse_ne:
                    for ra in coarse_ridge_alpha:
                        for feature_spec, feature_exposures in feature_exposure_cache.items():
                            feature_p_grid = coarse_feature_p if feature_exposures is not None else [0.0]
                            feature_ra_grid = (
                                coarse_feature_ridge_alpha if feature_exposures is not None else [0.0]
                            )
                            for fp in feature_p_grid:
                                for fra in feature_ra_grid:
                                    row = _evaluate_candidate(
                                        lam=float(lam),
                                        neutralize_benchmark=float(nb),
                                        neutralize_example=float(ne),
                                        ridge_alpha=float(ra),
                                        feature_spec=str(feature_spec),
                                        feature_neutralize=float(fp),
                                        feature_ridge_alpha=float(fra),
                                        raw_rank=raw_rank,
                                        benchmark_rank=benchmark_rank,
                                        example_rank=example_rank,
                                        benchmark_ref_centered=benchmark_ref_centered,
                                        benchmark_ref_denoms=benchmark_ref_denoms,
                                        example_ref_centered=example_ref_centered,
                                        example_ref_denoms=example_ref_denoms,
                                        feature_exposures=feature_exposures,
                                        era_slices=era_slices,
                                        benchmark_corr_per_era=benchmark_corr_per_era,
                                        corr_targets=corr_targets,
                                        bmc_targets=bmc_targets,
                                        benchmark_gauss=benchmark_gauss,
                                        benchmark_gauss_denoms=benchmark_gauss_denoms,
                                        early_era_max=int(args.early_era_max),
                                        max_corr_cap=float(args.max_corr_cap),
                                        min_delta_mean=float(args.min_delta_mean),
                                        min_delta_cumsum_end=float(args.min_delta_cumsum_end),
                                        payout_clip=float(args.payout_clip),
                                    )
                                    coarse_rows.append(row)
        coarse_df = pd.DataFrame(coarse_rows)
        coarse_best, coarse_status = _select_best(
            coarse_df, objective=str(args.selection_objective)
        )
        obj_col, _ = _objective_sort_spec(str(args.selection_objective))
        obj_key = obj_col[0]
        print(
            f"Coarse done ({len(coarse_df)} candidates, {time.time() - coarse_t0:.2f}s): "
            f"status={coarse_status} lam={coarse_best['lambda']:.4f} "
            f"nb={coarse_best['neutralize_benchmark']:.4f} "
            f"ridge_alpha={coarse_best['ridge_alpha']:.6g} "
            f"{obj_key}={coarse_best[obj_key]:.6f}",
            flush=True,
        )

        fine_lam = _grid_from_center(
            center=float(coarse_best["lambda"]),
            window=float(args.fine_window_lambda),
            step=float(args.fine_step_lambda),
            lo=0.0,
            hi=2.0,
        )
        fine_nb = _grid_from_center(
            center=float(coarse_best["neutralize_benchmark"]),
            window=float(args.fine_window_neutralize),
            step=float(args.fine_step_neutralize),
            lo=0.0,
            hi=2.0,
        )
        fine_ne = _grid_from_center(
            center=float(coarse_best["neutralize_example"]),
            window=float(args.fine_window_neutralize_example),
            step=float(args.fine_step_neutralize_example),
            lo=0.0,
            hi=2.0,
        )
        fine_feature_p = _grid_from_center(
            center=float(coarse_best["feature_neutralize"]),
            window=float(args.fine_window_feature_neutralize),
            step=float(args.fine_step_feature_neutralize),
            lo=0.0,
            hi=2.0,
        )
        fine_ridge_alpha = _ridge_fine_grid(
            center=float(coarse_best["ridge_alpha"]),
            coarse_grid=coarse_ridge_alpha,
            explicit_grid=str(args.fine_ridge_alpha_grid),
            multipliers=str(args.fine_ridge_alpha_multipliers),
        )
        fine_feature_ridge_alpha = _ridge_fine_grid(
            center=float(coarse_best["feature_ridge_alpha"]),
            coarse_grid=coarse_feature_ridge_alpha,
            explicit_grid=str(args.fine_feature_ridge_alpha_grid),
            multipliers=str(args.fine_feature_ridge_alpha_multipliers),
        )
        fine_feature_spec = str(coarse_best["feature_spec"])
        fine_feature_exposures = feature_exposure_cache[fine_feature_spec]
        if fine_feature_exposures is None:
            fine_feature_p = [0.0]
            fine_feature_ridge_alpha = [0.0]

        fine_t0 = time.time()
        fine_rows: list[dict] = []
        for lam in fine_lam:
            for nb in fine_nb:
                for ne in fine_ne:
                    for ra in fine_ridge_alpha:
                        for fp in fine_feature_p:
                            for fra in fine_feature_ridge_alpha:
                                row = _evaluate_candidate(
                                    lam=float(lam),
                                    neutralize_benchmark=float(nb),
                                    neutralize_example=float(ne),
                                    ridge_alpha=float(ra),
                                    feature_spec=fine_feature_spec,
                                    feature_neutralize=float(fp),
                                    feature_ridge_alpha=float(fra),
                                    raw_rank=raw_rank,
                                    benchmark_rank=benchmark_rank,
                                    example_rank=example_rank,
                                    benchmark_ref_centered=benchmark_ref_centered,
                                    benchmark_ref_denoms=benchmark_ref_denoms,
                                    example_ref_centered=example_ref_centered,
                                    example_ref_denoms=example_ref_denoms,
                                    feature_exposures=fine_feature_exposures,
                                    era_slices=era_slices,
                                    benchmark_corr_per_era=benchmark_corr_per_era,
                                    corr_targets=corr_targets,
                                    bmc_targets=bmc_targets,
                                    benchmark_gauss=benchmark_gauss,
                                    benchmark_gauss_denoms=benchmark_gauss_denoms,
                                    early_era_max=int(args.early_era_max),
                                    max_corr_cap=float(args.max_corr_cap),
                                    min_delta_mean=float(args.min_delta_mean),
                                    min_delta_cumsum_end=float(args.min_delta_cumsum_end),
                                    payout_clip=float(args.payout_clip),
                                )
                                fine_rows.append(row)
        fine_df = pd.DataFrame(fine_rows)
        fine_best, fine_status = _select_best(
            fine_df, objective=str(args.selection_objective)
        )
        print(
            f"Fine done ({len(fine_df)} candidates, {time.time() - fine_t0:.2f}s): "
            f"status={fine_status} lam={fine_best['lambda']:.4f} "
            f"nb={fine_best['neutralize_benchmark']:.4f} "
            f"ridge_alpha={fine_best['ridge_alpha']:.6g} "
            f"{obj_key}={fine_best[obj_key]:.6f}",
            flush=True,
        )
        final_best = fine_best

        model_payload = {
            "model_file": model_file,
            "rows": int(len(era_int)),
            "eras": int(len(era_slices)),
            "corr_example_vs_benchmark_rank": float(corr_ex_bench),
            "coarse": {
                "status": coarse_status,
                "best": coarse_best,
                "top5": coarse_df.sort_values(
                    obj_col, ascending=False
                )
                .head(5)
                .to_dict(orient="records"),
            },
            "fine": {
                "status": fine_status,
                "grid": {
                    "lambda": {
                        "start": float(min(fine_lam)),
                        "end": float(max(fine_lam)),
                        "step": float(args.fine_step_lambda),
                    },
                    "neutralize_benchmark": {
                        "start": float(min(fine_nb)),
                        "end": float(max(fine_nb)),
                        "step": float(args.fine_step_neutralize),
                    },
                    "neutralize_example": {
                        "start": float(min(fine_ne)),
                        "end": float(max(fine_ne)),
                        "step": float(args.fine_step_neutralize_example),
                    },
                    "feature_spec": fine_feature_spec,
                    "feature_neutralize": {
                        "start": float(min(fine_feature_p)),
                        "end": float(max(fine_feature_p)),
                        "step": float(args.fine_step_feature_neutralize),
                    },
                    "ridge_alpha": {
                        "start": float(min(fine_ridge_alpha)),
                        "end": float(max(fine_ridge_alpha)),
                        "values": [float(x) for x in fine_ridge_alpha],
                    },
                    "feature_ridge_alpha": {
                        "start": float(min(fine_feature_ridge_alpha)),
                        "end": float(max(fine_feature_ridge_alpha)),
                        "values": [float(x) for x in fine_feature_ridge_alpha],
                    },
                    "n_candidates": int(len(fine_rows)),
                },
                "best": final_best,
                "top5": fine_df.sort_values(
                    obj_col, ascending=False
                )
                .head(5)
                .to_dict(orient="records"),
            },
        }
        payload["models"].append(model_payload)
        overall_rows.append(
            {
                "model_file": model_file,
                "status": fine_status,
                "selection_objective": str(args.selection_objective),
                "delta_cumsum_end": float(final_best["delta_cumsum_end"]),
                "delta_mean": float(final_best["delta_mean"]),
                "corr_cap_used": float(final_best["corr_cap_used"]),
                "payout_mean": float(final_best["payout_mean"]),
                "payout_sortino": float(final_best["payout_sortino"]),
                "payout_cumsum_end": float(final_best["payout_cumsum_end"]),
                "bmc_mean": float(final_best["bmc_mean"]),
                "lambda": float(final_best["lambda"]),
                "neutralize_benchmark": float(final_best["neutralize_benchmark"]),
                "neutralize_example": float(final_best["neutralize_example"]),
                "ridge_alpha": float(final_best["ridge_alpha"]),
                "feature_spec": str(final_best["feature_spec"]),
                "feature_neutralize": float(final_best["feature_neutralize"]),
                "feature_ridge_alpha": float(final_best["feature_ridge_alpha"]),
            }
        )

    overall_df = pd.DataFrame(overall_rows)
    if str(args.selection_objective) == "delta_cumsum_end":
        positive_df = overall_df[overall_df["status"] == "positive_delta_feasible"]
        if not positive_df.empty:
            overall_best = positive_df.sort_values(
                ["delta_cumsum_end", "delta_mean"], ascending=False
            ).iloc[0]
        else:
            overall_best = overall_df.sort_values(
                ["delta_cumsum_end", "delta_mean"], ascending=False
            ).iloc[0]
    else:
        cap_df = overall_df[overall_df["status"].str.contains("cap_feasible", na=False)]
        pool = cap_df if not cap_df.empty else overall_df
        obj_cols, obj_asc = _objective_sort_spec(str(args.selection_objective))
        overall_best = pool.sort_values(obj_cols, ascending=obj_asc).iloc[0]
    payload["overall_best"] = overall_best.to_dict()
    payload["runtime_seconds"] = float(time.time() - started)

    out_path = results_dir / args.output_name
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved two-stage tuning results to {out_path}", flush=True)
    print(json.dumps(payload["overall_best"], indent=2), flush=True)


if __name__ == "__main__":
    main()
