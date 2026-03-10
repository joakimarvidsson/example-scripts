from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agents.experiments.nn_ender20_latest_residualized.strict_gbt_walkforward_research import (
    _artifact_name,
    _compute_delta_metrics,
    _load_example_preds_for_eval,
    _rank01_per_era,
    _resolve_input_path,
    _safe_corr,
    _write_result_json,
)


MODEL_PATHS = {
    "cat_dense": Path(
        "/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/"
        "nn_ender20_latest_residualized/predictions/cat_strict_resid008_medfaith64_walkfwd_strict_dense_e4_r700.parquet"
    ),
    "cat_smallfaith_dense": Path(
        "/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/"
        "nn_ender20_latest_residualized/predictions/cat_strict_resid010_smallfaith64_walkfwd_strict_dense_e4_r700.parquet"
    ),
    "lgbm_smallfaith_dense": Path(
        "/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/"
        "nn_ender20_latest_residualized/predictions/lgbm_dart_strict_resid010_smallfaith64_walkfwd_strict_dense_e4_r700.parquet"
    ),
    "mlp_roundM6": Path(
        "/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/"
        "nn_ender20_latest_residualized/predictions/torch_mlp_resid_latest_ender20_roundM6_base_w80_m3_w20.parquet"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search dense OOF blend weights with a soft correlation penalty."
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
        "--example-preds-path",
        type=Path,
        default=Path("numerai/v5.2/validation_example_preds.parquet"),
    )
    parser.add_argument("--benchmark-model", default="v52_lgbm_ender20")
    parser.add_argument("--target-col", default="target_ender_20")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--era-col", default="era")
    parser.add_argument("--min-eval-era", type=int, default=577)
    parser.add_argument("--max-eval-era", type=int, default=1197)
    parser.add_argument("--model-pack", default="cat_dense,cat_smallfaith_dense,lgbm_smallfaith_dense,mlp_roundM6")
    parser.add_argument("--seed", type=int, default=20260311)
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--corr-threshold", type=float, default=0.995)
    parser.add_argument("--benchmark-penalty", type=float, default=5.0)
    parser.add_argument("--example-penalty", type=float, default=2.0)
    parser.add_argument("--prefilter-top-n", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--summary-name",
        default="roundS40_softpenalty_dense_blend_search.json",
    )
    parser.add_argument(
        "--out-name",
        default="blend_softpenalty_dense_treepack_mlp_roundS40",
    )
    return parser.parse_args()


def _load_target_benchmark(
    *,
    full_data_path: Path,
    benchmark_data_path: Path,
    id_col: str,
    era_col: str,
    target_col: str,
    benchmark_model: str,
    min_eval_era: int,
    max_eval_era: int,
) -> pd.DataFrame:
    era_vals = [f"{era:04d}" for era in range(min_eval_era, max_eval_era + 1)]
    base = pd.read_parquet(
        full_data_path,
        columns=[id_col, era_col, target_col],
        filters=[(era_col, "in", era_vals)],
    )
    bench = pd.read_parquet(
        benchmark_data_path,
        columns=[id_col, era_col, benchmark_model],
        filters=[(era_col, "in", era_vals)],
    )
    base[era_col] = base[era_col].astype(str)
    bench[era_col] = bench[era_col].astype(str)
    return base.merge(bench, on=[id_col, era_col], how="inner", validate="one_to_one")


def _load_prediction_column(path: Path, name: str, *, id_col: str, era_col: str, min_eval_era: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "prediction" not in df.columns:
        raise ValueError(f"{path} missing prediction column")
    out = df[[id_col, era_col, "prediction"]].copy().rename(columns={"prediction": name})
    out = out[out[era_col].astype(int) >= int(min_eval_era)].copy()
    out[era_col] = out[era_col].astype(str)
    return out


def _dirichlet_weights(rng: np.random.Generator, n_models: int, n_samples: int) -> np.ndarray:
    base = rng.dirichlet(np.ones(n_models), size=n_samples)
    corners = np.eye(n_models)
    near_corners = []
    if n_models >= 2:
        for i in range(n_models):
            for j in range(i + 1, n_models):
                near_corners.append(np.array([0.0] * n_models, dtype=np.float64))
                near_corners[-1][i] = 0.95
                near_corners[-1][j] = 0.05
                near_corners.append(np.array([0.0] * n_models, dtype=np.float64))
                near_corners[-1][i] = 0.90
                near_corners[-1][j] = 0.10
    extra = np.vstack([corners, *near_corners]) if near_corners else corners
    return np.vstack([base, extra])


def _safe_np_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _build_era_groups(eras: pd.Series) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for _era, idx in eras.groupby(eras, sort=True).groups.items():
        out.append(np.asarray(idx, dtype=np.int64))
    return out


def _fast_rank01_per_era(values: np.ndarray, era_groups: list[np.ndarray]) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    for idx in era_groups:
        v = values[idx]
        order = np.argsort(v, kind="mergesort")
        sorted_v = v[order]
        n = len(sorted_v)
        ranks = np.empty(n, dtype=np.float64)
        start = 0
        while start < n:
            end = start + 1
            while end < n and sorted_v[end] == sorted_v[start]:
                end += 1
            avg_rank = ((start + 1) + end) / 2.0
            ranks[order[start:end]] = avg_rank / n
            start = end
        out[idx] = ranks
    return out


def _cheap_prefilter_score(
    pred: np.ndarray,
    *,
    target: np.ndarray,
    benchmark: np.ndarray,
    example: np.ndarray,
    era_groups: list[np.ndarray],
    early_group_count: int,
    corr_threshold: float,
    benchmark_penalty: float,
    example_penalty: float,
) -> tuple[float, float, float, float]:
    delta_vals: list[float] = []
    for idx in era_groups:
        delta_vals.append(
            _safe_np_corr(pred[idx], target[idx]) - _safe_np_corr(benchmark[idx], target[idx])
        )
    delta = np.asarray(delta_vals, dtype=np.float64)
    delta_mean = float(delta.mean()) if delta.size else 0.0
    early_delta_mean = float(delta[:early_group_count].mean()) if early_group_count > 0 else 0.0
    corr_bench = float(_safe_np_corr(pred, benchmark))
    corr_example = float(_safe_np_corr(pred, example))
    bench_excess = max(0.0, corr_bench - float(corr_threshold))
    ex_excess = max(0.0, corr_example - float(corr_threshold))
    score = (
        delta_mean
        + 0.5 * early_delta_mean
        - float(benchmark_penalty) * bench_excess
        - float(example_penalty) * ex_excess
    )
    return float(score), corr_bench, corr_example, delta_mean


def main() -> None:
    args = parse_args()
    args.full_data_path = _resolve_input_path(args.full_data_path)
    args.benchmark_data_path = _resolve_input_path(args.benchmark_data_path)
    args.example_preds_path = _resolve_input_path(args.example_preds_path)
    experiment_dir = args.experiment_dir.resolve()
    predictions_dir = experiment_dir / "predictions"
    results_dir = experiment_dir / "results"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_names = [s.strip() for s in str(args.model_pack).split(",") if s.strip()]
    for name in model_names:
        if name not in MODEL_PATHS:
            raise ValueError(f"Unknown model name: {name}")

    base = _load_target_benchmark(
        full_data_path=args.full_data_path.resolve(),
        benchmark_data_path=args.benchmark_data_path.resolve(),
        id_col=args.id_col,
        era_col=args.era_col,
        target_col=args.target_col,
        benchmark_model=args.benchmark_model,
        min_eval_era=int(args.min_eval_era),
        max_eval_era=int(args.max_eval_era),
    )
    merged = base.copy()
    for model_name in model_names:
        merged = merged.merge(
            _load_prediction_column(
                MODEL_PATHS[model_name],
                model_name,
                id_col=args.id_col,
                era_col=args.era_col,
                min_eval_era=int(args.min_eval_era),
            ),
            on=[args.id_col, args.era_col],
            how="inner",
        )

    eval_eras = sorted(merged[args.era_col].astype(int).unique())
    example_df = _load_example_preds_for_eval(
        example_preds_path=args.example_preds_path.resolve(),
        full_path=args.full_data_path.resolve(),
        id_col=args.id_col,
        era_col=args.era_col,
        eval_eras=eval_eras,
    )
    merged = merged.merge(
        example_df[[args.id_col, args.era_col, "example_prediction"]],
        on=[args.id_col, args.era_col],
        how="left",
        validate="one_to_one",
    )
    era_groups = _build_era_groups(merged[args.era_col])
    merged["benchmark_rank"] = _fast_rank01_per_era(
        merged[args.benchmark_model].to_numpy(dtype=np.float64),
        era_groups,
    )
    merged["example_rank"] = _fast_rank01_per_era(
        merged["example_prediction"].to_numpy(dtype=np.float64),
        era_groups,
    )
    rank_cols: list[str] = []
    for model_name in model_names:
        rank_col = f"{model_name}_rank"
        merged[rank_col] = _fast_rank01_per_era(
            merged[model_name].to_numpy(dtype=np.float64),
            era_groups,
        )
        rank_cols.append(rank_col)

    rng = np.random.default_rng(int(args.seed))
    weight_grid = _dirichlet_weights(rng, len(model_names), int(args.n_samples))

    top_rows: list[dict] = []
    best_row: dict | None = None
    best_pred: pd.Series | None = None
    x = merged[rank_cols].to_numpy(dtype=np.float64)
    bench_rank = merged["benchmark_rank"].astype(np.float64)
    example_rank = merged["example_rank"].astype(np.float64)
    eras = merged[args.era_col]
    target = merged[args.target_col].to_numpy(dtype=np.float64)
    early_group_count = sum(1 for idx in era_groups if int(eras.iloc[idx[0]]) <= 889)

    cheap_rows: list[dict] = []
    for weights in weight_grid:
        pred = x @ weights
        cheap_score, corr_bench, corr_example, delta_mean = _cheap_prefilter_score(
            pred,
            target=target,
            benchmark=bench_rank.to_numpy(dtype=np.float64),
            example=example_rank.to_numpy(dtype=np.float64),
            era_groups=era_groups,
            early_group_count=early_group_count,
            corr_threshold=float(args.corr_threshold),
            benchmark_penalty=float(args.benchmark_penalty),
            example_penalty=float(args.example_penalty),
        )
        cheap_rows.append(
            {
                "cheap_score": float(cheap_score),
                "corr_with_benchmark_global": float(corr_bench),
                "corr_with_example_global": float(corr_example),
                "cheap_delta_mean": float(delta_mean),
                "weights": {name: float(w) for name, w in zip(model_names, weights)},
                "pred": pred,
            }
        )

    cheap_rows = sorted(cheap_rows, key=lambda r: float(r["cheap_score"]), reverse=True)
    candidates = cheap_rows[: max(1, min(int(args.prefilter_top_n), len(cheap_rows)))]

    for row0 in candidates:
        pred_rank = _fast_rank01_per_era(np.asarray(row0["pred"], dtype=np.float64), era_groups)
        pred = pd.Series(pred_rank, index=merged.index, dtype=np.float64)
        corr_bench = float(_safe_corr(pred, bench_rank))
        corr_example = float(_safe_corr(pred, example_rank))
        tmp = merged[[args.id_col, args.era_col, args.target_col, args.benchmark_model]].copy()
        tmp["prediction"] = pred.to_numpy(dtype=np.float64)
        metrics = _compute_delta_metrics(
            tmp,
            pred_col="prediction",
            benchmark_col=args.benchmark_model,
            target_col=args.target_col,
            era_col=args.era_col,
            early_era_max=889,
            compute_bmc=True,
        )
        bench_excess = max(0.0, corr_bench - float(args.corr_threshold))
        ex_excess = max(0.0, corr_example - float(args.corr_threshold))
        objective = (
            float(metrics["payout_mean"])
            - float(args.benchmark_penalty) * bench_excess
            - float(args.example_penalty) * ex_excess
        )
        row = {
            "cheap_score": float(row0["cheap_score"]),
            "objective": float(objective),
            "corr_with_benchmark_global": corr_bench,
            "corr_with_example_global": corr_example,
            "weights": row0["weights"],
            **metrics,
        }
        if best_row is None or row["objective"] > best_row["objective"]:
            best_row = row
            best_pred = pred
        top_rows.append(row)

    assert best_row is not None
    assert best_pred is not None
    top_rows = sorted(top_rows, key=lambda r: float(r["objective"]), reverse=True)[: int(args.top_k)]

    out_name = _artifact_name(args.out_name, "")
    best_df = merged[[args.id_col, args.era_col, args.target_col, args.benchmark_model]].copy()
    best_df["prediction"] = best_pred.to_numpy(dtype=np.float64)
    out_path = predictions_dir / f"{out_name}.parquet"
    best_df.rename(columns={args.benchmark_model: "benchmark_prediction"}).to_parquet(out_path, index=False)

    payload = {
        "settings": {
            "model_pack": model_names,
            "n_samples": int(args.n_samples),
            "prefilter_top_n": int(args.prefilter_top_n),
            "corr_threshold": float(args.corr_threshold),
            "benchmark_penalty": float(args.benchmark_penalty),
            "example_penalty": float(args.example_penalty),
            "eval_era_min": int(min(eval_eras)),
            "eval_era_max": int(max(eval_eras)),
            "eval_era_count": int(len(eval_eras)),
        },
        "best": best_row,
        "cheap_prefilter_top": [
            {
                "cheap_score": float(r["cheap_score"]),
                "corr_with_benchmark_global": float(r["corr_with_benchmark_global"]),
                "corr_with_example_global": float(r["corr_with_example_global"]),
                "cheap_delta_mean": float(r["cheap_delta_mean"]),
                "weights": r["weights"],
            }
            for r in cheap_rows[: int(args.top_k)]
        ],
        "top_candidates": top_rows,
        "output": {
            "predictions_file": str(out_path.relative_to(experiment_dir.parent)),
        },
    }
    summary_path = results_dir / args.summary_name
    _write_result_json(summary_path, payload)
    print(json.dumps(payload["best"], indent=2))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
