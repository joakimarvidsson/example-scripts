from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from agents.experiments.nn_ender20_latest_residualized.strict_gbt_walkforward_research import (
    _artifact_name,
    _compute_delta_metrics,
    _load_example_preds_for_eval,
    _parse_candidate_modes,
    _parse_lambdas,
    _rank01_per_era,
    _resolve_input_path,
    _select_strict_blend,
    _write_result_json,
)
from agents.code.modeling.utils.target_transforms import subtract_scaled_invnorm_column


@dataclass(frozen=True)
class StackSpec:
    name: str
    model_kind: str
    feature_mode: str
    target_mode: str
    alpha: float
    positive: bool
    base_models: tuple[str, ...]
    residual_scale: float


MODEL_PATHS = {
    "cat_dense": Path(
        "/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/"
        "nn_ender20_latest_residualized/predictions/cat_strict_resid008_medfaith64_walkfwd_strict_dense_e4_r700.parquet"
    ),
    "mlp_roundM6": Path(
        "/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/"
        "nn_ender20_latest_residualized/predictions/torch_mlp_resid_latest_ender20_roundM6_base_w80_m3_w20.parquet"
    ),
    "xgb_hist": Path(
        "/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/"
        "nn_ender20_latest_residualized/predictions/xgb_full_offset0_medium_walkfwd_lam009_eval575plus.parquet"
    ),
    "ridge_smallfaith": Path(
        "/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/"
        "nn_ender20_latest_residualized/predictions/ridge_strict_resid008_smallfaith64_a10_walkfwd_strict_roundS26_ridge_refine.parquet"
    ),
    "lgbm_optuna_confirm": Path(
        "/Users/joakim/Documents/Projects/Numerai/example-scripts-codex-faith-scale/numerai/agents/experiments/"
        "nn_ender20_latest_residualized/predictions/optuna_lgbm_shallow_highlr_best_confirm.parquet"
    ),
    "mlp_resid010_dense_seedavg3": Path(
        "/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/"
        "nn_ender20_latest_residualized/predictions/mtmlp_strict_resid010_medfaith64_mainonly_walkfwd_"
        "raw_walkfwd_roundS44dense_seedavg3.parquet"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward prediction stacker over existing OOF Numerai candidates."
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
    parser.add_argument("--eval-era-step", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=26)
    parser.add_argument("--early-era-max", type=int, default=889)
    parser.add_argument(
        "--blend-lambdas",
        default="0.02,0.05,0.1,0.2,0.5,1.0",
    )
    parser.add_argument(
        "--neutralize-benchmark-grid",
        default="0.0,0.1,0.2,0.3",
    )
    parser.add_argument(
        "--neutralize-example-grid",
        default="0.0,0.1,0.2,0.3",
    )
    parser.add_argument(
        "--candidate-modes",
        default="blend,blend_neutralize",
    )
    parser.add_argument(
        "--spec-names",
        default="",
        help="Optional comma-separated stack spec names to run.",
    )
    parser.add_argument("--summary-name", default="walkforward_prediction_stacker_summary.json")
    parser.add_argument("--round-note-name", default="roundS34_prediction_stacker.md")
    return parser.parse_args()


def _load_prediction_column(path: Path, name: str, *, id_col: str, era_col: str, min_eval_era: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    pred_col = "prediction" if "prediction" in df.columns else "prediction_raw" if "prediction_raw" in df.columns else None
    if pred_col is None:
        raise ValueError(f"{path} missing prediction or prediction_raw column")
    cols = [id_col, era_col, pred_col]
    out = df[cols].copy().rename(columns={pred_col: name})
    out = out[out[era_col].astype(int) >= int(min_eval_era)].copy()
    return out


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
    eval_era_step: int,
) -> pd.DataFrame:
    era_vals = [f"{era:04d}" for era in range(min_eval_era, max_eval_era + 1) if (era - min_eval_era) % eval_era_step == 0]
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


def _build_stack_specs() -> list[StackSpec]:
    return [
        StackSpec(
            name="stack_ridge_treepack_mlp_r006_a10",
            model_kind="ridge",
            feature_mode="raw_ranks",
            target_mode="residualized",
            alpha=10.0,
            positive=False,
            base_models=("cat_dense", "xgb_hist", "ridge_smallfaith", "mlp_roundM6"),
            residual_scale=0.006,
        ),
        StackSpec(
            name="stack_ridge_treepack_mlp_lgbm_r006_a10",
            model_kind="ridge",
            feature_mode="raw_ranks",
            target_mode="residualized",
            alpha=10.0,
            positive=False,
            base_models=("cat_dense", "xgb_hist", "ridge_smallfaith", "lgbm_optuna_confirm", "mlp_roundM6"),
            residual_scale=0.006,
        ),
        StackSpec(
            name="stack_ridge_deltas_mlp_r006_a10",
            model_kind="ridge",
            feature_mode="tree_deltas_plus_mlp",
            target_mode="residualized",
            alpha=10.0,
            positive=False,
            base_models=("cat_dense", "xgb_hist", "ridge_smallfaith", "mlp_roundM6"),
            residual_scale=0.006,
        ),
        StackSpec(
            name="stack_ridge_cat_mlp_r006_a1",
            model_kind="ridge",
            feature_mode="raw_ranks",
            target_mode="residualized",
            alpha=1.0,
            positive=False,
            base_models=("cat_dense", "mlp_roundM6"),
            residual_scale=0.006,
        ),
        StackSpec(
            name="stack_ridge_treepackfull_mlp_r006_a10",
            model_kind="ridge",
            feature_mode="raw_ranks",
            target_mode="residualized",
            alpha=10.0,
            positive=False,
            base_models=("cat_dense", "xgb_hist", "lgbm_optuna_confirm", "mlp_roundM6"),
            residual_scale=0.006,
        ),
        StackSpec(
            name="stack_poslin_treepackfull_mlp_raw",
            model_kind="linear",
            feature_mode="raw_ranks",
            target_mode="raw",
            alpha=0.0,
            positive=True,
            base_models=("cat_dense", "xgb_hist", "lgbm_optuna_confirm", "mlp_roundM6"),
            residual_scale=0.0,
        ),
        StackSpec(
            name="stack_poslin_treepack_mlp_raw",
            model_kind="linear",
            feature_mode="raw_ranks",
            target_mode="raw",
            alpha=0.0,
            positive=True,
            base_models=("cat_dense", "xgb_hist", "ridge_smallfaith", "mlp_roundM6"),
            residual_scale=0.0,
        ),
        StackSpec(
            name="stack_poslin_cat_dense_resid010seedavg3_mlp_raw",
            model_kind="linear",
            feature_mode="raw_ranks",
            target_mode="raw",
            alpha=0.0,
            positive=True,
            base_models=("cat_dense", "mlp_resid010_dense_seedavg3", "mlp_roundM6"),
            residual_scale=0.0,
        ),
        StackSpec(
            name="stack_ridge_cat_dense_resid010seedavg3_mlp_r006_a1",
            model_kind="ridge",
            feature_mode="raw_ranks",
            target_mode="residualized",
            alpha=1.0,
            positive=False,
            base_models=("cat_dense", "mlp_resid010_dense_seedavg3", "mlp_roundM6"),
            residual_scale=0.006,
        ),
        StackSpec(
            name="stack_poslin_cat_dense_resid010seedavg3_raw",
            model_kind="linear",
            feature_mode="raw_ranks",
            target_mode="raw",
            alpha=0.0,
            positive=True,
            base_models=("cat_dense", "mlp_resid010_dense_seedavg3"),
            residual_scale=0.0,
        ),
    ]


def _make_features(df: pd.DataFrame, spec: StackSpec, benchmark_col: str) -> tuple[list[str], pd.DataFrame]:
    work = df.copy()
    feature_cols: list[str] = []
    bench_rank_col = f"{benchmark_col}_rank"
    if spec.feature_mode == "raw_ranks":
        for model in spec.base_models:
            col = f"{model}_rank"
            feature_cols.append(col)
    elif spec.feature_mode == "tree_deltas_plus_mlp":
        for model in spec.base_models:
            col = f"{model}_rank"
            if model.startswith("mlp"):
                feature_cols.append(col)
                continue
            delta_col = f"{model}_delta_rank"
            work[delta_col] = work[col] - work[bench_rank_col]
            feature_cols.append(delta_col)
    else:
        raise ValueError(f"Unsupported feature mode: {spec.feature_mode}")
    return feature_cols, work


def _build_estimator(spec: StackSpec):
    if spec.model_kind == "ridge":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=float(spec.alpha))),
            ]
        )
    if spec.model_kind == "linear":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LinearRegression(positive=bool(spec.positive))),
            ]
        )
    raise ValueError(f"Unsupported model kind: {spec.model_kind}")


def _train_walkforward_stack(
    merged: pd.DataFrame,
    spec: StackSpec,
    *,
    benchmark_col: str,
    target_col: str,
    id_col: str,
    era_col: str,
    eval_eras: list[int],
    block_size: int,
) -> pd.DataFrame:
    ranked = merged.copy()
    ranked[f"{benchmark_col}_rank"] = _rank01_per_era(ranked[benchmark_col], ranked[era_col])
    for model in spec.base_models:
        ranked[f"{model}_rank"] = _rank01_per_era(ranked[model], ranked[era_col])

    feature_cols, ranked = _make_features(ranked, spec, benchmark_col)
    preds: list[pd.DataFrame] = []
    blocks = [eval_eras[i : i + block_size] for i in range(0, len(eval_eras), block_size)]

    for block_idx, val_eras in enumerate(blocks):
        train_eras = [era for era in eval_eras if era < min(val_eras)]
        train_df = ranked[ranked[era_col].astype(int).isin(train_eras)].copy()
        val_df = ranked[ranked[era_col].astype(int).isin(val_eras)].copy()
        if train_df.empty or val_df.empty:
            continue

        y_train = train_df[target_col].astype(float)
        if spec.target_mode == "residualized" and spec.residual_scale > 0:
            y_train = subtract_scaled_invnorm_column(
                y_train,
                train_df[[benchmark_col, era_col]],
                benchmark_col=benchmark_col,
                era_col=era_col,
                scale=float(spec.residual_scale),
                per_era=True,
                use_rank=False,
                clip_eps=1e-6,
                center=True,
                center_per_era=False,
            )
        y_train = np.asarray(y_train, dtype=np.float64)
        est = _build_estimator(spec)
        est.fit(train_df[feature_cols].to_numpy(), y_train)
        pred = est.predict(val_df[feature_cols].to_numpy()).astype(np.float64)
        out = val_df[[id_col, era_col, target_col, benchmark_col]].copy()
        out["prediction_raw"] = pred
        preds.append(out)
        print(
            f"{spec.name}: block {block_idx + 1}/{len(blocks)} "
            f"train_eras={len(train_eras)} train_rows={len(train_df):,} "
            f"val_rows={len(val_df):,}",
            flush=True,
        )

    pred_df = pd.concat(preds, ignore_index=True)
    pred_df["prediction_raw"] = (
        _rank01_per_era(pred_df[benchmark_col], pred_df[era_col])
        + pred_df["prediction_raw"]
    )
    return pred_df


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

    base = _load_target_benchmark(
        full_data_path=args.full_data_path,
        benchmark_data_path=args.benchmark_data_path,
        id_col=args.id_col,
        era_col=args.era_col,
        target_col=args.target_col,
        benchmark_model=args.benchmark_model,
        min_eval_era=int(args.min_eval_era),
        max_eval_era=int(args.max_eval_era),
        eval_era_step=int(args.eval_era_step),
    )
    prediction_frames: dict[str, pd.DataFrame] = {}
    for name, path in MODEL_PATHS.items():
        prediction_frames[name] = _load_prediction_column(
            path,
            name,
            id_col=args.id_col,
            era_col=args.era_col,
            min_eval_era=int(args.min_eval_era),
        )
        prediction_frames[name][args.era_col] = prediction_frames[name][args.era_col].astype(str)

    rows: list[dict] = []
    note_lines = [
        "# Round S34: Walk-forward prediction stacker",
        "",
        "Goal: stack the strongest existing OOF models using a cheap walk-forward meta-model instead of another raw model family.",
        "",
    ]

    specs = _build_stack_specs()
    if args.spec_names.strip():
        wanted = {name.strip() for name in args.spec_names.split(",") if name.strip()}
        specs = [spec for spec in specs if spec.name in wanted]
        if not specs:
            raise ValueError(f"No matching --spec-names found: {sorted(wanted)}")

    for spec in specs:
        merged = base.copy()
        for model_name in spec.base_models:
            merged = merged.merge(
                prediction_frames[model_name],
                on=[args.id_col, args.era_col],
                how="inner",
            )
        eval_eras = sorted(
            {
                int(e)
                for e in merged[args.era_col].astype(int)
                if int(args.min_eval_era) <= int(e) <= int(args.max_eval_era)
            }
        )
        example_df = _load_example_preds_for_eval(
            example_preds_path=args.example_preds_path.resolve(),
            full_path=args.full_data_path.resolve(),
            id_col=args.id_col,
            era_col=args.era_col,
            eval_eras=eval_eras,
        )
        pred_df = _train_walkforward_stack(
            merged,
            spec,
            benchmark_col=args.benchmark_model,
            target_col=args.target_col,
            id_col=args.id_col,
            era_col=args.era_col,
            eval_eras=eval_eras,
            block_size=int(args.block_size),
        )
        strict_df, strict_metrics = _select_strict_blend(
            pred_df,
            id_col=args.id_col,
            example_df=example_df,
            benchmark_col=args.benchmark_model,
            target_col=args.target_col,
            era_col=args.era_col,
            early_era_max=args.early_era_max,
            blend_lambdas=_parse_lambdas(args.blend_lambdas),
            neutralize_benchmark_grid=_parse_lambdas(args.neutralize_benchmark_grid),
            neutralize_example_grid=_parse_lambdas(args.neutralize_example_grid),
            candidate_modes=_parse_candidate_modes(args.candidate_modes),
            max_corr_with_benchmark=1.1,
            max_corr_with_example=1.1,
            min_delta_mean=0.0,
            min_delta_cumsum_end=0.0,
            selection_objective="corr_sortino_vs_benchmark",
        )
        out_name = _artifact_name(spec.name, "")
        out_path = predictions_dir / f"{out_name}.parquet"
        strict_df.rename(columns={args.benchmark_model: "benchmark_prediction"}).to_parquet(out_path, index=False)
        payload = {
            "stack_spec": {
                "model_kind": spec.model_kind,
                "feature_mode": spec.feature_mode,
                "target_mode": spec.target_mode,
                "alpha": spec.alpha,
                "positive": spec.positive,
                "base_models": list(spec.base_models),
                "residual_scale": spec.residual_scale,
            },
            "metrics": strict_metrics,
            "output": {"predictions_file": str(out_path.relative_to(experiment_dir.parent))},
        }
        _write_result_json(results_dir / f"{out_name}.json", payload)
        pred_eras = sorted(pred_df[args.era_col].astype(int).unique())
        rows.append(
            {
                "model": out_name,
                "base_models": ",".join(spec.base_models),
                "feature_mode": spec.feature_mode,
                "target_mode": spec.target_mode,
                "alpha": spec.alpha,
                "merged_era_count": len(eval_eras),
                "merged_era_min": int(min(eval_eras)) if eval_eras else None,
                "merged_era_max": int(max(eval_eras)) if eval_eras else None,
                "predicted_era_count": len(pred_eras),
                "predicted_era_min": int(min(pred_eras)) if pred_eras else None,
                "predicted_era_max": int(max(pred_eras)) if pred_eras else None,
                **strict_metrics,
            }
        )
        note_lines.extend(
            [
                f"## {out_name}",
                "",
                f"- base models: `{', '.join(spec.base_models)}`",
                f"- feature mode: `{spec.feature_mode}`",
                f"- target mode: `{spec.target_mode}`",
                f"- merged eras: `{len(eval_eras)}` (`{min(eval_eras) if eval_eras else 'na'}` to `{max(eval_eras) if eval_eras else 'na'}`)",
                f"- predicted eras: `{len(pred_eras)}` (`{min(pred_eras) if pred_eras else 'na'}` to `{max(pred_eras) if pred_eras else 'na'}`)",
                f"- delta cumsum end: `{strict_metrics['delta_cumsum_end']:.6f}`",
                f"- bmc mean: `{strict_metrics.get('bmc_mean', float('nan')):.6f}`",
                f"- payout mean: `{strict_metrics.get('payout_mean', float('nan')):.6f}`",
                "",
            ]
        )

    summary_df = pd.DataFrame(rows).sort_values("delta_cumsum_end", ascending=False)
    summary_path = results_dir / args.summary_name
    _write_result_json(
        summary_path,
        {
            "settings": {
                "eval_era_step": int(args.eval_era_step),
                "block_size": int(args.block_size),
                "blend_lambdas": _parse_lambdas(args.blend_lambdas),
                "neutralize_benchmark_grid": _parse_lambdas(args.neutralize_benchmark_grid),
                "neutralize_example_grid": _parse_lambdas(args.neutralize_example_grid),
                "candidate_modes": _parse_candidate_modes(args.candidate_modes),
            },
            "top_models": summary_df.to_dict(orient="records"),
            "selected_best_model": str(summary_df.iloc[0]["model"]),
        },
    )
    (experiment_dir / args.round_note_name).write_text("\n".join(note_lines))
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
