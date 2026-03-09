from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from optuna.samplers import TPESampler

from agents.experiments.nn_ender20_latest_residualized.strict_gbt_walkforward_research import (
    ModelSpec,
    _feature_cols_for_spec,
    _load_example_preds_for_eval,
    _load_feature_sets,
    _parse_candidate_modes,
    _parse_lambdas,
    _resolve_features_json,
    _resolve_input_path,
    _select_strict_blend,
    _train_walkforward_model,
    _write_result_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optuna scout for shallow, higher-learning-rate tree models on the "
            "strict walk-forward ender20 benchmark-delta setup."
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
    parser.add_argument("--early-era-max", type=int, default=889)
    parser.add_argument("--families", default="xgb,lgbm,catboost")
    parser.add_argument("--n-trials", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2601)
    parser.add_argument("--depth-min", type=int, default=3)
    parser.add_argument("--depth-max", type=int, default=6)
    parser.add_argument("--learning-rate-min", type=float, default=0.05)
    parser.add_argument("--learning-rate-max", type=float, default=0.5)
    parser.add_argument(
        "--feature-choices",
        default="",
        help="Optional comma-separated feature sets to constrain the search.",
    )
    parser.add_argument(
        "--residual-scale-choices",
        default="",
        help="Optional comma-separated residual scales to constrain the search.",
    )
    parser.add_argument(
        "--lgbm-boosting-types",
        default="gbdt,dart",
        help="Comma-separated LightGBM boosting types to allow.",
    )
    parser.add_argument(
        "--xgb-offset-choices",
        default="full,0,1,2,3",
        help="Comma-separated XGBoost offset choices.",
    )
    parser.add_argument("--scout-eval-era-step", type=int, default=8)
    parser.add_argument("--scout-block-size", type=int, default=26)
    parser.add_argument("--scout-max-rows-per-era", type=int, default=500)
    parser.add_argument("--confirm-eval-era-step", type=int, default=4)
    parser.add_argument("--confirm-block-size", type=int, default=26)
    parser.add_argument("--confirm-max-rows-per-era", type=int, default=700)
    parser.add_argument(
        "--blend-lambdas",
        default="0.02,0.05,0.075,0.10,0.125,0.15,0.20,0.30,0.50,0.75,1.0",
    )
    parser.add_argument(
        "--neutralize-benchmark-grid",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0",
    )
    parser.add_argument(
        "--neutralize-example-grid",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0",
    )
    parser.add_argument(
        "--candidate-modes",
        default="blend,neutralize,blend_neutralize",
    )
    parser.add_argument("--summary-name", default="optuna_shallow_highlr_tree_summary.json")
    parser.add_argument("--round-note-name", default="roundS32_optuna_shallow_highlr.md")
    return parser.parse_args()


def _family_seed(base_seed: int, family: str) -> int:
    offsets = {"xgb": 0, "lgbm": 1000, "catboost": 2000}
    return int(base_seed) + offsets[family]


def _parse_families(raw: str) -> list[str]:
    allowed = {"xgb", "lgbm", "catboost"}
    out = [token.strip().lower() for token in raw.split(",") if token.strip()]
    invalid = [token for token in out if token not in allowed]
    if invalid:
        raise ValueError(f"Unsupported families: {invalid}")
    return out


def _parse_csv_choices(raw: str) -> list[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _parse_float_choices(raw: str) -> list[float]:
    return [float(token.strip()) for token in str(raw).split(",") if token.strip()]


def _load_all_eras(full_data_path: Path, era_col: str) -> list[int]:
    era_only = pd.read_parquet(full_data_path, columns=[era_col])
    return sorted({int(e) for e in era_only[era_col].astype(str).tolist()})


def _eval_eras(all_eras: list[int], *, min_eval_era: int, max_eval_era: int, eval_era_step: int) -> list[int]:
    return [
        era
        for era in all_eras
        if min_eval_era <= era <= max_eval_era and ((era - min_eval_era) % eval_era_step == 0)
    ]


def _feature_choices_for_family(family: str, override: list[str] | None = None) -> list[str]:
    if override:
        return override
    if family == "xgb":
        return ["medium", "small", "medium+faith2:64", "small+faith2:64"]
    if family == "lgbm":
        return ["small+faith2:64", "medium+faith2:64", "small", "medium"]
    if family == "catboost":
        return ["medium+faith2:64", "small+faith2:64", "medium", "small"]
    raise ValueError(f"Unsupported family: {family}")


def _spec_from_trial(
    family: str,
    trial: optuna.Trial,
    *,
    seed: int,
    args: argparse.Namespace,
) -> ModelSpec:
    feature_override = _parse_csv_choices(args.feature_choices) if args.feature_choices.strip() else None
    residual_choices = (
        _parse_float_choices(args.residual_scale_choices)
        if args.residual_scale_choices.strip()
        else [0.0, 0.006, 0.008, 0.010, 0.012]
    )
    feature_set = trial.suggest_categorical("feature_set", _feature_choices_for_family(family, feature_override))
    residual_scale = trial.suggest_categorical("residual_scale", residual_choices)
    depth = trial.suggest_int("max_depth", int(args.depth_min), int(args.depth_max))
    learning_rate = trial.suggest_float(
        "learning_rate",
        float(args.learning_rate_min),
        float(args.learning_rate_max),
        log=True,
    )

    if family == "xgb":
        offset_token = trial.suggest_categorical("offset", _parse_csv_choices(args.xgb_offset_choices))
        offset = None if offset_token == "full" else int(offset_token)
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": -1,
            "n_estimators": trial.suggest_int("n_estimators", 80, 320, step=20),
            "learning_rate": learning_rate,
            "max_depth": depth,
            "subsample": trial.suggest_float("subsample", 0.70, 1.00, step=0.05),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.15, 0.50, step=0.05),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0, step=0.1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 8.0, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 60.0),
            "random_state": seed,
        }
        name = f"optuna_xgb_shallow_highlr_t{trial.number:03d}"
        return ModelSpec(
            name=name,
            model_family="xgb",
            feature_set=feature_set,
            offset=offset,
            residual_scale=float(residual_scale),
            params=params,
        )

    if family == "lgbm":
        boosting_type = trial.suggest_categorical("boosting_type", _parse_csv_choices(args.lgbm_boosting_types))
        params: dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 420, step=20),
            "learning_rate": learning_rate,
            "num_leaves": min(63, 2**depth - 1),
            "max_depth": depth,
            "subsample": trial.suggest_float("subsample", 0.70, 1.00, step=0.05),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.15, 0.50, step=0.05),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0, step=0.1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 8.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 120, step=10),
            "objective": "regression",
            "boosting_type": boosting_type,
            "n_jobs": -1,
            "verbose": -1,
            "random_state": seed,
        }
        if boosting_type == "dart":
            params["drop_rate"] = trial.suggest_float("drop_rate", 0.05, 0.30, step=0.05)
            params["skip_drop"] = trial.suggest_float("skip_drop", 0.20, 0.80, step=0.10)
        name = f"optuna_lgbm_shallow_highlr_t{trial.number:03d}"
        return ModelSpec(
            name=name,
            model_family="lgbm",
            feature_set=feature_set,
            offset=None,
            residual_scale=float(residual_scale),
            params=params,
        )

    if family == "catboost":
        params = {
            "iterations": trial.suggest_int("iterations", 80, 500, step=20),
            "depth": depth,
            "learning_rate": learning_rate,
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.70, 1.00, step=0.05),
            "loss_function": "RMSE",
            "allow_writing_files": False,
            "thread_count": -1,
            "verbose": 0,
            "random_seed": seed,
        }
        name = f"optuna_cat_shallow_highlr_t{trial.number:03d}"
        return ModelSpec(
            name=name,
            model_family="catboost",
            feature_set=feature_set,
            offset=None,
            residual_scale=float(residual_scale),
            params=params,
        )

    raise ValueError(f"Unsupported family: {family}")


def _evaluate_spec(
    spec: ModelSpec,
    *,
    args: argparse.Namespace,
    feature_sets: dict[str, list[str]],
    all_eras: list[int],
    eval_eras: list[int],
    block_size: int,
    max_rows_per_era: int,
    example_df: pd.DataFrame | None,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_cols = _feature_cols_for_spec(feature_sets, spec.feature_set)
    raw_pred_df = _train_walkforward_model(
        spec,
        full_path=args.full_data_path.resolve(),
        bench_path=args.benchmark_data_path.resolve(),
        all_eras=all_eras,
        eval_eras=eval_eras,
        block_size=block_size,
        max_rows_per_era=max_rows_per_era,
        id_col=args.id_col,
        era_col=args.era_col,
        eval_target_col=args.target_col,
        train_target_col=args.target_col,
        train_target_mix=[],
        benchmark_model=args.benchmark_model,
        feature_cols=feature_cols,
        seed=seed,
    )
    strict_df, strict_metrics = _select_strict_blend(
        raw_pred_df,
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
    return strict_df, strict_metrics


def _trial_row(
    family: str,
    trial_number: int,
    spec: ModelSpec,
    metrics: dict[str, Any],
    *,
    scout_eval_era_step: int,
    scout_max_rows_per_era: int,
) -> dict[str, Any]:
    return {
        "family": family,
        "trial_number": int(trial_number),
        "model": spec.name,
        "feature_set": spec.feature_set,
        "offset": spec.offset if spec.offset is not None else "full",
        "residual_scale": spec.residual_scale,
        "params": spec.params,
        "scout_eval_era_step": int(scout_eval_era_step),
        "scout_max_rows_per_era": int(scout_max_rows_per_era),
        **metrics,
    }


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

    feature_sets = _load_feature_sets(_resolve_features_json())
    all_eras = _load_all_eras(args.full_data_path.resolve(), args.era_col)
    scout_eval_eras = _eval_eras(
        all_eras,
        min_eval_era=int(args.min_eval_era),
        max_eval_era=int(args.max_eval_era),
        eval_era_step=int(args.scout_eval_era_step),
    )
    confirm_eval_eras = _eval_eras(
        all_eras,
        min_eval_era=int(args.min_eval_era),
        max_eval_era=int(args.max_eval_era),
        eval_era_step=int(args.confirm_eval_era_step),
    )
    if not scout_eval_eras or not confirm_eval_eras:
        raise ValueError("Missing scout or confirm evaluation eras.")

    scout_example_df = _load_example_preds_for_eval(
        example_preds_path=args.example_preds_path.resolve(),
        full_path=args.full_data_path.resolve(),
        id_col=args.id_col,
        era_col=args.era_col,
        eval_eras=scout_eval_eras,
    )
    confirm_example_df = _load_example_preds_for_eval(
        example_preds_path=args.example_preds_path.resolve(),
        full_path=args.full_data_path.resolve(),
        id_col=args.id_col,
        era_col=args.era_col,
        eval_eras=confirm_eval_eras,
    )

    summary: dict[str, Any] = {
        "settings": {
            "families": _parse_families(args.families),
            "n_trials": int(args.n_trials),
            "target_col": args.target_col,
            "benchmark_model": args.benchmark_model,
            "scout_eval_era_step": int(args.scout_eval_era_step),
            "scout_block_size": int(args.scout_block_size),
            "scout_max_rows_per_era": int(args.scout_max_rows_per_era),
            "confirm_eval_era_step": int(args.confirm_eval_era_step),
            "confirm_block_size": int(args.confirm_block_size),
            "confirm_max_rows_per_era": int(args.confirm_max_rows_per_era),
            "blend_lambdas": _parse_lambdas(args.blend_lambdas),
            "neutralize_benchmark_grid": _parse_lambdas(args.neutralize_benchmark_grid),
            "neutralize_example_grid": _parse_lambdas(args.neutralize_example_grid),
            "candidate_modes": _parse_candidate_modes(args.candidate_modes),
        },
        "families": {},
    }
    note_lines = [
        "# Round S32: Optuna shallow/high-lr tree scout",
        "",
        "Goal: test shallower tree models with higher learning rates using Optuna and check whether they stay additive on `target_ender_20` versus `v52_lgbm_ender20`.",
        "",
    ]

    for family in _parse_families(args.families):
        family_seed = _family_seed(args.seed, family)
        sampler = TPESampler(seed=family_seed)
        study = optuna.create_study(
            study_name=f"s32_{family}_shallow_highlr",
            direction="maximize",
            sampler=sampler,
        )
        trial_rows: list[dict[str, Any]] = []
        trial_payloads: dict[int, dict[str, Any]] = {}

        def objective(trial: optuna.Trial) -> float:
            spec = _spec_from_trial(family, trial, seed=family_seed + trial.number, args=args)
            _, metrics = _evaluate_spec(
                spec,
                args=args,
                feature_sets=feature_sets,
                all_eras=all_eras,
                eval_eras=scout_eval_eras,
                block_size=int(args.scout_block_size),
                max_rows_per_era=int(args.scout_max_rows_per_era),
                example_df=scout_example_df,
                seed=family_seed + trial.number,
            )
            row = _trial_row(
                family,
                trial.number,
                spec,
                metrics,
                scout_eval_era_step=int(args.scout_eval_era_step),
                scout_max_rows_per_era=int(args.scout_max_rows_per_era),
            )
            trial_rows.append(row)
            trial_payloads[trial.number] = {
                "model": {
                    "family": family,
                    "name": spec.name,
                    "feature_set": spec.feature_set,
                    "offset": spec.offset,
                    "residual_scale": spec.residual_scale,
                    "params": spec.params,
                },
                "metrics": metrics,
            }
            trial.set_user_attr("feature_set", spec.feature_set)
            trial.set_user_attr("offset", spec.offset if spec.offset is not None else "full")
            trial.set_user_attr("residual_scale", spec.residual_scale)
            trial.set_user_attr("delta_cumsum_end", float(metrics["delta_cumsum_end"]))
            trial.set_user_attr("delta_mean", float(metrics["delta_mean"]))
            trial.set_user_attr("bmc_mean", float(metrics.get("bmc_mean", float("nan"))))
            trial.set_user_attr("payout_mean", float(metrics.get("payout_mean", float("nan"))))
            trial.set_user_attr("selected_feasible", bool(metrics.get("selected_feasible", False)))
            return float(metrics["delta_cumsum_end"])

        study.optimize(objective, n_trials=int(args.n_trials), show_progress_bar=False)

        trials_df = pd.DataFrame(trial_rows).sort_values(
            ["selected_feasible", "delta_cumsum_end", "bmc_mean", "payout_mean"],
            ascending=[False, False, False, False],
        )
        best_trial_number = int(trials_df.iloc[0]["trial_number"])
        best_trial_payload = trial_payloads[best_trial_number]
        best_trial_spec = ModelSpec(
            name=f"optuna_{family}_shallow_highlr_best_confirm",
            model_family=family,
            feature_set=str(best_trial_payload["model"]["feature_set"]),
            offset=best_trial_payload["model"]["offset"],
            residual_scale=float(best_trial_payload["model"]["residual_scale"]),
            params=dict(best_trial_payload["model"]["params"]),
        )

        confirm_df, confirm_metrics = _evaluate_spec(
            best_trial_spec,
            args=args,
            feature_sets=feature_sets,
            all_eras=all_eras,
            eval_eras=confirm_eval_eras,
            block_size=int(args.confirm_block_size),
            max_rows_per_era=int(args.confirm_max_rows_per_era),
            example_df=confirm_example_df,
            seed=family_seed + 10_000 + best_trial_number,
        )
        confirm_pred_path = predictions_dir / f"{best_trial_spec.name}.parquet"
        confirm_df.rename(columns={args.benchmark_model: "benchmark_prediction"}).to_parquet(
            confirm_pred_path, index=False
        )
        confirm_result_path = results_dir / f"{best_trial_spec.name}.json"
        _write_result_json(
            confirm_result_path,
            {
                "selection": {
                    "family": family,
                    "best_trial_number": best_trial_number,
                    "objective": "delta_cumsum_end",
                },
                "scout_best": best_trial_payload,
                "confirm": {
                    "feature_set": best_trial_spec.feature_set,
                    "offset": best_trial_spec.offset,
                    "residual_scale": best_trial_spec.residual_scale,
                    "params": best_trial_spec.params,
                    "metrics": confirm_metrics,
                    "predictions_file": str(confirm_pred_path.relative_to(experiment_dir.parent)),
                },
            },
        )

        top_trials_path = results_dir / f"optuna_{family}_shallow_highlr_trials.json"
        _write_result_json(
            top_trials_path,
            {
                "family": family,
                "top_trials": trials_df.head(10).to_dict(orient="records"),
                "best_trial_number": best_trial_number,
                "confirm_result_file": str(confirm_result_path.relative_to(experiment_dir.parent)),
            },
        )

        summary["families"][family] = {
            "best_trial_number": best_trial_number,
            "best_scout": best_trial_payload,
            "confirm_metrics": confirm_metrics,
            "top_trials": trials_df.head(5).to_dict(orient="records"),
            "trial_results_file": str(top_trials_path.relative_to(experiment_dir.parent)),
            "confirm_result_file": str(confirm_result_path.relative_to(experiment_dir.parent)),
        }

        note_lines.extend(
            [
                f"## {family}",
                "",
                f"- Best scout trial: `{best_trial_number}`",
                f"- Scout feature set: `{best_trial_payload['model']['feature_set']}`",
                f"- Scout residual scale: `{best_trial_payload['model']['residual_scale']}`",
                f"- Scout delta cumsum end: `{best_trial_payload['metrics']['delta_cumsum_end']:.6f}`",
                f"- Scout bmc mean: `{best_trial_payload['metrics'].get('bmc_mean', float('nan')):.6f}`",
                f"- Dense confirm delta cumsum end: `{confirm_metrics['delta_cumsum_end']:.6f}`",
                f"- Dense confirm bmc mean: `{confirm_metrics.get('bmc_mean', float('nan')):.6f}`",
                f"- Dense confirm payout mean: `{confirm_metrics.get('payout_mean', float('nan')):.6f}`",
                f"- Dense confirm additive feasible: `{bool(confirm_metrics.get('selected_feasible', False))}`",
                "",
            ]
        )

    summary_path = results_dir / args.summary_name
    _write_result_json(summary_path, summary)
    (experiment_dir / args.round_note_name).write_text("\n".join(note_lines))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
