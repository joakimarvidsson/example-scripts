"""CLI for HPO v3: Three-phase pipeline.

Usage:
    # Phase 1: HPO
    python -m agents.code.modeling.hpo_v3_cli hpo --n-trials 20

    # Phase 2: Neutralization tuning (after HPO)
    python -m agents.code.modeling.hpo_v3_cli neutralization --best-params-file results.json

    # Phase 3: Holdout evaluation (final, once only)
    python -m agents.code.modeling.hpo_v3_cli holdout --best-params-file results.json --neutralization-strength 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="HPO v3: Three-phase pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # HPO phase
    hpo_parser = subparsers.add_parser("hpo", help="Run HPO phase")
    hpo_parser.add_argument("--n-trials", type=int, default=20)
    hpo_parser.add_argument("--metric", default="bmc_sharpe")
    hpo_parser.add_argument("--n-strides", type=int, default=3)
    hpo_parser.add_argument("--folds-per-stride", type=int, default=2)
    hpo_parser.add_argument("--n-estimators", type=int, default=2000)
    hpo_parser.add_argument("--hpo-max-era", type=int, default=800)
    hpo_parser.add_argument("--wandb-project", default=None)
    hpo_parser.add_argument("--study-name", default=None)
    hpo_parser.add_argument("--model-type", default="LGBMRegressor",
                            choices=["LGBMRegressor", "XGBRegressor", "CatBoostRegressor"],
                            help="Model type to optimize")

    # Neutralization phase
    neut_parser = subparsers.add_parser("neutralization", help="Run neutralization tuning")
    neut_parser.add_argument("--best-params-file", type=Path, required=True)
    neut_parser.add_argument("--n-estimators", type=int, default=2000)
    neut_parser.add_argument("--strengths", type=str, default="0,0.25,0.5,0.75,1.0")
    neut_parser.add_argument("--model-type", default="LGBMRegressor",
                            choices=["LGBMRegressor", "XGBRegressor", "CatBoostRegressor"],
                            help="Model type to use")

    # Holdout phase
    holdout_parser = subparsers.add_parser("holdout", help="Run holdout evaluation (FINAL)")
    holdout_parser.add_argument("--best-params-file", type=Path, required=True)
    holdout_parser.add_argument("--neutralization-strength", type=float, required=True)
    holdout_parser.add_argument("--n-estimators", type=int, default=2000)
    holdout_parser.add_argument("--model-type", default="LGBMRegressor",
                            choices=["LGBMRegressor", "XGBRegressor", "CatBoostRegressor"],
                            help="Model type to use")

    args = parser.parse_args()

    if args.command == "hpo":
        from agents.code.modeling.utils.hpo_v3 import run_hpo_phase

        study, best_params = run_hpo_phase(
            n_trials=args.n_trials,
            metric=args.metric,
            n_strides=args.n_strides,
            folds_per_stride=args.folds_per_stride,
            n_estimators=args.n_estimators,
            hpo_max_era=args.hpo_max_era,
            wandb_project=args.wandb_project,
            study_name=args.study_name,
            model_type=args.model_type,
            # Use standard paths
            downsampled_path="v5.2/downsampled_full.parquet",
            benchmark_path="v5.2/downsampled_full_benchmark_models.parquet",
        )

        print(f"\nBest params: {json.dumps(best_params, indent=2)}")

    elif args.command == "neutralization":
        from agents.code.modeling.utils.hpo_v3 import evaluate_on_neutralization

        with open(args.best_params_file) as f:
            hpo_results = json.load(f)

        best_params = hpo_results["best_params"]
        strengths = [float(s) for s in args.strengths.split(",")]

        results = evaluate_on_neutralization(
            best_params,
            model_type=args.model_type,
            n_estimators=args.n_estimators,
            neutralization_strengths=strengths,
            downsampled_path="v5.2/downsampled_full.parquet",
            full_path="v5.2/full.parquet",
            benchmark_path="v5.2/full_benchmark_models.parquet",
        )

        # Save results
        output_file = args.best_params_file.parent / f"{args.best_params_file.stem}_neutralization.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {output_file}")

    elif args.command == "holdout":
        from agents.code.modeling.utils.hpo_v3 import evaluate_on_holdout

        with open(args.best_params_file) as f:
            hpo_results = json.load(f)

        best_params = hpo_results["best_params"]

        results = evaluate_on_holdout(
            best_params,
            args.neutralization_strength,
            model_type=args.model_type,
            n_estimators=args.n_estimators,
            full_path="v5.2/full.parquet",
            benchmark_path="v5.2/full_benchmark_models.parquet",
        )

        # Save results
        output_file = args.best_params_file.parent / f"{args.best_params_file.stem}_holdout.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
