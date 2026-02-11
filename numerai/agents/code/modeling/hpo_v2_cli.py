"""CLI for running improved HPO with era-stride CV and WandB."""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run HPO with era-stride CV, early stopping, and WandB logging"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of HPO trials (default: 20)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optuna study name (auto-generated if not provided)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="SQLite database path (auto-generated if not provided)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["bmc_sharpe", "bmc_mean", "corr_sharpe", "corr_mean"],
        default="bmc_sharpe",
        help="Metric to optimize (default: bmc_sharpe)",
    )
    # Era-stride CV settings
    parser.add_argument(
        "--n-strides",
        type=int,
        default=3,
        help="Number of era strides for CV (default: 3)",
    )
    parser.add_argument(
        "--folds-per-stride",
        type=int,
        default=2,
        help="Walk-forward folds per stride (default: 2)",
    )
    parser.add_argument(
        "--embargo",
        type=int,
        default=16,
        help="Era embargo for CV (default: 16)",
    )
    parser.add_argument(
        "--min-train-eras",
        type=int,
        default=20,
        help="Minimum training eras per fold (default: 20)",
    )
    # Early stopping settings
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=5000,
        help="Fixed n_estimators with early stopping (default: 5000)",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=100,
        help="Early stopping rounds (default: 100)",
    )
    # WandB settings
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="numerai-hpo",
        help="WandB project name (default: numerai-hpo, use 'none' to disable)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity/team",
    )
    # Other settings
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Timeout in seconds (optional)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1, sequential)",
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create a new config file with best params after HPO",
    )

    args = parser.parse_args()

    from agents.code.modeling.utils.hpo_v2 import create_hpo_config_v2, run_hpo_v2

    # Handle WandB disable
    wandb_project = args.wandb_project if args.wandb_project.lower() != "none" else None

    study, best_params = run_hpo_v2(
        config_path=args.config,
        n_trials=args.n_trials,
        study_name=args.study_name,
        db_path=args.db_path,
        metric=args.metric,
        n_strides=args.n_strides,
        folds_per_stride=args.folds_per_stride,
        embargo=args.embargo,
        min_train_eras=args.min_train_eras,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping,
        wandb_project=wandb_project,
        wandb_entity=args.wandb_entity,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
    )

    if args.create_config:
        recommended = study.best_trial.user_attrs.get("avg_n_estimators")
        create_hpo_config_v2(args.config, best_params, recommended_n_estimators=recommended)

    print(f"\nCompleted {len(study.trials)} trials")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best {args.metric}: {study.best_value:.6f}")


if __name__ == "__main__":
    main()
