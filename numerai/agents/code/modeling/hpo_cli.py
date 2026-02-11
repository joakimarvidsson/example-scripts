"""CLI for running Optuna HPO."""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run Optuna HPO for Numerai models")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of HPO trials (default: 50)",
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

    from agents.code.modeling.utils.hpo import create_hpo_config, run_hpo

    study, best_params = run_hpo(
        config_path=args.config,
        n_trials=args.n_trials,
        study_name=args.study_name,
        db_path=args.db_path,
        metric=args.metric,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
    )

    if args.create_config:
        create_hpo_config(args.config, best_params)

    print(f"\nCompleted {len(study.trials)} trials")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best {args.metric}: {study.best_value:.6f}")


if __name__ == "__main__":
    main()
