"""CLI for HPO v4: Enhanced pipeline with payout optimization and ensembling.

Usage:
    # Phase 1: HPO (same as v3)
    python -m agents.code.modeling.hpo_v3_cli hpo --n-trials 20

    # Phase 3: Ensemble training
    python -m agents.code.modeling.hpo_v4_cli ensemble --best-params-file results.json

    # Phase 4: Holdout evaluation
    python -m agents.code.modeling.hpo_v4_cli holdout --best-params-file results.json --neut-config-file neut.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Default data directory
DEFAULT_DATA_DIR = "/Users/joakim/Documents/Projects/Numerai/data"


def main():
    parser = argparse.ArgumentParser(
        description="HPO v4: Enhanced pipeline with payout optimization"
    )
    parser.add_argument("--data-dir", type=Path, default=Path(DEFAULT_DATA_DIR),
                       help="Base data directory containing v5.2/")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ensemble training
    ensemble_parser = subparsers.add_parser("ensemble", help="Train ensemble models")
    ensemble_parser.add_argument("--best-params-file", type=Path, required=True)
    ensemble_parser.add_argument("--n-estimators", type=int, default=2000)
    ensemble_parser.add_argument("--n-offsets", type=int, default=4)
    ensemble_parser.add_argument("--full-model-subsample", type=float, default=0.25,
                                help="Subsample fraction for full model (0 to skip, 0.25 default)")
    ensemble_parser.add_argument("--output-dir", type=Path, default=Path("numerai/agents/baselines/models"))

    # Holdout evaluation
    holdout_parser = subparsers.add_parser("holdout", help="Run holdout evaluation (FINAL)")
    holdout_parser.add_argument("--best-params-file", type=Path, required=True)
    holdout_parser.add_argument("--neut-config-file", type=Path, required=True)
    holdout_parser.add_argument("--ensemble-dir", type=Path, default=Path("numerai/agents/baselines/models"))

    # Full pipeline (ensemble + holdout)
    full_parser = subparsers.add_parser("full", help="Run ensemble training + holdout evaluation")
    full_parser.add_argument("--best-params-file", type=Path, required=True,
                            help="HPO best params from v3")
    full_parser.add_argument("--neut-config-file", type=Path, required=True,
                            help="Neutralization config from v4 neutralization phase")
    full_parser.add_argument("--n-estimators", type=int, default=2000)
    full_parser.add_argument("--full-model-subsample", type=float, default=0.25,
                            help="Subsample fraction for full model (0 to skip, 0.25 default)")
    full_parser.add_argument("--output-dir", type=Path, default=Path("numerai/agents/baselines"))

    args = parser.parse_args()
    data_dir = str(args.data_dir)

    if args.command == "ensemble":
        from agents.code.modeling.utils.hpo_v4 import train_ensemble_models

        with open(args.best_params_file) as f:
            hpo_results = json.load(f)

        best_params = hpo_results["best_params"]

        # Use memory-efficient mode with full data
        models = train_ensemble_models(
            best_params,
            n_estimators=args.n_estimators,
            n_offsets=args.n_offsets,
            data_dir=data_dir,
            memory_efficient=True,
            save_dir=args.output_dir,
            full_model_subsample=args.full_model_subsample,
        )

        # Save metadata (models already saved individually)
        meta_file = args.output_dir / "ensemble_metadata.json"
        with open(meta_file, "w") as f:
            json.dump({
                "n_models": len(models),
                "models": models,
                "best_params": best_params,
                "n_estimators": args.n_estimators,
            }, f, indent=2)
        print(f"\nSaved metadata: {meta_file}")

    elif args.command == "holdout":
        from agents.code.modeling.utils.hpo_v4 import (
            evaluate_on_holdout_v4,
            train_ensemble_models,
        )

        with open(args.best_params_file) as f:
            hpo_results = json.load(f)

        with open(args.neut_config_file) as f:
            neut_results = json.load(f)

        best_params = hpo_results["best_params"]
        neut_config = neut_results["best_config"]

        # Load ensemble from metadata or train
        meta_file = args.ensemble_dir / "ensemble_metadata.json"
        if meta_file.exists():
            print(f"Loading ensemble metadata from {meta_file}")
            with open(meta_file) as f:
                ensemble_meta = json.load(f)
            models = ensemble_meta["models"]
        else:
            print("Training ensemble models...")
            models = train_ensemble_models(
                best_params,
                n_estimators=2000,
                data_dir=data_dir,
                memory_efficient=True,
                save_dir=args.ensemble_dir,
            )

        results = evaluate_on_holdout_v4(
            models,
            neut_config,
            data_dir=data_dir,
            memory_efficient=True,
        )

        # Save results
        output_file = args.best_params_file.parent / f"{args.best_params_file.stem}_holdout_v4.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {output_file}")

    elif args.command == "full":
        from agents.code.modeling.utils.hpo_v4 import (
            train_ensemble_models,
            evaluate_on_holdout_v4,
        )

        with open(args.best_params_file) as f:
            hpo_results = json.load(f)

        with open(args.neut_config_file) as f:
            neut_results = json.load(f)

        best_params = hpo_results["best_params"]
        neut_config = neut_results["best_config"]
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 3: Ensemble training (memory-efficient)
        print("\n" + "="*70)
        print("PHASE 3: ENSEMBLE TRAINING")
        print("="*70)

        models_dir = output_dir / "models"
        models = train_ensemble_models(
            best_params,
            n_estimators=args.n_estimators,
            data_dir=data_dir,
            memory_efficient=True,
            save_dir=models_dir,
            full_model_subsample=args.full_model_subsample,
        )

        # Save metadata
        meta_file = models_dir / "ensemble_metadata.json"
        with open(meta_file, "w") as f:
            json.dump({
                "n_models": len(models),
                "models": models,
                "best_params": best_params,
                "n_estimators": args.n_estimators,
            }, f, indent=2)
        print(f"Saved: {meta_file}")

        # Phase 4: Holdout evaluation
        print("\n" + "="*70)
        print("PHASE 4: HOLDOUT EVALUATION")
        print("="*70)

        holdout_results = evaluate_on_holdout_v4(
            models,
            neut_config,
            data_dir=data_dir,
            memory_efficient=True,
        )

        holdout_file = output_dir / "holdout_results_v4.json"
        with open(holdout_file, "w") as f:
            json.dump(holdout_results, f, indent=2)
        print(f"Saved: {holdout_file}")

        # Summary
        print("\n" + "="*70)
        print("V4 PIPELINE COMPLETE")
        print("="*70)
        print(f"Ensemble models: {len(models)}")
        print(f"")
        print(f"Holdout Results:")
        print(f"  CORR Mean:   {holdout_results['corr_mean']:.6f}  Sharpe: {holdout_results['corr_sharpe']:.4f}")
        print(f"  BMC Mean:    {holdout_results['bmc_mean']:.6f}  Sharpe: {holdout_results['bmc_sharpe']:.4f}")
        print(f"  Payout Mean: {holdout_results['payout_mean']:.6f}  Sharpe: {holdout_results['payout_sharpe']:.4f}")


if __name__ == "__main__":
    main()
