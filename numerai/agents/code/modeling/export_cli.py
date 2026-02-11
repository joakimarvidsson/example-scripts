"""CLI for training and exporting Numerai models."""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train and export Numerai models with cloudpickle")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for model (auto-generated if not provided)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the exported model works after training",
    )

    args = parser.parse_args()

    from agents.code.modeling.utils.model_export import train_and_export, verify_export

    model_path, predictions_path = train_and_export(
        config_path=args.config,
        output_path=args.output,
    )

    print(f"\nModel exported to: {model_path}")
    print(f"Predictions saved to: {predictions_path}")

    if args.verify:
        import pandas as pd
        # Load predictions to get sample features for verification
        predictions = pd.read_parquet(predictions_path)
        # Load full data for verification
        from numerapi import NumerAPI
        from agents.code.modeling.utils.config import load_config
        from agents.code.modeling.utils.data import load_features, load_full_data

        config = load_config(args.config)
        data_config = config.get("data", {})
        data_version = data_config.get("data_version", "v5.2")
        feature_set = data_config.get("feature_set", "small")
        target_col = data_config.get("target_col", "target")
        era_col = data_config.get("era_col", "era")
        id_col = data_config.get("id_col", "id")
        full_data_path = data_config.get("full_data_path")

        napi = NumerAPI()
        features = load_features(napi, data_version, feature_set)
        full = load_full_data(
            napi, data_version, features, era_col, target_col, id_col,
            full_data_path=full_data_path,
        )

        sample = full.head(100)
        verify_export(model_path, sample)


if __name__ == "__main__":
    main()
