"""Export ensemble models as a self-contained pickle for Numerai compute.

Numerai's compute environment does NOT have the 'agents' package, so pickles
must only reference standard libraries (lightgbm, xgboost, catboost, numpy, pandas).

This script loads trained ensemble model pickles, extracts the raw underlying
models (e.g., lightgbm.LGBMRegressor), and re-exports as a self-contained pickle.

Usage:
    python -m agents.code.modeling.export_ensemble_cli \
        --ensemble-dir agents/baselines/models_live \
        --output agents/baselines/models_live/numerai_ensemble_model.pkl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd


class NumeraiEnsembleModel:
    """Self-contained ensemble model for Numerai live predictions.

    This class contains ONLY raw model objects (lightgbm.LGBMRegressor, etc.)
    and numpy/pandas operations. It does NOT depend on the 'agents' package.

    Numerai calls model.predict(live_features) where live_features is a DataFrame.
    """

    def __init__(
        self,
        models: list,
        feature_cols: list[str],
        weights: list[float] | None = None,
    ):
        self.models = models
        self.feature_cols = feature_cols
        self.weights = weights or [1.0 / len(models)] * len(models)

    def predict(self, live_features: pd.DataFrame) -> pd.DataFrame:
        """Generate ensemble predictions for Numerai live data."""
        features = live_features[self.feature_cols]

        predictions = np.zeros(len(features))
        for model, weight in zip(self.models, self.weights):
            preds = model.predict(features)
            predictions += weight * preds

        return pd.DataFrame({"prediction": predictions})


def _extract_raw_model(model):
    """Extract the raw underlying model from a custom wrapper.

    Our custom wrappers (LGBMRegressor, XGBRegressor, CatBoostRegressor)
    store the raw model in self._model. This function extracts it so the
    pickle only depends on the ML library, not our agents package.
    """
    if hasattr(model, "_model"):
        raw = model._model
        print(f"  Extracted raw {type(raw).__module__}.{type(raw).__name__}")
        return raw

    # Already a raw model (lightgbm.LGBMRegressor, etc.)
    print(f"  Already raw: {type(model).__module__}.{type(model).__name__}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Export ensemble as self-contained pickle for Numerai"
    )
    parser.add_argument(
        "--ensemble-dir", type=Path, required=True,
        help="Directory containing model pickles and ensemble_metadata.json",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output pickle path (default: <ensemble-dir>/numerai_ensemble_model.pkl)",
    )
    parser.add_argument(
        "--feature-set", default="all",
        help="Feature set to use (default: all)",
    )
    parser.add_argument(
        "--data-version", default="v5.2",
        help="Data version (default: v5.2)",
    )

    args = parser.parse_args()

    ensemble_dir = args.ensemble_dir
    meta_file = ensemble_dir / "ensemble_metadata.json"

    if not meta_file.exists():
        raise FileNotFoundError(f"ensemble_metadata.json not found in {ensemble_dir}")

    with open(meta_file) as f:
        metadata = json.load(f)

    print(f"Loading ensemble from {ensemble_dir}")
    print(f"  Models: {metadata['n_models']}")

    # Load feature columns
    from numerapi import NumerAPI
    from agents.code.modeling.utils.data import load_features

    napi = NumerAPI()
    feature_cols = load_features(napi, args.data_version, args.feature_set)
    print(f"  Features: {len(feature_cols)}")

    # Load and extract raw models
    raw_models = []
    for i, model_info in enumerate(metadata["models"]):
        model_path = Path(model_info["model_path"])
        # Resolve relative paths: try as-is, then relative to ensemble_dir
        if not model_path.exists():
            model_path = ensemble_dir / model_path.name
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_info['model_path']} (also tried {model_path})")
        print(f"\nModel {i + 1}/{metadata['n_models']}: {model_path}")

        with open(model_path, "rb") as f:
            model = cloudpickle.load(f)

        raw_model = _extract_raw_model(model)
        raw_models.append(raw_model)

    # Build self-contained ensemble
    ensemble = NumeraiEnsembleModel(
        models=raw_models,
        feature_cols=feature_cols,
    )

    # Export
    output_path = args.output or (ensemble_dir / "numerai_ensemble_model.pkl")
    print(f"\nExporting self-contained ensemble to {output_path}")

    with open(output_path, "wb") as f:
        cloudpickle.dump(ensemble, f)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Models: {len(raw_models)}")
    print(f"  Features: {len(feature_cols)}")

    # Verify the pickle can be loaded without agents module
    print(f"\nVerifying pickle loads correctly...")
    with open(output_path, "rb") as f:
        loaded = cloudpickle.load(f)
    print(f"  Loaded: {type(loaded).__name__} with {len(loaded.models)} models")

    # Check that no model references agents module
    for i, model in enumerate(loaded.models):
        module = type(model).__module__
        if "agents" in module:
            print(f"  WARNING: Model {i} still references agents module: {module}")
        else:
            print(f"  Model {i}: {module}.{type(model).__name__} (OK)")

    print(f"\nDone! Upload {output_path} to Numerai.")


if __name__ == "__main__":
    main()
