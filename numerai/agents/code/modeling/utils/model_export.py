"""Cloudpickle model export for Numerai live predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import pandas as pd


class NumeraiModelWrapper:
    """Wrapper class for Numerai model predictions.

    This wrapper is designed to be pickled with cloudpickle and uploaded
    to Numerai for live predictions. It handles:
    - Feature selection
    - Preprocessing (missing value handling)
    - Prediction
    """

    def __init__(
        self,
        model: Any,
        feature_cols: list[str],
        preprocessing: dict | None = None,
        metadata: dict | None = None,
    ):
        """Initialize the wrapper.

        Args:
            model: Trained model with a predict method
            feature_cols: List of feature column names
            preprocessing: Preprocessing config (missing_value, nan_missing_all_twos)
            metadata: Additional metadata (model_type, params, etc.)
        """
        self.model = model
        self.feature_cols = feature_cols
        self.preprocessing = preprocessing or {}
        self.metadata = metadata or {}

    def predict(self, live_features: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for live features.

        Args:
            live_features: DataFrame with live features from Numerai

        Returns:
            DataFrame with 'prediction' column
        """
        # Handle missing value preprocessing
        features = live_features[self.feature_cols].copy()

        nan_missing_all_twos = self.preprocessing.get("nan_missing_all_twos", False)
        missing_value = self.preprocessing.get("missing_value", 2.0)

        if nan_missing_all_twos:
            # Replace rows where all features equal missing_value with NaN
            mask = (features == missing_value).all(axis=1)
            features.loc[mask] = np.nan

        # Generate predictions
        predictions = self.model.predict(features)

        # Return as DataFrame
        return pd.DataFrame({"prediction": predictions})


def export_model(
    model: Any,
    feature_cols: list[str],
    output_path: Path | str,
    preprocessing: dict | None = None,
    metadata: dict | None = None,
) -> Path:
    """Export a trained model using cloudpickle.

    Args:
        model: Trained model with a predict method
        feature_cols: List of feature column names
        output_path: Path to save the pickled model
        preprocessing: Preprocessing config
        metadata: Additional metadata to include

    Returns:
        Path to the exported model file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = NumeraiModelWrapper(
        model=model,
        feature_cols=feature_cols,
        preprocessing=preprocessing,
        metadata=metadata,
    )

    with open(output_path, "wb") as f:
        cloudpickle.dump(wrapper, f)

    print(f"Exported model to {output_path}")

    # Also save metadata as JSON for reference
    metadata_path = output_path.with_suffix(".json")
    meta_to_save = {
        "n_features": len(feature_cols),
        "preprocessing": preprocessing,
        **(metadata or {}),
    }
    with open(metadata_path, "w") as f:
        json.dump(meta_to_save, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    return output_path


def load_model(model_path: Path | str) -> NumeraiModelWrapper:
    """Load a cloudpickled model.

    Args:
        model_path: Path to the pickled model

    Returns:
        NumeraiModelWrapper instance
    """
    with open(model_path, "rb") as f:
        return cloudpickle.load(f)


def train_and_export(
    config_path: Path,
    output_path: Path | str | None = None,
    train_on_all_data: bool = True,
) -> tuple[Path, Path]:
    """Train a model and export it for Numerai.

    Args:
        config_path: Path to config file
        output_path: Output path for model (auto-generated if None)
        train_on_all_data: If True, train on all data (not CV)

    Returns:
        Tuple of (model_path, predictions_path)
    """
    from numerapi import NumerAPI

    from agents.code.modeling.utils.config import load_config
    from agents.code.modeling.utils.constants import DEFAULT_BENCHMARK_MODEL
    from agents.code.modeling.utils.data import (
        apply_missing_all_twos_as_nan,
        attach_benchmark_models,
        load_features,
        load_full_data,
    )
    from agents.code.modeling.utils.model_data import (
        build_model_data_loader,
        build_x_cols,
        normalize_x_groups,
    )
    from agents.code.modeling.utils.model_factory import build_model
    from agents.code.modeling.utils.pipeline import resolve_model_config, resolve_output_locations

    config = load_config(config_path)
    data_config = config.get("data", {})
    preprocessing_config = config.get("preprocessing", {})
    model_config = config.get("model", {})

    # Extract config values
    data_version = data_config.get("data_version", "v5.2")
    feature_set = data_config.get("feature_set", "small")
    target_col = data_config.get("target_col", "target")
    era_col = data_config.get("era_col", "era")
    id_col = data_config.get("id_col", "id")
    full_data_path = data_config.get("full_data_path")
    benchmark_data_path = data_config.get("benchmark_data_path")

    nan_missing_all_twos = preprocessing_config.get("nan_missing_all_twos", False)
    missing_value = preprocessing_config.get("missing_value", 2.0)

    # Resolve paths
    output_dir, _, _, predictions_dir = resolve_output_locations(config, None)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    napi = NumerAPI()
    features = load_features(napi, data_version, feature_set)
    full = load_full_data(
        napi,
        data_version,
        features,
        era_col,
        target_col,
        id_col,
        full_data_path=full_data_path,
    )

    if nan_missing_all_twos:
        full = apply_missing_all_twos_as_nan(full, features, era_col, missing_value)

    # Handle x_groups for benchmark models
    raw_x_groups = model_config.get("x_groups") or model_config.get("data_needed")
    x_groups = normalize_x_groups(raw_x_groups)
    benchmark_cols: list[str] = []

    if "benchmark_models" in x_groups:
        if not id_col:
            raise ValueError("id_col is required to attach benchmark models.")
        full, benchmark_cols = attach_benchmark_models(
            full,
            napi,
            data_version,
            benchmark_data_path,
            era_col,
            id_col,
        )

    x_cols = build_x_cols(
        x_groups=x_groups,
        features=features,
        benchmark_cols=benchmark_cols,
        era_col=era_col,
        id_col=id_col,
        baseline_col=None,
    )

    # Build and train model
    model_type, model_params = resolve_model_config(model_config)
    model = build_model(model_type, model_params, model_config, feature_cols=features)

    print(f"Training {model_type} on {len(full)} samples...")
    X = full[x_cols]
    y = full[target_col]
    model.fit(X, y)
    print("Training complete")

    # Export model
    if output_path is None:
        output_path = models_dir / f"{config_path.stem}.pkl"
    else:
        output_path = Path(output_path)

    metadata = {
        "model_type": model_type,
        "params": model_params,
        "data_version": data_version,
        "feature_set": feature_set,
        "config_path": str(config_path),
        "training_samples": len(full),
    }

    model_path = export_model(
        model=model,
        feature_cols=x_cols,
        output_path=output_path,
        preprocessing=preprocessing_config,
        metadata=metadata,
    )

    # Generate predictions on full data for verification
    predictions = pd.DataFrame({
        id_col: full[id_col].values if id_col in full.columns else range(len(full)),
        era_col: full[era_col].values,
        "prediction": model.predict(X),
    })

    predictions_path = predictions_dir / f"{config_path.stem}_full.parquet"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(predictions_path, index=False)
    print(f"Saved full predictions to {predictions_path}")

    return model_path, predictions_path


def verify_export(model_path: Path | str, sample_features: pd.DataFrame) -> bool:
    """Verify an exported model works correctly.

    Args:
        model_path: Path to the pickled model
        sample_features: Sample features to test prediction

    Returns:
        True if verification passes
    """
    print(f"Verifying model: {model_path}")

    wrapper = load_model(model_path)
    print(f"  Loaded wrapper with {len(wrapper.feature_cols)} features")

    # Check features are available
    missing = set(wrapper.feature_cols) - set(sample_features.columns)
    if missing:
        print(f"  WARNING: Missing features: {list(missing)[:5]}...")
        return False

    # Test prediction
    predictions = wrapper.predict(sample_features)
    print(f"  Generated {len(predictions)} predictions")
    print(f"  Prediction range: [{predictions['prediction'].min():.4f}, {predictions['prediction'].max():.4f}]")
    print(f"  Prediction mean: {predictions['prediction'].mean():.4f}")

    return True
