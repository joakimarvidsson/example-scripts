"""
Upload trained model predictions to Numerai.
Usage: python -m agents.code.upload_model --predictions pred.parquet --model-name mymodel
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd
import numerapi


def upload_predictions(predictions_path: str, model_name: str, public_id: str, secret_key: str):
    """Upload predictions to Numerai tournament."""
    napi = numerapi.NumerAPI(public_id=public_id, secret_key=secret_key)

    # Get model ID
    models = napi.get_models()
    if model_name not in models:
        print(f"Model '{model_name}' not found.")
        print(f"Available models (first 20): {list(models.keys())[:20]}")
        return False

    model_id = models[model_name]

    # Load predictions and format for submission
    df = pd.read_parquet(predictions_path)
    print(f"Loaded predictions: {len(df)} rows")

    # Numerai expects 'id' and 'prediction' columns
    if 'id' not in df.columns:
        print("Warning: 'id' column not found, using index")
        df = df.reset_index()
        if 'index' in df.columns:
            df = df.rename(columns={'index': 'id'})

    submission = df[['id', 'prediction']].copy()
    print(f"Submission shape: {submission.shape}")

    # Save to temp CSV and upload
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
        submission.to_csv(f.name, index=False)
        temp_path = f.name

    try:
        print(f"Uploading to model '{model_name}' (id: {model_id})...")
        napi.upload_predictions(temp_path, model_id=model_id)
        print(f"Successfully uploaded predictions to {model_name}")
        return True
    finally:
        os.unlink(temp_path)


def main():
    parser = argparse.ArgumentParser(description="Upload predictions to Numerai")
    parser.add_argument("--predictions", required=True, help="Prediction parquet file")
    parser.add_argument("--model-name", required=True, help="Numerai model name")
    parser.add_argument("--public-id",
                        default=os.environ.get("NUMERAI_PUBLIC_ID", ""),
                        help="Numerai public ID (or set NUMERAI_PUBLIC_ID env var)")
    parser.add_argument("--secret-key",
                        default=os.environ.get("NUMERAI_SECRET_KEY", ""),
                        help="Numerai secret key (or set NUMERAI_SECRET_KEY env var)")
    args = parser.parse_args()

    if not args.public_id or not args.secret_key:
        print("Error: Numerai credentials required. Set NUMERAI_PUBLIC_ID and NUMERAI_SECRET_KEY env vars.")
        return

    upload_predictions(args.predictions, args.model_name, args.public_id, args.secret_key)


if __name__ == "__main__":
    main()
