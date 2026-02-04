"""
Ensemble builder for combining multiple model predictions.
Usage: python -m agents.code.modeling.ensemble --predictions pred1.parquet pred2.parquet --output ensemble.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def rank_average(predictions_list: list[pd.DataFrame], era_col: str = "era") -> pd.DataFrame:
    """Combine predictions via per-era rank averaging."""
    base = predictions_list[0].copy()
    all_preds = []

    for i, df in enumerate(predictions_list):
        pred_col = f"pred_{i}"
        base[pred_col] = df["prediction"].values
        all_preds.append(pred_col)

    def era_rank_avg(group):
        ranks = []
        for col in all_preds:
            ranks.append(group[col].rank(pct=True))
        return pd.Series(np.mean(ranks, axis=0), index=group.index)

    base["prediction"] = base.groupby(era_col, group_keys=False).apply(era_rank_avg)
    base = base.drop(columns=all_preds)

    return base


def simple_average(predictions_list: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine predictions via simple averaging."""
    base = predictions_list[0].copy()
    pred_sum = base["prediction"].copy()

    for df in predictions_list[1:]:
        pred_sum += df["prediction"].values

    base["prediction"] = pred_sum / len(predictions_list)
    return base


def main():
    parser = argparse.ArgumentParser(description="Ensemble model predictions")
    parser.add_argument("--predictions", nargs="+", required=True, help="Prediction parquet files")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument("--method", default="rank_average", choices=["rank_average", "simple_average"])
    parser.add_argument("--era-col", default="era", help="Era column name")
    args = parser.parse_args()

    predictions_list = [pd.read_parquet(p) for p in args.predictions]
    print(f"Loaded {len(predictions_list)} prediction files")

    if args.method == "rank_average":
        result = rank_average(predictions_list, args.era_col)
    else:
        result = simple_average(predictions_list)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    print(f"Saved ensemble predictions to {args.output}")


if __name__ == "__main__":
    main()
