from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
EXPORT_SCRIPT = REPO_ROOT / "numerai/agents/experiments/nn_ender20_latest_residualized/export_strict_gbt_pickle.py"
SOURCE_EXAMPLE_SCRIPTS = Path("/Users/joakim/Documents/Projects/Numerai/example-scripts")
SOURCE_V52 = SOURCE_EXAMPLE_SCRIPTS / "numerai/v5.2"
LIVE_PATH = Path("/Users/joakim/Documents/Projects/Numerai/Classic/ender/v5.2/live.parquet")
LIVE_BENCH_PATH = Path("/Users/joakim/Documents/Projects/Numerai/Classic/ender/v5.2/live_benchmark_models.parquet")
OUTPUT_DIR = REPO_ROOT / "numerai/agents/experiments/nn_ender20_latest_residualized/models/season2026_supply"
SUMMARY_PATH = REPO_ROOT / "numerai/agents/experiments/nn_ender20_latest_residualized/results/season2026_xgb_supply_batch_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and smoke-test unique live XGB supply variants.")
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter with xgboost and cloudpickle available.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=SUMMARY_PATH,
    )
    return parser.parse_args()


def variant_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "xgb_strict_resid006_medfaith64_phuber_lam005_nb002_live_seed1441_py312_20260322",
            "feature_set": "medium+faith2:64",
            "residual_scale": 0.006,
            "sample_seed": 1441,
            "max_rows_per_era": 800,
        },
        {
            "name": "xgb_strict_resid008_medfaith64_phuber_lam005_nb002_live_seed1553_py312_20260322",
            "feature_set": "medium+faith2:64",
            "residual_scale": 0.008,
            "sample_seed": 1553,
            "max_rows_per_era": 800,
        },
        {
            "name": "xgb_strict_resid008_medium_phuber_lam005_nb002_live_seed1667_py312_20260322",
            "feature_set": "medium",
            "residual_scale": 0.008,
            "sample_seed": 1667,
            "max_rows_per_era": 800,
        },
    ]


def run_export(python_bin: str, spec: dict[str, Any]) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifacts_dir = OUTPUT_DIR / f"{spec['name']}_artifacts"
    output_pkl = OUTPUT_DIR / f"{spec['name']}.pkl"
    cmd = [
        python_bin,
        str(EXPORT_SCRIPT),
        "--mode",
        "all",
        "--train-profile",
        "live",
        "--feature-set",
        str(spec["feature_set"]),
        "--target-col",
        "target_ender_20",
        "--benchmark-col",
        "v52_lgbm_ender20",
        "--residual-scale",
        str(spec["residual_scale"]),
        "--lambda-blend",
        "0.05",
        "--neutralize-benchmark",
        "0.02",
        "--xgb-variant",
        "d5lr3e2",
        "--xgb-objective",
        "reg:pseudohubererror",
        "--full-data-path",
        str(SOURCE_V52 / "full.parquet"),
        "--full-benchmark-path",
        str(SOURCE_V52 / "full_benchmark_models.parquet"),
        "--features-path",
        str(SOURCE_V52 / "features.json"),
        "--max-rows-per-era",
        str(spec["max_rows_per_era"]),
        "--sample-seed",
        str(spec["sample_seed"]),
        "--artifacts-dir",
        str(artifacts_dir.relative_to(REPO_ROOT)),
        "--output-pkl",
        str(output_pkl.relative_to(REPO_ROOT)),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return artifacts_dir, output_pkl


def smoke_test(path: Path) -> dict[str, Any]:
    live = pd.read_parquet(LIVE_PATH)
    bench = pd.read_parquet(LIVE_BENCH_PATH)
    if "id" in live.columns and "id" in bench.columns:
        bench = bench.set_index("id").reindex(live["id"]).reset_index(drop=True)

    with path.open("rb") as fh:
        predict = cloudpickle.load(fh)
    pred = predict(live, bench)
    if isinstance(pred, pd.DataFrame):
        if "prediction" in pred.columns:
            arr = pred["prediction"].to_numpy(dtype=float)
        else:
            arr = pred.iloc[:, 0].to_numpy(dtype=float)
    elif isinstance(pred, pd.Series):
        arr = pred.to_numpy(dtype=float)
    else:
        arr = np.asarray(pred, dtype=float).reshape(-1)

    return {
        "len": int(len(arr)),
        "nan": int(np.isnan(arr).sum()),
        "std": float(np.nanstd(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
    }


def load_metadata(artifacts_dir: Path) -> dict[str, Any]:
    return json.loads((artifacts_dir / "metadata.json").read_text())


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    for spec in variant_specs():
        artifacts_dir, output_pkl = run_export(args.python_bin, spec)
        records.append(
            {
                "name": spec["name"],
                "artifact_path": str(output_pkl),
                "artifacts_dir": str(artifacts_dir),
                "spec": spec,
                "metadata": load_metadata(artifacts_dir),
                "smoke": smoke_test(output_pkl),
            }
        )

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(args.summary_path)


if __name__ == "__main__":
    main()
