from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.code.modeling.utils.target_transforms import subtract_scaled_invnorm_column


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export strict residualized GBT as Numerai-compatible cloudpickle."
    )
    parser.add_argument(
        "--mode",
        choices=["train-artifacts", "build-pickle", "all"],
        default="all",
    )
    parser.add_argument(
        "--train-profile",
        choices=["live", "diagnostics", "custom"],
        default="live",
        help=(
            "live: all available eras, diagnostics: cap max era to 560 unless overridden, "
            "custom: use explicit min/max."
        ),
    )
    parser.add_argument("--min-train-era", type=int, default=None)
    parser.add_argument("--max-train-era", type=int, default=None)
    parser.add_argument(
        "--max-rows-per-era",
        type=int,
        default=800,
        help="Row cap per era. 0 means keep all rows for each era.",
    )
    parser.add_argument("--sample-seed", type=int, default=1337)
    parser.add_argument("--feature-set", default="medium")
    parser.add_argument("--target-col", default="target_ender_20")
    parser.add_argument("--benchmark-col", default="v52_lgbm_ender20")
    parser.add_argument("--residual-scale", type=float, default=0.008)
    parser.add_argument(
        "--lambda-blend",
        type=float,
        default=1.0,
        help="Final prediction = benchmark_rank + lambda * (raw_rank - benchmark_rank).",
    )
    parser.add_argument(
        "--neutralize-benchmark",
        type=float,
        default=0.0,
        help=(
            "Per-era neutralization strength vs benchmark rank after lambda blend. "
            "Set to tuned value (e.g. 0.07 for payout-optimized config)."
        ),
    )
    parser.add_argument(
        "--xgb-variant",
        choices=["d4lr5e2", "d5lr3e2", "d3lr7e2", "d6lr2e2"],
        default="d5lr3e2",
        help="XGBoost hyperparameter preset matching strict walk-forward experiments.",
    )
    parser.add_argument(
        "--xgb-objective",
        default="reg:squarederror",
        help="XGBoost objective used for training, for example reg:squarederror or reg:pseudohubererror.",
    )
    parser.add_argument(
        "--full-data-path",
        type=Path,
        default=Path("numerai/v5.2/full.parquet"),
    )
    parser.add_argument(
        "--full-benchmark-path",
        type=Path,
        default=Path("numerai/v5.2/full_benchmark_models.parquet"),
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("numerai/v5.2/features.json"),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(
            "numerai/agents/experiments/nn_ender20_latest_residualized/models/xgb_strict_resid008_live_artifacts"
        ),
    )
    parser.add_argument(
        "--output-pkl",
        type=Path,
        default=Path(
            "numerai/agents/experiments/nn_ender20_latest_residualized/models/xgb_strict_resid008_live.pkl"
        ),
    )
    return parser.parse_args()


def _resolve_feature_set_name(name: str) -> str:
    aliases = {"faith2": "faith"}
    return aliases.get(name, name)


def _load_feature_list(features_path: Path, feature_set: str) -> list[str]:
    payload = json.loads(features_path.read_text())
    sets = payload.get("feature_sets", {})
    tokens = [t.strip() for t in str(feature_set).split("+") if t.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        base = token
        cap: int | None = None
        if ":" in token:
            base, cap_str = token.split(":", 1)
            base = base.strip()
            cap = int(cap_str.strip())
        key = _resolve_feature_set_name(base)
        if key not in sets:
            raise KeyError(f"Feature set '{base}' not found in {features_path}")
        cols = list(sets[key])
        if cap is not None:
            cols = cols[:cap]
        for col in cols:
            if col not in seen:
                seen.add(col)
                out.append(col)
    return out


def _sample_rows_by_era(df: pd.DataFrame, max_rows_per_era: int, seed: int) -> pd.DataFrame:
    if max_rows_per_era <= 0:
        return df
    if df.empty:
        return df
    parts: list[pd.DataFrame] = []
    for _, group in df.groupby("era", sort=False):
        if len(group) <= max_rows_per_era:
            parts.append(group)
        else:
            parts.append(group.sample(n=max_rows_per_era, random_state=seed, replace=False))
    return pd.concat(parts, ignore_index=True)


def _load_training_frame(
    *,
    full_data_path: Path,
    full_benchmark_path: Path,
    feature_cols: list[str],
    target_col: str,
    benchmark_col: str,
    min_train_era: int | None,
    max_train_era: int | None,
) -> pd.DataFrame:
    cols = ["id", "era", target_col] + feature_cols
    frame = pd.read_parquet(full_data_path, columns=cols)
    frame["era"] = frame["era"].astype(str)
    era_int = frame["era"].astype(int)

    if min_train_era is not None:
        frame = frame.loc[era_int >= int(min_train_era)].copy()
        era_int = frame["era"].astype(int)
    if max_train_era is not None:
        frame = frame.loc[era_int <= int(max_train_era)].copy()

    bench = pd.read_parquet(full_benchmark_path, columns=["era", benchmark_col]).reset_index()
    bench = bench.rename(columns={"index": "id"})
    bench["era"] = bench["era"].astype(str)

    frame = frame.merge(
        bench[["id", "era", benchmark_col]],
        on=["id", "era"],
        how="inner",
        validate="one_to_one",
    )
    frame = frame.dropna(subset=[target_col, benchmark_col])
    return frame


def _collect_nodes(node: dict, out: dict[int, dict]) -> None:
    nid = int(node["nodeid"])
    out[nid] = node
    for child in node.get("children", []):
        _collect_nodes(child, out)


def _parse_tree_json(
    tree_json: dict,
    feature_name_to_idx: dict[str, int],
) -> dict[str, Any]:
    nodes: dict[int, dict] = {}
    _collect_nodes(tree_json, nodes)
    node_ids = sorted(nodes.keys())
    idx_of = {nid: i for i, nid in enumerate(node_ids)}

    n = len(node_ids)
    feature = np.full(n, -1, dtype=np.int32)
    threshold = np.zeros(n, dtype=np.float64)
    left = np.full(n, -1, dtype=np.int32)
    right = np.full(n, -1, dtype=np.int32)
    missing = np.full(n, -1, dtype=np.int32)
    value = np.zeros(n, dtype=np.float64)

    for nid in node_ids:
        node = nodes[nid]
        i = idx_of[nid]
        if "leaf" in node:
            value[i] = float(node["leaf"])
            continue

        split = str(node["split"])
        if split.startswith("f") and split[1:].isdigit():
            feat_idx = int(split[1:])
        else:
            if split not in feature_name_to_idx:
                raise KeyError(f"Unknown split feature '{split}'")
            feat_idx = int(feature_name_to_idx[split])

        feature[i] = feat_idx
        threshold[i] = float(node["split_condition"])
        left[i] = int(idx_of[int(node["yes"])])
        right[i] = int(idx_of[int(node["no"])])
        missing[i] = int(idx_of[int(node["missing"])])

    root_idx = int(idx_of[int(tree_json["nodeid"])])
    return {
        "root": root_idx,
        "feature": feature.tolist(),
        "threshold": threshold.tolist(),
        "left": left.tolist(),
        "right": right.tolist(),
        "missing": missing.tolist(),
        "value": value.tolist(),
    }


def train_and_save_artifacts(
    *,
    repo_root: Path,
    artifacts_dir: Path,
    feature_set: str,
    full_data_path: Path,
    full_benchmark_path: Path,
    features_path: Path,
    target_col: str,
    benchmark_col: str,
    residual_scale: float,
    lambda_blend: float,
    neutralize_benchmark: float,
    xgb_variant: str,
    xgb_objective: str,
    train_profile: str,
    min_train_era: int | None,
    max_train_era: int | None,
    max_rows_per_era: int,
    sample_seed: int,
) -> None:
    from xgboost import XGBRegressor

    resolved_min = min_train_era
    resolved_max = max_train_era
    if train_profile == "diagnostics" and resolved_max is None:
        resolved_max = 560

    feature_cols = _load_feature_list((repo_root / features_path).resolve(), feature_set=feature_set)
    frame = _load_training_frame(
        full_data_path=(repo_root / full_data_path).resolve(),
        full_benchmark_path=(repo_root / full_benchmark_path).resolve(),
        feature_cols=feature_cols,
        target_col=target_col,
        benchmark_col=benchmark_col,
        min_train_era=resolved_min,
        max_train_era=resolved_max,
    )
    if frame.empty:
        raise ValueError("No training rows after applying era filters.")

    era_values = sorted(frame["era"].unique(), key=lambda x: int(x))
    print(
        f"Loaded {len(frame):,} rows across {len(era_values)} eras "
        f"({era_values[0]}..{era_values[-1]}) before row sampling"
    )

    train = _sample_rows_by_era(frame, max_rows_per_era=max_rows_per_era, seed=sample_seed).copy()
    train = train.sort_values("era").reset_index(drop=True)
    print(f"Training rows after per-era cap: {len(train):,}")

    y = subtract_scaled_invnorm_column(
        train[target_col],
        train[[benchmark_col, "era"]],
        benchmark_col=benchmark_col,
        era_col="era",
        scale=float(residual_scale),
        per_era=True,
        use_rank=False,
        clip_eps=1e-6,
        center=True,
        center_per_era=False,
    )

    variant_params = {
        "d4lr5e2": {
            "n_estimators": 120,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.85,
            "colsample_bytree": 0.2,
            "reg_alpha": 0.1,
            "reg_lambda": 3.0,
            "min_child_weight": 30.0,
        },
        "d5lr3e2": {
            "n_estimators": 240,
            "learning_rate": 0.03,
            "max_depth": 5,
            "subsample": 0.90,
            "colsample_bytree": 0.35,
            "reg_alpha": 0.0,
            "reg_lambda": 2.0,
            "min_child_weight": 20.0,
        },
        "d3lr7e2": {
            "n_estimators": 180,
            "learning_rate": 0.07,
            "max_depth": 3,
            "subsample": 0.90,
            "colsample_bytree": 0.25,
            "reg_alpha": 0.2,
            "reg_lambda": 4.0,
            "min_child_weight": 40.0,
        },
        "d6lr2e2": {
            "n_estimators": 320,
            "learning_rate": 0.02,
            "max_depth": 6,
            "subsample": 0.80,
            "colsample_bytree": 0.40,
            "reg_alpha": 0.0,
            "reg_lambda": 2.0,
            "min_child_weight": 35.0,
        },
    }
    params = {
        **variant_params[str(xgb_variant)],
        "objective": str(xgb_objective),
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": int(sample_seed),
    }
    model = XGBRegressor(**params)
    model.fit(train[feature_cols], y.to_numpy(dtype=np.float64, copy=False))

    booster = model.get_booster()
    config = json.loads(booster.save_config())
    base_score_raw = str(config["learner"]["learner_model_param"]["base_score"])
    base_score_raw = base_score_raw.strip().strip("[]")
    if "," in base_score_raw:
        base_score_raw = base_score_raw.split(",")[0]
    base_score = float(base_score_raw)

    feature_name_to_idx = {name: i for i, name in enumerate(feature_cols)}
    tree_dicts = []
    for tree_str in booster.get_dump(dump_format="json"):
        tree_json = json.loads(tree_str)
        tree_dicts.append(_parse_tree_json(tree_json, feature_name_to_idx))

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "feature_cols": feature_cols,
        "benchmark_col": benchmark_col,
        "target_col": target_col,
        "residual_scale": float(residual_scale),
        "lambda_blend": float(lambda_blend),
        "neutralize_benchmark": float(neutralize_benchmark),
        "xgb_variant": str(xgb_variant),
        "xgb_objective": str(xgb_objective),
        "training_profile": train_profile,
        "min_train_era": resolved_min,
        "max_train_era": resolved_max,
        "max_rows_per_era": int(max_rows_per_era),
        "sample_seed": int(sample_seed),
        "n_rows": int(len(train)),
        "n_eras": int(len(train["era"].unique())),
        "era_min": str(min(train["era"], key=lambda x: int(x))),
        "era_max": str(max(train["era"], key=lambda x: int(x))),
        "xgb_params": params,
        "base_score": float(base_score),
        "n_trees": int(len(tree_dicts)),
    }
    (artifacts_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (artifacts_dir / "trees.json").write_text(json.dumps(tree_dicts))
    print(f"Saved artifacts to {artifacts_dir}")


def _load_trees(artifacts_dir: Path) -> list[dict[str, np.ndarray | int]]:
    trees_raw = json.loads((artifacts_dir / "trees.json").read_text())
    trees: list[dict[str, np.ndarray | int]] = []
    for tree in trees_raw:
        trees.append(
            {
                "root": int(tree["root"]),
                "feature": np.asarray(tree["feature"], dtype=np.int32),
                "threshold": np.asarray(tree["threshold"], dtype=np.float64),
                "left": np.asarray(tree["left"], dtype=np.int32),
                "right": np.asarray(tree["right"], dtype=np.int32),
                "missing": np.asarray(tree["missing"], dtype=np.int32),
                "value": np.asarray(tree["value"], dtype=np.float64),
            }
        )
    return trees


def _predict_raw_numpy(X: np.ndarray, trees: list[dict[str, Any]], base_score: float) -> np.ndarray:
    n = X.shape[0]
    pred = np.full(n, float(base_score), dtype=np.float64)
    for tree in trees:
        feat = tree["feature"]
        thr = tree["threshold"]
        left = tree["left"]
        right = tree["right"]
        missing = tree["missing"]
        value = tree["value"]
        root = int(tree["root"])

        for i in range(n):
            node = root
            while True:
                f = int(feat[node])
                if f < 0:
                    pred[i] += float(value[node])
                    break
                x = float(X[i, f])
                if np.isnan(x):
                    node = int(missing[node])
                elif x < float(thr[node]):
                    node = int(left[node])
                else:
                    node = int(right[node])
    return pred


def build_cloudpickle_model(*, artifacts_dir: Path, output_pkl: Path) -> None:
    import cloudpickle

    meta = json.loads((artifacts_dir / "metadata.json").read_text())
    feature_cols = list(meta["feature_cols"])
    benchmark_col = str(meta["benchmark_col"])
    lambda_blend = float(meta["lambda_blend"])
    neutralize_benchmark = float(meta.get("neutralize_benchmark", 0.0))
    base_score = float(meta["base_score"])
    trees = _load_trees(artifacts_dir)

    def _neutralize_per_era(
        pred: pd.Series,
        ref: pd.Series,
        eras: pd.Series,
        proportion: float,
    ) -> pd.Series:
        if proportion <= 0.0:
            return pred
        out = pred.astype(np.float64, copy=True)
        work = pd.DataFrame({"pred": pred, "ref": ref, "era": eras})
        for _, g in work.groupby("era", sort=False):
            idx = g.index
            p = g["pred"].to_numpy(dtype=np.float64, copy=False)
            r = g["ref"].to_numpy(dtype=np.float64, copy=False)
            mask = np.isfinite(p) & np.isfinite(r)
            if mask.sum() < 2:
                continue
            p_valid = p[mask]
            r_valid = r[mask]
            p_center = p_valid - p_valid.mean()
            r_center = r_valid - r_valid.mean()
            denom = float(np.dot(r_center, r_center))
            if denom <= 1e-12:
                continue
            beta = float(np.dot(p_center, r_center) / denom)
            p_valid_adj = p_center - float(proportion) * beta * r_center
            p_adj = p.copy()
            p_adj[mask] = p_valid_adj + p_valid.mean()
            out.loc[idx] = p_adj
        return out

    def predict(
        live_features: pd.DataFrame,
        live_benchmark_models: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        X = live_features.reindex(columns=feature_cols, fill_value=2.0)
        X_np = X.to_numpy(dtype=np.float64, copy=False)
        raw_pred = _predict_raw_numpy(X_np, trees=trees, base_score=base_score)

        if "era" in live_features.columns:
            eras = live_features["era"].astype(str)
        else:
            eras = pd.Series("0", index=live_features.index)

        raw_rank = pd.Series(raw_pred, index=live_features.index).groupby(
            eras, sort=False
        ).rank(method="average", pct=True)

        benchmark_series: pd.Series | None = None
        if live_benchmark_models is not None and benchmark_col in live_benchmark_models.columns:
            benchmark_series = pd.Series(
                live_benchmark_models[benchmark_col], index=live_benchmark_models.index
            ).reindex(live_features.index)
        elif benchmark_col in live_features.columns:
            benchmark_series = pd.Series(
                live_features[benchmark_col], index=live_features.index
            )

        if benchmark_series is not None:
            bench_rank = benchmark_series.groupby(eras, sort=False).rank(
                method="average", pct=True
            )
            pred = bench_rank + lambda_blend * (raw_rank - bench_rank)
            if neutralize_benchmark > 0.0:
                pred = _neutralize_per_era(
                    pred,
                    bench_rank,
                    eras,
                    proportion=neutralize_benchmark,
                )
            pred = pred.fillna(raw_rank)
        else:
            pred = raw_rank

        # Always enforce final per-era ranking for Numerai submissions.
        pred = pred.groupby(eras, sort=False).rank(method="average", pct=True)
        out = pd.DataFrame({"prediction": pred.astype(np.float64)}, index=live_features.index)
        return out

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, "wb") as f:
        cloudpickle.dump(predict, f)
    print(f"Wrote cloudpickle model to {output_pkl}")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[4]
    artifacts_dir = (repo_root / args.artifacts_dir).resolve()
    output_pkl = (repo_root / args.output_pkl).resolve()

    if args.mode in {"train-artifacts", "all"}:
        train_and_save_artifacts(
            repo_root=repo_root,
            artifacts_dir=artifacts_dir,
            feature_set=str(args.feature_set),
            full_data_path=Path(args.full_data_path),
            full_benchmark_path=Path(args.full_benchmark_path),
            features_path=Path(args.features_path),
            target_col=str(args.target_col),
            benchmark_col=str(args.benchmark_col),
            residual_scale=float(args.residual_scale),
            lambda_blend=float(args.lambda_blend),
            neutralize_benchmark=float(args.neutralize_benchmark),
            xgb_variant=str(args.xgb_variant),
            xgb_objective=str(args.xgb_objective),
            train_profile=str(args.train_profile),
            min_train_era=args.min_train_era,
            max_train_era=args.max_train_era,
            max_rows_per_era=int(args.max_rows_per_era),
            sample_seed=int(args.sample_seed),
        )

    if args.mode in {"build-pickle", "all"}:
        build_cloudpickle_model(artifacts_dir=artifacts_dir, output_pkl=output_pkl)


if __name__ == "__main__":
    main()
