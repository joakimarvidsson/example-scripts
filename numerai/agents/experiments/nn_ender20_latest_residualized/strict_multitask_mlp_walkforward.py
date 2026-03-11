from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from agents.code.modeling.utils.target_transforms import (
    subtract_scaled_invnorm_column,
)
try:
    from strict_gbt_walkforward_research import (
        _artifact_name,
        _feature_cols_for_spec,
        _load_example_preds_for_eval,
        _load_feature_sets,
        _load_rows_for_eras,
        _parse_candidate_modes,
        _parse_lambdas,
        _resolve_input_path,
        _resolve_features_json,
        _sample_train_rows_per_era,
        _select_strict_blend,
        _write_result_json,
    )
except ModuleNotFoundError:
    from agents.experiments.nn_ender20_latest_residualized.strict_gbt_walkforward_research import (  # type: ignore
        _artifact_name,
        _feature_cols_for_spec,
        _load_example_preds_for_eval,
        _load_feature_sets,
        _load_rows_for_eras,
        _parse_candidate_modes,
        _parse_lambdas,
        _resolve_input_path,
        _resolve_features_json,
        _sample_train_rows_per_era,
        _select_strict_blend,
        _write_result_json,
    )


@dataclass(frozen=True)
class MultitaskSpec:
    name: str
    feature_set: str
    residual_scale: float
    main_target_mix: tuple[tuple[str, float], ...] | None
    aux_target_col: str | None
    aux_weight: float
    hidden_layer_sizes: tuple[int, ...]
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    val_era_fraction: float
    clip_grad_norm: float | None
    arch_type: str = "plain"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict walk-forward multitask Torch MLP scout on target_ender_20, "
            "optionally adding an auxiliary head during training."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("numerai/agents/experiments/nn_ender20_latest_residualized"),
    )
    parser.add_argument(
        "--full-data-path",
        type=Path,
        default=Path("numerai/v5.2/full.parquet"),
    )
    parser.add_argument(
        "--benchmark-data-path",
        type=Path,
        default=Path("numerai/v5.2/full_benchmark_models.parquet"),
    )
    parser.add_argument(
        "--benchmark-model",
        default="v52_lgbm_ender20",
    )
    parser.add_argument("--target-col", default="target_ender_20")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--era-col", default="era")
    parser.add_argument("--min-eval-era", type=int, default=577)
    parser.add_argument("--max-eval-era", type=int, default=1197)
    parser.add_argument(
        "--eval-era-step",
        type=int,
        default=8,
        help="Evaluate every Nth era (default=8 for scout speed).",
    )
    parser.add_argument(
        "--early-era-max",
        type=int,
        default=889,
        help="Upper bound for early-window consistency stats.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=26,
        help="Number of eras per walk-forward validation block.",
    )
    parser.add_argument(
        "--max-rows-per-era",
        type=int,
        default=700,
        help="Cap rows per training era for memory/speed.",
    )
    parser.add_argument(
        "--train-era-step",
        type=int,
        default=4,
        help="Use every Nth historical training era before loading rows (default=4).",
    )
    parser.add_argument(
        "--train-era-offset",
        type=int,
        default=0,
        help="Modulo offset for --train-era-step (default=0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, or mps.",
    )
    parser.add_argument(
        "--blend-lambdas",
        default="0.05,0.075,0.10,0.125,0.15,0.20,0.30,0.40,0.50,0.75,1.0",
    )
    parser.add_argument(
        "--neutralize-benchmark-grid",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0",
    )
    parser.add_argument(
        "--neutralize-example-grid",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0",
    )
    parser.add_argument(
        "--candidate-modes",
        default="neutralize,blend,blend_neutralize",
    )
    parser.add_argument(
        "--example-preds-path",
        type=Path,
        default=Path("numerai/v5.2/validation_example_preds.parquet"),
    )
    parser.add_argument(
        "--max-corr-with-benchmark",
        type=float,
        default=1.0,
        help="Loose scout cap; set below 1 if needed.",
    )
    parser.add_argument(
        "--max-corr-with-example",
        type=float,
        default=1.0,
        help="Loose scout cap; set below 1 if needed.",
    )
    parser.add_argument(
        "--min-delta-mean",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--min-delta-cumsum-end",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--selection-objective",
        choices=["strict_score", "delta_cumsum_end", "corr_sortino_vs_benchmark"],
        default="strict_score",
    )
    parser.add_argument(
        "--spec-names",
        default="",
        help="Optional comma-separated model spec names to run.",
    )
    parser.add_argument(
        "--reuse-raw-preds",
        action="store_true",
        help="Reuse cached raw walk-forward predictions if present.",
    )
    parser.add_argument(
        "--skip-strict-score",
        action="store_true",
        help="Write raw walk-forward caches only and skip strict scoring.",
    )
    parser.add_argument(
        "--best-out-name",
        default="mtmlp_strict_best_ender20_walkfwd",
    )
    parser.add_argument(
        "--summary-name",
        default="mtmlp_strict_walkforward_summary.json",
    )
    parser.add_argument(
        "--artifact-suffix",
        default="",
    )
    return parser.parse_args()


def _detect_device(device_name: str) -> str:
    import torch

    requested = str(device_name).strip().lower()
    if requested and requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


class _MultitaskTorchModel:
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_layer_sizes: tuple[int, ...],
        dropout: float,
        learning_rate: float,
        weight_decay: float,
        batch_size: int,
        max_epochs: int,
        patience: int,
        clip_grad_norm: float | None,
        aux_weight: float,
        arch_type: str,
        device_name: str,
        seed: int,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "strict_multitask_mlp_walkforward.py requires torch. "
                "Install with `.venv/bin/pip install torch`."
            ) from exc

        self._torch = torch
        self._nn = torch.nn
        self._optim = torch.optim
        self._input_dim = int(input_dim)
        self._hidden_layer_sizes = tuple(int(v) for v in hidden_layer_sizes)
        self._dropout = float(dropout)
        self._learning_rate = float(learning_rate)
        self._weight_decay = float(weight_decay)
        self._batch_size = int(batch_size)
        self._max_epochs = int(max_epochs)
        self._patience = int(patience)
        self._clip_grad_norm = clip_grad_norm
        self._aux_weight = float(aux_weight)
        self._arch_type = str(arch_type)
        self._device_name = str(device_name)
        self._seed = int(seed)
        self._torch.manual_seed(self._seed)
        if self._torch.backends.mps.is_available():
            self._torch.mps.manual_seed(self._seed)
        self._model = self._build_network()

    def _tensor(self, values: np.ndarray):
        arr = np.ascontiguousarray(values, dtype=np.float32)
        return self._torch.from_numpy(arr).to(
            device=self._torch.device(self._device_name), dtype=self._torch.float32
        )

    def _build_network(self):
        torch = self._torch
        nn = self._nn

        if self._arch_type == "plain":
            layers: list = []
            in_dim = self._input_dim
            for hidden_dim in self._hidden_layer_sizes:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.GELU())
                if self._dropout > 0.0:
                    layers.append(nn.Dropout(self._dropout))
                in_dim = hidden_dim
            trunk = nn.Sequential(*layers)
        elif self._arch_type == "resnet":
            class ResidualBlock(nn.Module):
                def __init__(self, in_dim: int, out_dim: int, dropout: float):
                    super().__init__()
                    self.fc1 = nn.Linear(in_dim, out_dim)
                    self.fc2 = nn.Linear(out_dim, out_dim)
                    self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)
                    self.norm = nn.LayerNorm(out_dim)
                    self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

                def forward(self, x):
                    h = self.fc1(x)
                    h = nn.functional.gelu(h)
                    h = self.dropout(h)
                    h = self.fc2(h)
                    h = self.skip(x) + h
                    h = self.norm(h)
                    h = nn.functional.gelu(h)
                    return self.dropout(h)

            blocks: list = []
            in_dim = self._input_dim
            for hidden_dim in self._hidden_layer_sizes:
                blocks.append(ResidualBlock(in_dim, hidden_dim, self._dropout))
                in_dim = hidden_dim
            trunk = nn.Sequential(*blocks)
        else:
            raise ValueError(f"Unsupported arch_type: {self._arch_type}")

        class Net(nn.Module):
            def __init__(self, trunk_module, trunk_dim: int):
                super().__init__()
                self.trunk = trunk_module
                self.main_head = nn.Linear(trunk_dim, 1)
                self.aux_head = nn.Linear(trunk_dim, 1)

            def forward(self, x):
                h = self.trunk(x)
                return self.main_head(h), self.aux_head(h)

        device = torch.device(self._device_name)
        return Net(trunk, in_dim).to(device)

    def fit(
        self,
        x_train: np.ndarray,
        y_main_train: np.ndarray,
        y_aux_train: np.ndarray | None,
        x_val: np.ndarray,
        y_main_val: np.ndarray,
        y_aux_val: np.ndarray | None,
    ) -> "_MultitaskTorchModel":
        torch = self._torch
        device = torch.device(self._device_name)
        mse = self._nn.MSELoss()
        optimizer = self._optim.AdamW(
            self._model.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )

        x_train_tensor = self._tensor(x_train)
        y_main_train_tensor = self._tensor(y_main_train.reshape(-1, 1))
        y_aux_train_tensor = (
            self._tensor(y_aux_train.reshape(-1, 1))
            if y_aux_train is not None and self._aux_weight > 0.0
            else None
        )
        x_val_tensor = self._tensor(x_val)
        y_main_val_tensor = self._tensor(y_main_val.reshape(-1, 1))
        n_train = x_train_tensor.shape[0]
        indices = np.arange(n_train)
        rng = np.random.default_rng(self._seed)
        best_val = float("inf")
        best_state = None
        patience_left = self._patience

        for _epoch in range(self._max_epochs):
            self._model.train()
            rng.shuffle(indices)
            for start in range(0, n_train, self._batch_size):
                batch_idx = indices[start : start + self._batch_size]
                xb = x_train_tensor[batch_idx]
                yb_main = y_main_train_tensor[batch_idx]

                optimizer.zero_grad(set_to_none=True)
                pred_main, pred_aux = self._model(xb)
                loss = mse(pred_main, yb_main)
                if y_aux_train_tensor is not None:
                    yb_aux = y_aux_train_tensor[batch_idx]
                    loss = loss + self._aux_weight * mse(pred_aux, yb_aux)
                loss.backward()
                if self._clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(), float(self._clip_grad_norm)
                    )
                optimizer.step()

            self._model.eval()
            with torch.no_grad():
                pred_main_val, _pred_aux_val = self._model(x_val_tensor)
                val_loss = float(mse(pred_main_val, y_main_val_tensor).item())

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self._model.state_dict().items()
                }
                patience_left = self._patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        return self

    def predict_main(self, x: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        torch = self._torch
        device = torch.device(self._device_name)
        self._model.eval()
        preds = np.empty(x.shape[0], dtype=np.float32)
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                end = start + batch_size
                xb = self._tensor(x[start:end])
                pred_main, _pred_aux = self._model(xb)
                preds[start:end] = (
                    pred_main.detach().cpu().numpy().reshape(-1).astype(np.float32)
                )
        return preds.astype(np.float64, copy=False)


def _base_specs() -> list[MultitaskSpec]:
    common = {
        "residual_scale": 0.008,
        "feature_set": "medium:256+faith2:64",
        "hidden_layer_sizes": (768, 384, 192),
        "dropout": 0.10,
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "batch_size": 4096,
        "max_epochs": 50,
        "patience": 6,
        "val_era_fraction": 0.12,
        "clip_grad_norm": 1.0,
    }
    common_no_resid = {k: v for k, v in common.items() if k != "residual_scale"}
    common_no_resid_or_feat = {
        k: v for k, v in common.items() if k not in {"residual_scale", "feature_set"}
    }
    small_common = {
        "feature_set": "small+faith2:64",
        "hidden_layer_sizes": (384, 192, 96),
        "dropout": 0.05,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "batch_size": 4096,
        "max_epochs": 40,
        "patience": 5,
        "val_era_fraction": 0.12,
        "clip_grad_norm": 1.0,
    }
    medium_compact = {
        "feature_set": "medium:128+faith2:64",
        "hidden_layer_sizes": (512, 256, 128),
        "dropout": 0.08,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "batch_size": 4096,
        "max_epochs": 40,
        "patience": 5,
        "val_era_fraction": 0.12,
        "clip_grad_norm": 1.0,
    }
    res_common = {
        "feature_set": "medium:256+faith2:64",
        "hidden_layer_sizes": (512, 512, 512),
        "dropout": 0.10,
        "learning_rate": 2.5e-4,
        "weight_decay": 1e-4,
        "batch_size": 4096,
        "max_epochs": 50,
        "patience": 6,
        "val_era_fraction": 0.12,
        "clip_grad_norm": 1.0,
        "arch_type": "resnet",
    }
    return [
        MultitaskSpec(
            name="mtmlp_strict_resid008_medfaith64_mainonly_walkfwd",
            main_target_mix=None,
            aux_target_col=None,
            aux_weight=0.0,
            **common,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid008_medfaith64_auxe60_w025_walkfwd",
            main_target_mix=None,
            aux_target_col="target_ender_60",
            aux_weight=0.25,
            **common,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid008_medfaith64_auxe60_w050_walkfwd",
            main_target_mix=None,
            aux_target_col="target_ender_60",
            aux_weight=0.50,
            **common,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid008_medfaith64_auxt60_w025_walkfwd",
            main_target_mix=None,
            aux_target_col="target_teager2b_60",
            aux_weight=0.25,
            **common,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid008_medfaith64_auxt60_w050_walkfwd",
            main_target_mix=None,
            aux_target_col="target_teager2b_60",
            aux_weight=0.50,
            **common,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid006_medfaith64_mainonly_walkfwd",
            main_target_mix=None,
            residual_scale=0.006,
            aux_target_col=None,
            aux_weight=0.0,
            **common_no_resid,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid009_medfaith64_mainonly_walkfwd",
            main_target_mix=None,
            residual_scale=0.009,
            aux_target_col=None,
            aux_weight=0.0,
            **common_no_resid,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid010_medfaith64_mainonly_walkfwd",
            main_target_mix=None,
            residual_scale=0.010,
            aux_target_col=None,
            aux_weight=0.0,
            **common_no_resid,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid010_medium256only_mainonly_walkfwd",
            main_target_mix=None,
            feature_set="medium:256",
            residual_scale=0.010,
            aux_target_col=None,
            aux_weight=0.0,
            **common_no_resid_or_feat,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid010_faith128only_mainonly_walkfwd",
            main_target_mix=None,
            feature_set="faith2:128",
            residual_scale=0.010,
            aux_target_col=None,
            aux_weight=0.0,
            **common_no_resid_or_feat,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid010_faith192only_mainonly_walkfwd",
            main_target_mix=None,
            feature_set="faith2:192",
            residual_scale=0.010,
            aux_target_col=None,
            aux_weight=0.0,
            **common_no_resid_or_feat,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid010_medium128faith128_mainonly_walkfwd",
            main_target_mix=None,
            feature_set="medium:128+faith2:128",
            residual_scale=0.010,
            aux_target_col=None,
            aux_weight=0.0,
            **common_no_resid_or_feat,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid010_medmask40a_faith64_mainonly_walkfwd",
            main_target_mix=None,
            feature_set="medium_mask40a+faith2:64",
            residual_scale=0.010,
            aux_target_col=None,
            aux_weight=0.0,
            **common_no_resid_or_feat,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid010_medmask40b_faith64_mainonly_walkfwd",
            main_target_mix=None,
            feature_set="medium_mask40b+faith2:64",
            residual_scale=0.010,
            aux_target_col=None,
            aux_weight=0.0,
            **common_no_resid_or_feat,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid010_medmask60a_faith64_mainonly_walkfwd",
            main_target_mix=None,
            feature_set="medium_mask60a+faith2:64",
            residual_scale=0.010,
            aux_target_col=None,
            aux_weight=0.0,
            **common_no_resid_or_feat,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid010_medmask60b_faith64_mainonly_walkfwd",
            main_target_mix=None,
            feature_set="medium_mask60b+faith2:64",
            residual_scale=0.010,
            aux_target_col=None,
            aux_weight=0.0,
            **common_no_resid_or_feat,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid010_medmask20a_faith192_mainonly_walkfwd",
            main_target_mix=None,
            feature_set="medium_mask20a+faith2:192",
            residual_scale=0.010,
            aux_target_col=None,
            aux_weight=0.0,
            **common_no_resid_or_feat,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid011_medfaith64_mainonly_walkfwd",
            main_target_mix=None,
            residual_scale=0.011,
            aux_target_col=None,
            aux_weight=0.0,
            **common_no_resid,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid012_medfaith64_mainonly_walkfwd",
            main_target_mix=None,
            residual_scale=0.012,
            aux_target_col=None,
            aux_weight=0.0,
            **common_no_resid,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid006_medfaith64_mix_e20e60_7525_walkfwd",
            main_target_mix=(("target_ender_20", 0.75), ("target_ender_60", 0.25)),
            residual_scale=0.006,
            aux_target_col=None,
            aux_weight=0.0,
            **common_no_resid,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid006_medfaith64_mix_e20t60_7525_walkfwd",
            main_target_mix=(("target_ender_20", 0.75), ("target_teager2b_60", 0.25)),
            residual_scale=0.006,
            aux_target_col=None,
            aux_weight=0.0,
            **common_no_resid,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid008_medfaith64_mix_e20e60_7525_walkfwd",
            main_target_mix=(("target_ender_20", 0.75), ("target_ender_60", 0.25)),
            residual_scale=0.008,
            aux_target_col=None,
            aux_weight=0.0,
            **common_no_resid,
        ),
        MultitaskSpec(
            name="resmlp_strict_resid010_medfaith64_mainonly_walkfwd",
            main_target_mix=None,
            residual_scale=0.010,
            aux_target_col=None,
            aux_weight=0.0,
            **res_common,
        ),
        MultitaskSpec(
            name="resmlp_strict_resid011_medfaith64_mainonly_walkfwd",
            main_target_mix=None,
            residual_scale=0.011,
            aux_target_col=None,
            aux_weight=0.0,
            **res_common,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid006_smallfaith64_mainonly_walkfwd",
            main_target_mix=None,
            residual_scale=0.006,
            aux_target_col=None,
            aux_weight=0.0,
            **small_common,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid008_smallfaith64_mainonly_walkfwd",
            main_target_mix=None,
            residual_scale=0.008,
            aux_target_col=None,
            aux_weight=0.0,
            **small_common,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid006_smallfaith64_auxt60_w025_walkfwd",
            main_target_mix=None,
            residual_scale=0.006,
            aux_target_col="target_teager2b_60",
            aux_weight=0.25,
            **small_common,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid006_medcompactfaith64_mainonly_walkfwd",
            main_target_mix=None,
            residual_scale=0.006,
            aux_target_col=None,
            aux_weight=0.0,
            **medium_compact,
        ),
        MultitaskSpec(
            name="mtmlp_strict_resid008_medcompactfaith64_auxt60_w025_walkfwd",
            main_target_mix=None,
            residual_scale=0.008,
            aux_target_col="target_teager2b_60",
            aux_weight=0.25,
            **medium_compact,
        ),
    ]


def _split_train_val_by_era(
    train_df: pd.DataFrame,
    *,
    era_col: str,
    val_era_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_eras = sorted({str(e) for e in train_df[era_col].astype(str).tolist()}, key=int)
    if len(unique_eras) < 8:
        split = max(1, len(unique_eras) // 5)
    else:
        split = max(2, int(np.ceil(len(unique_eras) * float(val_era_fraction))))
    if split >= len(unique_eras):
        split = max(1, len(unique_eras) - 1)
    val_eras = set(unique_eras[-split:])
    val_mask = train_df[era_col].astype(str).isin(val_eras)
    fit_df = train_df.loc[~val_mask].reset_index(drop=True)
    val_df = train_df.loc[val_mask].reset_index(drop=True)
    if fit_df.empty or val_df.empty:
        raise ValueError("Internal train/validation split failed.")
    return fit_df, val_df


def _residualize_target(
    y: pd.Series,
    benchmark: pd.Series,
    eras: pd.Series,
    scale: float,
) -> np.ndarray:
    if scale <= 0.0:
        return y.to_numpy(dtype=np.float32, copy=False)
    resid = subtract_scaled_invnorm_column(
        y,
        pd.DataFrame({"benchmark": benchmark, "era": eras}),
        benchmark_col="benchmark",
        era_col="era",
        scale=float(scale),
        per_era=True,
        use_rank=False,
        clip_eps=1e-6,
        center=True,
        center_per_era=False,
    )
    return resid.to_numpy(dtype=np.float32, copy=False)


def _mixed_residual_target(
    df: pd.DataFrame,
    *,
    target_col: str,
    benchmark_model: str,
    era_col: str,
    residual_scale: float,
    main_target_mix: tuple[tuple[str, float], ...] | None,
) -> np.ndarray:
    if not main_target_mix:
        return _residualize_target(
            df[target_col],
            df[benchmark_model],
            df[era_col],
            scale=residual_scale,
        )
    weights = np.asarray([float(weight) for _col, weight in main_target_mix], dtype=np.float64)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        raise ValueError(f"Invalid main target mix weights: {main_target_mix}")
    weights = weights / float(weights.sum())
    mixed = np.zeros(len(df), dtype=np.float64)
    for (mix_col, _weight), norm_weight in zip(main_target_mix, weights):
        mixed += float(norm_weight) * _residualize_target(
            df[mix_col],
            df[benchmark_model],
            df[era_col],
            scale=residual_scale,
        ).astype(np.float64, copy=False)
    return mixed.astype(np.float32, copy=False)


def _build_custom_feature_subspaces(
    feature_sets: dict[str, list[str]],
) -> dict[str, list[str]]:
    medium_cols = list(feature_sets["medium"])

    def sample_medium(name: str, frac: float, seed: int) -> tuple[str, list[str]]:
        rng = np.random.default_rng(seed)
        size = max(1, int(round(len(medium_cols) * float(frac))))
        idx = np.sort(rng.choice(len(medium_cols), size=size, replace=False))
        cols = [medium_cols[int(i)] for i in idx]
        return name, cols

    return dict(
        [
            sample_medium("medium_mask20a", 0.20, 20260321),
            sample_medium("medium_mask40a", 0.40, 20260341),
            sample_medium("medium_mask40b", 0.40, 20260342),
            sample_medium("medium_mask60a", 0.60, 20260361),
            sample_medium("medium_mask60b", 0.60, 20260362),
        ]
    )


def _train_walkforward_model(
    spec: MultitaskSpec,
    *,
    full_path: Path,
    bench_path: Path,
    all_eras: list[int],
    eval_eras: list[int],
    block_size: int,
    max_rows_per_era: int,
    train_era_step: int,
    train_era_offset: int,
    id_col: str,
    era_col: str,
    target_col: str,
    benchmark_model: str,
    feature_cols: list[str],
    seed: int,
    device_name: str,
) -> pd.DataFrame:
    blocks = [
        eval_eras[i : i + block_size] for i in range(0, len(eval_eras), block_size)
    ]
    preds: list[pd.DataFrame] = []
    extra_targets = [target_col]
    if spec.main_target_mix:
        for mix_col, _weight in spec.main_target_mix:
            if mix_col not in extra_targets:
                extra_targets.append(mix_col)
    if spec.aux_target_col:
        extra_targets.append(spec.aux_target_col)

    for block_idx, val_eras in enumerate(blocks):
        train_end = min(val_eras)
        train_eras = [era for era in all_eras if era < train_end]
        if int(train_era_step) > 1:
            train_eras = [
                era
                for idx, era in enumerate(train_eras)
                if idx % int(train_era_step) == int(train_era_offset)
            ]
        if not train_eras:
            raise ValueError(f"{spec.name}: no train eras for block {block_idx}.")

        train_df = _load_rows_for_eras(
            full_path=full_path,
            bench_path=bench_path,
            eras=train_eras,
            feature_cols=feature_cols,
            id_col=id_col,
            era_col=era_col,
            target_cols=extra_targets,
            benchmark_model=benchmark_model,
        )
        train_df = _sample_train_rows_per_era(
            train_df,
            era_col,
            max_rows_per_era=max_rows_per_era,
            seed=seed + block_idx,
        )
        drop_cols = [target_col]
        if spec.main_target_mix:
            for mix_col, _weight in spec.main_target_mix:
                if mix_col not in drop_cols:
                    drop_cols.append(mix_col)
        if spec.aux_target_col and spec.aux_weight > 0.0:
            drop_cols.append(spec.aux_target_col)
        train_df = train_df.dropna(subset=drop_cols).reset_index(drop=True)

        val_df = _load_rows_for_eras(
            full_path=full_path,
            bench_path=bench_path,
            eras=val_eras,
            feature_cols=feature_cols,
            id_col=id_col,
            era_col=era_col,
            target_cols=[target_col],
            benchmark_model=benchmark_model,
        )

        fit_df, internal_val_df = _split_train_val_by_era(
            train_df, era_col=era_col, val_era_fraction=spec.val_era_fraction
        )

        scaler = StandardScaler()
        x_fit = scaler.fit_transform(
            fit_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
        ).astype(np.float32, copy=False)
        x_internal_val = scaler.transform(
            internal_val_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
        ).astype(np.float32, copy=False)
        x_block_val = scaler.transform(
            val_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
        ).astype(np.float32, copy=False)

        y_main_fit = _mixed_residual_target(
            fit_df,
            target_col=target_col,
            benchmark_model=benchmark_model,
            era_col=era_col,
            residual_scale=spec.residual_scale,
            main_target_mix=spec.main_target_mix,
        )
        y_main_internal_val = _mixed_residual_target(
            internal_val_df,
            target_col=target_col,
            benchmark_model=benchmark_model,
            era_col=era_col,
            residual_scale=spec.residual_scale,
            main_target_mix=spec.main_target_mix,
        )

        y_aux_fit: np.ndarray | None = None
        y_aux_internal_val: np.ndarray | None = None
        if spec.aux_target_col and spec.aux_weight > 0.0:
            y_aux_fit = _residualize_target(
                fit_df[spec.aux_target_col],
                fit_df[benchmark_model],
                fit_df[era_col],
                scale=spec.residual_scale,
            )
            y_aux_internal_val = _residualize_target(
                internal_val_df[spec.aux_target_col],
                internal_val_df[benchmark_model],
                internal_val_df[era_col],
                scale=spec.residual_scale,
            )

        model = _MultitaskTorchModel(
            input_dim=x_fit.shape[1],
            hidden_layer_sizes=spec.hidden_layer_sizes,
            dropout=spec.dropout,
            learning_rate=spec.learning_rate,
            weight_decay=spec.weight_decay,
            batch_size=spec.batch_size,
            max_epochs=spec.max_epochs,
            patience=spec.patience,
            clip_grad_norm=spec.clip_grad_norm,
            aux_weight=spec.aux_weight,
            arch_type=spec.arch_type,
            device_name=device_name,
            seed=seed + block_idx,
        )
        model.fit(
            x_fit,
            y_main_fit,
            y_aux_fit,
            x_internal_val,
            y_main_internal_val,
            y_aux_internal_val,
        )
        pred = model.predict_main(x_block_val)

        out = val_df[[id_col, era_col, target_col, benchmark_model]].copy()
        out["prediction_raw"] = pred
        preds.append(out)

        print(
            f"{spec.name}: block {block_idx + 1}/{len(blocks)} "
            f"train_eras={len(train_eras)} fit_rows={len(fit_df):,} "
            f"int_val_rows={len(internal_val_df):,} block_val_rows={len(val_df):,}",
            flush=True,
        )

    pred_df = pd.concat(preds, ignore_index=True)
    if pred_df[id_col].duplicated().any():
        raise ValueError(f"{spec.name}: duplicate ids in predictions.")
    return pred_df


def main() -> None:
    if sys.version_info >= (3, 14):
        raise RuntimeError(
            "strict_multitask_mlp_walkforward.py is unstable with torch on Python 3.14 "
            "in this environment. Use Python 3.12 or 3.13."
        )
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    predictions_dir = experiment_dir / "predictions"
    results_dir = experiment_dir / "results"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    _set_all_seeds(int(args.seed))
    device_name = _detect_device(args.device)
    print(f"Using torch device: {device_name}", flush=True)

    feature_sets = _load_feature_sets(_resolve_features_json())
    feature_sets.update(_build_custom_feature_subspaces(feature_sets))
    full_data_path = _resolve_input_path(args.full_data_path)
    benchmark_data_path = _resolve_input_path(args.benchmark_data_path)
    example_preds_path = _resolve_input_path(args.example_preds_path)
    specs = _base_specs()
    if args.spec_names.strip():
        wanted = {name.strip() for name in args.spec_names.split(",") if name.strip()}
        specs = [spec for spec in specs if spec.name in wanted]
        if not specs:
            raise ValueError(f"No matching specs in --spec-names: {sorted(wanted)}")

    era_only = pd.read_parquet(full_data_path, columns=[args.era_col])
    all_eras = sorted({int(e) for e in era_only[args.era_col].astype(str).tolist()})
    eval_eras = [
        era
        for era in all_eras
        if (
            int(args.min_eval_era) <= era <= int(args.max_eval_era)
            and ((era - int(args.min_eval_era)) % int(args.eval_era_step) == 0)
        )
    ]
    if not eval_eras:
        raise ValueError("No evaluation eras found.")

    example_df = _load_example_preds_for_eval(
        example_preds_path=example_preds_path,
        full_path=full_data_path,
        id_col=args.id_col,
        era_col=args.era_col,
        eval_eras=eval_eras,
    )

    lambdas = _parse_lambdas(args.blend_lambdas)
    neutralize_benchmark_grid = _parse_lambdas(args.neutralize_benchmark_grid)
    neutralize_example_grid = _parse_lambdas(args.neutralize_example_grid)
    candidate_modes = _parse_candidate_modes(args.candidate_modes)

    run_rows: list[dict] = []
    strict_candidates: list[tuple[str, pd.DataFrame, dict]] = []

    for spec in specs:
        feature_cols = _feature_cols_for_spec(feature_sets, spec.feature_set)
        raw_cache_name = _artifact_name(f"{spec.name}_raw_walkfwd", args.artifact_suffix)
        raw_cache_path = predictions_dir / f"{raw_cache_name}.parquet"
        raw_pred_df: pd.DataFrame | None = None
        if args.reuse_raw_preds and raw_cache_path.exists():
            cached = pd.read_parquet(raw_cache_path)
            required = {
                args.id_col,
                args.era_col,
                args.target_col,
                args.benchmark_model,
                "prediction_raw",
            }
            missing = required.difference(cached.columns)
            if not missing:
                raw_pred_df = cached
                print(f"Reused cached raw predictions: {raw_cache_path}", flush=True)

        if raw_pred_df is None:
            raw_pred_df = _train_walkforward_model(
                spec,
                full_path=full_data_path,
                bench_path=benchmark_data_path,
                all_eras=all_eras,
                eval_eras=eval_eras,
                block_size=args.block_size,
                max_rows_per_era=args.max_rows_per_era,
                train_era_step=args.train_era_step,
                train_era_offset=args.train_era_offset,
                id_col=args.id_col,
                era_col=args.era_col,
                target_col=args.target_col,
                benchmark_model=args.benchmark_model,
                feature_cols=feature_cols,
                seed=args.seed,
                device_name=device_name,
            )
            raw_pred_df.to_parquet(raw_cache_path, index=False)
            print(f"Saved raw predictions cache: {raw_cache_path}", flush=True)

        if args.skip_strict_score:
            run_rows.append(
                {
                    "model": raw_cache_name,
                    "feature_set": spec.feature_set,
                    "arch_type": spec.arch_type,
                    "main_target_mix": (
                        ",".join(f"{col}:{weight:g}" for col, weight in spec.main_target_mix)
                        if spec.main_target_mix
                        else "none"
                    ),
                    "aux_target": spec.aux_target_col or "none",
                    "aux_weight": spec.aux_weight,
                    "raw_only": True,
                    "oof_rows": int(raw_pred_df.shape[0]),
                    "oof_eras": int(raw_pred_df[args.era_col].astype(int).nunique()),
                }
            )
            continue

        strict_df, strict_metrics = _select_strict_blend(
            raw_pred_df,
            id_col=args.id_col,
            example_df=example_df,
            benchmark_col=args.benchmark_model,
            target_col=args.target_col,
            era_col=args.era_col,
            early_era_max=args.early_era_max,
            blend_lambdas=lambdas,
            neutralize_benchmark_grid=neutralize_benchmark_grid,
            neutralize_example_grid=neutralize_example_grid,
            candidate_modes=candidate_modes,
            max_corr_with_benchmark=float(args.max_corr_with_benchmark),
            max_corr_with_example=float(args.max_corr_with_example),
            min_delta_mean=float(args.min_delta_mean),
            min_delta_cumsum_end=float(args.min_delta_cumsum_end),
            selection_objective=str(args.selection_objective),
        )
        strict_name = _artifact_name(f"{spec.name}_strict", args.artifact_suffix)
        strict_path = predictions_dir / f"{strict_name}.parquet"
        strict_df.rename(columns={args.benchmark_model: "benchmark_prediction"}).to_parquet(
            strict_path, index=False
        )

        result_payload = {
            "data": {
                "data_version": "v5.2",
                "feature_set": spec.feature_set,
                "target": args.target_col,
                "arch_type": spec.arch_type,
                "main_target_mix": (
                    [[col, float(weight)] for col, weight in spec.main_target_mix]
                    if spec.main_target_mix
                    else None
                ),
                "aux_target": spec.aux_target_col,
                "oof_rows": int(strict_df.shape[0]),
                "oof_eras": int(len(eval_eras)),
                "walkforward_block_size_eras": int(args.block_size),
                "max_rows_per_era": int(args.max_rows_per_era),
                "train_era_step": int(args.train_era_step),
                "train_era_offset": int(args.train_era_offset),
                "device": device_name,
            },
            "benchmark": {
                "model": args.benchmark_model,
                "file": str(benchmark_data_path),
            },
            "model": {
                "type": "multitask_torch_mlp_walkforward_strict_delta",
                "base_model_name": spec.name,
                "main_target_mix": (
                    [[col, float(weight)] for col, weight in spec.main_target_mix]
                    if spec.main_target_mix
                    else None
                ),
                "aux_target": spec.aux_target_col,
                "aux_weight": spec.aux_weight,
                "arch_type": spec.arch_type,
                "residual_scale": spec.residual_scale,
                "hidden_layer_sizes": list(spec.hidden_layer_sizes),
                "dropout": spec.dropout,
                "learning_rate": spec.learning_rate,
                "weight_decay": spec.weight_decay,
                "batch_size": spec.batch_size,
                "max_epochs": spec.max_epochs,
                "patience": spec.patience,
                "val_era_fraction": spec.val_era_fraction,
                "clip_grad_norm": spec.clip_grad_norm,
            },
            "output": {"predictions_file": str(strict_path.relative_to(experiment_dir.parent))},
            "metrics": strict_metrics,
        }
        _write_result_json(results_dir / f"{strict_name}.json", result_payload)
        run_rows.append(
            {
                "model": strict_name,
                "feature_set": spec.feature_set,
                "arch_type": spec.arch_type,
                "main_target_mix": (
                    ",".join(f"{col}:{weight:g}" for col, weight in spec.main_target_mix)
                    if spec.main_target_mix
                    else "none"
                ),
                "aux_target": spec.aux_target_col or "none",
                "aux_weight": spec.aux_weight,
                **strict_metrics,
            }
        )
        strict_candidates.append((strict_name, strict_df, strict_metrics))

    summary_df = pd.DataFrame(run_rows)
    if "strict_score" in summary_df.columns:
        summary_df = summary_df.sort_values("strict_score", ascending=False)
    feasible_count = int(summary_df["feasible"].sum()) if "feasible" in summary_df.columns else 0
    if not summary_df.empty and "strict_score" in summary_df.columns:
        print("\nTop multitask MLP candidates")
        print(
            summary_df[
                [
                    "model",
                    "aux_target",
                    "aux_weight",
                    "mode",
                    "lambda",
                    "neutralize_bench",
                    "neutralize_example",
                    "feasible",
                    "delta_mean",
                    "early_delta_mean",
                    "delta_roll20_min",
                    "delta_cumsum_end",
                    "delta_cumsum_min",
                    "corr_with_benchmark_max_abs",
                    "corr_with_example_max_abs",
                    "bmc_mean",
                    "payout_mean",
                    "strict_score",
                ]
            ].to_string(index=False)
        )
        print(f"\nFeasible models: {feasible_count}/{len(summary_df)}", flush=True)
    elif not summary_df.empty:
        print("\nRaw-only multitask MLP caches")
        print(summary_df.to_string(index=False), flush=True)

    best_name: str | None = None
    if "strict_score" in summary_df.columns and not summary_df.empty:
        feasible_df = summary_df[summary_df["feasible"] == True] if "feasible" in summary_df.columns else pd.DataFrame()
        if not feasible_df.empty:
            if args.selection_objective == "delta_cumsum_end":
                best_name = str(feasible_df.sort_values("delta_cumsum_end", ascending=False).iloc[0]["model"])
            elif args.selection_objective == "corr_sortino_vs_benchmark":
                best_name = str(feasible_df.sort_values("delta_sortino", ascending=False).iloc[0]["model"])
            else:
                best_name = str(feasible_df.sort_values("strict_score", ascending=False).iloc[0]["model"])
        else:
            best_name = str(summary_df.iloc[0]["model"])

    if best_name is not None:
        best_df = next(df for name, df, _ in strict_candidates if name == best_name)
        best_metrics = next(m for name, _, m in strict_candidates if name == best_name)

        best_out_name = args.best_out_name
        best_out_path = predictions_dir / f"{best_out_name}.parquet"
        best_df.rename(columns={args.benchmark_model: "benchmark_prediction"}).to_parquet(
            best_out_path, index=False
        )
        _write_result_json(
            results_dir / f"{best_out_name}.json",
            {
                "selection": {
                    "best_base_model": best_name,
                    "selection_objective": args.selection_objective,
                    "required_constraints": {
                        "max_corr_with_benchmark": args.max_corr_with_benchmark,
                        "max_corr_with_example": args.max_corr_with_example,
                        "min_delta_mean": args.min_delta_mean,
                        "min_delta_cumsum_end": args.min_delta_cumsum_end,
                    },
                },
                "metrics": best_metrics,
                "output": {
                    "predictions_file": str(best_out_path.relative_to(experiment_dir.parent))
                },
            },
        )

    summary_path = results_dir / args.summary_name
    _write_result_json(
        summary_path,
        {
            "settings": {
                "min_eval_era": args.min_eval_era,
                "max_eval_era": args.max_eval_era,
                "eval_era_step": args.eval_era_step,
                "early_era_max": args.early_era_max,
                "block_size": args.block_size,
                "max_rows_per_era": args.max_rows_per_era,
                "train_era_step": args.train_era_step,
                "train_era_offset": args.train_era_offset,
                "blend_lambdas": lambdas,
                "neutralize_benchmark_grid": neutralize_benchmark_grid,
                "neutralize_example_grid": neutralize_example_grid,
                "candidate_modes": candidate_modes,
                "example_preds_path": str(example_preds_path),
                "max_corr_with_benchmark": args.max_corr_with_benchmark,
                "max_corr_with_example": args.max_corr_with_example,
                "min_delta_mean": args.min_delta_mean,
                "min_delta_cumsum_end": args.min_delta_cumsum_end,
                "selection_objective": args.selection_objective,
                "device": device_name,
            },
            "top_models": summary_df.to_dict(orient="records"),
            "selected_best_model": best_name,
            "feasible_model_count": feasible_count,
        },
    )
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
