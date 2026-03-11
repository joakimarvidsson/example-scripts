from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from agents.code.modeling.utils.target_transforms import subtract_scaled_invnorm_column

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


ERA_PARTITIONS: dict[str, tuple[int, int]] = {
    "research_pool": (577, 956),
    "holdout_a": (964, 1079),
    "holdout_b": (1084, 1199),
}


@dataclass(frozen=True)
class NeuralCvSpec:
    name: str
    feature_set: str
    residual_scale: float
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
    aux_target_col: str | None = None
    aux_weight: float = 0.0
    era_decay_halflife: float | None = None


ArrayInput = np.ndarray | tuple[np.ndarray, np.ndarray]
TensorInput = Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict neural walk-forward CV harness with purged outer blocks, internal "
            "embargoed validation, optional seed averaging, and optional WandB logging."
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
    parser.add_argument(
        "--era-partition",
        choices=["custom", *ERA_PARTITIONS.keys()],
        default="research_pool",
    )
    parser.add_argument("--min-eval-era", type=int, default=577)
    parser.add_argument("--max-eval-era", type=int, default=956)
    parser.add_argument(
        "--eval-era-step",
        type=int,
        default=4,
        help="Evaluate every Nth era. Training still uses all available prior eras by default.",
    )
    parser.add_argument(
        "--early-era-max",
        type=int,
        default=-1,
        help="If negative, use the midpoint of the active eval partition.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=20,
        help="Number of eras per walk-forward validation block.",
    )
    parser.add_argument(
        "--max-rows-per-era",
        type=int,
        default=800,
        help="Cap rows per training era for memory and speed.",
    )
    parser.add_argument(
        "--train-era-step",
        type=int,
        default=1,
        help="Use every Nth historical training era before loading rows.",
    )
    parser.add_argument(
        "--train-era-offset",
        type=int,
        default=0,
        help="Modulo offset for --train-era-step.",
    )
    parser.add_argument(
        "--outer-embargo-eras",
        type=int,
        default=4,
        help="Hold out this many eras between outer-train and outer-validation.",
    )
    parser.add_argument(
        "--inner-embargo-eras",
        type=int,
        default=4,
        help="Hold out this many eras between fit and internal validation.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--seed-average-count", type=int, default=1)
    parser.add_argument("--seed-stride", type=int, default=97)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, or mps.",
    )
    parser.add_argument(
        "--blend-lambdas",
        default="0.0,0.02,0.05,0.075,0.10,0.125,0.15,0.20",
    )
    parser.add_argument(
        "--neutralize-benchmark-grid",
        default="0.0,0.02,0.05,0.1,0.15,0.2",
    )
    parser.add_argument(
        "--neutralize-example-grid",
        default="0.0,0.02,0.05,0.1,0.15,0.2",
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
    parser.add_argument("--max-corr-with-benchmark", type=float, default=1.0)
    parser.add_argument("--max-corr-with-example", type=float, default=1.0)
    parser.add_argument("--min-delta-mean", type=float, default=0.0)
    parser.add_argument("--min-delta-cumsum-end", type=float, default=0.0)
    parser.add_argument(
        "--selection-objective",
        choices=["strict_score", "delta_cumsum_end", "corr_sortino_vs_benchmark"],
        default="strict_score",
    )
    parser.add_argument(
        "--spec-names",
        default="",
        help="Optional comma-separated spec names to run.",
    )
    parser.add_argument(
        "--reuse-raw-preds",
        action="store_true",
        help="Reuse cached per-seed raw walk-forward predictions if present.",
    )
    parser.add_argument(
        "--skip-strict-score",
        action="store_true",
        help="Write averaged raw walk-forward caches only and skip strict scoring.",
    )
    parser.add_argument(
        "--best-out-name",
        default="strict_neural_cv_best",
    )
    parser.add_argument(
        "--summary-name",
        default="strict_neural_cv_summary.json",
    )
    parser.add_argument(
        "--artifact-suffix",
        default="",
    )
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-group", default="")
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument(
        "--wandb-mode",
        choices=["disabled", "offline", "online"],
        default="disabled",
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


class _GatedBlock:
    def __init__(self, nn_module: Any, in_dim: int, out_dim: int, dropout: float) -> None:
        self._nn = nn_module
        self.block = nn_module.ModuleDict(
            {
                "main": nn_module.Linear(in_dim, out_dim),
                "gate": nn_module.Linear(in_dim, out_dim),
                "drop": nn_module.Dropout(dropout) if dropout > 0.0 else nn_module.Identity(),
            }
        )

    def module(self):
        nn = self._nn

        class Block(nn.Module):
            def __init__(self, pieces):
                super().__init__()
                self.main = pieces["main"]
                self.gate = pieces["gate"]
                self.drop = pieces["drop"]

            def forward(self, x):
                h = nn.functional.gelu(self.main(x))
                g = torch.sigmoid(self.gate(x))
                return self.drop(h * g)

        import torch

        return Block(self.block)


class _NeuralTorchModel:
    def __init__(
        self,
        *,
        input_dims: int | tuple[int, int],
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
                "strict_neural_cv_walkforward.py requires torch. "
                "Install it in the selected runtime, e.g. `.venv-py312-torch/bin/pip install torch`."
            ) from exc

        self._torch = torch
        self._nn = torch.nn
        self._optim = torch.optim
        self._input_dims = input_dims
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

    def _tensor_input(self, values: ArrayInput) -> TensorInput:
        if isinstance(values, tuple):
            return tuple(self._tensor(v) for v in values)
        return self._tensor(values)

    def _slice_input(self, values: TensorInput, batch_idx: np.ndarray | slice) -> TensorInput:
        if isinstance(values, tuple):
            return tuple(v[batch_idx] for v in values)
        return values[batch_idx]

    def _forward(self, values: TensorInput):
        if self._arch_type == "twotower":
            assert isinstance(values, tuple)
            return self._model(values[0], values[1])
        return self._model(values)

    def _flat_trunk(
        self,
        input_dim: int,
        block_type: str,
        hidden_layer_sizes: tuple[int, ...] | None = None,
    ):
        nn = self._nn
        layers: list[Any] = []
        in_dim = input_dim
        hidden_sizes = (
            tuple(int(v) for v in hidden_layer_sizes)
            if hidden_layer_sizes is not None
            else self._hidden_layer_sizes
        )
        if not hidden_sizes:
            raise ValueError("hidden_layer_sizes must be non-empty")
        for hidden_dim in hidden_sizes:
            if block_type == "plain":
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.GELU())
                if self._dropout > 0.0:
                    layers.append(nn.Dropout(self._dropout))
            elif block_type == "gated":
                layers.append(_GatedBlock(nn, in_dim, hidden_dim, self._dropout).module())
            else:
                raise ValueError(f"Unsupported flat block type: {block_type}")
            in_dim = hidden_dim
        return nn.Sequential(*layers), in_dim

    def _build_network(self):
        torch = self._torch
        nn = self._nn

        class FlatNet(nn.Module):
            def __init__(self, trunk_module, trunk_dim: int):
                super().__init__()
                self.trunk = trunk_module
                self.main_head = nn.Linear(trunk_dim, 1)
                self.aux_head = nn.Linear(trunk_dim, 1)

            def forward(self, x):
                h = self.trunk(x)
                return self.main_head(h), self.aux_head(h)

        class TwoTowerNet(nn.Module):
            def __init__(
                self,
                tower_a: nn.Module,
                tower_b: nn.Module,
                dim_a: int,
                dim_b: int,
                fusion: nn.Module,
                fusion_dim: int,
            ):
                super().__init__()
                self.tower_a = tower_a
                self.tower_b = tower_b
                self.fusion = fusion
                self.main_head = nn.Linear(fusion_dim, 1)
                self.aux_head = nn.Linear(fusion_dim, 1)
                self._dim_a = dim_a
                self._dim_b = dim_b

            def forward(self, x_a, x_b):
                h_a = self.tower_a(x_a)
                h_b = self.tower_b(x_b)
                h = torch.cat([h_a, h_b], dim=1)
                h = self.fusion(h)
                return self.main_head(h), self.aux_head(h)

        if self._arch_type in {"plain", "gated"}:
            assert isinstance(self._input_dims, int)
            trunk, trunk_dim = self._flat_trunk(self._input_dims, self._arch_type)
            model = FlatNet(trunk, trunk_dim)
        elif self._arch_type == "twotower":
            if not isinstance(self._input_dims, tuple) or len(self._input_dims) != 2:
                raise ValueError("twotower requires two input dimensions")
            if len(self._hidden_layer_sizes) < 2:
                raise ValueError("twotower requires at least two hidden sizes")
            tower_hidden = self._hidden_layer_sizes[:-1]
            fusion_hidden = (self._hidden_layer_sizes[-1],)
            tower_a, dim_a = self._flat_trunk(
                self._input_dims[0], "plain", hidden_layer_sizes=tower_hidden
            )
            tower_b, dim_b = self._flat_trunk(
                self._input_dims[1], "plain", hidden_layer_sizes=tower_hidden
            )
            fusion_layers: list[Any] = []
            fusion_in = dim_a + dim_b
            for hidden_dim in fusion_hidden:
                fusion_layers.append(nn.Linear(fusion_in, hidden_dim))
                fusion_layers.append(nn.GELU())
                if self._dropout > 0.0:
                    fusion_layers.append(nn.Dropout(self._dropout))
                fusion_in = hidden_dim
            fusion = nn.Sequential(*fusion_layers)
            model = TwoTowerNet(tower_a, tower_b, dim_a, dim_b, fusion, fusion_in)
        else:
            raise ValueError(f"Unsupported arch_type: {self._arch_type}")
        return model.to(self._torch.device(self._device_name))

    @staticmethod
    def _weighted_mse(pred, target, weight):
        err = (pred - target) ** 2
        return (err * weight).sum() / weight.sum().clamp(min=1e-8)

    def fit(
        self,
        x_train: ArrayInput,
        y_main_train: np.ndarray,
        y_aux_train: np.ndarray | None,
        sample_weight_train: np.ndarray,
        x_val: ArrayInput,
        y_main_val: np.ndarray,
        y_aux_val: np.ndarray | None,
    ) -> "_NeuralTorchModel":
        torch = self._torch
        optimizer = self._optim.AdamW(
            self._model.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )

        x_train_tensor = self._tensor_input(x_train)
        y_main_train_tensor = self._tensor(y_main_train.reshape(-1, 1))
        y_aux_train_tensor = (
            self._tensor(y_aux_train.reshape(-1, 1))
            if y_aux_train is not None and self._aux_weight > 0.0
            else None
        )
        sample_weight_train_tensor = self._tensor(sample_weight_train.reshape(-1, 1))
        x_val_tensor = self._tensor_input(x_val)
        y_main_val_tensor = self._tensor(y_main_val.reshape(-1, 1))
        y_aux_val_tensor = (
            self._tensor(y_aux_val.reshape(-1, 1))
            if y_aux_val is not None and self._aux_weight > 0.0
            else None
        )

        n_train = y_main_train_tensor.shape[0]
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
                xb = self._slice_input(x_train_tensor, batch_idx)
                yb_main = y_main_train_tensor[batch_idx]
                wb = sample_weight_train_tensor[batch_idx]

                optimizer.zero_grad(set_to_none=True)
                pred_main, pred_aux = self._forward(xb)
                loss = self._weighted_mse(pred_main, yb_main, wb)
                if y_aux_train_tensor is not None:
                    yb_aux = y_aux_train_tensor[batch_idx]
                    loss = loss + self._aux_weight * self._weighted_mse(pred_aux, yb_aux, wb)
                loss.backward()
                if self._clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(), float(self._clip_grad_norm)
                    )
                optimizer.step()

            self._model.eval()
            with torch.no_grad():
                pred_main_val, pred_aux_val = self._forward(x_val_tensor)
                val_loss = float(self._weighted_mse(
                    pred_main_val,
                    y_main_val_tensor,
                    torch.ones_like(y_main_val_tensor),
                ).item())
                if y_aux_val_tensor is not None and self._aux_weight > 0.0:
                    val_loss += float(
                        self._aux_weight
                        * self._weighted_mse(
                            pred_aux_val,
                            y_aux_val_tensor,
                            torch.ones_like(y_aux_val_tensor),
                        ).item()
                    )

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

    def predict_main(self, x: ArrayInput, batch_size: int = 8192) -> np.ndarray:
        self._model.eval()
        n_rows = x[0].shape[0] if isinstance(x, tuple) else x.shape[0]
        preds = np.empty(n_rows, dtype=np.float32)
        x_tensor = self._tensor_input(x)
        with self._torch.no_grad():
            for start in range(0, n_rows, batch_size):
                end = start + batch_size
                xb = self._slice_input(x_tensor, slice(start, end))
                pred_main, _pred_aux = self._forward(xb)
                preds[start:end] = (
                    pred_main.detach().cpu().numpy().reshape(-1).astype(np.float32)
                )
        return preds.astype(np.float64, copy=False)



def _initial_specs() -> list[NeuralCvSpec]:
    return [
        NeuralCvSpec(
            name="ncv_plain_resid010_medfaith64_nodecay",
            feature_set="medium:256+faith2:64",
            residual_scale=0.010,
            hidden_layer_sizes=(768, 384, 192),
            dropout=0.10,
            learning_rate=2.0e-4,
            weight_decay=1.0e-4,
            batch_size=4096,
            max_epochs=45,
            patience=6,
            val_era_fraction=0.14,
            clip_grad_norm=1.0,
            arch_type="plain",
            era_decay_halflife=None,
        ),
        NeuralCvSpec(
            name="ncv_gated_resid010_medfaith64_hl256",
            feature_set="medium:256+faith2:64",
            residual_scale=0.010,
            hidden_layer_sizes=(768, 384, 192),
            dropout=0.10,
            learning_rate=2.0e-4,
            weight_decay=1.0e-4,
            batch_size=4096,
            max_epochs=45,
            patience=6,
            val_era_fraction=0.14,
            clip_grad_norm=1.0,
            arch_type="gated",
            era_decay_halflife=256.0,
        ),
        NeuralCvSpec(
            name="ncv_twotower_resid010_medfaith64_hl256",
            feature_set="medium:256+faith2:64",
            residual_scale=0.010,
            hidden_layer_sizes=(512, 256, 128),
            dropout=0.10,
            learning_rate=2.0e-4,
            weight_decay=1.0e-4,
            batch_size=4096,
            max_epochs=45,
            patience=6,
            val_era_fraction=0.14,
            clip_grad_norm=1.0,
            arch_type="twotower",
            era_decay_halflife=256.0,
        ),
        NeuralCvSpec(
            name="ncv_twotower_resid010_medfaith128_hl128",
            feature_set="medium:192+faith2:128",
            residual_scale=0.010,
            hidden_layer_sizes=(512, 256, 128),
            dropout=0.10,
            learning_rate=2.0e-4,
            weight_decay=1.0e-4,
            batch_size=4096,
            max_epochs=45,
            patience=6,
            val_era_fraction=0.14,
            clip_grad_norm=1.0,
            arch_type="twotower",
            era_decay_halflife=128.0,
        ),
    ]



def _resolve_eval_partition(
    partition: str,
    min_eval_era: int,
    max_eval_era: int,
) -> tuple[int, int]:
    if partition == "custom":
        return int(min_eval_era), int(max_eval_era)
    return ERA_PARTITIONS[partition]



def _resolve_early_era_max(eval_eras: list[int], requested: int) -> int:
    if not eval_eras:
        raise ValueError("No evaluation eras found.")
    if int(requested) > 0:
        return min(int(requested), max(eval_eras))
    midpoint = max(1, len(eval_eras) // 2)
    return int(eval_eras[midpoint - 1])



def _split_train_val_by_era(
    train_df: pd.DataFrame,
    *,
    era_col: str,
    val_era_fraction: float,
    inner_embargo_eras: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_eras = sorted({int(e) for e in train_df[era_col].astype(int).tolist()})
    if len(unique_eras) < 6:
        raise ValueError("Not enough eras for embargoed internal validation split.")
    if len(unique_eras) < 10:
        split = max(1, len(unique_eras) // 5)
    else:
        split = max(2, int(np.ceil(len(unique_eras) * float(val_era_fraction))))
    split = min(split, len(unique_eras) - 1)

    while split >= 1:
        val_eras = unique_eras[-split:]
        val_start_idx = len(unique_eras) - split
        fit_stop_idx = max(0, val_start_idx - int(inner_embargo_eras))
        fit_eras = set(unique_eras[:fit_stop_idx])
        if fit_eras:
            fit_mask = train_df[era_col].astype(int).isin(fit_eras)
            val_mask = train_df[era_col].astype(int).isin(set(val_eras))
            fit_df = train_df.loc[fit_mask].reset_index(drop=True)
            val_df = train_df.loc[val_mask].reset_index(drop=True)
            if not fit_df.empty and not val_df.empty:
                return fit_df, val_df
        split -= 1
    raise ValueError("Embargoed internal train/validation split failed.")



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



def _era_decay_weights(eras: pd.Series, halflife: float | None) -> np.ndarray:
    eras_int = eras.astype(int).to_numpy(dtype=np.int32, copy=False)
    if halflife is None or float(halflife) <= 0.0:
        return np.ones(len(eras_int), dtype=np.float32)
    max_era = int(eras_int.max())
    weights = 0.5 ** ((max_era - eras_int) / float(halflife))
    weights = weights / float(np.mean(weights))
    return weights.astype(np.float32, copy=False)



def _feature_groups_for_twotower(
    feature_sets: dict[str, list[str]],
    feature_spec: str,
) -> tuple[list[str], list[str]]:
    tokens = [t.strip() for t in str(feature_spec).split("+") if t.strip()]
    base_cols: list[str] = []
    faith_cols: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        base = token
        cap: int | None = None
        if ":" in token:
            base, cap_str = token.split(":", 1)
            base = base.strip()
            cap = int(cap_str.strip())
        key = base.strip()
        resolved = "faith" if key == "faith2" else key
        cols = list(feature_sets[resolved])
        if cap is not None:
            cols = cols[:cap]
        target = faith_cols if resolved.startswith("faith") else base_cols
        for col in cols:
            if col not in seen:
                seen.add(col)
                target.append(col)
    if not base_cols or not faith_cols:
        raise ValueError(f"twotower requires both non-faith and faith features, got '{feature_spec}'")
    return base_cols, faith_cols



def _to_model_input(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    feature_groups: tuple[list[str], list[str]] | None,
    scalers: Any,
    arch_type: str,
) -> ArrayInput:
    if arch_type != "twotower":
        scaler = scalers
        return scaler.transform(df[feature_cols].to_numpy(dtype=np.float32, copy=False)).astype(np.float32, copy=False)
    assert feature_groups is not None
    scaler_a, scaler_b = scalers
    cols_a, cols_b = feature_groups
    x_a = scaler_a.transform(df[cols_a].to_numpy(dtype=np.float32, copy=False)).astype(np.float32, copy=False)
    x_b = scaler_b.transform(df[cols_b].to_numpy(dtype=np.float32, copy=False)).astype(np.float32, copy=False)
    return x_a, x_b



def _fit_scalers(
    fit_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    feature_groups: tuple[list[str], list[str]] | None,
    arch_type: str,
):
    if arch_type != "twotower":
        scaler = StandardScaler()
        scaler.fit(fit_df[feature_cols].to_numpy(dtype=np.float32, copy=False))
        return scaler
    assert feature_groups is not None
    cols_a, cols_b = feature_groups
    scaler_a = StandardScaler()
    scaler_b = StandardScaler()
    scaler_a.fit(fit_df[cols_a].to_numpy(dtype=np.float32, copy=False))
    scaler_b.fit(fit_df[cols_b].to_numpy(dtype=np.float32, copy=False))
    return scaler_a, scaler_b



def _train_walkforward_model(
    spec: NeuralCvSpec,
    *,
    full_path: Path,
    bench_path: Path,
    all_eras: list[int],
    eval_eras: list[int],
    block_size: int,
    max_rows_per_era: int,
    train_era_step: int,
    train_era_offset: int,
    outer_embargo_eras: int,
    inner_embargo_eras: int,
    id_col: str,
    era_col: str,
    target_col: str,
    benchmark_model: str,
    feature_cols: list[str],
    feature_groups: tuple[list[str], list[str]] | None,
    seed: int,
    device_name: str,
) -> pd.DataFrame:
    blocks = [eval_eras[i : i + block_size] for i in range(0, len(eval_eras), block_size)]
    preds: list[pd.DataFrame] = []
    extra_targets = [target_col]
    if spec.aux_target_col:
        extra_targets.append(spec.aux_target_col)

    for block_idx, val_eras in enumerate(blocks):
        train_stop_exclusive = min(val_eras) - int(outer_embargo_eras)
        train_eras = [era for era in all_eras if era < train_stop_exclusive]
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
            train_df,
            era_col=era_col,
            val_era_fraction=spec.val_era_fraction,
            inner_embargo_eras=inner_embargo_eras,
        )

        scalers = _fit_scalers(
            fit_df,
            feature_cols=feature_cols,
            feature_groups=feature_groups,
            arch_type=spec.arch_type,
        )
        x_fit = _to_model_input(
            fit_df,
            feature_cols=feature_cols,
            feature_groups=feature_groups,
            scalers=scalers,
            arch_type=spec.arch_type,
        )
        x_internal_val = _to_model_input(
            internal_val_df,
            feature_cols=feature_cols,
            feature_groups=feature_groups,
            scalers=scalers,
            arch_type=spec.arch_type,
        )
        x_block_val = _to_model_input(
            val_df,
            feature_cols=feature_cols,
            feature_groups=feature_groups,
            scalers=scalers,
            arch_type=spec.arch_type,
        )

        y_main_fit = _residualize_target(
            fit_df[target_col],
            fit_df[benchmark_model],
            fit_df[era_col],
            scale=spec.residual_scale,
        )
        y_main_internal_val = _residualize_target(
            internal_val_df[target_col],
            internal_val_df[benchmark_model],
            internal_val_df[era_col],
            scale=spec.residual_scale,
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

        sample_weight_fit = _era_decay_weights(fit_df[era_col], spec.era_decay_halflife)

        input_dims: int | tuple[int, int]
        if spec.arch_type == "twotower":
            assert isinstance(x_fit, tuple)
            input_dims = (x_fit[0].shape[1], x_fit[1].shape[1])
        else:
            assert isinstance(x_fit, np.ndarray)
            input_dims = x_fit.shape[1]

        run_seed = seed + block_idx
        _set_all_seeds(run_seed)
        model = _NeuralTorchModel(
            input_dims=input_dims,
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
            seed=run_seed,
        )
        model.fit(
            x_fit,
            y_main_fit,
            y_aux_fit,
            sample_weight_fit,
            x_internal_val,
            y_main_internal_val,
            y_aux_internal_val,
        )
        pred = model.predict_main(x_block_val)

        out = val_df[[id_col, era_col, target_col, benchmark_model]].copy()
        out[era_col] = out[era_col].astype(str)
        out["prediction_raw"] = pred
        preds.append(out)

        print(
            f"{spec.name}: block {block_idx + 1}/{len(blocks)} train_eras={len(train_eras)} "
            f"fit_rows={len(fit_df):,} int_val_rows={len(internal_val_df):,} block_val_rows={len(val_df):,}",
            flush=True,
        )

    pred_df = pd.concat(preds, ignore_index=True)
    if pred_df[id_col].duplicated().any():
        raise ValueError(f"{spec.name}: duplicate ids in predictions.")
    return pred_df



def _merge_seed_predictions(
    seed_frames: list[pd.DataFrame],
    *,
    id_col: str,
    era_col: str,
    target_col: str,
    benchmark_model: str,
) -> pd.DataFrame:
    if not seed_frames:
        raise ValueError("No seed frames to average.")
    base = seed_frames[0][[id_col, era_col, target_col, benchmark_model, "prediction_raw"]].copy()
    base = base.rename(columns={"prediction_raw": "prediction_raw_avg"})
    for idx, frame in enumerate(seed_frames[1:], start=1):
        cols = [id_col, era_col, target_col, benchmark_model, "prediction_raw"]
        other = frame[cols].copy().rename(columns={"prediction_raw": f"prediction_raw_{idx}"})
        base = base.merge(
            other,
            on=[id_col, era_col, target_col, benchmark_model],
            how="inner",
            validate="one_to_one",
        )
    pred_cols = [c for c in base.columns if c.startswith("prediction_raw")]
    base["prediction_raw"] = base[pred_cols].mean(axis=1)
    return base[[id_col, era_col, target_col, benchmark_model, "prediction_raw"]].copy()



def _wandb_init(args: argparse.Namespace, summary_settings: dict[str, Any]):
    if not args.wandb_project.strip() or args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("wandb requested but package is not installed in this environment.", flush=True)
        return None
    init_kwargs: dict[str, Any] = {
        "project": args.wandb_project.strip(),
        "config": summary_settings,
        "mode": args.wandb_mode,
    }
    if args.wandb_entity.strip():
        init_kwargs["entity"] = args.wandb_entity.strip()
    if args.wandb_run_group.strip():
        init_kwargs["group"] = args.wandb_run_group.strip()
    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    if tags:
        init_kwargs["tags"] = tags
    return wandb.init(**init_kwargs)



def main() -> None:
    if sys.version_info >= (3, 14):
        raise RuntimeError(
            "strict_neural_cv_walkforward.py is unstable with torch on Python 3.14 in this environment. "
            "Use Python 3.12 or 3.13."
        )

    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    predictions_dir = experiment_dir / "predictions"
    results_dir = experiment_dir / "results"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    min_eval_era, max_eval_era = _resolve_eval_partition(
        args.era_partition,
        args.min_eval_era,
        args.max_eval_era,
    )

    _set_all_seeds(int(args.seed))
    device_name = _detect_device(args.device)
    print(f"Using torch device: {device_name}", flush=True)

    feature_sets = _load_feature_sets(_resolve_features_json())
    full_data_path = _resolve_input_path(args.full_data_path)
    benchmark_data_path = _resolve_input_path(args.benchmark_data_path)
    example_preds_path = _resolve_input_path(args.example_preds_path)

    specs = _initial_specs()
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
        if min_eval_era <= era <= max_eval_era and ((era - min_eval_era) % int(args.eval_era_step) == 0)
    ]
    if not eval_eras:
        raise ValueError("No evaluation eras found.")
    early_era_max = _resolve_early_era_max(eval_eras, args.early_era_max)

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

    summary_settings = {
        "era_partition": args.era_partition,
        "min_eval_era": min_eval_era,
        "max_eval_era": max_eval_era,
        "eval_era_step": args.eval_era_step,
        "early_era_max": early_era_max,
        "block_size": args.block_size,
        "max_rows_per_era": args.max_rows_per_era,
        "train_era_step": args.train_era_step,
        "train_era_offset": args.train_era_offset,
        "outer_embargo_eras": args.outer_embargo_eras,
        "inner_embargo_eras": args.inner_embargo_eras,
        "seed": args.seed,
        "seed_average_count": args.seed_average_count,
        "seed_stride": args.seed_stride,
        "device": device_name,
    }
    wandb_run = _wandb_init(args, summary_settings)

    run_rows: list[dict[str, Any]] = []
    strict_candidates: list[tuple[str, pd.DataFrame, dict[str, Any]]] = []

    for spec in specs:
        feature_cols = _feature_cols_for_spec(feature_sets, spec.feature_set)
        feature_groups = (
            _feature_groups_for_twotower(feature_sets, spec.feature_set)
            if spec.arch_type == "twotower"
            else None
        )
        seed_frames: list[pd.DataFrame] = []
        seed_values = [int(args.seed) + i * int(args.seed_stride) for i in range(int(args.seed_average_count))]
        avg_name = _artifact_name(
            f"{spec.name}_raw_walkfwd_seedavg{int(args.seed_average_count)}",
            args.artifact_suffix,
        )
        avg_cache_path = predictions_dir / f"{avg_name}.parquet"
        if args.reuse_raw_preds and avg_cache_path.exists():
            raw_pred_df = pd.read_parquet(avg_cache_path)
            seed_frames = [raw_pred_df]
            print(f"Reused averaged raw predictions: {avg_cache_path}", flush=True)
        else:
            for seed_value in seed_values:
                seed_name = _artifact_name(
                    f"{spec.name}_raw_walkfwd_seed{seed_value}",
                    args.artifact_suffix,
                )
                seed_path = predictions_dir / f"{seed_name}.parquet"
                if args.reuse_raw_preds and seed_path.exists():
                    seed_df = pd.read_parquet(seed_path)
                    print(f"Reused seed raw predictions: {seed_path}", flush=True)
                else:
                    seed_df = _train_walkforward_model(
                        spec,
                        full_path=full_data_path,
                        bench_path=benchmark_data_path,
                        all_eras=all_eras,
                        eval_eras=eval_eras,
                        block_size=args.block_size,
                        max_rows_per_era=args.max_rows_per_era,
                        train_era_step=args.train_era_step,
                        train_era_offset=args.train_era_offset,
                        outer_embargo_eras=args.outer_embargo_eras,
                        inner_embargo_eras=args.inner_embargo_eras,
                        id_col=args.id_col,
                        era_col=args.era_col,
                        target_col=args.target_col,
                        benchmark_model=args.benchmark_model,
                        feature_cols=feature_cols,
                        feature_groups=feature_groups,
                        seed=seed_value,
                        device_name=device_name,
                    )
                    seed_df.to_parquet(seed_path, index=False)
                    print(f"Saved raw predictions cache: {seed_path}", flush=True)
                seed_frames.append(seed_df)
            raw_pred_df = _merge_seed_predictions(
                seed_frames,
                id_col=args.id_col,
                era_col=args.era_col,
                target_col=args.target_col,
                benchmark_model=args.benchmark_model,
            )
            raw_pred_df.to_parquet(avg_cache_path, index=False)
            print(f"Saved averaged raw predictions cache: {avg_cache_path}", flush=True)

        if args.skip_strict_score:
            row = {
                "model": avg_name,
                "feature_set": spec.feature_set,
                "arch_type": spec.arch_type,
                "aux_target": spec.aux_target_col or "none",
                "aux_weight": spec.aux_weight,
                "era_decay_halflife": spec.era_decay_halflife,
                "seed_average_count": int(args.seed_average_count),
                "raw_only": True,
                "oof_rows": int(raw_pred_df.shape[0]),
                "oof_eras": int(raw_pred_df[args.era_col].astype(int).nunique()),
            }
            run_rows.append(row)
            if wandb_run is not None:
                wandb_run.log({f"raw_only/{spec.name}/oof_rows": row["oof_rows"]})
            continue

        strict_df, strict_metrics = _select_strict_blend(
            raw_pred_df,
            id_col=args.id_col,
            example_df=example_df,
            benchmark_col=args.benchmark_model,
            target_col=args.target_col,
            era_col=args.era_col,
            early_era_max=early_era_max,
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
        strict_name = _artifact_name(f"{spec.name}_strict_seedavg{int(args.seed_average_count)}", args.artifact_suffix)
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
                "aux_target": spec.aux_target_col,
                "oof_rows": int(strict_df.shape[0]),
                "oof_eras": int(len(eval_eras)),
                "era_partition": args.era_partition,
                "walkforward_block_size_eras": int(args.block_size),
                "max_rows_per_era": int(args.max_rows_per_era),
                "train_era_step": int(args.train_era_step),
                "train_era_offset": int(args.train_era_offset),
                "outer_embargo_eras": int(args.outer_embargo_eras),
                "inner_embargo_eras": int(args.inner_embargo_eras),
                "seed_average_count": int(args.seed_average_count),
                "seed_stride": int(args.seed_stride),
                "device": device_name,
            },
            "benchmark": {
                "model": args.benchmark_model,
                "file": str(benchmark_data_path),
            },
            "model": {
                "type": "strict_neural_cv_walkforward",
                "base_model_name": spec.name,
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
                "era_decay_halflife": spec.era_decay_halflife,
            },
            "output": {"predictions_file": str(strict_path.relative_to(experiment_dir.parent))},
            "metrics": strict_metrics,
        }
        _write_result_json(results_dir / f"{strict_name}.json", result_payload)

        row = {
            "model": strict_name,
            "feature_set": spec.feature_set,
            "arch_type": spec.arch_type,
            "aux_target": spec.aux_target_col or "none",
            "aux_weight": spec.aux_weight,
            "era_decay_halflife": spec.era_decay_halflife,
            "seed_average_count": int(args.seed_average_count),
            **strict_metrics,
        }
        run_rows.append(row)
        strict_candidates.append((strict_name, strict_df, strict_metrics))
        if wandb_run is not None:
            wandb_run.log({
                "spec_name": spec.name,
                "strict_score": strict_metrics.get("strict_score"),
                "delta_cumsum_end": strict_metrics.get("delta_cumsum_end"),
                "delta_mean": strict_metrics.get("delta_mean"),
                "bmc_mean": strict_metrics.get("bmc_mean"),
                "payout_mean": strict_metrics.get("payout_mean"),
            })

    summary_df = pd.DataFrame(run_rows)
    if not summary_df.empty and "strict_score" in summary_df.columns:
        summary_df = summary_df.sort_values("strict_score", ascending=False)
    feasible_count = int(summary_df["feasible"].sum()) if "feasible" in summary_df.columns else 0

    if not summary_df.empty and "strict_score" in summary_df.columns:
        print("\nTop strict neural CV candidates")
        print(
            summary_df[
                [
                    "model",
                    "arch_type",
                    "era_decay_halflife",
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
        print("\nRaw-only strict neural CV caches")
        print(summary_df.to_string(index=False), flush=True)

    best_name: str | None = None
    best_metrics: dict[str, Any] | None = None
    if "strict_score" in summary_df.columns and not summary_df.empty:
        feasible_df = summary_df[summary_df["feasible"] == True] if "feasible" in summary_df.columns else pd.DataFrame()
        source_df = feasible_df if not feasible_df.empty else summary_df
        if args.selection_objective == "delta_cumsum_end":
            best_name = str(source_df.sort_values("delta_cumsum_end", ascending=False).iloc[0]["model"])
        elif args.selection_objective == "corr_sortino_vs_benchmark":
            best_name = str(source_df.sort_values("delta_sortino", ascending=False).iloc[0]["model"])
        else:
            best_name = str(source_df.sort_values("strict_score", ascending=False).iloc[0]["model"])
        best_df = next(df for name, df, _ in strict_candidates if name == best_name)
        best_metrics = next(m for name, _, m in strict_candidates if name == best_name)
        best_out_path = predictions_dir / f"{args.best_out_name}.parquet"
        best_df.rename(columns={args.benchmark_model: "benchmark_prediction"}).to_parquet(best_out_path, index=False)
        _write_result_json(
            results_dir / f"{args.best_out_name}.json",
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
                "output": {"predictions_file": str(best_out_path.relative_to(experiment_dir.parent))},
            },
        )
        if wandb_run is not None:
            wandb_run.summary["best_model"] = best_name
            for key, value in best_metrics.items():
                if isinstance(value, (int, float, bool)):
                    wandb_run.summary[f"best_{key}"] = value

    summary_path = results_dir / args.summary_name
    _write_result_json(
        summary_path,
        {
            "settings": {
                **summary_settings,
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
                "wandb_project": args.wandb_project,
                "wandb_entity": args.wandb_entity,
                "wandb_mode": args.wandb_mode,
            },
            "top_models": summary_df.to_dict(orient="records"),
            "selected_best_model": best_name,
            "selected_best_metrics": best_metrics,
            "feasible_model_count": feasible_count,
        },
    )
    print(f"\nSaved summary to {summary_path}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
