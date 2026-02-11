from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class _SplitIndices:
    train_idx: np.ndarray
    val_idx: np.ndarray | None


class TorchMLPRegressor:
    """PyTorch MLP wrapper compatible with the existing fit/predict pipeline."""

    def __init__(self, feature_cols: list[str] | None = None, **params):
        try:
            import torch
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise ImportError(
                "TorchMLPRegressor requires torch and scikit-learn. "
                "Install with `.venv/bin/pip install torch scikit-learn`."
            ) from exc

        self._torch = torch
        self._nn = torch.nn
        self._optim = torch.optim
        self._feature_cols = feature_cols
        self._params = dict(params)
        self._scaler_cls = StandardScaler

        hidden_sizes = self._params.pop("hidden_layer_sizes", (512, 256, 128))
        self._hidden_layer_sizes = tuple(int(v) for v in hidden_sizes)
        if not self._hidden_layer_sizes:
            raise ValueError("hidden_layer_sizes must contain at least one layer size.")

        self._activation = str(self._params.pop("activation", "gelu")).lower()
        self._dropout = float(self._params.pop("dropout", 0.1))
        self._learning_rate = float(self._params.pop("learning_rate_init", 3e-4))
        self._weight_decay = float(self._params.pop("weight_decay", 0.0))
        self._batch_size = int(self._params.pop("batch_size", 4096))
        self._max_epochs = int(self._params.pop("max_iter", 120))
        self._early_stopping = bool(self._params.pop("early_stopping", True))
        self._early_stopping_mode = str(
            self._params.pop("early_stopping_mode", "row")
        ).lower()
        self._validation_fraction = float(self._params.pop("validation_fraction", 0.1))
        self._val_era_fraction = float(self._params.pop("val_era_fraction", 0.1))
        self._patience = int(self._params.pop("patience", 10))
        self._min_delta = float(self._params.pop("min_delta", 1e-5))
        self._clip_grad_norm = self._params.pop("clip_grad_norm", None)
        self._scale = bool(self._params.pop("scale", True))
        self._prediction_batch_size = int(
            self._params.pop("prediction_batch_size", 8192)
        )
        self._random_state = int(self._params.pop("random_state", 1337))
        self._device_name = str(self._params.pop("device", "cpu"))
        self._verbose = bool(self._params.pop("verbose", False))
        self._dtype = self._params.pop("dtype", "float32")

        if self._params:
            unknown = ", ".join(sorted(self._params.keys()))
            raise ValueError(f"Unknown TorchMLPRegressor params: {unknown}")

        self._model = None
        self._scaler = None
        self._train_history: dict[str, float | int] = {}

    def fit(self, X, y, **kwargs):
        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise ValueError(f"TorchMLPRegressor.fit got unexpected kwargs: {unknown}")

        X_df = self._ensure_frame(X)
        y_series = self._ensure_series(y, X_df.index)

        features = self._filter_features(X_df, self._feature_cols)
        x_np = features.to_numpy(dtype=np.float32, copy=False)
        y_np = y_series.to_numpy(dtype=np.float32, copy=False).reshape(-1, 1)

        split = self._build_split_indices(X_df)
        train_idx = split.train_idx
        val_idx = split.val_idx

        x_train = x_np[train_idx]
        x_val = x_np[val_idx] if val_idx is not None else None
        y_train = y_np[train_idx]
        y_val = y_np[val_idx] if val_idx is not None else None

        if self._scale:
            self._scaler = self._scaler_cls()
            x_train = self._scaler.fit_transform(x_train).astype(np.float32, copy=False)
            if x_val is not None:
                x_val = self._scaler.transform(x_val).astype(np.float32, copy=False)
        else:
            self._scaler = None

        torch = self._torch
        device = torch.device(self._device_name)
        model = self._build_network(x_train.shape[1]).to(device)
        criterion = self._nn.MSELoss()
        optimizer = self._optim.AdamW(
            model.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)

        if x_val is not None and y_val is not None:
            x_val_tensor = torch.tensor(x_val, dtype=torch.float32, device=device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device)
        else:
            x_val_tensor = None
            y_val_tensor = None

        n_train = x_train_tensor.shape[0]
        indices = np.arange(n_train)
        rng = np.random.default_rng(self._random_state)

        best_val = float("inf")
        best_state = None
        best_epoch = -1
        patience_left = self._patience

        for epoch in range(self._max_epochs):
            model.train()
            rng.shuffle(indices)

            epoch_loss = 0.0
            total = 0
            for start in range(0, n_train, self._batch_size):
                batch_idx = indices[start : start + self._batch_size]
                xb = x_train_tensor[batch_idx]
                yb = y_train_tensor[batch_idx]

                optimizer.zero_grad(set_to_none=True)
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                if self._clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(self._clip_grad_norm)
                    )
                optimizer.step()

                batch_size = int(xb.shape[0])
                epoch_loss += float(loss.item()) * batch_size
                total += batch_size

            train_loss = epoch_loss / max(total, 1)

            if x_val_tensor is None or y_val_tensor is None:
                if self._verbose:
                    print(f"epoch={epoch:03d} train_loss={train_loss:.6f}")
                continue

            model.eval()
            with torch.no_grad():
                val_preds = model(x_val_tensor)
                val_loss = float(criterion(val_preds, y_val_tensor).item())

            if self._verbose:
                print(
                    f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
                )

            if val_loss < (best_val - self._min_delta):
                best_val = val_loss
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                patience_left = self._patience
            else:
                patience_left -= 1
                if self._early_stopping and patience_left <= 0:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        else:
            best_val = float("nan")

        self._model = model
        self._train_history = {
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "epochs_ran": epoch + 1 if self._max_epochs > 0 else 0,
        }
        return self

    def predict(self, X):
        if self._model is None:
            raise ValueError("Model is not fit yet.")

        X_df = self._ensure_frame(X)
        features = self._filter_features(X_df, self._feature_cols)
        x_np = features.to_numpy(dtype=np.float32, copy=False)
        if self._scaler is not None:
            x_np = self._scaler.transform(x_np).astype(np.float32, copy=False)

        torch = self._torch
        device = torch.device(self._device_name)
        self._model.eval()

        preds = np.empty(x_np.shape[0], dtype=np.float32)
        with torch.no_grad():
            for start in range(0, x_np.shape[0], self._prediction_batch_size):
                end = start + self._prediction_batch_size
                xb = torch.tensor(x_np[start:end], dtype=torch.float32, device=device)
                out = self._model(xb).detach().cpu().numpy().reshape(-1)
                preds[start:end] = out

        return preds.astype(np.float64, copy=False)

    def _build_network(self, input_dim: int):
        if input_dim <= 0:
            raise ValueError("TorchMLPRegressor requires at least one input feature.")

        layers = []
        in_dim = input_dim
        for hidden_dim in self._hidden_layer_sizes:
            layers.append(self._nn.Linear(in_dim, hidden_dim))
            layers.append(self._activation_layer())
            if self._dropout > 0.0:
                layers.append(self._nn.Dropout(self._dropout))
            in_dim = hidden_dim
        layers.append(self._nn.Linear(in_dim, 1))
        return self._nn.Sequential(*layers)

    def _activation_layer(self):
        if self._activation == "gelu":
            return self._nn.GELU()
        if self._activation == "relu":
            return self._nn.ReLU()
        if self._activation == "tanh":
            return self._nn.Tanh()
        if self._activation == "silu":
            return self._nn.SiLU()
        raise ValueError(
            f"Unsupported activation '{self._activation}'. Use gelu/relu/tanh/silu."
        )

    def _build_split_indices(self, X_df: pd.DataFrame) -> _SplitIndices:
        n_rows = len(X_df)
        if n_rows < 2:
            idx = np.arange(n_rows, dtype=np.int64)
            return _SplitIndices(train_idx=idx, val_idx=None)

        if not self._early_stopping:
            idx = np.arange(n_rows, dtype=np.int64)
            return _SplitIndices(train_idx=idx, val_idx=None)

        mode = self._early_stopping_mode
        if mode in {"era", "era-wise", "era_wise"}:
            split = self._era_split_indices(X_df)
            if split is not None:
                return split
            mode = "row"

        if mode not in {"row", "row-wise", "row_wise"}:
            raise ValueError(
                f"Unknown early_stopping_mode '{self._early_stopping_mode}'. "
                "Use row or era."
            )
        return self._row_split_indices(n_rows)

    def _row_split_indices(self, n_rows: int) -> _SplitIndices:
        n_val = int(round(n_rows * self._validation_fraction))
        n_val = min(max(n_val, 1), n_rows - 1)

        rng = np.random.default_rng(self._random_state)
        perm = rng.permutation(n_rows)
        val_idx = np.sort(perm[:n_val])
        train_idx = np.sort(perm[n_val:])
        return _SplitIndices(train_idx=train_idx, val_idx=val_idx)

    def _era_split_indices(self, X_df: pd.DataFrame) -> _SplitIndices | None:
        if "era" not in X_df.columns:
            return None

        eras = X_df["era"]
        unique_eras = sorted(set(eras), key=_era_sort_key)
        if len(unique_eras) < 2:
            return None

        n_val_eras = int(round(len(unique_eras) * self._val_era_fraction))
        n_val_eras = min(max(n_val_eras, 1), len(unique_eras) - 1)
        val_eras = set(unique_eras[-n_val_eras:])
        val_mask = eras.isin(val_eras).to_numpy()
        train_mask = ~val_mask
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            return None
        return _SplitIndices(
            train_idx=np.flatnonzero(train_mask),
            val_idx=np.flatnonzero(val_mask),
        )

    @staticmethod
    def _filter_features(X: pd.DataFrame, feature_cols: Iterable[str] | None) -> pd.DataFrame:
        if not feature_cols:
            return X
        missing = [col for col in feature_cols if col not in X.columns]
        if missing:
            raise ValueError(
                f"Missing feature columns for TorchMLPRegressor: {missing[:5]}"
                + ("..." if len(missing) > 5 else "")
            )
        return X[list(feature_cols)]

    @staticmethod
    def _ensure_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

    @staticmethod
    def _ensure_series(y, index) -> pd.Series:
        if isinstance(y, pd.Series):
            return y
        return pd.Series(y, index=index)

    def __getattr__(self, name: str):
        if name in {"_model", "_train_history"}:
            raise AttributeError(name)
        return getattr(self._model, name)


def _era_sort_key(era):
    try:
        return int(era)
    except (TypeError, ValueError):
        return str(era)
