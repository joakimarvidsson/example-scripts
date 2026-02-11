from __future__ import annotations


class MLPRegressor:
    """Minimal wrapper exposing fit/predict for the training pipeline."""

    def __init__(self, feature_cols: list[str] | None = None, **params):
        try:
            from sklearn.neural_network import MLPRegressor as SklearnMLPRegressor
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for MLPRegressor. "
                "Install with `.venv/bin/pip install scikit-learn`."
            ) from exc

        self._feature_cols = feature_cols
        self._params = dict(params)
        scale = bool(self._params.pop("scale", True))

        self._mlp = SklearnMLPRegressor(**self._params)
        if scale:
            self._model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("mlp", self._mlp),
                ]
            )
        else:
            self._model = self._mlp

    def fit(self, X, y, **kwargs):
        X = self._filter_features(X, self._feature_cols)
        X_values = X.to_numpy() if hasattr(X, "to_numpy") else X
        self._model.fit(X_values, y, **kwargs)
        return self

    def predict(self, X):
        X = self._filter_features(X, self._feature_cols)
        X_values = X.to_numpy() if hasattr(X, "to_numpy") else X
        return self._model.predict(X_values)

    @staticmethod
    def _filter_features(X, feature_cols):
        if not feature_cols or not hasattr(X, "columns"):
            return X
        missing = [col for col in feature_cols if col not in X.columns]
        if missing:
            raise ValueError(
                f"Missing feature columns for MLPRegressor: {missing[:5]}"
                + ("..." if len(missing) > 5 else "")
            )
        return X[feature_cols]

    def __getattr__(self, name: str):
        return getattr(self._model, name)
