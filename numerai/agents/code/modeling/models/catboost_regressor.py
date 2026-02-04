from __future__ import annotations


class CatBoostRegressor:
    """Minimal wrapper exposing fit/predict for the training pipeline."""

    def __init__(self, feature_cols: list[str] | None = None, **params):
        try:
            from catboost import CatBoostRegressor as _CatBoostRegressor
        except ImportError as exc:
            raise ImportError(
                "catboost is required for CatBoostRegressor. Install with `.venv/bin/pip install catboost`."
            ) from exc
        self._CatBoostRegressor = _CatBoostRegressor
        # Set silent mode by default
        params.setdefault("verbose", 0)
        self._params = dict(params)
        self._model = _CatBoostRegressor(**params)
        self._feature_cols = feature_cols

    def fit(self, X, y, **kwargs):
        X = self._filter_features(X, self._feature_cols)
        self._model.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        X = self._filter_features(X, self._feature_cols)
        return self._model.predict(X)

    @staticmethod
    def _filter_features(X, feature_cols):
        if not feature_cols or not hasattr(X, "columns"):
            return X
        missing = [col for col in feature_cols if col not in X.columns]
        if missing:
            raise ValueError(
                f"Missing feature columns for CatBoostRegressor: {missing[:5]}"
                + ("..." if len(missing) > 5 else "")
            )
        return X[feature_cols]

    def __getattr__(self, name: str):
        return getattr(self._model, name)
