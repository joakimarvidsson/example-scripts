from __future__ import annotations

def build_model(
    model_type: str,
    model_params: dict,
    model_config: dict | None = None,
    *,
    feature_cols: list[str] | None = None,
):
    model_config = model_config or {}
    if model_type == "LGBMRegressor":
        from agents.code.modeling.models.lgbm_regressor import LGBMRegressor
        model = LGBMRegressor(feature_cols=feature_cols, **model_params)
    elif model_type == "MLPRegressor":
        from agents.code.modeling.models.mlp_regressor import MLPRegressor
        model = MLPRegressor(feature_cols=feature_cols, **model_params)
    elif model_type == "XGBRegressor":
        from agents.code.modeling.models.xgboost_regressor import XGBRegressor
        model = XGBRegressor(feature_cols=feature_cols, **model_params)
    elif model_type == "CatBoostRegressor":
        from agents.code.modeling.models.catboost_regressor import CatBoostRegressor
        model = CatBoostRegressor(feature_cols=feature_cols, **model_params)
    else:
        raise ValueError(
            "Unsupported model type: "
            f"{model_type}. Supported types: LGBMRegressor, MLPRegressor, XGBRegressor, CatBoostRegressor"
        )

    target_transform = model_config.get("target_transform")
    if target_transform:
        from agents.code.modeling.utils.target_transforms import TargetTransformWrapper

        model = TargetTransformWrapper(model, target_transform)
    return model
