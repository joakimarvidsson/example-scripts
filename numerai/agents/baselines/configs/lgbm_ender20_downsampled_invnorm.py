# LGBM with inverse-normal benchmark subtraction
# Different target transform increases ensemble diversity
CONFIG = {
    "data": {
        "data_version": "v5.2",
        "feature_set": "all",
        "target_col": "target",
        "era_col": "era",
        "embargo_eras": 13,
        "full_data_path": "numerai/v5.2/downsampled_full.parquet",
        "benchmark_data_path": "numerai/v5.2/downsampled_full_benchmark_models.parquet",
        "benchmark_model": "v52_lgbm_ender20",
    },
    "model": {
        "type": "LGBMRegressor",
        "x_groups": ["features", "era", "benchmark_models"],
        "params": {
            "learning_rate": 0.0899,
            "max_depth": 6,
            "num_leaves": 173,
            "colsample_bytree": 0.772,
            "min_data_in_leaf": 5929,
            "reg_alpha": 0.0188,
            "reg_lambda": 0.297,
            "subsample": 0.711,
            "n_estimators": 2000,
            "n_jobs": -1,
            "random_state": 1337,
        },
        "target_transform": {
            "type": "subtract_benchmark_inverse_normal",
            "benchmark_col": "v52_lgbm_ender20",
            "era_col": "era",
            "per_era": True,
            "scale": 0.07,
        },
    },
    "training": {
        "cv": {
            "enabled": True,
            "n_splits": 5,
            "embargo": 13,
            "mode": "expanding",
            "min_train_size": 0,
        },
    },
    "preprocessing": {"missing_value": 2.0, "nan_missing_all_twos": False},
    "output": {
        "output_dir": "baselines",
        "results_name": "lgbm_ender20_downsampled_invnorm",
    },
}
