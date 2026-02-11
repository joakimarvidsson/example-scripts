CONFIG = {
    "data": {
        "data_version": "v5.2",
        "feature_set": "all",
        "target_col": "target",
        "era_col": "era",
        "embargo_eras": 13,
        "full_data_path": "numerai/v5.2/downsampled_full.parquet",
        "benchmark_data_path": "numerai/v5.2/downsampled_full_benchmark_models.parquet",
    },
    "model": {
        "type": "CatBoostRegressor",
        "x_groups": ["features", "era", "benchmark_models"],
        "params": {
            "iterations": 2000,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
            "subsample": 0.8,
            "colsample_bylevel": 0.1,
            "random_seed": 1337,
            "verbose": 0,
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
        "results_name": "catboost_ender20_downsampled",
    },
}
