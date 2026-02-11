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
        "type": "TorchMLPRegressor",
        "x_groups": ["features", "era", "benchmark_models"],
        "params": {
            "hidden_layer_sizes": [512, 256, 128],
            "activation": "gelu",
            "learning_rate_init": 0.0003,
            "batch_size": 4096,
            "max_iter": 120,
            "dropout": 0.1,
            "weight_decay": 0.0,
            "early_stopping": True,
            "patience": 10,
            "validation_fraction": 0.1,
            "validation_split_mode": "era",
            "scale": True,
            "random_state": 1337,
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
        "results_name": "torch_mlp_ender20_downsampled",
    },
}
