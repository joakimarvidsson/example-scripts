CONFIG = {
    'data': {
        'data_version': 'v5.2',
        'feature_set': 'all',
        'target_col': 'target',
        'era_col': 'era',
        'embargo_eras': 13,
        'full_data_path': 'numerai/v5.2/downsampled_full.parquet',
        'benchmark_data_path': 'numerai/v5.2/downsampled_full_benchmark_models.parquet',
    },
    'model': {
        'type': 'LGBMRegressor',
        'x_groups': ['features', 'era', 'benchmark_models'],
        'params': {
            'max_depth': 8,
            'n_estimators': 2000,
            'learning_rate': 0.005,
            'num_leaves': 256,
            'colsample_bytree': 0.1,
            'min_data_in_leaf': 5000,
            'n_jobs': -1,
            'random_state': 1337,
        },
    },
    'training': {
        'cv': {
            'enabled': True,
            'n_splits': 5,
            'embargo': 13,
            'mode': 'expanding',
            'min_train_size': 0,
        },
    },
    'preprocessing': {
        'missing_value': 2.0,
        'nan_missing_all_twos': False,
    },
    'output': {
        'output_dir': 'baselines',
        'results_name': 'lgbm_ender20_downsampled',
    },
}
