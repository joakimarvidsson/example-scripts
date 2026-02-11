CONFIG = {
    'data': {
        'data_version': 'v5.2',
        'feature_set': 'medium',  # Smaller feature set for speed
        'target_col': 'target',
        'era_col': 'era',
        'embargo_eras': 13,
        'full_data_path': 'numerai/v5.2/downsampled_full.parquet',
        'benchmark_data_path': 'numerai/v5.2/downsampled_full_benchmark_models.parquet',
    },
    'model': {
        'type': 'LGBMRegressor',
        'x_groups': ['features'],
        'params': {
            'max_depth': 5,
            'n_estimators': 100,  # Quick test
            'learning_rate': 0.05,
            'num_leaves': 31,
            'colsample_bytree': 0.5,
            'min_data_in_leaf': 5000,
            'n_jobs': -1,
            'random_state': 1337,
            'verbose': -1,  # Suppress warnings
        },
    },
    'training': {
        'cv': {
            'enabled': True,
            'n_splits': 3,  # Fewer folds for speed
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
        'results_name': 'lgbm_quick_test',
    },
}
