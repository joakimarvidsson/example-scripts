# HPO v2 optimized config (2026-02-06)
# Best BMC Sharpe: 0.169 from trial #5
# NOTE: n_estimators set to 2000 (HPO early stopping suggested 53, which is unreliable
#       due to era-stride CV creating very different train/val distributions)
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
            # HPO-optimized hyperparameters
            'learning_rate': 0.0457,
            'max_depth': 10,
            'num_leaves': 437,
            'colsample_bytree': 0.644,
            'min_data_in_leaf': 6316,
            'reg_alpha': 0.867,
            'reg_lambda': 0.068,
            'subsample': 0.731,
            # n_estimators: manually set (early stopping unreliable with era-stride CV)
            'n_estimators': 2000,
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
        'results_name': 'lgbm_ender20_downsampled_hpo_v2',
    },
}
