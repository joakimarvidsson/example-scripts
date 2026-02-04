CONFIG = {
    'data': {
        'data_version': 'v5.2',
        'feature_set': 'all',
        'target_col': 'target',
        'era_col': 'era',
        'embargo_eras': 13,
    },
    'model': {
        'type': 'XGBRegressor',
        'x_groups': ['features', 'era', 'benchmark_models'],
        'params': {
            'n_estimators': 2000,
            'max_depth': 6,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 1337,
            'n_jobs': -1,
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
        'results_name': 'xgb_ender20_baseline',
    },
}
