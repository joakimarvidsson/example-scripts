# Round S33: Optuna dense-narrow LGBM

Goal: test shallower tree models with higher learning rates using Optuna and check whether they stay additive on `target_ender_20` versus `v52_lgbm_ender20`.

## lgbm

- Best scout trial: `1`
- Scout feature set: `medium`
- Scout residual scale: `0.006`
- Scout delta cumsum end: `0.009443`
- Scout bmc mean: `0.000089`
- Dense confirm delta cumsum end: `-0.012311`
- Dense confirm bmc mean: `0.000036`
- Dense confirm payout mean: `0.024355`
- Dense confirm additive feasible: `False`
