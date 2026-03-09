# Round S32: Optuna shallow/high-lr tree scout

Goal: test shallower tree models with higher learning rates using Optuna and check whether they stay additive on `target_ender_20` versus `v52_lgbm_ender20`.

## xgb

- Best scout trial: `2`
- Scout feature set: `medium`
- Scout residual scale: `0.006`
- Scout delta cumsum end: `0.023479`
- Scout bmc mean: `0.000330`
- Dense confirm delta cumsum end: `-0.064806`
- Dense confirm bmc mean: `-0.000020`
- Dense confirm payout mean: `0.023995`
- Dense confirm additive feasible: `False`

## lgbm

- Best scout trial: `0`
- Scout feature set: `medium`
- Scout residual scale: `0.006`
- Scout delta cumsum end: `0.017890`
- Scout bmc mean: `0.000330`
- Dense confirm delta cumsum end: `-0.002480`
- Dense confirm bmc mean: `0.000211`
- Dense confirm payout mean: `0.024840`
- Dense confirm additive feasible: `False`

## catboost

- Best scout trial: `1`
- Scout feature set: `small`
- Scout residual scale: `0.0`
- Scout delta cumsum end: `0.007941`
- Scout bmc mean: `0.000216`
- Dense confirm delta cumsum end: `-0.074316`
- Dense confirm bmc mean: `-0.000132`
- Dense confirm payout mean: `0.023662`
- Dense confirm additive feasible: `False`
