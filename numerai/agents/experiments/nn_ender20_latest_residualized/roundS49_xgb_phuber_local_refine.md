# Round S49: XGB Pseudo-Huber Local Refinement

## Goal

Run a tight local neighborhood around the newly deployed pseudo-Huber XGBoost incumbent instead of widening to unrelated families.

Neighborhood tested:

- residual scale `0.006`, `0.008`, `0.010` on the incumbent `d5lr3e2` profile
- structural variants at residual scale `0.008`
  - `d4lr5e2`
  - `d3lr7e2`
  - `d6lr2e2`

All runs used `medium+faith2:64`.

## Scout Setup

Raw scout caches:

- eval era step `8`
- block size `26`
- max rows per era `500`

Scout raw-cache summary:

- `/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/nn_ender20_latest_residualized/results/gbt_strict_walkforward_roundS49_phuber_scout_rawonly_summary.json`

Scout compact tuner:

- `/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/nn_ender20_latest_residualized/results/corrcap_two_stage_roundS49_xgb_phuber_scout_light.json`

## Scout Result

Best scout candidate:

- `xgb_strict_resid010_medfaith64_phuber_d5lr3e2_walkfwd_raw_walkfwd_roundS49scout.parquet`
  - `status = positive_delta_feasible`
  - `delta_cumsum_end = 0.0334272`
  - `delta_mean = 0.0004286`
  - `bmc_mean = 0.0005432`
  - `payout_mean = 0.0251190`
  - selected post-processing:
    - `lambda = 0.07`
    - `neutralize_benchmark = 0.0`
    - `neutralize_example = 0.01`

Other outcomes:

- `resid006 d5lr3e2`: negative
- `resid008 d5lr3e2`: positive but weaker than `resid010`
- `resid008 d4lr5e2`: negative
- `resid008 d3lr7e2`: negative
- `resid008 d6lr2e2`: positive but much weaker

## Dense Confirmation

Dense raw-only summary:

- `/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/nn_ender20_latest_residualized/results/gbt_strict_walkforward_roundS49_phuber_dense_rawonly_summary.json`

Dense compact tuner:

- `/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/nn_ender20_latest_residualized/results/corrcap_two_stage_roundS49_xgb_phuber_dense_light.json`

Dense result for the scout winner:

- `xgb_strict_resid010_medfaith64_phuber_d5lr3e2_walkfwd_raw_walkfwd_roundS49dense.parquet`
  - `status = no_positive_delta_feasible`
  - `delta_cumsum_end = -0.0401763`
  - `delta_mean = -0.0002575`
  - `bmc_mean = 0.0002274`
  - `payout_mean = 0.0246154`
  - selected post-processing:
    - `lambda = 0.05`
    - `neutralize_benchmark = 0.0`
    - `neutralize_example = 0.01`

## Decision

- Do not replace the deployed pseudo-Huber XGBoost incumbent with the `resid010` variant.
- The local pseudo-Huber neighborhood again showed the same pattern as earlier rounds:
  - strong scout uplift
  - collapse under dense confirmation
- Keep the currently deployed `resid008` pseudo-Huber XGBoost as the tree incumbent.
