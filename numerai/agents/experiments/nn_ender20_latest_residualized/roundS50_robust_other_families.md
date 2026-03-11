# Round S50: Robust Non-XGB Families

## Deployed Model Status

The currently assigned pseudo-Huber XGBoost on `truecontribution` is healthy:

- current round: `1221`
- round open time: `2026-03-11T13:00:00Z`
- round close time: `2026-03-12T13:00:00Z`
- latest compute pickle diagnostics: `succeeded`
- latest trigger status: `submission_succeeded`

This round treated that deployed XGBoost as the incumbent and searched for a genuinely different raw family.

## Goal

Complete the unfinished robust-loss non-XGB med-faith neighborhood already wired into the strict walk-forward harness:

- `lgbm_gbdt_strict_resid008_medfaith64_l1_walkfwd`
- `lgbm_gbdt_strict_resid008_medfaith64_huber_walkfwd`
- `cat_strict_resid008_medfaith64_mae_walkfwd`
- `cat_strict_resid008_medfaith64_logcosh_walkfwd`

All runs used `medium+faith2:64`.

## Scout Setup

Raw scout caches:

- eval era step `8`
- block size `26`
- max rows per era `500`

Scout raw-cache summary:

- `/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/nn_ender20_latest_residualized/results/gbt_strict_walkforward_roundS50_robust_otherfamilies_scout_rawonly_summary.json`

Scout compact tuner:

- `/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/nn_ender20_latest_residualized/results/corrcap_two_stage_roundS50_robust_otherfamilies_scout_light.json`

## Scout Result

Negative at scout scale:

- `lgbm_gbdt_strict_resid008_medfaith64_l1_walkfwd`
  - `delta_cumsum_end = -0.034519`
- `lgbm_gbdt_strict_resid008_medfaith64_huber_walkfwd`
  - `delta_cumsum_end = -0.020174`
- `cat_strict_resid008_medfaith64_mae_walkfwd`
  - `delta_cumsum_end = -0.015432`

Best scout candidate:

- `cat_strict_resid008_medfaith64_logcosh_walkfwd_raw_walkfwd_roundS50scout.parquet`
  - `status = positive_delta_feasible`
  - `delta_cumsum_end = 0.0251410`
  - `delta_mean = 0.0003223`
  - `bmc_mean = 0.0006316`
  - `payout_mean = 0.0252143`
  - selected post-processing:
    - `lambda = 0.08`
    - `neutralize_benchmark = 0.0`
    - `neutralize_example = 0.03`

This was strong enough to justify dense confirmation.

## Dense Confirmation

Dense raw-only summary:

- `/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/nn_ender20_latest_residualized/results/gbt_strict_walkforward_roundS50_robust_otherfamilies_dense_rawonly_summary.json`

Dense compact tuner:

- `/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/nn_ender20_latest_residualized/results/corrcap_two_stage_roundS50_robust_otherfamilies_dense_light.json`

Dense result for `LogCosh`:

- `cat_strict_resid008_medfaith64_logcosh_walkfwd_raw_walkfwd_roundS50dense.parquet`
  - `status = no_positive_delta_feasible`
  - `delta_cumsum_end = -0.0214238`
  - `delta_mean = -0.0001373`
  - `bmc_mean = 0.0002506`
  - `payout_mean = 0.0248177`
  - selected post-processing:
    - `lambda = 0.04`
    - `neutralize_benchmark = 0.03`
    - `neutralize_example = 0.07`

## Decision

- Do not promote any S50 non-XGB robust-loss candidate.
- Keep the deployed pseudo-Huber XGBoost as the incumbent.
- `CatBoost LogCosh` was the only scout-positive non-XGB raw candidate in this batch, but it failed dense confirmation.
