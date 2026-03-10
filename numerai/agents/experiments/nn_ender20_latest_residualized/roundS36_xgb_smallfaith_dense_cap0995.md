# Round S36: Dense Confirm for `xgb_strict_resid010_smallfaith64_full_d5lr3e2_walkfwd`

Goal: scale the `Round S35` capped XGB winner from scout settings to the denser validation profile before promoting it.

## Dense profile

- eval step: `4`
- block size: `26`
- max rows per era: `700`
- hard corr cap: `0.995`
- objective: `delta_cumsum_end`

Dense raw cache:

- `predictions/xgb_strict_resid010_smallfaith64_full_d5lr3e2_walkfwd_raw_walkfwd_dense_e4_r700_cap0995.parquet`

Dense strict result:

- `/Users/joakim/Documents/Projects/Numerai/example-scripts-codex-faith-scale/numerai/agents/experiments/nn_ender20_latest_residualized/results/xgb_strict_resid010_smallfaith64_full_d5lr3e2_walkfwd_strict_dense_e4_r700_cap0995.json`

Metrics:

- `delta_mean = -0.000257`
- `delta_cumsum_end = -0.040159`
- `delta_sortino = -0.124095`
- `bmc_mean = 0.000147`
- `payout_mean = 0.024403`
- `corr_cap_used = 0.998654`
- `feasible = false`

Selected post-processing:

- mode: `blend`
- lambda: `0.05`
- benchmark neutralization: `0.0`
- example neutralization: `0.0`

## Decision

- The `small+faith2:64` XGB winner did **not** survive dense confirmation.
- It reverted to negative cumulative CORR delta and failed the hard `0.995` corr cap.
- Do not promote this model.

## Follow-up

- Reopened the unfinished capped LGBM/CatBoost scout branch after the XGB failure.
- That follow-up run was started but stopped before any result file completed, to avoid leaving a long-running training process open at the end of the turn.
- Resume next from:
  - `lgbm_dart_strict_resid008_medfaith64_walkfwd`
  - `cat_strict_resid008_medfaith64_walkfwd`
  - `cat_strict_resid008_smallfaith64_walkfwd`
  - `cat_strict_resid010_smallfaith64_walkfwd`
