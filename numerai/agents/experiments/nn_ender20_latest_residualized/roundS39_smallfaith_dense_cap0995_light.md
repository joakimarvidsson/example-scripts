# Round S39: Light Dense Capped Follow-up on Remaining Small-Faith Tree Caches

Goal: close out the remaining dense cached small-faith tree candidates under the same hard `0.995` cap, using a lighter post-processing sweep after the heavier feature-neutralization attempt proved too expensive.

## Settings

- script: `tune_corrcap_two_stage.py`
- hard corr cap: `0.995`
- objective: `delta_cumsum_end`
- search restricted to:
  - lambda blend
  - benchmark neutralization
  - example neutralization
- feature neutralization disabled (`feature_spec = none` only)

## Dense CatBoost small-faith

Result file:

- `results/corrcap_two_stage_roundS39a_cat_smallfaith_dense_cap0995_light_2026-03-11.json`

Best result:

- model file: `cat_strict_resid010_smallfaith64_walkfwd_raw_walkfwd_dense_e4_r700.parquet`
- status: `no_positive_delta_feasible`
- `delta_mean = -0.000219`
- `delta_cumsum_end = -0.034100`
- `bmc_mean = 0.000458`
- `payout_mean = 0.025043`
- `payout_sortino = 10.637247`
- `corr_cap_used = 0.994915`
- selected lambda: `0.09`
- benchmark neutralize: `0.03`
- example neutralize: `0.01`

## Dense LightGBM small-faith

Result file:

- `results/corrcap_two_stage_roundS39b_lgbm_smallfaith_dense_cap0995_light_2026-03-11.json`

Best result:

- model file: `lgbm_dart_strict_resid010_smallfaith64_walkfwd_raw_walkfwd_dense_e4_r700.parquet`
- status: `no_positive_delta_feasible`
- `delta_mean = -0.000362`
- `delta_cumsum_end = -0.056491`
- `bmc_mean = 0.000501`
- `payout_mean = 0.025019`
- `payout_sortino = 10.758547`
- `corr_cap_used = 0.994990`
- selected lambda: `0.09`
- benchmark neutralize: `0.03`
- example neutralize: `0.00`

## Decision

- Neither remaining dense small-faith cached tree candidate survived the hard `0.995` cap.
- The post-processing optimum again stayed in the same light-touch regime: `lambda ~ 0.09` with small benchmark/example neutralization and no feature neutralization.
- At this point the dense capped tree branch is effectively exhausted for the currently cached CatBoost and LightGBM candidates.
