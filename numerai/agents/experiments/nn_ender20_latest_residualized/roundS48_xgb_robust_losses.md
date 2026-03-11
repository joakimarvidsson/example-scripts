# Round S48: Robust-Loss Tree Pivot

## Goal

Pivot away from the exhausted plain-MLP neighborhood and test a genuinely different raw-model idea: robust-loss tree objectives around the stronger `medium+faith2:64` residualized tree neighborhood.

## Implementation

Added robust-loss specs to the strict GBT harness:

- `xgb_strict_resid008_medfaith64_abs_d5lr3e2_walkfwd`
- `xgb_strict_resid008_medfaith64_phuber_d5lr3e2_walkfwd`
- `lgbm_gbdt_strict_resid008_medfaith64_l1_walkfwd`
- `lgbm_gbdt_strict_resid008_medfaith64_huber_walkfwd`
- `cat_strict_resid008_medfaith64_mae_walkfwd`
- `cat_strict_resid008_medfaith64_logcosh_walkfwd`

Also added a `--skip-strict-score` mode to the strict GBT harness so raw walk-forward caches can be generated independently of the slower strict candidate scorer.

## Scout Result: XGB Robust Losses

Scout scoring artifact:

- `/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/nn_ender20_latest_residualized/results/corrcap_two_stage_roundS48_xgb_robustloss_light.json`

Completed XGB results:

- `xgb_strict_resid008_medfaith64_abs_d5lr3e2_walkfwd_raw_walkfwd_roundS48robust.parquet`
  - `delta_cumsum_end = -0.021903`
  - `bmc_mean = 0.000223` (compact tuner run)
  - verdict: reject

- `xgb_strict_resid008_medfaith64_phuber_d5lr3e2_walkfwd_raw_walkfwd_roundS48robust.parquet`
  - `delta_cumsum_end = 0.004760`
  - `delta_mean = 0.0000610`
  - `bmc_mean = 0.0002665`
  - `payout_mean = 0.0242651`
  - `corr_cap_used = 0.9986589`
  - verdict: dense-confirm

The LightGBM/CatBoost robust-loss half of the combined raw-cache batch was not completed in this round after the split, so this note only records the completed XGB evidence.

## Dense Confirmation: XGB Pseudo-Huber

Dense raw-only cache summary:

- `/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/nn_ender20_latest_residualized/results/gbt_strict_walkforward_roundS48_xgb_phuber_dense_rawonly_summary.json`

Dense compact scoring artifact:

- `/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/nn_ender20_latest_residualized/results/corrcap_two_stage_roundS48_xgb_phuber_dense_light.json`

Dense result:

- `xgb_strict_resid008_medfaith64_phuber_d5lr3e2_walkfwd_raw_walkfwd_roundS48phuberdense.parquet`
  - `delta_cumsum_end = 0.0174684`
  - `delta_mean = 0.0001120`
  - `bmc_mean = 0.0003732`
  - `payout_mean = 0.0252364`
  - `corr_cap_used = 0.9986303`
  - selected post-processing:
    - `lambda = 0.05`
    - `neutralize_benchmark = 0.0`
    - `neutralize_example = 0.02`

## Comparison To Current Dense Tree Incumbent

Current dense CatBoost incumbent:

- `/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/nn_ender20_latest_residualized/results/cat_strict_resid008_medfaith64_walkfwd_strict_dense_e4_r700.json`
  - `delta_cumsum_end = 0.0035496`
  - `delta_mean = 0.0000228`
  - `bmc_mean = 0.0003604`
  - `payout_mean = 0.0251358`

Dense pseudo-Huber XGB is better on all four headline OOS metrics on this dense validation slice:

- higher `delta_cumsum_end`
- higher `delta_mean`
- slightly higher `bmc_mean`
- slightly higher `payout_mean`

## Decision

- Promote `xgb_strict_resid008_medfaith64_phuber_d5lr3e2_walkfwd` to the next stage.
- Do not promote the absolute-error XGB variant.
- Keep the CatBoost model as the current deployed incumbent until this pseudo-Huber XGB is scaled to the live-training profile and packaged cleanly.

## Live Training, Pickle Export, And Upload

Live artifact training was completed on the full available era range using the benchmark-only deployable tuning:

- training profile: `live`
- eras: `0158..1199`
- sampled rows after per-era cap: `833600`
- feature set: `medium+faith2:64`
- residual scale: `0.008`
- XGBoost objective: `reg:pseudohubererror`
- deployable post-processing:
  - `lambda = 0.05`
  - `neutralize_benchmark = 0.02`
  - `neutralize_example = 0.0`

Artifacts:

- `/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/nn_ender20_latest_residualized/models/xgb_strict_resid008_medfaith64_phuber_lam005_nb002_live_artifacts/metadata.json`
- `/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/nn_ender20_latest_residualized/models/xgb_strict_resid008_medfaith64_phuber_lam005_nb002_live_py312_20260311.pkl`

The Python 3.12 cloudpickle was smoke-tested locally with the Numerai upload venv and a sample `predict(features, benchmark_models)` call.

Comparison plot for the dense OOS slice:

- `/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/nn_ender20_latest_residualized/plots/v52_lgbm_ender20_vs_cat_dense_vs_xgb_phuber_dense_liveproxy_dark.png`
- metrics:
  - benchmark `corr_cumsum_end = 5.09086`
  - CatBoost dense incumbent `delta_cumsum_end = 0.00355`
  - pseudo-Huber XGB live proxy `delta_cumsum_end = 0.01746`

Upload state at the end of this round:

- target slot: `truecontribution`
- model id: `521734a9-af12-41b4-b511-e7d726878497`
- uploaded pickle id: `690dcb40-e241-4ef4-b1cf-f8455eab2d6e`
- Numerai status: `validationStatus = validating`, `triggerStatus = queued`

The upload was started successfully, but Numerai validation had not reached a terminal state by the time this round was checkpointed, so assignment was deferred.
