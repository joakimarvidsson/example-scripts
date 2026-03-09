# Round S35: Parallel Decorrelation-First Raw Search + Feature Orthogonalization

Goal: run two different idea classes in parallel:

- raw model scout with a hard benchmark/example rank-correlation cap
- post-hoc feature-subset orthogonalization on a strong cached raw model

## Branch A: decorrelation-first raw scout

Settings:

- script: `strict_gbt_walkforward_research.py`
- eval step: `8`
- block size: `13`
- max rows per era: `500`
- hard corr cap: `0.995`
- objective: `delta_cumsum_end`

The full scout was launched across XGB, LGBM, and CatBoost faith-heavy specs, but I stopped it after the XGB family checkpoint to avoid leaving a long-running training job open for the rest of the turn.

Finished XGB results:

### `xgb_strict_resid010_smallfaith64_full_d5lr3e2_walkfwd`

- result file: `results/xgb_strict_resid010_smallfaith64_full_d5lr3e2_walkfwd_strict_roundS35_decorfirst.json`
- feature set: `small+faith2:64`
- feasible under cap: `yes`
- delta mean: `0.000173`
- delta cumsum end: `0.013465`
- BMC mean: `0.000770`
- payout mean: `0.025149`
- selected mode: `blend`
- lambda: `0.10`
- corr cap used: `0.994129`

### `xgb_strict_direct_faith96_full_d5lr3e2_walkfwd`

- result file: `results/xgb_strict_direct_faith96_full_d5lr3e2_walkfwd_strict_roundS35_decorfirst.json`
- feature set: `faith2:96`
- feasible under cap: `no`
- delta mean: `0.000132`
- delta cumsum end: `0.010294`
- BMC mean: `0.000094`
- payout mean: `0.023918`
- selected mode: `blend`
- lambda: `0.02`
- corr cap used: `0.999797`

Branch A decision:

- current leader is `xgb_strict_resid010_smallfaith64_full_d5lr3e2_walkfwd`
- it is the first clear cap-feasible additive hit from this round

## Branch B: compact feature-subset orthogonalization scout

Settings:

- script: `tune_corrcap_two_stage.py`
- model file: `cat_strict_resid008_medfaith64_walkfwd_raw_walkfwd.parquet`
- objective: `payout_sortino`
- max corr cap: `0.997`
- feature specs: `none`, `faith2:64`, `small+faith2:64`

Result file:

- `/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/nn_ender20_latest_residualized/results/corrcap_two_stage_roundS35_feature_ortho_cat_compact_2026-03-10.json`

Best cap-feasible setting:

- lambda: `0.05`
- benchmark neutralize: `0.09`
- example neutralize: `0.0`
- feature spec: `small+faith2:64`
- feature neutralize: `0.10`
- ridge alpha: `0.0`
- feature ridge alpha: `0.0`

Metrics:

- payout mean: `0.024344`
- payout sortino: `12.358432`
- BMC mean: `0.000465`
- delta mean: `-0.000095`
- delta cumsum end: `-0.007390`
- corr cap used: `0.996989`

Branch B decision:

- the compact orthogonalization scout found a cap-feasible payout-improving setting
- but it is not additive on CORR delta
- so it is not better than the raw XGB small-faith scout if additive CORR remains a hard requirement

## Overall decision

- best result from this parallel round: `xgb_strict_resid010_smallfaith64_full_d5lr3e2_walkfwd`
- best orthogonalization setting on the dense CatBoost raw model is not additive
- next step should be to scale or confirm the `small+faith2:64` XGB leader, then decide whether the unfinished LGBM/CatBoost branch is still worth resuming
