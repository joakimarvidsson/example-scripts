# Round S37: Compact Capped Cache Scout on Existing Tree Raw Predictions

Goal: resume the capped LGBM/CatBoost branch without retraining, using the existing raw walk-forward caches and a compact two-stage post-processing sweep.

## Settings

- script: `tune_corrcap_two_stage.py`
- validation slice: cached walk-forward OOF from the existing raw prediction files
- hard corr cap: `0.995`
- objective: `delta_cumsum_end`
- feature specs searched: `none`, `small+faith2:64`
- feature neutralization proportions: `0.00`, `0.05`, `0.10`
- benchmark/example neutralization: compact coarse+fine local sweep

Result file:

- `results/corrcap_two_stage_roundS37_tree_cache_compact_2026-03-11.json`

## Finished models

### `cat_strict_resid008_medfaith64_walkfwd_raw_walkfwd.parquet`

- status: `positive_delta_feasible`
- `delta_mean = 0.000320`
- `delta_cumsum_end = 0.024949`
- `bmc_mean = 0.000787`
- `payout_mean = 0.025292`
- `corr_cap_used = 0.994154`
- selected lambda: `0.10`
- benchmark neutralize: `0.00`
- example neutralize: `0.01`
- feature spec: `none`

### `cat_strict_resid010_smallfaith64_walkfwd_raw_walkfwd.parquet`

- status: `positive_delta_feasible`
- `delta_mean = 0.000264`
- `delta_cumsum_end = 0.020571`
- `bmc_mean = 0.000693`
- `payout_mean = 0.025388`
- `corr_cap_used = 0.994952`
- selected lambda: `0.09`
- benchmark neutralize: `0.02`
- example neutralize: `0.02`
- feature spec: `none`

### `lgbm_dart_strict_resid008_medfaith64_walkfwd_raw_walkfwd.parquet`

- status: `no_positive_delta_feasible`
- `delta_mean = -0.000054`
- `delta_cumsum_end = -0.008414`
- `bmc_mean = 0.000700`
- `payout_mean = 0.025791`
- `corr_cap_used = 0.994963`

## Decision

- Best scout winner from this round: `cat_strict_resid008_medfaith64_walkfwd_raw_walkfwd`.
- Both CatBoost caches were cap-feasible and additive at scout scale.
- The compact sweep again preferred very light post-processing, with `feature_spec = none` on the best rows.
- Next step: confirm the best scout winner on the dense cached prediction file before considering promotion.
