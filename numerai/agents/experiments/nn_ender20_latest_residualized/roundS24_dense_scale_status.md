# Round S24 Dense Scale Status

Date: 2026-03-09

## Dense profile

- `eval-era-step=4`
- `max_rows_per_era=700`
- uncapped selection (`max_corr_with_benchmark=2.0`, `max_corr_with_example=2.0`)

## Completed dense result

Model: `lgbm_dart_strict_resid010_smallfaith64_walkfwd_strict_dense_e4_r700`

- `delta_mean = -0.0000813`
- `delta_cumsum_end = -0.0126777`
- `bmc_mean = 0.0002861`
- `payout_mean = 0.0248414`
- verdict: scout winner did not survive denser evaluation

## Partial dense challenger state

Model: `cat_strict_resid008_medfaith64_walkfwd`

- dense raw cache completed:
  - `cat_strict_resid008_medfaith64_walkfwd_raw_walkfwd_dense_e4_r700.parquet`
- strict scoring was started but not completed in this round before stop

## Decision

Do not proceed to tree+MLP payout blending off the dense LightGBM result.

Finish dense CatBoost strict scoring first, then choose the surviving dense tree candidate before any blend optimization.
