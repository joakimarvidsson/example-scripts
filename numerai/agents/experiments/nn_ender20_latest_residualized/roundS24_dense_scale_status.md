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
- dense strict scoring completed:
  - `cat_strict_resid008_medfaith64_walkfwd_strict_dense_e4_r700`
  - `delta_mean = 0.0000228`
  - `delta_cumsum_end = 0.0035496`
  - `bmc_mean = 0.0003604`
  - `payout_mean = 0.0251358`
  - verdict: survives dense evaluation, but only marginally

Model: `cat_strict_resid010_smallfaith64_walkfwd`

- dense raw cache completed:
  - `cat_strict_resid010_smallfaith64_walkfwd_raw_walkfwd_dense_e4_r700.parquet`
- dense strict scoring completed:
  - `cat_strict_resid010_smallfaith64_walkfwd_strict_dense_e4_r700`
  - `delta_mean = 0.0000219`
  - `delta_cumsum_end = 0.0034174`
  - `bmc_mean = 0.0002735`
  - `payout_mean = 0.0249306`
  - verdict: survives dense evaluation, but is slightly worse than the medfaith64 dense incumbent on delta, BMC, and payout

## Dense tree + MLP blend check

Dense tree winner:

- `cat_strict_resid008_medfaith64_walkfwd_strict_dense_e4_r700`

MLP partners checked:

1. `torch_resid_latest_ender20_roundC_blend_noi12_base_rowfull_refined`
2. `torch_mlp_resid_latest_ender20_roundM6_base_w80_m3_w20`

Best blend found:

- `95%` dense CatBoost
- `5%` `torch_mlp_resid_latest_ender20_roundM6_base_w80_m3_w20`
- artifact:
  - `blend_cat_strict_resid008_medfaith64_walkfwd_strict_dense_e4_r700_w95_torch_mlp_resid_latest_ender20_roundm6_base_w80_m3_w20_minera577.parquet`
- metrics:
  - `corr_mean = 0.0326598`
  - `corr_sortino = 3.7650`
  - `bmc_mean = 0.0003706`
  - `payout_mean = 0.0251497`
  - `payout_sharpe = 1.8492`

Interpretation:

- The best blend is only a very small adjustment on top of the dense CatBoost standalone model.
- `roundM6_base_w80_m3_w20` is a slightly better blend partner than the older `roundC` row-full blend in this dense comparison.
- The blend improvement is real but small, so this is not yet a major step-change model.

## Early vs late split sanity check

Dense incumbent: `cat_strict_resid008_medfaith64_walkfwd_strict_dense_e4_r700`

- eras `577-889`
  - `delta_cumsum_end = 0.0224311`
  - `bmc_mean = 0.0005578`
  - `payout_mean = 0.0303372`
- eras `893-1197`
  - `delta_cumsum_end = -0.0188815`
  - `bmc_mean = 0.0001579`
  - `payout_mean = 0.0197994`

Best dense blend: `95%` CatBoost + `5%` `roundM6_base_w80_m3_w20`

- eras `577-889`
  - `delta_cumsum_end = 0.0238303`
  - `bmc_mean = 0.0005700`
  - `payout_mean = 0.0303511`
- eras `893-1197`
  - `delta_cumsum_end = -0.0197620`
  - `bmc_mean = 0.0001660`
  - `payout_mean = 0.0198132`

Interpretation:

- The current dense edge is not concentrated only in the later eras.
- It is stronger in the earlier half of the validation window and gives some of that edge back later.
- The `95/5` blend behaves almost identically to the dense CatBoost standalone model on this split, which confirms that the blend is a very small adjustment rather than a meaningfully different regime.

## Decision

- Reject the dense LightGBM candidate.
- Keep `cat_strict_resid008_medfaith64_walkfwd` as the current surviving dense tree.
- Reject `cat_strict_resid010_smallfaith64_walkfwd` as a replacement candidate.
- If deploying a tree+MLP blend from this round, prefer the `95/5` CatBoost + `roundM6_base_w80_m3_w20` blend.
- Continue searching for a stronger dense additive tree before treating this as a final winner.
