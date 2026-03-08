# Round S24 Uncapped Tree Faith Scout

Date: 2026-03-09

## Goal

Retest additive tree models on the main target without the 0.99 correlation cap, using faith-heavy feature sets and uncapped strict selection.

## Code changes used

- `strict_gbt_walkforward_research.py`
  - family-specific parameter wiring fixed so CatBoost no longer receives XGBoost-only arguments
  - output names made configurable to avoid overwriting prior artifacts
- `tune_corrcap_two_stage.py`
  - added explicit support for `feature_spec=none`

## Scout setup

- Script: `numerai/agents/experiments/nn_ender20_latest_residualized/strict_gbt_walkforward_research.py`
- Selection objective: `delta_cumsum_end`
- Caps: disabled via `--max-corr-with-benchmark 2.0 --max-corr-with-example 2.0`
- Minimum additive constraints kept at `delta_mean >= 0` and `delta_cumsum_end >= 0`
- Fast scout profile: `eval-era-step=8`, `max_rows_per_era=500`

## Results

| Model | delta_mean | delta_cumsum_end | bmc_mean | payout_mean | feasible |
| --- | ---: | ---: | ---: | ---: | --- |
| `lgbm_dart_strict_resid010_smallfaith64_walkfwd_strict` | 0.0003544 | 0.0276442 | 0.0006072 | 0.0251153 | yes |
| `cat_strict_resid008_medfaith64_walkfwd_strict` | 0.0003417 | 0.0266522 | 0.0005978 | 0.0250626 | yes |
| `cat_strict_resid010_smallfaith64_walkfwd_strict` | 0.0003342 | 0.0260707 | 0.0005955 | 0.0252529 | yes |
| `cat_strict_direct_medfaith64_walkfwd_strict` | -0.0000994 | -0.0077527 | 0.0000238 | 0.0236352 | no |

## Readout

- Dropping the cap did not rescue the old residual XGB family.
- It did expose a better family/feature-set combination: residual tree models on `small+faith2` and `medium+faith2` are additive in scout mode.
- The best additive CORR-delta model in this round was `lgbm_dart_strict_resid010_smallfaith64_walkfwd_strict`.
- The highest payout among the additive tree scouts was the slightly more aggressive `cat_strict_resid010_smallfaith64_walkfwd_strict`, but it trailed the LightGBM on cumulative CORR delta.

## Next actions

1. Scale `lgbm_dart_strict_resid010_smallfaith64_walkfwd` on a denser evaluation setup.
2. Scale `cat_strict_resid008_medfaith64_walkfwd` as the main challenger.
3. Blend the best additive tree model with the strongest residual MLP and retune for payout proxy.
