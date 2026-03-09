# Round S31: Scout-Level Tree/Ridge Blend Follow-up

## Goal
After rejecting the multitask MLP branch, test whether cheap orthogonal blending can improve the current scout-level additive tree candidates without another retrain.

## Blend Sweep
Used existing scout-level prediction files and blended them against:
- `torch_mlp_resid_latest_ender20_roundM6_base_w80_m3_w20`

Prediction models tested:
- `cat_strict_resid008_medfaith64_walkfwd_strict`
- `cat_strict_resid010_smallfaith64_walkfwd_strict`
- `lgbm_dart_strict_resid010_smallfaith64_walkfwd_strict`
- `ridge_strict_resid008_smallfaith64_a10_walkfwd_strict_roundS26_ridge_refine`

Artifact:
- `/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/nn_ender20_latest_residualized/results/roundS31_scout_tree_ridge_vs_roundM6_blends.json`

## Result
Best standalone by CORR Sortino:
- `cat_strict_resid010_smallfaith64_walkfwd_strict`
- `corr_sortino = 3.5144`
- `bmc_mean = 0.0005955`
- `payout_mean = 0.0252529`

Best blend by payout Sharpe:
- still `cat_strict_resid010_smallfaith64_walkfwd_strict`
- `gbt_weight = 1.0`
- `mlp_weight = 0.0`

Interpretation:
- none of the simple `tree/ridge + roundM6 MLP` blends improved on the standalone CatBoost scout winner
- simple orthogonal blending is exhausted on this scout set

## Neutralization Follow-up
A payout-oriented unified neutralization sweep was started on:
- `cat_strict_resid010_smallfaith64_walkfwd_raw_walkfwd.parquet`

First wide grid and then a narrower grid were both stopped.

Reason:
- local runtime remained too high for the value of this follow-up before a result landed
- this needs either caching/profiling in `tune_corrcap_two_stage.py` or an even tighter hand-picked grid

## Decision
- keep `cat_strict_resid010_smallfaith64_walkfwd_strict` as the current scout-level winner from this branch
- do not promote any new blend from this round
- no Numerai upload from this round
