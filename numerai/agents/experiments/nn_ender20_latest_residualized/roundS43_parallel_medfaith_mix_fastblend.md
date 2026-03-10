# Round S43: Faster Dense Blend Rerun + Med-Faith Py3.12 MLP Mix Scout

Goal: continue both branches in parallel, but this time correct the neural branch back to the `medium:256+faith2:64` family and make the dense blend search actually finish.

## Branch A: dense soft-penalty blend rerun

Code change:

- `softpenalty_dense_blend_search.py`
  - the cheap prefilter now scores `x @ weights` directly, where `x` is already built from per-era ranked component predictions
  - only the short-listed candidates are re-ranked per era before full metric evaluation

Run:

- `n_samples = 1200`
- `prefilter_top_n = 24`
- result file:
  - `results/roundS43_softpenalty_dense_blend_search_prefilter_norank.json`

Best finished blend:

- `cat_dense = 0.6157`
- `cat_smallfaith_dense = 0.0259`
- `lgbm_smallfaith_dense = 0.2797`
- `mlp_roundM6 = 0.0787`

Metrics:

- `delta_mean = -0.000550`
- `delta_cumsum_end = -0.085771`
- `bmc_mean = 0.000488`
- `payout_mean = 0.024853`
- `corr_with_benchmark_global = 0.994883`

Comparison vs the old finished `Round S40` blend:

- `payout_mean` improved slightly: `0.024844 -> 0.024853`
- `delta_cumsum_end` improved materially but stayed negative: `-0.119350 -> -0.085771`
- `bmc_mean` got slightly worse: `0.000558 -> 0.000488`

Decision:

- the implementation bottleneck is largely fixed: this branch now finishes in one turn
- the model result is still not promotable because cumulative CORR delta remains clearly negative

## Branch B: med-faith py3.12 raw-only MLP scout with mixed main targets

Code change:

- `strict_multitask_mlp_walkforward.py`
  - added `main_target_mix` to `MultitaskSpec`
  - main training target can now be a weighted mix of residualized targets, rather than only `target_ender_20`
  - added med-faith residual-scale variants and two mixed-main-target variants

New raw-only specs run under Python `3.12`:

- `mtmlp_strict_resid006_medfaith64_mainonly_walkfwd`
- `mtmlp_strict_resid010_medfaith64_mainonly_walkfwd`
- `mtmlp_strict_resid006_medfaith64_mix_e20e60_7525_walkfwd`
- `mtmlp_strict_resid006_medfaith64_mix_e20t60_7525_walkfwd`
- `mtmlp_strict_resid008_medfaith64_mix_e20e60_7525_walkfwd`

Raw-only summary:

- `results/mtmlp_roundS43_medfaith_mix_rawonly_summary.json`

Light scorer used:

- `tune_corrcap_two_stage.py`
- hard corr cap `0.997`
- no feature neutralization
- compact lambda / benchmark / example neutralization grid
- result file:
  - `results/corrcap_two_stage_roundS43_py312_mtmlp_medfaith_mix_light.json`

Scored results:

### `mtmlp_strict_resid010_medfaith64_mainonly_walkfwd`

- `delta_mean = -0.000173`
- `delta_cumsum_end = -0.013501`
- `bmc_mean = 0.000332`
- `payout_mean = 0.024076`

### `mtmlp_strict_resid006_medfaith64_mainonly_walkfwd`

- `delta_mean = -0.000301`
- `delta_cumsum_end = -0.023459`
- `bmc_mean = 0.000225`
- `payout_mean = 0.023800`

### `mtmlp_strict_resid006_medfaith64_mix_e20t60_7525_walkfwd`

- `delta_mean = -0.000277`
- `delta_cumsum_end = -0.021635`
- `bmc_mean = 0.000195`
- `payout_mean = 0.023781`

### `mtmlp_strict_resid008_medfaith64_mix_e20e60_7525_walkfwd`

- `delta_mean = -0.000678`
- `delta_cumsum_end = -0.052885`
- `bmc_mean = 0.000062`
- `payout_mean = 0.023083`

### `mtmlp_strict_resid006_medfaith64_mix_e20e60_7525_walkfwd`

- `delta_mean = -0.000796`
- `delta_cumsum_end = -0.062060`
- `bmc_mean = -0.000097`
- `payout_mean = 0.022620`

Comparison vs the earlier least-bad py3.12 neural result from `Round S40`:

- old best: `mtmlp_strict_resid008_medfaith64_mainonly_walkfwd_raw_walkfwd_roundS40_py312.parquet`
- old `delta_cumsum_end = -0.021238`
- old `payout_mean = 0.023560`

New conclusion:

- `resid010` med-faith main-only is the new least-bad neural variant so far
- it improved both delta and payout vs the old `resid008` main-only baseline
- mixed-main-target training did not help in this round
- none of the new py3.12 variants became additive

## Overall decision

- Branch A is now operationally usable again, but still not finding a promotable dense ensemble.
- Branch B found a better neural direction: increase residual scale to `0.010` on the med-faith main-only setup.
- No model upload or replacement is justified from this round.
