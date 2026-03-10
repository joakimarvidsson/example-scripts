# Round S42: Parallel Small-Faith Py3.12 MLP Scout + Fast-Rank Dense Blend Attempt

Goal: continue both branches again in parallel.

- Branch A: broaden the cheap py3.12 MLP scout beyond the original `medium:256+faith2:64` family.
- Branch B: remove the pandas groupby rank bottleneck from the dense soft-penalty blend search.

## Branch A: broader cheap py3.12 MLP scout

New multitask specs added:

- `mtmlp_strict_resid006_smallfaith64_mainonly_walkfwd`
- `mtmlp_strict_resid006_smallfaith64_auxt60_w025_walkfwd`
- `mtmlp_strict_resid006_medcompactfaith64_mainonly_walkfwd`

Feature sets:

- `small+faith2:64`
- `medium:128+faith2:64`

All three were run in raw-only mode under Python `3.12` and then scored with the light two-stage tuner.

Raw-only summary:

- `results/mtmlp_roundS42_rawonly_summary.json`

Scored result file:

- `results/corrcap_two_stage_roundS42_py312_mtmlp_light.json`

Results:

### `mtmlp_strict_resid006_smallfaith64_mainonly_walkfwd`

- `delta_mean = -0.001047`
- `delta_cumsum_end = -0.081662`
- `payout_mean = 0.022083`

### `mtmlp_strict_resid006_smallfaith64_auxt60_w025_walkfwd`

- `delta_mean = -0.001430`
- `delta_cumsum_end = -0.111568`
- `payout_mean = 0.021636`

### `mtmlp_strict_resid006_medcompactfaith64_mainonly_walkfwd`

- `delta_mean = -0.000539`
- `delta_cumsum_end = -0.042015`
- `payout_mean = 0.023037`

Decision:

- None of the broadened cheap py3.12 variants beat the earlier `Round S40` med-faith main-only cache.
- The least-bad neural result remains:
  - `mtmlp_strict_resid008_medfaith64_mainonly_walkfwd_raw_walkfwd_roundS40_py312.parquet`
  - from `results/corrcap_two_stage_roundS40_py312_mtmlp_light.json`
  - `delta_cumsum_end = -0.021238`

## Branch B: fast-rank dense blend search

Change made:

- replaced the pandas per-era ranking path in `softpenalty_dense_blend_search.py` with a custom numpy ranker over precomputed era groups

Run attempted:

- `n_samples = 1000`
- `prefilter_top_n = 20`
- output target: `results/roundS42_softpenalty_dense_blend_search_prefilter_fast.json`

Outcome:

- the run still did not finish within the turn budget
- no final `Round S42` ensemble artifact was produced
- the remaining hot path is still the prefilter ranking loop over all candidate blends

Decision:

- the dense blend search implementation is improved again, but still not practical enough yet for a full 1000-sample run on this machine
- current best completed ensemble result therefore remains the finished `Round S40` result in:
  - `results/roundS40_softpenalty_dense_blend_search.json`

## Overall decision

- Branch A is now better mapped out: the cheap small-faith and med-compact py3.12 MLP variants are weaker than the original med-faith main-only variant.
- Branch B still needs another performance pass before it becomes a productive research loop.
- No model upload or replacement is justified from this round.
