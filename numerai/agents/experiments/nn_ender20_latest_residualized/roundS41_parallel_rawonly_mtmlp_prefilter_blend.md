# Round S41: Parallel Raw-Only Py3.12 MLP Cache + Prefiltered Dense Blend Search

Goal: continue both branches from `Round S40` in parallel.

- Branch A: finish the remaining Python 3.12 auxiliary MLP cache and score it quickly.
- Branch B: optimize the dense soft-penalty blend search so it prefilters cheap candidates before full BMC/payout evaluation.

## Code changes

### `strict_multitask_mlp_walkforward.py`

Added:

- `--skip-strict-score`

Purpose:

- write raw walk-forward prediction caches without paying the slow strict scorer in the same process
- useful when the raw model training is the expensive/valuable step and strict scoring can be delegated to a lighter post-hoc tuner

### `softpenalty_dense_blend_search.py`

Added:

- cheap prefilter stage
- `--prefilter-top-n`

Method:

- generate candidate blend weights
- compute a cheap proxy score based on:
  - approximate per-era delta vs benchmark on the target
  - soft penalties for benchmark/example correlation above threshold
- run the full `_compute_delta_metrics()` path only on the top prefiltered candidates

## Branch A: remaining py3.12 auxiliary MLP

Ran raw-only scout for:

- `mtmlp_strict_resid008_medfaith64_auxt60_w025_walkfwd`

Raw cache written:

- `predictions/mtmlp_strict_resid008_medfaith64_auxt60_w025_walkfwd_raw_walkfwd_roundS40_py312.parquet`

Then scored with the light two-stage tuner:

- result file: `results/corrcap_two_stage_roundS41_py312_mtmlp_auxt_light.json`

Best result:

- status: `no_positive_delta_feasible`
- `delta_mean = -0.000472`
- `delta_cumsum_end = -0.036828`
- `bmc_mean = 0.000036`
- `payout_mean = 0.023220`
- `corr_cap_used = 0.996973`
- selected lambda: `0.07`
- benchmark neutralize: `0.03`
- example neutralize: `0.02`

Combined with `Round S40`, the first three completed py3.12 MLP raw caches are now:

- `mainonly`: negative
- `auxe60_w025`: worse
- `auxt60_w025`: negative, between the two

Decision:

- the reopened py3.12 neural branch is now operational and reusable
- but the first three completed variants are all negative on cumulative CORR delta
- no neural candidate is promotable from this line yet

## Branch B: prefiltered dense soft-penalty ensemble search

Script:

- `softpenalty_dense_blend_search.py`

Run attempted with:

- `n_samples = 1500`
- `prefilter_top_n = 30`
- dense pack:
  - `cat_dense`
  - `cat_smallfaith_dense`
  - `lgbm_smallfaith_dense`
  - `mlp_roundM6`

Outcome:

- the optimized script reduced CPU/memory pressure relative to the original all-candidate full-metric sweep
- but the run still did not finish within the turn budget on this machine
- no final `Round S41` ensemble artifact was produced

Current best completed ensemble result therefore remains the finished `Round S40` run:

- `results/roundS40_softpenalty_dense_blend_search.json`
- best completed weights:
  - `cat_dense = 0.90`
  - `mlp_roundM6 = 0.10`
- still negative on CORR delta

## Overall decision

- Branch A completed successfully and closed the first three py3.12 MLP variants.
- Branch B improved the search implementation, but not enough yet to produce a completed new ensemble result.
- No model upload or replacement is justified from this round.
