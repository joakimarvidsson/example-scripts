# Round S56: Benchmark-Correction Neural Objective

## Idea
Stop asking the network to be a standalone benchmark replacement.
Instead, train it as a bounded additive correction on top of the benchmark:
- final prediction used in the main loss is `benchmark + scale * tanh(network_output)`
- main loss remains era-wise correlation against the raw `target_ender_20`

## Implementation
Added `benchmark_combine_scale` to `strict_neural_cv_walkforward.py`.
For these specs:
- main target switches to raw `target_ender_20`
- the loss is computed on the combined prediction
- prediction export also uses the combined prediction

## Smoke
Best completed smoke candidate:
- `ncv_gated_corr_benchs05_medfaith64_strict_seedavg1_roundS56smoke`
- `delta_mean = 0.0002643`
- `delta_cumsum_end = 0.0063424`
- `delta_cumsum_min = 0.0007239`
- `bmc_mean = 0.0001266`
- `payout_mean = 0.0267353`

This was promising because cumulative CORR delta stayed positive throughout the smoke window.

## Research Pool
Scaled research-pool result:
- `ncv_gated_corr_benchs05_medfaith64_strict_seedavg2_roundS56rp`
- `delta_mean = 0.0000950`
- `delta_cumsum_end = 0.0045602`
- `delta_cumsum_min = -0.0017869`
- `bmc_mean = 0.0000731`
- `payout_mean = 0.0276712`
- selected post-processing:
  - `mode = blend_neutralize`
  - `lambda = 0.05`
  - `neutralize_bench = 0.10`
  - `neutralize_example = 0.10`

Compared with the smoke, the scaled result was smaller and more benchmark-like, but still additive.

## Holdout A
Holdout-A result for the same winner:
- `ncv_gated_corr_benchs05_medfaith64_strict_seedavg2_roundS56ha`
- exact benchmark fallback selected again:
  - `mode = blend`
  - `lambda = 0.0`
  - `delta_mean = 0.0`
  - `delta_cumsum_end = 0.0`
  - `bmc_mean = 0.0`
  - `corr_with_benchmark_global = 1.0`

## Decision
- Keep the benchmark-correction code path; it is a better inductive bias than standalone neural prediction.
- Reject this branch as a standalone deployment candidate.
- Next step: use the neural correction outputs as candidate blend components around the already additive pseudo-Huber XGB incumbent, instead of expecting the neural model to survive on its own.
