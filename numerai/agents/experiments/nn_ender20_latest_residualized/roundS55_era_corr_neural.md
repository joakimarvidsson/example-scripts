# Round S55: Era-Correlation Neural Objective

## Idea
Replace row-shuffled residual MSE with an era-aware Pearson-correlation objective so the model is trained directly on the per-era structure that Numerai scores.

## Implementation
- Added `main_loss="era_corr"` to `strict_neural_cv_walkforward.py`
- Training batches switch from shuffled rows to shuffled eras for `era_corr` specs
- Main loss becomes mean `1 - corr(pred, target)` across eras, weighted by era-decay weights
- Auxiliary head, when present, remains weighted MSE

## Smoke
`ncv_gated_corr_resid010_medfaith64_nodecay`
- `delta_cumsum_end = 0.00899`
- `bmc_mean = 0.000395`
- `payout_mean = 0.027271`

This was enough to justify a full research-pool run.

## Research Pool
Best completed research-pool result:
- `ncv_gated_corr_resid010_medfaith64_nodecay_strict_seedavg2_roundS55rp`
- `delta_mean = 0.0003814`
- `delta_cumsum_end = 0.0183060`
- `bmc_mean = 0.0006115`
- `payout_mean = 0.0290913`
- selected post-processing:
  - `mode = blend_neutralize`
  - `lambda = 0.05`
  - `neutralize_bench = 0.05`
  - `neutralize_example = 0.10`

Other completed research-pool result:
- `ncv_gated_corr_resid008_medfaith64_nodecay_strict_seedavg2_roundS55rp`
- `delta_cumsum_end = 0.0091040`
- `bmc_mean = 0.0001973`
- `payout_mean = 0.0280191`

## Holdout A
Best research-pool winner tested:
- `ncv_gated_corr_resid010_medfaith64_nodecay_strict_seedavg2_roundS55ha`

Holdout-A result:
- benchmark fallback selected again
- `mode = blend`
- `lambda = 0.0`
- `delta_mean = 0.0`
- `delta_cumsum_end = 0.0`
- `bmc_mean = 0.0`
- `corr_with_benchmark_global = 1.0`

## Decision
- Keep the era-corr objective code; it improved research-pool behavior materially.
- Reject the standalone era-corr neural signal as a deployment candidate for now.
- Next iteration should stop treating the network as a standalone predictor and instead train it as a direct correction model around the benchmark.
