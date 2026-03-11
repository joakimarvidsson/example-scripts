# Round S51: Random Fourier Feature Ridge

## Goal

Open a genuinely different raw idea class from the tree and plain-ridge branches by trying kernelized linear models:

- standardize features
- project through random Fourier features (RBF approximation)
- fit a ridge head on the transformed representation

This keeps training cheap while adding nonlinearity in a different way than trees or MLPs.

## Implementation

Added `rff` feature-transform support to the strict walk-forward harness using `RBFSampler`, with the transformed arrays cast back to `float32` to keep memory under control.

Initial larger `RFF` sizes were too memory-heavy, so the live scout used:

- `rff128_0.5`
- `rff256_1.0`

with ridge alphas:

- `1.0`
- `10.0`

All runs used `medium+faith2:64` and residual scale `0.008`.

## Scout Setup

Raw scout caches:

- eval era step `8`
- block size `26`
- max rows per era `500`

Scout raw-cache summary:

- `/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/nn_ender20_latest_residualized/results/gbt_strict_walkforward_roundS51_rff_ridge_scout_rawonly_summary.json`

Scout compact tuner:

- `/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/nn_ender20_latest_residualized/results/corrcap_two_stage_roundS51_rff_ridge_scout_light.json`

## Result

All four `RFF`-ridge variants were negative at scout scale.

Best of the four:

- `ridge_strict_resid008_medfaith64_a10_rff128_g0p5_walkfwd_raw_walkfwd_roundS51scout.parquet`
  - `status = no_positive_delta_feasible`
  - `delta_cumsum_end = -0.0413635`
  - `delta_mean = -0.0005303`
  - `bmc_mean = -0.0000154`
  - `payout_mean = 0.0232167`

The `rff256_1.0` variants were materially worse, with `delta_cumsum_end` around `-0.07463`.

## Decision

- Do not dense-confirm this family.
- Keep the deployed pseudo-Huber XGBoost as the incumbent.
- Keep the `RFF` transform support in the harness as a completed negative result for future reference.
