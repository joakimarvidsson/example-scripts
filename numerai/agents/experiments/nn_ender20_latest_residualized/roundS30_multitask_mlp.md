# Round S30: Multitask Torch MLP Scout

## Goal
Test whether a shared-trunk two-head MLP can improve `target_ender_20` OOF performance by adding an auxiliary head during training while still evaluating only on `target_ender_20`.

## Implementation
- Added `strict_multitask_mlp_walkforward.py`.
- Reused strict walk-forward loading and strict post-processing from `strict_gbt_walkforward_research.py`.
- Main head trains on residualized `target_ender_20`.
- Auxiliary head trains on residualized `target_ender_60` or `target_teager2b_60`.
- Feature slice: `medium:256+faith2:64`.
- Internal validation now early-stops on the main head only.
- Added a Python 3.14 runtime guard because local Torch was segfaulting there; stable runs used a local Python 3.12 scout env.

## Cheap Smoke Comparison
Short-window smoke setup:
- eval eras: `577, 609, 641`
- train era step: `8`
- max rows per era: `100`

| model | aux target | aux weight | delta_mean | delta_cumsum_end | bmc_mean | payout_mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `mtmlp_strict_resid008_medfaith64_mainonly_walkfwd` | none | 0.00 | -0.065314 | -0.195942 | -0.002697 | -0.011914 |
| `mtmlp_strict_resid008_medfaith64_auxe60_w025_walkfwd` | `target_ender_60` | 0.25 | -0.064969 | -0.194908 | -0.001769 | -0.009566 |
| `mtmlp_strict_resid008_medfaith64_auxt60_w025_walkfwd` | `target_teager2b_60` | 0.25 | -0.064989 | -0.194967 | -0.001843 | -0.009748 |

## Runtime Note
A wider scout on `eval-era-step=8`, `max_rows_per_era=300`, and more specs was started multiple times and stopped. Even after fixing early stopping and preloading tensors onto MPS, the first auxiliary spec remained too expensive for a local scout loop before showing any positive signal.

## Decision
Reject this branch for now.

Reasons:
- All smoke variants are still clearly negative on `target_ender_20` delta and BMC.
- Auxiliary heads help only marginally relative to a bad main-only control.
- The broader scout is too slow locally to justify more budget without a positive smoke result first.

## Next Step
Return to cheaper, orthogonal model families or blending/neutralization work where scout throughput is much higher.
