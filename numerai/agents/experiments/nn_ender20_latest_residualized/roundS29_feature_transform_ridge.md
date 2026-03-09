# Round S29 Ridge Feature Transforms

Date: 2026-03-09

## Goal

Test whether the `small+faith2:64` ridge family is being limited by collinearity or noisy geometry rather than by the target itself.

Model family held fixed:

- `ridge`
- `feature_set = small+faith2:64`
- `alpha = 10.0`
- evaluation target = `target_ender_20`

## Variants

1. baseline raw features
2. standardized features
3. standardized + PCA 32
4. standardized + PCA 64
5. standardized + PCA 96

## Scout profile

- `eval-era-step=8`
- `max_rows_per_era=500`
- strict selection grid:
  - `blend_lambdas = [0.05, 0.075, 0.10, 0.125]`
  - `neutralize_benchmark = [0.0, 0.1, 0.2]`
  - `neutralize_example = [0.0, 0.1, 0.2]`
  - `candidate_modes = blend, neutralize`

Summary:

- `results/gbt_strict_walkforward_roundS29_ridge_featx_summary.json`

## Results

| model | delta_cumsum_end | delta_mean | bmc_mean | payout_mean | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| raw baseline | `0.0034055` | `0.0000437` | `0.0001892` | `0.0239281` | best |
| PCA 96 | `0.0028958` | `0.0000371` | `0.0001967` | `0.0239424` | close, but worse on additive delta |
| standardize | `0.0026947` | `0.0000345` | `0.0001852` | `0.0239115` | worse |
| PCA 32 | `-0.0042714` | `-0.0000548` | `0.0002206` | `0.0239448` | reject |
| PCA 64 | `-0.0220460` | `-0.0002826` | `0.0000303` | `0.0233841` | reject |

## Decision

- Keep the new preprocessing support in the harness.
- Do not dense-scale any S29 variant.
- For this ridge family, simple feature transforms do not beat the raw `small+faith2:64` representation on additive CORR delta.
