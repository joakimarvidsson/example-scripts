# Round S27 Auxiliary-Target Ridge

Date: 2026-03-09

## Goal

Test whether training on alternate Numerai targets can improve additive performance on the main benchmark-relative objective:

- train on auxiliary target
- evaluate strictly on `target_ender_20`
- keep model family fixed to the fastest viable orthogonal family:
  - `ridge_strict_resid008_smallfaith64_a10_walkfwd`

## Target inventory used

Quick scout target-to-`target_ender_20` relationships on `downsampled_full.parquet`:

- `target_teager2b_20`: era-mean corr `0.797`
- `target_waldo_20`: era-mean corr `0.761`
- `target_alpha_20`: era-mean corr `0.698`
- `target_ender_60`: era-mean corr `0.468`
- `target_teager2b_60`: era-mean corr `0.433`

## Scout round

Scout profile:

- `eval-era-step=8`
- `max_rows_per_era=500`
- strict selection grid:
  - `blend_lambdas = [0.05, 0.075, 0.10, 0.125]`
  - `neutralize_benchmark = [0.0, 0.1, 0.2]`
  - `neutralize_example = [0.0, 0.1, 0.2]`
  - `candidate_modes = blend, neutralize`

Summary artifact:

- `results/gbt_strict_walkforward_roundS27_ridge_aux_targets_summary.json`

### Scout results

| train target | delta_cumsum_end | delta_mean | bmc_mean | payout_mean | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `target_teager2b_60` | `0.0109266` | `0.0001401` | `0.0002107` | `0.0240654` | best scout |
| `target_ender_60` | `0.0105482` | `0.0001352` | `0.0002172` | `0.0240137` | second-best scout |
| `target_alpha_20` | `-0.0064248` | `-0.0000824` | `0.0001188` | `0.0236099` | reject |
| `target_teager2b_20` | `-0.0216070` | `-0.0002770` | `0.0000018` | `0.0232287` | reject |
| `target_waldo_20` | `-0.0240074` | `-0.0003078` | `-0.0000165` | `0.0232192` | reject |

Interpretation:

- For this ridge family, the 20-day auxiliary targets did not help.
- The only promising direction was a longer horizon:
  - `target_teager2b_60`
  - `target_ender_60`

## Dense confirmation

Dense profile:

- `eval-era-step=4`
- `max_rows_per_era=700`

Summary artifact:

- `results/gbt_strict_walkforward_roundS27_ridge_aux_dense_summary.json`

### Dense results

| train target | delta_cumsum_end | delta_mean | bmc_mean | payout_mean | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `target_ender_60` | `-0.0442677` | `-0.0002838` | `0.0000996` | `0.0242975` | failed dense confirmation |
| `target_teager2b_60` | `-0.0464389` | `-0.0002977` | `0.0000901` | `0.0242969` | failed dense confirmation |

## Decision

- Keep the new train-target/eval-target split in the harness.
- Reject the auxiliary-target ridge idea in its current form.
- Do not promote any S27 auxiliary-target ridge variant to a heavier family or to deployment.

## What this round taught us

- Auxiliary targets can look attractive at scout scale and still fail badly at dense scale.
- For this line of research, horizon-shifted labels were more promising than alternate 20-day labels, but still not robust enough.
- The code change remains useful because it enables future auxiliary-target research with stronger model families or explicit multitask/blended-target designs.
