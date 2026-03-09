# Round S28 Blended-Target Ridge

Date: 2026-03-09

## Goal

Test whether keeping some `target_ender_20` mass in the training label can stabilize the longer-horizon signal that looked promising in the S27 scout.

Model family held fixed:

- `ridge_strict_resid008_smallfaith64_a10_walkfwd`

Evaluation target held fixed:

- `target_ender_20`

## Scout profile

- `eval-era-step=8`
- `max_rows_per_era=500`
- strict selection grid:
  - `blend_lambdas = [0.05, 0.075, 0.10, 0.125]`
  - `neutralize_benchmark = [0.0, 0.1, 0.2]`
  - `neutralize_example = [0.0, 0.1, 0.2]`
  - `candidate_modes = blend, neutralize`

Scout summary:

- `results/gbt_strict_walkforward_roundS28_ridge_mix_summary.json`

### Scout results

| train target mix | delta_cumsum_end | delta_mean | bmc_mean | payout_mean | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `target_ender_20:0.5,target_ender_60:0.5` | `0.0152320` | `0.0001953` | `0.0002454` | `0.0241184` | best scout |
| `target_ender_20:0.5,target_ender_60:0.25,target_teager2b_60:0.25` | `0.0134005` | `0.0001718` | `0.0002347` | `0.0240904` | second-best scout |
| `target_ender_20:0.75,target_teager2b_60:0.25` | `0.0093156` | `0.0001194` | `0.0002146` | `0.0240273` | positive scout |
| `target_ender_20:0.75,target_ender_60:0.25` | `0.0084806` | `0.0001087` | `0.0002096` | `0.0239922` | positive scout |
| `target_ender_20:1.0` | `0.0033569` | `0.0000430` | `0.0001883` | `0.0239263` | baseline |

Interpretation:

- Blending `target_ender_20` with 60-day targets is materially better than pure `target_ender_20` at scout scale for this ridge family.
- The strongest scout mix is `50/50` between `target_ender_20` and `target_ender_60`.

## Dense confirmation

- `eval-era-step=4`
- `max_rows_per_era=700`

Dense summary:

- `results/gbt_strict_walkforward_roundS28_ridge_mix_dense_summary.json`

### Dense results

| train target mix | delta_cumsum_end | delta_mean | bmc_mean | payout_mean | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `target_ender_20:0.5,target_ender_60:0.5` | `-0.0437055` | `-0.0002802` | `0.0000981` | `0.0242982` | failed dense confirmation |
| `target_ender_20:0.5,target_ender_60:0.25,target_teager2b_60:0.25` | `-0.0450579` | `-0.0002888` | `0.0000961` | `0.0242974` | failed dense confirmation |

## Decision

- Keep blended-target support in the harness.
- Reject the blended-target ridge idea in its current form.
- Do not carry these label mixes into CatBoost or deployment, because they failed dense validation too strongly.

## Takeaway

- The blended-target idea is more interesting than pure auxiliary-target substitution.
- But for this ridge family, the scout improvement is not robust enough to survive the denser walk-forward profile.
