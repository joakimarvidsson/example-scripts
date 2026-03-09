# Round S25 CatBoost Medfaith Refinement

Date: 2026-03-09

## Goal

Refine the surviving CatBoost scout family around the current medfaith incumbent:

- baseline: `cat_strict_resid008_medfaith64_walkfwd`
- objective: improve scout `delta_cumsum_end` without sacrificing BMC/payout
- scout profile:
  - `eval-era-step=8`
  - `max_rows_per_era=500`
  - trimmed strict-selection grid:
    - `blend_lambdas = [0.05, 0.075, 0.10, 0.125]`
    - `neutralize_benchmark = [0.0, 0.1, 0.2]`
    - `neutralize_example = [0.0, 0.1, 0.2]`
    - `candidate_modes = blend, neutralize`

## Variants run

1. `cat_strict_resid008_medfaith64_walkfwd`
2. `cat_strict_resid006_medfaith64_walkfwd`
3. `cat_strict_resid010_medfaith64_walkfwd`
4. `cat_strict_resid008_medfaith64_d5_walkfwd`

## Results

| model | delta_cumsum_end | delta_mean | bmc_mean | payout_mean | lambda | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cat_strict_resid008_medfaith64_walkfwd` | `0.0266522` | `0.0003417` | `0.0005978` | `0.0250626` | `0.075` | incumbent remains best |
| `cat_strict_resid006_medfaith64_walkfwd` | `0.0206110` | `0.0002642` | `0.0002745` | `0.0244582` | `0.05` | worse |
| `cat_strict_resid008_medfaith64_d5_walkfwd` | `0.0159630` | `0.0002047` | `0.0003401` | `0.0245341` | `0.05` | worse |
| `cat_strict_resid010_medfaith64_walkfwd` | `-0.0278694` | `-0.0003573` | `0.0001088` | `0.0236207` | `0.05` | reject |

## Interpretation

- `residual_scale=0.008` remains the local optimum in this CatBoost medfaith neighborhood.
- Lowering the residual scale to `0.006` reduces both additive CORR delta and BMC.
- Raising the residual scale to `0.010` breaks the additive CORR constraint entirely.
- Shallowing the trees from depth `6` to depth `5` reduces additive strength rather than improving generalization.
- The scout winner from the earlier round is locally stable. More CatBoost micro-tuning in this neighborhood is unlikely to produce a materially better dense model.

## Decision

- Keep `cat_strict_resid008_medfaith64_walkfwd` as the CatBoost scout incumbent.
- Do not promote any S25 medfaith refinement variant to dense scale.
- Next experiment should change something more fundamental than local CatBoost hyperparameters:
  - a different tree family,
  - a materially different feature construction,
  - or a more orthogonal model family for ensemble diversification.
