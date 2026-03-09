# Round S26 Ridge Scout

Date: 2026-03-09

## Goal

Test a faster, more orthogonal residual model family after the CatBoost medfaith neighborhood plateaued.

- family: `ridge`
- scout profile:
  - `eval-era-step=8`
  - `max_rows_per_era=500`
  - trimmed strict-selection grid:
    - `blend_lambdas = [0.05, 0.075, 0.10, 0.125]`
    - `neutralize_benchmark = [0.0, 0.1, 0.2]`
    - `neutralize_example = [0.0, 0.1, 0.2]`
    - `candidate_modes = blend, neutralize`

## Variants run

1. `ridge_strict_direct_smallfaith64_a1_walkfwd`
2. `ridge_strict_resid008_smallfaith64_a01_walkfwd`
3. `ridge_strict_resid008_smallfaith64_a1_walkfwd`
4. `ridge_strict_resid008_smallfaith64_a10_walkfwd`
5. `ridge_strict_resid008_medfaith64_a1_walkfwd`

## Results

| model | delta_cumsum_end | delta_mean | bmc_mean | payout_mean | lambda | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ridge_strict_resid008_smallfaith64_a10_walkfwd` | `0.0033572` | `0.0000430` | `0.0001884` | `0.0239260` | `0.05` | best ridge, but weak |
| `ridge_strict_resid008_smallfaith64_a01_walkfwd` | `0.0033211` | `0.0000426` | `0.0001881` | `0.0239253` | `0.05` | essentially tied, weak |
| `ridge_strict_resid008_smallfaith64_a1_walkfwd` | `0.0033152` | `0.0000425` | `0.0001880` | `0.0239252` | `0.05` | essentially tied, weak |
| `ridge_strict_direct_smallfaith64_a1_walkfwd` | `-0.0066715` | `-0.0000855` | `0.0000319` | `0.0234614` | `0.05` | reject |
| `ridge_strict_resid008_medfaith64_a1_walkfwd` | `-0.0105562` | `-0.0001353` | `0.0001296` | `0.0236388` | `0.05` | reject |

## Interpretation

- Residualization matters for ridge. The direct smallfaith64 ridge model is not additive.
- Ridge is only viable on `small+faith2:64`; moving to `medium+faith2:64` made the model worse.
- Ridge alpha barely matters in the tested range `0.1 -> 10.0`; all three residual smallfaith64 variants are effectively the same model.
- The ridge family is much weaker than the current CatBoost scout incumbent:
  - best ridge `delta_cumsum_end = 0.0033572`
  - CatBoost medfaith incumbent `delta_cumsum_end = 0.0266522`

## Decision

- Keep the new ridge family support in the research harness because it provides a fast orthogonal scout loop.
- Do not promote any ridge candidate to dense scale.
- Keep CatBoost as the stronger additive tree/model-family candidate for this line of research.
