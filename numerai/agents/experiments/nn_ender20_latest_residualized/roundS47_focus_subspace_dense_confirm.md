# Round S47: Focused Subspace Neighborhood After Fixed-Seeding

## Goal

Continue only from the corrected `S46` fixed-seed results and test a tighter neighborhood around the least-bad seed-averaged plain MLP subspace:

- baseline from corrected `S46`: `medium_mask40b+faith2:64`
- keep plain MLP, `resid010`, main-only, 3-seed averaging
- vary medium-mask fraction and faith cap locally instead of changing targets or architecture

## New Focused Specs

- `mtmlp_strict_resid010_medmask40b_faith64_mainonly_walkfwd`
- `mtmlp_strict_resid010_medmask40b_faith96_mainonly_walkfwd`
- `mtmlp_strict_resid010_medmask40b_faith128_mainonly_walkfwd`
- `mtmlp_strict_resid010_medmask35b_faith64_mainonly_walkfwd`
- `mtmlp_strict_resid010_medmask45b_faith64_mainonly_walkfwd`
- `mtmlp_strict_resid010_medmask40c_faith64_mainonly_walkfwd`

Custom medium masks added:

- `medium_mask35b`
- `medium_mask40c`
- `medium_mask45b`

## Scout Setup

- Python 3.12 torch env
- plain MLP
- target: `target_ender_20`
- training target style: residualized to `v52_lgbm_ender20` with `resid010`
- eval eras: `577-1197`
- scout evaluation: every `8`th era
- `max_rows_per_era = 700`
- train-era-step `4`
- seeds: `1337`, `2021`, `2401`
- score seed-averaged raw caches with compact two-stage tuner

Scout artifact:

- `/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/nn_ender20_latest_residualized/results/corrcap_two_stage_roundS47_focus_seedavg3_light.json`

## Scout Result

Only one candidate became additive at corrected scout scale:

- `mtmlp_strict_resid010_medmask35b_faith64_mainonly_walkfwd_raw_walkfwd_roundS47_seedavg3.parquet`
  - `delta_cumsum_end = 0.0048337`
  - `delta_mean = 0.0000620`
  - `bmc_mean = 0.0004190`
  - `payout_mean = 0.0245638`
  - `corr_cap_used = 0.9969567`
  - selected post-processing:
    - `lambda = 0.07`
    - `neutralize_benchmark = 0.0`
    - `neutralize_example = 0.05`

The rest of the neighborhood was worse than the corrected `S46` least-bad candidate:

- `medmask40b+faith64`: `delta_cumsum_end = -0.0045495`
- `medmask40b+faith96`: `delta_cumsum_end = -0.0186184`
- `medmask40b+faith128`: `delta_cumsum_end = -0.0181865`
- `medmask45b+faith64`: `delta_cumsum_end = -0.0266539`
- `medmask40c+faith64`: `delta_cumsum_end = -0.0003266`

## Dense Confirmation

Dense confirm was run only for the scout winner:

- same model spec
- seeds: `1337`, `2021`, `2401`
- eval step `4`
- `156` OOS eras
- `max_rows_per_era = 700`

Dense artifact:

- `/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/nn_ender20_latest_residualized/results/corrcap_two_stage_roundS47_dense_focus_seedavg3_light.json`

Dense result:

- `mtmlp_strict_resid010_medmask35b_faith64_mainonly_walkfwd_raw_walkfwdroundS47dense_seedavg3.parquet`
  - `delta_cumsum_end = -0.0615973`
  - `delta_mean = -0.0003949`
  - `bmc_mean = 0.0001933`
  - `payout_mean = 0.0243198`
  - `corr_cap_used = 0.9969636`
  - selected post-processing:
    - `lambda = 0.07`
    - `neutralize_benchmark = 0.02`
    - `neutralize_example = 0.03`

## Decision

- Do not promote `S47`.
- The focused subspace neighborhood produced a real corrected scout hit, but it failed dense confirmation decisively.
- Keep the existing dense CatBoost incumbent unchanged.

## Code Note

While scoring `S47`, the compact tuner incorrectly tried to resolve `features.json` even when invoked with `--feature-specs none`. That bug is now fixed by treating `none/null` as disabled feature neutralization rather than a real feature spec.
