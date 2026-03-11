# Round S44: Seed-Averaged Med-Faith Main-Only MLP Residual-Scale Sweep

Goal: follow the `Round S43` recommendation directly.

1. sweep `medium:256+faith2:64` main-only py3.12 MLP around `resid010`
2. use `residual_scale` in `{0.009, 0.010, 0.011, 0.012}`
3. train `3` seeds for each scale
4. average raw OOF predictions across seeds before scoring
5. dense-confirm immediately if any averaged scout candidate becomes additive

## Scout setup

Runtime:

- Python `3.12`
- `mps`

Walk-forward scout setup:

- `eval-era-step = 8`
- `train-era-step = 4`
- `block-size = 26`
- `max_rows_per_era = 700`

Seeds:

- `1337`
- `2021`
- `2401`

Raw-only per-seed summaries:

- `results/mtmlp_roundS44_medfaith_seed1337_rawonly_summary.json`
- `results/mtmlp_roundS44_medfaith_seed2021_rawonly_summary.json`
- `results/mtmlp_roundS44_medfaith_seed2401_rawonly_summary.json`

Seed-averaged raw caches created:

- `predictions/mtmlp_strict_resid009_medfaith64_mainonly_walkfwd_raw_walkfwd_roundS44_seedavg3.parquet`
- `predictions/mtmlp_strict_resid010_medfaith64_mainonly_walkfwd_raw_walkfwd_roundS44_seedavg3.parquet`
- `predictions/mtmlp_strict_resid011_medfaith64_mainonly_walkfwd_raw_walkfwd_roundS44_seedavg3.parquet`
- `predictions/mtmlp_strict_resid012_medfaith64_mainonly_walkfwd_raw_walkfwd_roundS44_seedavg3.parquet`

Light scorer:

- `tune_corrcap_two_stage.py`
- hard corr cap `0.997`
- no feature neutralization
- compact lambda / benchmark / example neutralization search
- result file:
  - `results/corrcap_two_stage_roundS44_py312_mtmlp_medfaith_seedavg3_light.json`

## Seed-averaged scout results

### `resid009`

- status: `no_positive_delta_feasible`
- `delta_mean = -0.000461`
- `delta_cumsum_end = -0.035983`
- `bmc_mean = 0.000135`
- `payout_mean = 0.023337`

### `resid010`

- status: `positive_delta_feasible`
- `delta_mean = 0.000193`
- `delta_cumsum_end = 0.015033`
- `bmc_mean = 0.000496`
- `payout_mean = 0.024726`

### `resid011`

- status: `no_positive_delta_feasible`
- `delta_mean = -0.000073`
- `delta_cumsum_end = -0.005691`
- `bmc_mean = 0.000341`
- `payout_mean = 0.024013`

### `resid012`

- status: `no_positive_delta_feasible`
- `delta_mean = -0.000228`
- `delta_cumsum_end = -0.017815`
- `bmc_mean = 0.000360`
- `payout_mean = 0.024150`

Decision after scout:

- seed averaging helped materially
- `resid010` became the first py3.12 med-faith neural configuration in this branch to clear the positive-delta scout gate
- `resid010` was selected for immediate dense confirmation

## Dense confirmation

Dense setup:

- same 3 seeds: `1337, 2021, 2401`
- same architecture: `mtmlp_strict_resid010_medfaith64_mainonly_walkfwd`
- `eval-era-step = 4`
- `train-era-step = 4`
- `block-size = 26`
- `max_rows_per_era = 700`

Dense raw-only summaries:

- `results/mtmlp_roundS44_dense_resid010_seed1337_rawonly_summary.json`
- `results/mtmlp_roundS44_dense_resid010_seed2021_rawonly_summary.json`
- `results/mtmlp_roundS44_dense_resid010_seed2401_rawonly_summary.json`

Dense seed-averaged raw cache:

- `predictions/mtmlp_strict_resid010_medfaith64_mainonly_walkfwd_raw_walkfwd_roundS44dense_seedavg3.parquet`

Dense light scorer result:

- `results/corrcap_two_stage_roundS44_py312_mtmlp_resid010_dense_seedavg3_light.json`

Dense result:

- status: `no_positive_delta_feasible`
- `delta_mean = -0.000446`
- `delta_cumsum_end = -0.069587`
- `bmc_mean = 0.000153`
- `payout_mean = 0.024224`

## Overall decision

- The full `1-5` loop is now completed for this branch.
- Seed averaging is useful at scout scale and should remain part of the neural workflow.
- The apparent `resid010` scout edge did **not** survive dense confirmation.
- No model upload or replacement is justified from this round.
- Current conclusion remains: the py3.12 med-faith MLP line is improving, but it is still not robust enough to replace the dense CatBoost incumbent.
