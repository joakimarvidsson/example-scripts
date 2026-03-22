# Round S58: Season 2026 Supply And Duplicate Cleanup

## Goal

Implement the immediate slot-cleanup tranche from the 2026 season plan while also generating fresh unique model supply around the current `truecontribution` XGB anchor.

## Slot Cleanup

Completed:

- `eisheth` was replaced successfully and is now running:
  - `residual_nn_ender20_spearman_delink05-mOYhew3IJj7q.pkl`
  - source artifact:
    - `/Users/joakim/Documents/Projects/Numerai/numerai-classic-ultimate/artifacts/models/residual_nn_ender20_spearman_delink05.pkl`

In progress:

- `aelva` was submitted and accepted by Numerai:
  - `residual_nn_teager2b_delink20-hhbc8cGLPBCr.pkl`
  - current state:
    - `validationStatus = validating`
    - `triggerStatus = queued`
  - source artifact:
    - `/Users/joakim/Documents/Projects/Numerai/numerai-classic-ultimate/artifacts/models/residual_nn_teager2b_delink20.pkl`

Queued behind age guard:

- `joakim18`
  - current file:
    - `lgbm_ender20-LOv9qZYw7L1h.pkl`
  - dry-run status:
    - `blocked_recent_slot`
    - `age_days = 29.07`
  - prepared replacement:
    - `/Users/joakim/Documents/Projects/Numerai/numerai-classic-ultimate/artifacts/models/residual_nn_ender20_delink05.pkl`
  - dry-run report:
    - `/Users/joakim/Documents/Projects/Numerai/numerai-classic-ultimate/reports/upload_no_duplicate_queue_2026-03-22_joakim18_dryrun.json`

The production anchor was intentionally left untouched:

- `truecontribution`
  - `/Users/joakim/Documents/Projects/Numerai/example-scripts/numerai/agents/experiments/nn_ender20_latest_residualized/models/xgb_strict_resid008_medfaith64_phuber_lam005_nb002_live_py312_20260311.pkl`

## Fresh Unique XGB Supply

Added a reproducible batch exporter:

- `/Users/joakim/Documents/Projects/Numerai/example-scripts-codex-faith-scale/numerai/agents/experiments/nn_ender20_latest_residualized/build_xgb_supply_batch.py`

It trains on the same live-profile recipe as `truecontribution`:

- objective: `reg:pseudohubererror`
- variant: `d5lr3e2`
- deployable post-processing:
  - `lambda = 0.05`
  - `neutralize_benchmark = 0.02`
- training profile:
  - `0158..1199`
  - `max_rows_per_era = 800`
  - `n_rows = 833600`

Built and smoke-tested on:

- `/Users/joakim/Documents/Projects/Numerai/Classic/ender/v5.2/live.parquet`
- `/Users/joakim/Documents/Projects/Numerai/Classic/ender/v5.2/live_benchmark_models.parquet`

Artifacts created:

1. `/Users/joakim/Documents/Projects/Numerai/example-scripts-codex-faith-scale/numerai/agents/experiments/nn_ender20_latest_residualized/models/season2026_supply/xgb_strict_resid006_medfaith64_phuber_lam005_nb002_live_seed1441_py312_20260322.pkl`
2. `/Users/joakim/Documents/Projects/Numerai/example-scripts-codex-faith-scale/numerai/agents/experiments/nn_ender20_latest_residualized/models/season2026_supply/xgb_strict_resid008_medfaith64_phuber_lam005_nb002_live_seed1553_py312_20260322.pkl`
3. `/Users/joakim/Documents/Projects/Numerai/example-scripts-codex-faith-scale/numerai/agents/experiments/nn_ender20_latest_residualized/models/season2026_supply/xgb_strict_resid008_medium_phuber_lam005_nb002_live_seed1667_py312_20260322.pkl`

For all three smoke tests:

- `len = 6710`
- `nan = 0`
- `std = 0.28867513138902823`
- `min = 0.00014903129657228018`
- `max = 1.0`

Batch summary:

- `/Users/joakim/Documents/Projects/Numerai/example-scripts-codex-faith-scale/numerai/agents/experiments/nn_ender20_latest_residualized/results/season2026_xgb_supply_batch_summary.json`

## Orthogonal Full-Era NN Supply

Three full-era seed-averaged NNs were smoke-tested successfully on the live frame:

- `/Users/joakim/Documents/Projects/Numerai/numerai-classic-ultimate/artifacts/models/residual_nn_rowan60_full_s20.pkl`
- `/Users/joakim/Documents/Projects/Numerai/numerai-classic-ultimate/artifacts/models/residual_nn_jasper60_full_s20.pkl`
- `/Users/joakim/Documents/Projects/Numerai/numerai-classic-ultimate/artifacts/models/residual_nn_ender60_full_s20.pkl`

Additional last-60-era holdout evidence against `target_ender_20`:

- `residual_nn_rowan60_full_s20.pkl`
  - `corr_mean = 0.0177648`
  - `bmc_mean = 0.0145493`
  - `payout_mean = 0.0254197`
  - `payout_sharpe = 1.0956`
  - `benchmark_corr = 0.1954`
- `residual_nn_jasper60_full_s20.pkl`
  - `corr_mean = 0.0190077`
  - `bmc_mean = 0.0159480`
  - `payout_mean = 0.0281646`
  - `payout_sharpe = 1.0674`
  - `benchmark_corr = 0.1713`
- `residual_nn_ender60_full_s20.pkl`
  - `corr_mean = 0.0204396`
  - `bmc_mean = 0.0167866`
  - `payout_mean = 0.0286057`
  - `payout_sharpe = 1.2043`
  - `benchmark_corr = 0.2207`

Recorded in:

- `/Users/joakim/Documents/Projects/Numerai/numerai-classic-ultimate/reports/nn_full_s20_holdout60_2026-03-22.json`

## Decision

- Immediate stale-negative cleanup is materially advanced:
  - `eisheth` done
  - `aelva` accepted and validating
  - `joakim18` queued behind the age guard
- The season portfolio now has three additional fresh XGB artifacts beyond `truecontribution`.
- The first orthogonal full-era NN supply pool is ready and stronger than expected on the last-60-era local holdout.

## Next Step

1. Replace `joakim18` the moment it clears the 30-day guard.
2. Rank the three new XGB supply artifacts against the current XGB anchor on dense OOS before any upload.
3. Use the full-era `ender60`, `jasper60`, and `rowan60` NNs as the next unique replacement pool for stale duplicate slots.
