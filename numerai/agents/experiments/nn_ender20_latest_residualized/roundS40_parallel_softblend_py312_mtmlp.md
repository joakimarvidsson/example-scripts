# Round S40: Parallel Soft-Penalty Dense Blend Search + Python 3.12 MLP Scout

Goal: run two new branches in parallel after exhausting the dense hard-cap tree line.

- Branch A: search dense ensemble weights with a soft correlation penalty instead of a hard cap.
- Branch B: reopen the neural branch in a dedicated Python 3.12 environment and test whether a multitask residual MLP can produce a more orthogonal signal family.

## Branch A: soft-penalty dense blend search

Script added:

- `softpenalty_dense_blend_search.py`

Settings:

- dense OOF window: eras `577` to `1197`
- model pack:
  - `cat_dense`
  - `cat_smallfaith_dense`
  - `lgbm_smallfaith_dense`
  - `mlp_roundM6`
- objective:
  - `payout_mean`
  - minus a soft penalty for benchmark/example rank correlation above `0.995`
- final fast run used only the deterministic corner / near-corner weight set (`n_samples = 0`) after the broader randomized search was too expensive.

Result file:

- `results/roundS40_softpenalty_dense_blend_search.json`

Best candidate:

- weights:
  - `cat_dense = 0.90`
  - `mlp_roundM6 = 0.10`
  - `cat_smallfaith_dense = 0.00`
  - `lgbm_smallfaith_dense = 0.00`
- `corr_with_benchmark_global = 0.991912`
- `delta_mean = -0.000765`
- `delta_cumsum_end = -0.119350`
- `bmc_mean = 0.000558`
- `payout_mean = 0.024844`

Decision:

- The soft-penalty ensemble search did **not** find a better dense payout candidate.
- The best answer collapsed back to the existing `90/10` CatBoost + MLP neighborhood, and it was still strongly negative on CORR delta.

## Branch B: Python 3.12 multitask residual MLP scout

Runtime:

- Python `3.12` env: `.venv-py312-torch`
- device: `cpu`

Requested scout specs:

- `mtmlp_strict_resid008_medfaith64_mainonly_walkfwd`
- `mtmlp_strict_resid008_medfaith64_auxe60_w025_walkfwd`
- `mtmlp_strict_resid008_medfaith64_auxt60_w025_walkfwd`

The long scout was interrupted after the second raw cache was safely written, because the strict scoring pass remained too expensive for the turn.

Raw caches written:

- `predictions/mtmlp_strict_resid008_medfaith64_mainonly_walkfwd_raw_walkfwd_roundS40_py312.parquet`
- `predictions/mtmlp_strict_resid008_medfaith64_auxe60_w025_walkfwd_raw_walkfwd_roundS40_py312.parquet`

Quick scorer used for completed evaluation:

- `tune_corrcap_two_stage.py` with:
  - hard corr cap `0.997`
  - no feature neutralization
  - compact lambda / benchmark / example neutralization search

Result file:

- `results/corrcap_two_stage_roundS40_py312_mtmlp_light.json`

### `mainonly`

- status: `no_positive_delta_feasible`
- `delta_mean = -0.000272`
- `delta_cumsum_end = -0.021238`
- `bmc_mean = 0.000176`
- `payout_mean = 0.023560`
- `corr_cap_used = 0.996959`
- selected lambda: `0.07`
- benchmark neutralize: `0.01`
- example neutralize: `0.04`

### `auxe60_w025`

- status: `no_positive_delta_feasible`
- `delta_mean = -0.000696`
- `delta_cumsum_end = -0.054288`
- `bmc_mean = 0.000015`
- `payout_mean = 0.023100`
- `corr_cap_used = 0.996965`
- selected lambda: `0.07`
- benchmark neutralize: `0.03`
- example neutralize: `0.02`

Decision:

- The Python 3.12 neural branch ran correctly and produced reusable raw caches.
- The completed neural evaluations were both negative on cumulative CORR delta.
- `mainonly` was less bad than `auxe60_w025`, but neither is promotable.

## Overall decision

- Branch A failed to improve the current dense CatBoost line.
- Branch B successfully reopened the neural path on the correct runtime, but the first two completed py3.12 evaluations were negative.
- No model upload or replacement is justified from this round.
