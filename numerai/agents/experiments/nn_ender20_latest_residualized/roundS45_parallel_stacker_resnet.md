# Round S45: CatBoost+Neural Stacker vs ResNet-Style Py3.12 Scout

Goal: run both next ideas in parallel.

- Branch A: try orthogonal stacking around the dense CatBoost incumbent plus the new dense seed-averaged neural component.
- Branch B: try a genuinely different neural architecture in the Python `3.12` environment.

## Branch A: CatBoost + dense neural stacker

Code changes:

- `walkforward_prediction_stacker.py`
  - allow loading `prediction_raw` files as stacker inputs
  - add the dense seed-averaged neural raw file as `mlp_resid010_dense_seedavg3`
  - add `--spec-names` so only the targeted stackers run

Dense neural input used:

- `predictions/mtmlp_strict_resid010_medfaith64_mainonly_walkfwd_raw_walkfwd_roundS44dense_seedavg3.parquet`

Run summary:

- `results/roundS45_stacker_neuralcat_summary.json`

Targeted stackers run:

- `stack_ridge_cat_dense_resid010seedavg3_mlp_r006_a1`
- `stack_poslin_cat_dense_resid010seedavg3_mlp_raw`
- `stack_poslin_cat_dense_resid010seedavg3_raw`

Results:

### `stack_ridge_cat_dense_resid010seedavg3_mlp_r006_a1`

- `delta_mean = 0.000106`
- `delta_cumsum_end = 0.013845`
- `delta_sortino = 0.586474`
- `bmc_mean = 0.000047`
- `payout_mean = 0.023646`
- `corr_with_benchmark_global = 0.999999`

### `stack_poslin_cat_dense_resid010seedavg3_mlp_raw`

- `delta_mean = 0.000101`
- `delta_cumsum_end = 0.013137`
- `delta_sortino = 0.546362`
- `bmc_mean = 0.000045`
- `payout_mean = 0.023639`
- `corr_with_benchmark_global = 0.999999`

### `stack_poslin_cat_dense_resid010seedavg3_raw`

- `delta_mean = 0.000085`
- `delta_cumsum_end = 0.011040`
- `delta_sortino = 0.476142`
- `bmc_mean = 0.000038`
- `payout_mean = 0.023610`
- `corr_with_benchmark_global = 0.999999`

Interpretation:

- all three targeted stackers are slightly additive on benchmark-relative CORR over their dense overlap window
- the ridge stack is the best of the three
- however, the stackers are effectively tiny perturbations on top of the benchmark, with global benchmark correlation essentially `1.0`
- `bmc_mean` is far below the dense CatBoost incumbent, so this is not a payout winner

Decision:

- promising as a directional orthogonalization check
- not strong enough to promote or upload

## Branch B: ResNet-style py3.12 neural scout

Code changes:

- `strict_multitask_mlp_walkforward.py`
  - add `arch_type` support
  - keep existing `plain` MLP path
  - add a `resnet` trunk built from residual blocks with learned skip projections and LayerNorm
  - add two resnet scout specs

Specs run:

- `resmlp_strict_resid010_medfaith64_mainonly_walkfwd`
- `resmlp_strict_resid011_medfaith64_mainonly_walkfwd`

Raw-only summary:

- `results/mtmlp_roundS45_resmlp_seed1337_rawonly_summary.json`

Light scorer result:

- `results/corrcap_two_stage_roundS45_resmlp_light.json`

Results:

### `resmlp_strict_resid010_medfaith64_mainonly_walkfwd`

- `delta_cumsum_end = -0.059270`
- `delta_mean = -0.000760`
- `bmc_mean = -0.000092`
- `payout_mean = 0.022860`

### `resmlp_strict_resid011_medfaith64_mainonly_walkfwd`

- `delta_cumsum_end = -0.028627`
- `delta_mean = -0.000367`
- `bmc_mean = 0.000207`
- `payout_mean = 0.023635`

Interpretation:

- the residual-block architecture does not beat the plain py3.12 MLP branch
- both tested resnet variants were negative on cumulative CORR delta
- this branch is not worth scaling yet

## Overall decision

- Branch A is better than Branch B.
- The best result from this round is `stack_ridge_cat_dense_resid010seedavg3_mlp_r006_a1`.
- Even that best stack is too benchmark-like and too weak on BMC to justify promotion.
- No model upload or replacement is justified from this round.
