# Round S52: Strict Neural CV Framework

## Objective

Build a stricter neural research harness that avoids future leakage and reduces selection distortion, then run an initial scout round on the research pool and a single confirmatory check on Holdout A.

## CV protocol

Implemented in:
- `/Users/joakim/Documents/Projects/Numerai/example-scripts-codex-faith-scale/numerai/agents/experiments/nn_ender20_latest_residualized/strict_neural_cv_walkforward.py`

Key changes:
- explicit era partitions:
  - `research_pool = 0577..0956`
  - `holdout_a = 0964..1079`
  - `holdout_b = 1084..1199`
- outer walk-forward uses a `4`-era embargo before each validation block
- internal early stopping split also uses a `4`-era embargo
- default `train-era-step=1` so scout training no longer changes era support by default
- seed averaging is built into the harness
- optional era-decay weighting is supported
- optional `wandb` logging is supported
- new neural architectures supported:
  - `plain`
  - `gated`
  - `twotower`

## Runtime fixes

The py3.12 neural runtime was blocked by macOS code-signing policy on native extensions.
I fixed the local runtime by re-signing the native libraries in:
- `scikit-learn`
- `torch`

I also installed:
- `matplotlib`
- `wandb`

inside:
- `/Users/joakim/Documents/Projects/Numerai/example-scripts-codex-faith-scale/.venv-py312-torch`

## Research-pool scout

Command profile:
- partition: `research_pool`
- `eval_step=8`
- `block_size=13`
- `max_rows_per_era=350`
- `seed_average_count=2`

Summary artifact:
- `/Users/joakim/Documents/Projects/Numerai/example-scripts-codex-faith-scale/numerai/agents/experiments/nn_ender20_latest_residualized/results/roundS52_neural_cv_research_pool_scout_summary.json`

Best model:
- `ncv_gated_resid010_medfaith64_hl256_strict_seedavg2_roundS52rp_scout`

Best metrics:
- `delta_mean = 0.0002795`
- `delta_cumsum_end = 0.0134179`
- `early_delta_mean = 0.0002471`
- `delta_roll20_min = 0.0002073`
- `bmc_mean = 0.0005690`
- `payout_mean = 0.0289864`
- selected post-processing:
  - `mode = blend_neutralize`
  - `lambda = 0.05`
  - `neutralize_example = 0.10`
  - `neutralize_bench = 0.0`

Other scout models:
- `ncv_twotower_resid010_medfaith64_hl256_strict_seedavg2_roundS52rp_scout`
  - `delta_cumsum_end = 0.006729`
  - `bmc_mean = 0.000116`
- `ncv_plain_resid010_medfaith64_nodecay_strict_seedavg2_roundS52rp_scout`
  - effectively collapsed to the benchmark on this scout slice

## Holdout A check

I ran only the research-pool winner on `holdout_a` and kept `holdout_b` untouched.

Summary artifact:
- `/Users/joakim/Documents/Projects/Numerai/example-scripts-codex-faith-scale/numerai/agents/experiments/nn_ender20_latest_residualized/results/roundS52_neural_cv_holdout_a_summary.json`

Holdout A result:
- `ncv_gated_resid010_medfaith64_hl256_strict_seedavg2_roundS52ha`
- `delta_mean = 0.0`
- `delta_cumsum_end = 0.0`
- `bmc_mean = 0.0`
- `payout_mean = 0.020696`

Interpretation:
- the gated model looked additive on the research-pool scout
- on Holdout A it reverted to the benchmark
- that means the framework is doing its job: the idea is not robust enough yet to promote

## Plots

Research-pool winner vs benchmark:
- `/Users/joakim/Documents/Projects/Numerai/example-scripts-codex-faith-scale/numerai/agents/experiments/nn_ender20_latest_residualized/plots/v52_lgbm_ender20_vs_ncv_gated_resid010_medfaith64_hl256_strict_seedavg2_roundS52rp_scout_dark.png`

Research-pool comparison plot for all three architectures:
- `/Users/joakim/Documents/Projects/Numerai/example-scripts-codex-faith-scale/numerai/agents/experiments/nn_ender20_latest_residualized/plots/v52_lgbm_ender20_vs_ncv_plain_resid010_medfaith64_nodecay_strict_seedavg2_roundS52rp_scout_plus_2_dark.png`

Holdout A winner check:
- `/Users/joakim/Documents/Projects/Numerai/example-scripts-codex-faith-scale/numerai/agents/experiments/nn_ender20_latest_residualized/plots/v52_lgbm_ender20_vs_ncv_gated_resid010_medfaith64_hl256_strict_seedavg2_roundS52ha_dark.png`

## WandB

Harness support is now present.
To log runs online, set:
- `WANDB_API_KEY`
- `--wandb-project <project>`
- optionally `--wandb-entity <team-or-user>`
- optionally `--wandb-run-group <group>`
- optionally `--wandb-tags tag1,tag2`
- `--wandb-mode online`

Until an API key is available, use `--wandb-mode disabled` or `offline`.

## Decision

- keep `holdout_b` untouched
- do not promote this neural line yet
- use this harness for the next neural rounds rather than the older scout loop
