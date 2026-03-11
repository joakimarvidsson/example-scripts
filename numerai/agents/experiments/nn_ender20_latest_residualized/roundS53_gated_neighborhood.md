# Round S53: Gated Neighborhood With Online W&B

## Objective

Continue the new strict neural CV line using the online W&B setup. Focus on the gated neighborhood around the prior `S52` winner, then check the current leader on `holdout_a`.

## W&B status

The strict neural harness now auto-loads repo-local `.env` credentials and logs online without manual sourcing.

Verified online runs:
- auth smoke: `fast-dew-2`
  - https://wandb.ai/joakimarvidsson/Numerai%20Classic/runs/3onx7x91
- research round `S53`: `warm-leaf-3`
  - https://wandb.ai/joakimarvidsson/Numerai%20Classic/runs/t71f93ue
- holdout A check: `dainty-water-4`
  - https://wandb.ai/joakimarvidsson/Numerai%20Classic/runs/crq8jwfw

Important correction:
- requested entity `Codex` returned `403 permission denied`
- writable entity for this account is `joakimarvidsson`

## Research-pool scout setup

Partition:
- `research_pool = 0577..0956`

Settings:
- `eval_step=8`
- `block_size=13`
- `max_rows_per_era=300`
- `seed_average_count=2`
- outer embargo `4`
- inner embargo `4`

## Completed gated results

Completed result JSONs:
- `/Users/joakim/Documents/Projects/Numerai/example-scripts-codex-faith-scale/numerai/agents/experiments/nn_ender20_latest_residualized/results/ncv_gated_resid008_medfaith64_hl256_strict_seedavg2_roundS53rp.json`
- `/Users/joakim/Documents/Projects/Numerai/example-scripts-codex-faith-scale/numerai/agents/experiments/nn_ender20_latest_residualized/results/ncv_gated_resid010_medfaith64_nodecay_strict_seedavg2_roundS53rp.json`
- `/Users/joakim/Documents/Projects/Numerai/example-scripts-codex-faith-scale/numerai/agents/experiments/nn_ender20_latest_residualized/results/ncv_gated_resid010_medfaith64_hl128_strict_seedavg2_roundS53rp.json`
- `/Users/joakim/Documents/Projects/Numerai/example-scripts-codex-faith-scale/numerai/agents/experiments/nn_ender20_latest_residualized/results/ncv_gated_resid010_medfaith64_hl256_strict_seedavg2_roundS53rp.json`

Headline metrics:

1. `ncv_gated_resid010_medfaith64_nodecay_strict_seedavg2_roundS53rp`
- `delta_mean = 0.0002434`
- `delta_cumsum_end = 0.0116841`
- `bmc_mean = 0.0006416`
- `payout_mean = 0.0290890`
- `mode = blend_neutralize`
- `lambda = 0.05`
- `neutralize_bench = 0.10`
- `neutralize_example = 0.10`
- `corr_with_benchmark_global = 0.9979408`

2. `ncv_gated_resid010_medfaith64_hl256_strict_seedavg2_roundS53rp`
- `delta_mean = 0.0001122`
- `delta_cumsum_end = 0.0053855`
- `bmc_mean = 0.0005801`
- `payout_mean = 0.0289024`

3. `ncv_gated_resid010_medfaith64_hl128_strict_seedavg2_roundS53rp`
- `delta_mean = 0.0000335`
- `delta_cumsum_end = 0.0016079`
- `bmc_mean = 0.0005375`
- `payout_mean = 0.0287683`

4. `ncv_gated_resid008_medfaith64_hl256_strict_seedavg2_roundS53rp`
- `delta_mean = 0.0000200`
- `delta_cumsum_end = 0.0009611`
- `bmc_mean = 0.0001382`
- `payout_mean = 0.0277994`

Interpretation:
- the strongest completed research-pool candidate is `gated resid010 nodecay`
- in this neighborhood, era decay hurt performance rather than helping it
- `resid010` remained the right scale; `resid008` was much weaker

## Why the round was cut short

The full `S53` queue also included additional gated and twotower variants.
I stopped it once the gated neighborhood had already given a clear ranking signal.
Reason:
- avoid spending more local budget on obviously weaker decay variants and twotower tails before checking the current leader on `holdout_a`

## Holdout A check

Result JSON:
- `/Users/joakim/Documents/Projects/Numerai/example-scripts-codex-faith-scale/numerai/agents/experiments/nn_ender20_latest_residualized/results/roundS53_neural_cv_holdout_a_summary.json`

Best holdout-A candidate:
- `ncv_gated_resid010_medfaith64_nodecay_strict_seedavg2_roundS53ha`

Metrics:
- `delta_mean = 0.0`
- `delta_cumsum_end = 0.0`
- `bmc_mean = 0.0`
- `payout_mean = 0.0206963`
- selected setting collapsed to benchmark:
  - `mode = blend`
  - `lambda = 0.0`
  - `neutralize_bench = 0.0`
  - `neutralize_example = 0.0`

Interpretation:
- the best gated research-pool candidate did **not** survive `holdout_a`
- it reverted to the benchmark on unseen eras
- this neural line is still not promotable

## Plots

Completed gated research-pool comparison:
- `/Users/joakim/Documents/Projects/Numerai/example-scripts-codex-faith-scale/numerai/agents/experiments/nn_ender20_latest_residualized/plots/v52_lgbm_ender20_vs_ncv_gated_resid008_medfaith64_hl256_strict_seedavg2_roundS53rp_plus_3_dark.png`

Holdout A check for the selected gated leader:
- `/Users/joakim/Documents/Projects/Numerai/example-scripts-codex-faith-scale/numerai/agents/experiments/nn_ender20_latest_residualized/plots/v52_lgbm_ender20_vs_ncv_gated_resid010_medfaith64_nodecay_strict_seedavg2_roundS53ha_dark.png`

## Decision

- keep `holdout_b` untouched
- do not promote this neural family yet
- next neural budget should change something more fundamental than decay around the same gated residual MLP
