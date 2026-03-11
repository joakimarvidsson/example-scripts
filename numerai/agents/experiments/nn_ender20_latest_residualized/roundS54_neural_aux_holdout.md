# Round S54: Aux-Head Neural CV Holdout Check

## Setup
- Harness: `strict_neural_cv_walkforward.py`
- Partitioning: `research_pool=0577..0956`, `holdout_a=0964..1079`, `holdout_b=1084..1199`
- Embargo: `4` eras outer and inner
- Seeds: `2` (`1337`, `1434`)
- Feature family: `medium:256+faith2:64`
- Base architecture: gated MLP
- Main target: residualized `target_ender_20` with `residual_scale=0.010`
- Aux heads tested:
  - `target_ender_60` with `aux_weight=0.25`
  - `target_teager2b_60` with `aux_weight=0.25`

## Research-Pool Result
Both aux-head variants materially improved over the earlier gated neighborhood on the research pool.

Best research-pool result:
- `ncv_gated_resid010_medfaith64_nodecay_auxe60w025_strict_seedavg2_roundS54rp`
- `delta_cumsum_end = 0.02257`
- `bmc_mean = 0.000758`
- `payout_mean = 0.029545`

Close second:
- `ncv_gated_resid010_medfaith64_nodecay_auxt60w025_strict_seedavg2_roundS54rp`
- `delta_cumsum_end = 0.02223`
- `bmc_mean = 0.000751`
- `payout_mean = 0.029521`

## Holdout A Result
Neither aux-head variant survived holdout.

Holdout-A summary:
- `roundS54_neural_cv_aux_holdout_a_summary.json`
- both candidates selected `mode=blend`, `lambda=0.0`
- both collapsed to exact benchmark fallback:
  - `delta_mean = 0.0`
  - `delta_cumsum_end = 0.0`
  - `bmc_mean = 0.0`
  - `corr_with_benchmark_global = 1.0`

Raw holdout-A cache inspection for the `ender60` aux-head variant showed:
- raw model was not benchmark-like (`global corr to benchmark ~= -0.0335`)
- raw model had slightly positive absolute corr mean (`~0.00169`)
- but still badly underperformed the benchmark on additive delta:
  - `delta_mean ~= -0.02597`
  - `delta_cumsum_end ~= -0.75315`

## Decision
- Reject the aux-head gated MLP branch as currently trained.
- The improvement was real on the research pool but not robust outside it.
- The failure mode is not plotting or post-hoc selection; the raw holdout model is simply too weak relative to the benchmark.

## Next Step
Change the training objective, not the post-processing:
- move from row-shuffled residual MSE to an era-aware correlation objective designed to optimize additive per-era behavior directly.
