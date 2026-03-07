# Residualized Latest Ender20 Target + Neural Network Research

Date: 2026-02-08

## Abstract
Continued the neural-network research until we reached a full-data model that is consistently additive on BMC. Single MLP models either had positive corr with weak/negative BMC, or strong BMC with negative corr (no-intercept residualization). A weighted ensemble of two residualized NNs solved this tradeoff and produced positive `corr_mean`, positive `bmc_mean`, and positive `bmc_last_200_eras.mean` on full OOF.

Selected final model:
- `mlp_resid_latest_ender20_blend_noi09_base400_full`
- Blend weights:
  - `0.09 * mlp_resid_latest_ender20_nointercept_full`
  - `0.91 * mlp_resid_latest_ender20_base400k_full`

## Hypothesis / Motivation
Using residualized Ender20 (`target_ender_20` residualized to `v52_lgbm_ender20`) should increase additive signal vs benchmark. If single-model objectives conflict (corr vs BMC), blending residualized NNs can recover both positive corr and positive BMC.

## Method
- Dataset snapshot: `v5.2` (latest Ender20 pair available here)
  - Target: `target_ender_20`
  - Benchmark model column: `v52_lgbm_ender20`
- CV: 5-fold expanding, 13-era embargo
- Primary objective: positive `bmc_mean` and positive `bmc_last_200_eras.mean`, with non-negative corr
- Model family: sklearn `MLPRegressor` with target residualization transform

## Experiments Run
### Round 1 (downsampled scout)
- `mlp_resid_latest_ender20_base_downsampled`
- `mlp_resid_latest_ender20_wider_downsampled`
- `mlp_resid_latest_ender20_alpha1e3_downsampled`
- `mlp_resid_latest_ender20_lr5e4_iter60_downsampled`
- `mlp_resid_latest_ender20_prop085_downsampled`

### Round 2 (full confirmatory)
- `mlp_resid_latest_ender20_alpha1e3_full`
- `mlp_resid_latest_ender20_base_full`

### Round 3 (full consistency push)
- `mlp_resid_latest_ender20_base400k_full`
- `mlp_resid_latest_ender20_medium_base_full`
- `mlp_resid_latest_ender20_medium_alpha1e3_full`
- `mlp_resid_latest_ender20_nointercept_full`
- `mlp_resid_latest_ender20_nointercept_prop085_full`

### Round 4 (full NN blend search)
- Linear and rank blends between no-intercept and positive-corr full models.
- Finalized two additive blends:
  - `mlp_resid_latest_ender20_blend_noi09_base400_full`
  - `mlp_resid_latest_ender20_blend_noi10_base400_full`

## Results
Primary metric: `bmc_last_200_eras.mean` with `bmc_mean` as co-requirement.

| run | split | corr_mean | bmc_mean | bmc_last_200_eras_mean | avg_corr_with_benchmark |
|---|---|---:|---:|---:|---:|
| mlp_resid_latest_ender20_alpha1e3_downsampled | downsampled | 0.002804 | 0.001434 | 0.001223 | 0.041289 |
| mlp_resid_latest_ender20_base_full | full | 0.001442 | -0.000173 | 0.000197 | 0.051533 |
| mlp_resid_latest_ender20_base400k_full | full | 0.002413 | 0.000062 | -0.000318 | 0.064735 |
| mlp_resid_latest_ender20_medium_base_full | full | 0.002737 | 0.000258 | -0.000610 | 0.077411 |
| mlp_resid_latest_ender20_nointercept_full | full | -0.011292 | 0.001664 | 0.003719 | -0.403639 |
| mlp_resid_latest_ender20_nointercept_prop085_full | full | -0.010421 | 0.001036 | 0.002713 | -0.352342 |
| mlp_resid_latest_ender20_blend_noi10_base400_full | full | 0.000042 | 0.000464 | 0.000542 | -0.021826 |
| **mlp_resid_latest_ender20_blend_noi09_base400_full** | **full** | **0.000296** | **0.000425** | **0.000460** | **-0.012784** |

Interpretation:
- No-intercept residualization created the strongest additive signal but with negative corr.
- Blending a small fraction of no-intercept with a positive-corr model produced full-data consistency: corr > 0 and both BMC metrics > 0.

## Final Selection
Selected: `mlp_resid_latest_ender20_blend_noi09_base400_full`

Why this one:
- Consistently additive on full OOF (`bmc_mean > 0`, `bmc_last_200_eras.mean > 0`)
- Positive corr (small but positive), unlike pure no-intercept models
- Better corr/BMC balance than the `noi10` variant while staying clearly additive

## Standard Plot
Generated with:

```bash
PYTHONPATH=numerai .venv/bin/python -m agents.code.analysis.show_experiment benchmark mlp_resid_latest_ender20_blend_noi09_base400_full \
  --base-benchmark-model v52_lgbm_ender20 \
  --benchmark-data-path numerai/v5.2/full_benchmark_models.parquet \
  --start-era 575 --dark \
  --target-col target_ender_20 \
  --output-dir numerai/agents/experiments/nn_ender20_latest_residualized \
  --baselines-dir numerai/agents/baselines
```

![benchmark vs final blend](plots/v52_lgbm_ender20_vs_mlp_resid_latest_ender20_blend_noi09_base400_full_dark.png)

## Decisions Made
- Kept `target_ender_20` + `v52_lgbm_ender20` as the latest Ender20 pair in current data.
- Expanded beyond single-model tuning into NN blending after corr/BMC conflict on full data.
- Chose low-weight no-intercept injection to preserve positive corr while retaining additive signal.

## Stopping Rationale
Stopped this cycle because the selected blend is already consistently additive on full OOF under the requested criteria:
- `corr_mean > 0`
- `bmc_mean > 0`
- `bmc_last_200_eras.mean > 0`

## Findings
- Residualization mode is the dominant lever for additivity in this NN setup.
- Pure no-intercept models are highly additive but anti-correlated with target.
- Controlled blending is an effective way to recover practical corr while keeping additive BMC.

## Gradient-Boosted Tree Follow-up (CORR Sortino + Blend)
Date: 2026-02-09

Goal:
- Find a GBT-derived model that can beat benchmark on CORR Sortino.
- Blend with the residual MLP for payout proxy optimization:
  - `clip(0.75 * CORR + 2.25 * BMC, -0.05, 0.05)`

### What was tested
- Standalone GBT baselines:
  - `lgbm_ender20_downsampled`
  - `xgb_ender20_downsampled`
- Both standalone baselines were below benchmark CORR Sortino.
- Added a benchmark-anchored post-transform on XGB:
  - `pred_final = benchmark + lambda * (xgb_pred - benchmark) * 1(|z_resid| >= threshold)`
  - Chosen transform: `lambda=0.10`, `threshold=1.5`, per-era z-score of residual.
  - Saved as `xgb_ender20_benchgate_lam010_th15_downsampled`.

### Key results
From `results/gbt_sortino_blend_summary_with_benchgate.json` (eras >= 575):

- Best standalone CORR Sortino:
  - `xgb_ender20_benchgate_lam010_th15_downsampled`
  - `corr_sortino = 5.5727`
  - Benchmark `corr_sortino = 5.0267`
  - Delta vs benchmark: `+0.5460`

- Best blend by payout Sharpe (GBT + residual MLP):
  - `0.95 * xgb_ender20_benchgate_lam010_th15_downsampled`
  - `0.05 * torch_resid_latest_ender20_std008_row_downsampled`
  - `payout_sharpe = 1.8257`
  - `corr_sortino = 5.5605`

### Stability check (walk-forward hyperparameter selection)
To reduce overfitting risk, lambda/threshold were also selected in expanding walk-forward blocks and evaluated only on later blocks:

- File: `results/xgb_benchgate_walk_forward_check.json`
- Walk-forward delta vs benchmark CORR Sortino: `+0.0230`

Interpretation:
- Strong full-window lift is present.
- Walk-forward lift is smaller but still positive, so this behaves more like a modest robust improvement than a large guaranteed gain.

### Artifacts
- Standalone transformed GBT predictions:
  - `predictions/xgb_ender20_benchgate_lam010_th15_downsampled.parquet`
- Best GBT+MLP blend predictions:
  - `predictions/blend_xgb_ender20_benchgate_lam010_th15_downsampled_w95_torch_resid_latest_ender20_std008_row_downsampled_minera575.parquet`
- Comparison plot (corr, delta-corr, cumulative BMC):
  - `plots/v52_lgbm_ender20_vs_xgb_ender20_downsampled_plus_2_dark.png`

## Next Experiments
- Tighten blend weights around 0.08–0.10 with era-conditional blending rules.
- Try small positive-corr lift (e.g., slight rank-averaged contribution from `base_full`) while preserving BMC.
- Move from static linear blend to per-era adaptive blend learned on prior eras.

## Repro Commands
From `/Users/joakim/Documents/Projects/Numerai/example-scripts`:

```bash
# Full models used in final blend
PYTHONPATH=numerai .venv/bin/python -m agents.code.modeling --config numerai/agents/experiments/nn_ender20_latest_residualized/configs/mlp_resid_latest_ender20_base400k_full.py
PYTHONPATH=numerai .venv/bin/python -m agents.code.modeling --config numerai/agents/experiments/nn_ender20_latest_residualized/configs/mlp_resid_latest_ender20_nointercept_full.py

# Build additive blend (default = 0.09 no-intercept, 0.91 base400)
PYTHONPATH=numerai .venv/bin/python numerai/agents/experiments/nn_ender20_latest_residualized/build_blend.py

# Optional: alternate blend weight
PYTHONPATH=numerai .venv/bin/python numerai/agents/experiments/nn_ender20_latest_residualized/build_blend.py --w-noi 0.10 --name mlp_resid_latest_ender20_blend_noi10_base400_full

# GBT CORR Sortino + payout-proxy sweep
PYTHONPATH=numerai .venv/bin/python numerai/agents/experiments/nn_ender20_latest_residualized/analyze_gbt_sortino_and_blends.py \
  --output-dir numerai/agents/experiments/nn_ender20_latest_residualized \
  --prediction-models xgb_ender20_downsampled xgb_ender20_benchgate_lam010_th15_downsampled lgbm_ender20_downsampled \
  --mlp-model torch_resid_latest_ender20_std008_row_downsampled \
  --gbt-weight-grid 0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00 \
  --summary-name gbt_sortino_blend_summary_with_benchgate.json

# Plot final model
PYTHONPATH=numerai .venv/bin/python -m agents.code.analysis.show_experiment benchmark mlp_resid_latest_ender20_blend_noi09_base400_full \
  --base-benchmark-model v52_lgbm_ender20 \
  --benchmark-data-path numerai/v5.2/full_benchmark_models.parquet \
  --start-era 575 --dark \
  --target-col target_ender_20 \
  --output-dir numerai/agents/experiments/nn_ender20_latest_residualized \
  --baselines-dir numerai/agents/baselines
```

## Strict Walk-Forward GBT Consistency Round
Date: 2026-02-09

Goal:
- Improve standalone GBT consistency against `v52_lgbm_ender20` by directly optimizing downside-robust delta-CORR.
- Keep BMC non-negative while reducing early-era underperformance.

Method:
- New script: `strict_gbt_walkforward_research.py`
- Data: `v5.2/full.parquet`, target `target_ender_20`
- Evaluation eras: `577..1197` sampled every 4 eras (`156` eras total)
- Walk-forward blocks: 6 blocks, expanding train before each block
- Strict selection objective:
  - reward: `delta_mean`, `delta_sortino`, `bmc_mean`, `payout_mean`
  - penalties: `delta_roll20_min < 0`, `delta_cumsum_min < 0`, early-era (`<=889`) negative mean delta

### Round Results (strict objective)

| model | lambda | delta_mean | early_delta_mean | delta_cumsum_end | delta_cumsum_min | bmc_mean | strict_score |
|---|---:|---:|---:|---:|---:|---:|---:|
| xgb_strict_direct_medium_full_walkfwd_strict | 0.005 | -0.000018 | -0.000095 | -0.002736 | -0.008095 | 0.000000 | -0.936852 |
| xgb_strict_direct_medium_off0_walkfwd_strict | 0.005 | -0.000040 | -0.000111 | -0.006189 | -0.012162 | -0.000010 | -1.783757 |
| xgb_strict_direct_medium_off2_walkfwd_strict | 0.005 | -0.000002 | -0.000051 | -0.000326 | -0.006264 | 0.000006 | -0.097124 |
| **xgb_strict_resid008_medium_full_walkfwd_strict** | **0.005** | **0.000055** | **0.000024** | **0.008633** | **-0.005423** | **0.000049** | **2.016191** |
| xgb_strict_resid008_medium_off0_walkfwd_strict | 0.010 | -0.000077 | -0.000033 | -0.011984 | -0.015722 | 0.000014 | -1.966585 |
| xgb_strict_resid008_medium_off2_walkfwd_strict | 0.005 | 0.000011 | -0.000031 | 0.001788 | -0.007203 | 0.000019 | 0.353637 |

Selected standalone GBT:
- `xgb_strict_resid008_medium_full_walkfwd_strict`

Interpretation:
- Residualized target (`scale=0.008`) fixed the sign problem seen in direct-target strict runs.
- Early-era mean delta moved positive (small), and total cumsum delta finished positive.
- Remaining weakness: intermittent drawdowns still exist (`delta_cumsum_min < 0`).

### GBT + MLP payout blend sweep

Blend candidates were evaluated by payout proxy mean (`clip(0.75*corr + 2.25*bmc, ±0.05)`).

Best payout blend:
- `0.75 * gbt + 0.25 * mlp` (rank blend per era)
- File:
  - `predictions/blend_xgb_strict_resid008_medium_full_walkfwd_strict_with_torch_resid_latest_ender20_std008_row_full_blend_noi09_live_ds575plus.parquet`
- Metrics:
  - `delta_mean = 0.003182`
  - `early_delta_mean = 0.002221`
  - `delta_cumsum_end = 0.496324`
  - `bmc_mean = 0.006140`
  - `payout_mean = 0.031618`

### Artifacts
- Standalone strict GBT predictions:
  - `predictions/xgb_strict_resid008_medium_full_walkfwd_strict.parquet`
- Standalone strict GBT metrics:
  - `results/xgb_strict_resid008_medium_full_walkfwd_strict.json`
- Strict round summary:
  - `results/gbt_strict_walkforward_summary.json`
- GBT+MLP blend sweep:
  - `results/blend_xgb_strict_resid008_medium_full_walkfwd_strict_with_torch_resid_latest_ender20_std008_row_full_blend_noi09_live_ds575plus.json`

### Updated plots
- Standalone strict GBT vs benchmark:
  - `plots/v52_lgbm_ender20_vs_xgb_strict_resid008_medium_full_walkfwd_strict_dark.png`
- GBT+MLP blend vs benchmark:
  - `plots/v52_lgbm_ender20_vs_blend_xgb_strict_resid008_medium_full_walkfwd_strict_with_torch_resid_latest_ender20_std008_row_full_blend_noi09_live_ds575plus_dark.png`
- Combined comparison (benchmark + strict GBT + blend):
  - `plots/v52_lgbm_ender20_vs_xgb_strict_resid008_medium_full_walkfwd_strict_plus_1_dark.png`

### Repro (strict round)
```bash
cd /Users/joakim/Documents/Projects/Numerai/example-scripts

PYTHONPATH=numerai .venv/bin/python numerai/agents/experiments/nn_ender20_latest_residualized/strict_gbt_walkforward_research.py \
  --spec-names xgb_strict_resid008_medium_full_walkfwd,xgb_strict_resid008_medium_off0_walkfwd,xgb_strict_resid008_medium_off2_walkfwd \
  --min-eval-era 577 --max-eval-era 1197 --max-rows-per-era 800 --skip-mlp-blend

PYTHONPATH=numerai .venv/bin/python -m agents.code.analysis.show_experiment benchmark xgb_strict_resid008_medium_full_walkfwd_strict \
  --base-benchmark-model v52_lgbm_ender20 \
  --benchmark-data-path numerai/v5.2/full_benchmark_models.parquet \
  --start-era 577 --dark \
  --target-col target_ender_20 \
  --output-dir numerai/agents/experiments/nn_ender20_latest_residualized \
  --baselines-dir numerai/agents/baselines
```

## Constrained Decorrelation Round (`corr <= 0.90`)

Goal:
- Require positive additive behavior vs benchmark (`delta_mean > 0`, `delta_cumsum_end > 0`)
- Require decorrelation cap: max correlation with benchmark/example predictions <= `0.90`

What was tested:
- Updated `strict_gbt_walkforward_research.py` to enforce hard feasibility constraints:
  - `max_corr_with_benchmark=0.90`
  - `max_corr_with_example=0.90`
  - `min_delta_mean=0.0`
  - `min_delta_cumsum_end=0.0`
- Expanded GBT spec bank (multiple depth/lr/residual-scale/offset variants).
- Additional manual probes:
  - XGBoost and LightGBM residual/direct walk-forward (medium features).
  - XGBoost and LightGBM with `small` features and much larger row caps (`max_rows_per_era=2000`).
  - Threshold-gated benchmark blending (residual-z gate) to try to improve cap-vs-delta tradeoff.

Result:
- No tested model satisfied both constraints simultaneously.
- In all strong candidates, positive cumsum delta occurs only when correlation with benchmark/example remains very high (`~0.99+`).
- Once correlation is pushed below `0.90`, cumsum delta becomes negative.

Representative evidence (best medium residual model):

| lambda | delta_cumsum_end | corr_with_benchmark_max_abs |
|---:|---:|---:|
| 0.05 | +0.004576 | 0.998681 |
| 0.30 | -0.283771 | 0.930307 |
| 0.40 | -0.515567 | 0.854782 |

Representative evidence (strongest `small`-feature LGBM):

| lambda | delta_cumsum_end | corr_with_benchmark_max_abs |
|---:|---:|---:|
| 0.10 | +0.069922 | 0.994318 |
| 0.30 | -0.157611 | 0.929855 |
| 0.40 | -0.391475 | 0.853285 |

Conclusion:
- Under current target/setup (`target_ender_20` with benchmark-residualized training), we found no feasible GBT candidate that is both:
  - additive on cumsum CORR vs benchmark, and
  - decorrelated to <= 0.90 vs benchmark/example.

## Upload Safety Guard (30-day rule)

Added:
- `upload_pickle_with_age_guard.py`

Path:
- `agents/experiments/nn_ender20_latest_residualized/upload_pickle_with_age_guard.py`

Behavior:
- Blocks upload if target model slot has a latest cloudpickle newer than `--min-age-days` (default `30`).
- Uses NumerAPI model upload flow and polls terminal status.

Validation:
- Guard correctly blocks recent slots, e.g. `degen`:
  - `Age guard blocked upload: latest pickle ... is only 0.19 days old (min required: 30 days)`

## Constrained Follow-up (Feb 9, 2026)

Goal:
- Find a standalone GBT model with:
  - positive cumulative CORR delta vs `v52_lgbm_ender20`
  - hard decorrelation cap `corr <= 0.90` vs benchmark/example

What was changed in code:
- `strict_gbt_walkforward_research.py`
  - added candidate modes: `neutralize`, `blend`, `blend_neutralize`
  - added per-era neutralization grids for benchmark/example
  - added raw walk-forward prediction cache reuse:
    - `predictions/<spec_name>_raw_walkfwd.parquet`
  - optimized candidate screening for CORR-sortino selection and computed BMC only for selected output
- `export_strict_gbt_pickle.py`
  - changed default `--lambda-blend` from `0.005` to `1.0` so exported pickles are not benchmark-anchored by default.

What was run:
- Full-era walk-forward strict run (step=1) for residual medium `d5lr3e2`.
- Scout sweeps on eras `577..1197`, step=2, block size 52:
  - medium + small direct/residual variants
  - focused residual-medium family:
    - `d3lr7e2`, `d4lr5e2`, `d5lr3e2`, `d6lr2e2`, residual scales `0.008/0.010/0.012`
- Targeted scale-up on the closest near-feasible family (`resid012_medium_d5lr3e2`) with `max_rows_per_era=1200`.
- Additional all-feature test:
  - `xgb_strict_resid012_all_full_d5lr3e2_walkfwd` (`feature_set=all`, `max_rows_per_era=500`)
  - Result remained infeasible with negative delta under cap.

Raw cached model variants created:
- `xgb_strict_direct_medium_full_d4lr5e2_walkfwd_raw_walkfwd.parquet`
- `xgb_strict_direct_small_full_d4lr5e2_walkfwd_raw_walkfwd.parquet`
- `xgb_strict_resid008_medium_full_d3lr7e2_walkfwd_raw_walkfwd.parquet`
- `xgb_strict_resid008_medium_full_d4lr5e2_walkfwd_raw_walkfwd.parquet`
- `xgb_strict_resid008_medium_full_d5lr3e2_walkfwd_raw_walkfwd.parquet`
- `xgb_strict_resid008_medium_full_d6lr2e2_walkfwd_raw_walkfwd.parquet`
- `xgb_strict_resid010_medium_full_d5lr3e2_walkfwd_raw_walkfwd.parquet`
- `xgb_strict_resid012_medium_full_d5lr3e2_walkfwd_raw_walkfwd.parquet`

Best unconstrained model in this follow-up:
- `xgb_strict_resid008_medium_full_d6lr2e2_walkfwd_strict`
  - `delta_mean = +0.000037`
  - `delta_cumsum_end = +0.011447`
  - but `corr_with_benchmark_max_abs = 0.998697` (fails cap)

Closest under-cap frontiers found (manual frontier checks on cached OOF):
- `resid008_medium_d5lr3e2`:
  - best `delta_mean` under cap: `-0.001409` at cap `0.8838`
- `resid008_medium_d6lr2e2`:
  - best `delta_mean` under cap: `-0.001179` at cap `0.8989`
- `resid012_medium_d5lr3e2` (700 rows/era run):
  - best `delta_mean` under cap: `-0.000396` at cap `0.8992`
- `direct_medium_d4lr5e2`:
  - best `delta_mean` under cap: `-0.004599` at cap `0.8323`

Conclusion of this follow-up:
- No tested GBT configuration achieved both:
  - `delta_mean > 0` and `delta_cumsum_end > 0`
  - `corr <= 0.90` vs benchmark/example
- The constrained frontier is very close to zero for residual-medium variants, but still negative with the tested setups.

Frontier artifact:
- `results/gbt_corrcap_frontier_manual_2026-02-09.json`

### Tier-1 Extension: LightGBM DART

Additional code update:
- `strict_gbt_walkforward_research.py`
  - added `model_family` support (`xgb`, `lgbm`) and LightGBM DART spec bank.

DART specs run:
- `lgbm_dart_strict_resid008_medium_walkfwd`
- `lgbm_dart_strict_resid010_medium_walkfwd`
- `lgbm_dart_strict_resid012_medium_walkfwd`
- `lgbm_dart_strict_resid010_small_walkfwd`

DART frontier summary:
- Positive unconstrained deltas were found, but only at high benchmark correlation (`~0.98-0.99`).
- Under hard cap `<= 0.90`, all tested DART variants remained negative on `delta_mean`.
- Closest DART under cap:
  - `lgbm_dart_strict_resid012_medium_walkfwd`
  - `delta_mean = -0.000682`
  - `delta_cumsum_end = -0.212214`
  - `corr_cap_value = 0.881518`

Artifact:
- `results/gbt_corrcap_frontier_dart_2026-02-09.json`

### Tier-2 Probe: XGB + DART Ensembles

Probe setup:
- Used cached raw OOF from:
  - `xgb_strict_resid012_medium_full_d5lr3e2_walkfwd_raw_walkfwd`
  - `xgb_strict_resid008_medium_full_d6lr2e2_walkfwd_raw_walkfwd`
  - `lgbm_dart_strict_resid012_medium_walkfwd_raw_walkfwd`
- Evaluated small set of pair/triple blends with benchmark blend + benchmark neutralization.
- **Important**: this probe used Numerai correlation metrics (not plain Pearson).

Result:
- No positive under-cap ensemble found in this probe.
- Best unconstrained ensemble still had negative delta:
  - `tri_040_030_030`, `lambda=0.1`, `neutralize=0.2`
  - `delta_mean = -0.000166`, `corr_cap_value = 0.990872`
- Best under-cap point:
  - `tri_040_030_030`, `lambda=0.3`, `neutralize=0.2`
  - `delta_mean = -0.002390`, `corr_cap_value = 0.886538`

Artifact:
- `results/gbt_corrcap_ensemble_probe_2026-02-09.json`

### Corr-Cap vs Payout Threshold Check

Question checked:
- \"How much do we need to relax max corr with example predictions to get consistently positive payout proxy?\"

Finding:
- No relaxation was needed from `0.90` for payout proxy.
- Under `cap <= 0.90`, a strong payout-proxy point already exists:
  - `lambda=0.3`, `neutralize_benchmark=0.2`
  - cap `0.8837`
  - `payout_mean ~ 0.0277`
  - `payout_pos_share ~ 0.916`
  - positive cumulative payout proxy.

Important nuance:
- `validation_example_preds` and `v52_lgbm_ender20` are effectively identical on these eras (corr ~ 1.0), so example-cap and benchmark-cap are equivalent in this setup.
- Positive payout proxy at cap `<=0.90` does **not** imply positive CORR delta trend; delta remained negative at that setting.

Artifact:
- `results/corrcap_payout_threshold_probe_2026-02-09.json`

### Two-Stage Corrcap Tune (0.99) With Memory-Optimized Evaluator

New script:
- `tune_corrcap_two_stage.py`
  - Implements coarse `0.1` then fine `0.01` neutralization tuning around the coarse optimum.
  - Uses memory-efficient evaluation:
    - load only required columns,
    - `float32` arrays for large vectors,
    - per-era slice evaluation in NumPy,
    - BMC/payout computed only for the final selected candidate per model.

Run settings:
- cap: `max_corr_cap = 0.99`
- constraints: `delta_mean >= 0`, `delta_cumsum_end >= 0`
- coarse grids:
  - `lambda: 0.05..0.50 (step 0.05)`
  - `neutralize_benchmark: 0.0..1.0 (step 0.1)`
- fine grids:
  - centered on coarse best, `±0.05` window
  - `step 0.01` for both lambda and neutralization

Models tuned:
- `xgb_strict_resid012_medium_full_d5lr3e2_walkfwd_raw_walkfwd`
- `xgb_strict_resid008_medium_full_d6lr2e2_walkfwd_raw_walkfwd`
- `lgbm_dart_strict_resid012_medium_walkfwd_raw_walkfwd`

Result:
- No model achieved positive feasible CORR-delta trend under cap `<=0.99`.
- Best overall under the configured constraints objective:
  - `xgb_strict_resid008_medium_full_d6lr2e2_walkfwd_raw_walkfwd`
  - fine best: `lambda=0.09`, `neutralize_benchmark=0.34`
  - `corr_cap_used=0.989959`
  - `delta_mean=-0.000351`
  - `delta_cumsum_end=-0.109306`
  - `payout_mean=0.019149`
  - `bmc_mean=0.000748`

Artifact:
- `results/corrcap_two_stage_tune_0p99_2026-02-10.json`

### Payout-Objective Tuning (Alpha + Neutralization)

New direction:
- Since strict positive CORR-delta trend was not feasible under the tested constraints, we ran a direct payout-proxy optimization over:
  - benchmark blend alpha (`lambda`)
  - benchmark neutralization proportion (`neutralize_benchmark`)

Objective:
- maximize proxy payout mean:
  - `payout = clip(0.75 * CORR + 2.25 * BMC, -0.05, 0.05)`

Method:
- Reused `tune_corrcap_two_stage.py` with:
  - coarse: neutralization step `0.1`, alpha step `0.05`
  - fine: local refinement with step `0.01`
  - objective: `selection_objective = payout_mean`
- Ran both:
  - capped (`max_corr_cap=0.99`)
  - effectively uncapped (`max_corr_cap=1.0`)

Result:
- Both runs converged to the same payout-optimal setting:
  - model: `xgb_strict_resid012_medium_full_d5lr3e2_walkfwd_raw_walkfwd`
  - `lambda = 0.36`
  - `neutralize_benchmark = 0.07`
  - `corr_cap_used = 0.8550`
  - `payout_mean = 0.027749`
  - `payout_sortino = 5.8805`
  - `payout_cumsum_end = 8.6300`
  - `bmc_mean = 0.003639`
- This improves payout proxy relative to other tested candidates, while remaining far from the 0.99 cap.

Important tradeoff:
- The payout-optimal setting still has negative CORR delta vs benchmark:
  - `delta_mean = -0.003405`
  - `delta_cumsum_end = -1.058811`
- So this is a deliberate shift in objective from “beat benchmark CORR trend” to “maximize payout proxy”.

Artifacts:
- `results/payout_two_stage_tune_cap099_2026-02-10.json`
- `results/payout_two_stage_tune_cap100_2026-02-10.json`
- payout-tuned predictions:
  - `predictions/xgb_strict_resid012_medium_full_d5lr3e2_walkfwd_tuned_payout_cap099.parquet`

### Cross-Feature-Set Payout Optimization (cap 0.99)

Question checked:
- \"Are we tuning neutralization + alpha/lambda across different feature sets (`small` / `medium` / `all`) to maximize payout proxy?\"

Method:
- Ran `tune_corrcap_two_stage.py` on all cached strict raw walk-forward models:
  - objective: `selection_objective=payout_mean`
  - cap: `max_corr_cap=0.99`
  - coarse/fine search:
    - coarse: `lambda` step `0.05`, `neutralize_benchmark` step `0.1`
    - fine: local refinement with `0.01` steps

Best per feature set (by payout_mean):
| feature_set | model | lambda | neutralize_benchmark | payout_mean | payout_sortino | corr_cap_used |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| medium | `xgb_strict_resid012_medium_full_d5lr3e2_walkfwd_raw_walkfwd` | 0.36 | 0.07 | 0.027749 | 5.8805 | 0.8550 |
| all | `xgb_strict_resid012_all_full_d5lr3e2_walkfwd_raw_walkfwd` | 0.20 | 0.05 | 0.025056 | 8.5156 | 0.9688 |
| small | `lgbm_dart_strict_resid010_small_walkfwd_raw_walkfwd` | 0.11 | 0.35 | 0.025039 | 9.2072 | 0.9832 |

Takeaway:
- Yes, optimization is now done per feature set.
- `medium` remains the best payout-mean candidate in this run; `all` and `small` trail in payout_mean but have strong Sortino.

Artifacts:
- `results/payout_two_stage_tune_all_features_cap099_2026-02-10.json`
- `results/payout_two_stage_tune_all_features_cap099_2026-02-10_feature_best.csv`
- `results/payout_two_stage_tune_all_features_cap099_2026-02-10_top10.csv`
- Feature-best tuned predictions:
  - `predictions/xgb_strict_resid012_medium_full_d5lr3e2_walkfwd_tuned_payout_cap099_fsbest.parquet`
  - `predictions/xgb_strict_resid012_all_full_d5lr3e2_walkfwd_tuned_payout_cap099_fsbest.parquet`
  - `predictions/lgbm_dart_strict_resid010_small_walkfwd_tuned_payout_cap099_fsbest.parquet`
- Comparison plot:
  - `plots/v52_lgbm_ender20_vs_feature_best_payout_cap099_dark.png`

### Cloudpickle Upload Status

Latest cloudpickle deployment (age-guarded; no overwrite of recent pickle slots):
- model slot: `deep_baseline`
- model id: `d1cf4921-b8c2-4cd3-9c6c-c4cab58adf50`
- upload id: `7cac819e-96cb-4693-b352-43a075ed2fbb`
- status:
  - `validationStatus=validated`
  - `triggerStatus=submission_succeeded`
- pickle artifact:
  - `models/xgb_strict_resid012_d5_live_payout_cap099_py312.pkl`

## 2026-02-28 Refresh: Residualized Torch NN (Latest Benchmark-Aligned Ender20)

Objective:
- Re-run a strict neural-network-only research cycle on the latest benchmark-aligned Ender20 target setup:
  - target: `target_ender_20`
  - residualization: `target - scale * invnorm(v52_lgbm_ender20)`
  - benchmark/BMC reference: `v52_lgbm_ender20`

Data refresh and alignment:
- Rebuilt `v5.2/full.parquet` and downsampled variants from local train/validation.
- Verified benchmark-aligned ceiling:
  - `full.parquet`: eras `0001..1199`
  - `full_benchmark_models.parquet`: eras `0158..1199`
- This means residualized training against benchmark predictions is currently benchmark-limited to era `1199`.

### Round A (era-wise stopping, 5 configs)

Configs:
- `torch_mlp_resid_latest_ender20_roundA_std007_all_wider2_lr15e4_dsfull`
- `torch_mlp_resid_latest_ender20_roundA_std008_all_wider2_lr15e4_dsfull`
- `torch_mlp_resid_latest_ender20_roundA_std009_all_wider2_lr15e4_dsfull`
- `torch_mlp_resid_latest_ender20_roundA_std008_all_wider2_lr1e4_do05_dsfull`
- `torch_mlp_resid_latest_ender20_roundA_std008_medium_wider2_lr15e4_dsfull`

Best Round A:
- `torch_mlp_resid_latest_ender20_roundA_std009_all_wider2_lr15e4_dsfull`
  - `corr_mean = 0.004271`
  - `bmc_mean = 0.001728`
  - `bmc_last_200_eras.mean = 0.002675`

### Round B (MikeP-style row-wise replication + DenseNet probe, 5 configs)

Configs:
- `torch_mlp_resid_latest_ender20_roundB_std008_all_row_wider2_clip1_mts400`
- `torch_mlp_resid_latest_ender20_roundB_std009_all_row_wider2_clip1_mts400`
- `torch_mlp_resid_latest_ender20_roundB_std008_all_row_wider2_rankinv_mts400`
- `torch_mlp_resid_latest_ender20_roundB_std008_all_row_wider2_clip1_mts550`
- `torch_dense_resid_latest_ender20_roundB_std008_all_row_m3_mts400`

Finding:
- Row-wise early stopping increased CORR but did not improve BMC in this setup.
- Best Round B `bmc_last_200_eras.mean` remained below Round A.
- Higher row-wise sample budget (`mts550`) turned BMC negative.

### Round C (NN blend refinement)

Rationale:
- Pure model-training sweeps plateaued below prior blend performance.
- Refined the full-data NN blend between:
  - `torch_resid_latest_ender20_std008_row_full`
  - `mlp_resid_latest_ender20_nointercept_full`

Refined best blend:
- `torch_resid_latest_ender20_roundC_blend_noi12_base_rowfull_refined`
  - blend: `0.88 * torch_resid_latest_ender20_std008_row_full + 0.12 * mlp_resid_latest_ender20_nointercept_full`
  - `corr_mean = 0.001094`
  - `bmc_mean = 0.002659`
  - `bmc_last_200_eras.mean = 0.003866`
  - `avg_corr_with_benchmark = -0.073526`

Interpretation:
- This slightly improves over the previous `noi09` blend (`bmc_last_200_eras.mean 0.003849`).
- In this run, no pure-training NN variant exceeded the refined blend on `bmc_last_200_eras.mean`.

### Updated Standard Plot

Generated with:

```bash
PYTHONPATH=numerai .venv/bin/python -m agents.code.analysis.show_experiment benchmark torch_resid_latest_ender20_roundC_blend_noi12_base_rowfull_refined \
  --base-benchmark-model v52_lgbm_ender20 \
  --benchmark-data-path numerai/v5.2/full_benchmark_models.parquet \
  --start-era 575 --dark \
  --target-col target_ender_20 \
  --output-dir numerai/agents/experiments/nn_ender20_latest_residualized \
  --baselines-dir numerai/agents/baselines
```

Plot:
- `plots/v52_lgbm_ender20_vs_torch_resid_latest_ender20_roundC_blend_noi12_base_rowfull_refined_dark.png`

### Cloudpickle Export + Upload (2026-02-28)

Artifacts built:
- Live artifacts (`all eras via uniform era sampling`):
  - `models/benchscale_blend_noi12_live_artifacts/`
  - `models/torch_resid_latest_ender20_roundC_blend_noi12_live.pkl`
  - Python-3.12 rebuild for Numerai runtime compatibility:
    - `models/torch_resid_latest_ender20_roundC_blend_noi12_live_py312.pkl`
- Diagnostics artifacts (`max_train_era=560`):
  - `models/benchscale_blend_noi12_diag560_artifacts/`
  - `models/torch_resid_latest_ender20_roundC_blend_noi12_diag560.pkl`

Upload attempts:
1. Slot `triangleman`, upload id `005ede41-3b03-41c3-bcfd-9eca9ed3cbc4`:
   - failed (`validationStatus=invalid`, `triggerStatus=model_failed`)
   - trigger description: `Segmentation fault! Ensure python and library versions match our environment.`
2. Rebuilt pickle with Python 3.12 and uploaded to slot `nielsbohr`:
   - upload id: `eedaf1c2-c896-45cb-aec0-f921fe2ec2cc`
   - final status:
     - `validationStatus=validated`
     - `triggerStatus=submission_succeeded`

Slot selection notes:
- Applied 30-day age guard.
- Queried model history; no eligible slot showed explicit recent failed-submission counts via available submission fields, so selection fell back to an age-eligible slot.

## 2026-03-01 Medium-Only Continuation (keep iterating)

Scope:
- Feature set constrained to `medium` only.
- Goal: improve consistency/additivity while preserving positive BMC.

### Round M5s (fast scout; baseline-only benchmark input)

Hypothesis:
- Replace `x_groups=["features","era","benchmark_models"]` with `["features","era","baseline"]` (single `v52_lgbm_ender20` column) to reduce benchmark-noise leakage and improve stability.

Configs:
- `torch_mlp_resid_latest_ender20_roundM5s_std008_medium_baseline_only_era_ds_mts450`
- `torch_mlp_resid_latest_ender20_roundM5s_std008_medium_baseline_only_noera_ds_mts450`
- `torch_mlp_resid_latest_ender20_roundM5s_std008_medium_baseline_only_era_centerera_ds_mts450`
- `torch_mlp_resid_latest_ender20_roundM5s_std009_medium_baseline_only_era_ds_mts450`
- `torch_mlp_resid_latest_ender20_roundM5s_std008_medium_baseline_only_era_ds_mts450_seed2026`

Results:

| model | corr_mean | bmc_mean | bmc_last_200 | avg_corr_with_benchmark |
| --- | ---:| ---:| ---:| ---:|
| `torch_mlp_resid_latest_ender20_roundM5s_std008_medium_baseline_only_era_ds_mts450` | 0.004183 | 0.000634 | 0.000333 | 0.089714 |
| `torch_mlp_resid_latest_ender20_roundM5s_std008_medium_baseline_only_noera_ds_mts450` | 0.004741 | 0.001584 | 0.001649 | 0.086646 |
| `torch_mlp_resid_latest_ender20_roundM5s_std008_medium_baseline_only_era_centerera_ds_mts450` | 0.004948 | 0.000219 | -0.000081 | 0.135106 |
| `torch_mlp_resid_latest_ender20_roundM5s_std009_medium_baseline_only_era_ds_mts450` | 0.004102 | 0.002755 | 0.002748 | 0.029300 |
| `torch_mlp_resid_latest_ender20_roundM5s_std008_medium_baseline_only_era_ds_mts450_seed2026` | 0.003752 | 0.001330 | 0.001127 | 0.051153 |

Decision:
- Baseline-only input did not beat the existing full-data medium leaders on `bmc_last_200`.
- Best scout from this branch (`std009`) remained below `roundM2`/`roundM4` medium leaders.

### Round M6 (medium-only blend optimization on full OOF predictions)

Blend pool:
- `torch_mlp_resid_latest_ender20_roundM2_medium_full_blend_row08` (high corr baseline)
- `torch_mlp_resid_latest_ender20_roundM3_std008_medium_era_rankinv_centerera_full_mts1000` (corr stabilizer)
- `torch_mlp_resid_latest_ender20_roundM4_medium_full_blend_noi04` (high-BMC component)
- `torch_mlp_resid_latest_ender20_roundM2_std008_medium_era_rankinv_full_mts1000`

Saved candidates:

| model | corr_mean | bmc_mean | bmc_last_200 | avg_corr_with_benchmark | proxy (0.75*corr + 2.25*bmc) |
| --- | ---:| ---:| ---:| ---:| ---:|
| `torch_mlp_resid_latest_ender20_roundM6_base_w80_m3_w20` | 0.007741 | 0.002833 | 0.003520 | 0.139195 | 0.012180 |
| `torch_mlp_resid_latest_ender20_roundM6_m4_w70_m3_w30` | 0.002999 | 0.003161 | 0.004075 | -0.020847 | 0.009362 |

Interpretation:
- `roundM6_base_w80_m3_w20` is the best balanced medium-only update (small but real improvement over `roundM2_medium_full_blend_row08` on both corr and bmc).
- `roundM6_m4_w70_m3_w30` is the highest-BMC candidate but sacrifices too much CORR.

### Updated Plots

Benchmark comparison:
- `plots/v52_lgbm_ender20_vs_torch_mlp_resid_latest_ender20_roundM6_base_w80_m3_w20_plus_1_dark.png`

Additivity vs current medium baseline:
- `plots/torch_mlp_resid_latest_ender20_roundM2_medium_full_blend_row08_vs_torch_mlp_resid_latest_ender20_roundM6_base_w80_m3_w20_plus_1_dark.png`

## 2026-03-03 Round S4 (requested: small + faith2)

Implementation note:
- v5.2 feature metadata in this repo has `faith` but no `faith2` key.
- Added a loader alias `faith2 -> faith` in:
  - `agents/code/modeling/utils/data.py`
- This keeps configs explicit (`feature_set="faith2"`) while using current metadata.

Configs run (downsampled scout):
- `torch_mlp_resid_latest_ender20_roundS4_std009_small_era_rankinv_ds_mts450`
- `torch_mlp_resid_latest_ender20_roundS4_std010_small_era_rankinv_ds_mts450`
- `torch_mlp_resid_latest_ender20_roundS4_std008_faith2_era_rankinv_ds_mts450`
- `torch_mlp_resid_latest_ender20_roundS4_std009_faith2_era_rankinv_ds_mts450`

Results:

| model | feature_set | corr_mean | bmc_mean | bmc_last_200 | avg_corr_with_benchmark | proxy (0.75*corr + 2.25*bmc) |
| --- | --- | ---:| ---:| ---:| ---:| ---:|
| `torch_mlp_resid_latest_ender20_roundS4_std009_small_era_rankinv_ds_mts450` | small | 0.006046 | 0.002921 | 0.002428 | 0.081011 | 0.011107 |
| `torch_mlp_resid_latest_ender20_roundS4_std010_small_era_rankinv_ds_mts450` | small | 0.004723 | 0.003652 | 0.003479 | 0.021313 | 0.011758 |
| `torch_mlp_resid_latest_ender20_roundS4_std008_faith2_era_rankinv_ds_mts450` | faith2(alias faith) | 0.000913 | 0.000387 | 0.000657 | 0.023855 | 0.001555 |
| `torch_mlp_resid_latest_ender20_roundS4_std009_faith2_era_rankinv_ds_mts450` | faith2(alias faith) | 0.000191 | -0.000769 | -0.000537 | 0.021936 | -0.001587 |

Decision:
- `small` remains viable; best in this S4 scout is `roundS4_std010_small`.
- `faith2` (aliased to current `faith`) materially underperformed and was not scaled.
- Existing full-data candidates still dominate:
  - `torch_mlp_resid_latest_ender20_roundS2_std009_small_era_rankinv_full_mts1000` (proxy 0.015193)
  - `torch_mlp_resid_latest_ender20_roundS3_m6_small_blend_w50` (proxy 0.017560, current best)

Updated plots:
- `plots/v52_lgbm_ender20_vs_torch_mlp_resid_latest_ender20_roundS4_std010_small_era_rankinv_ds_mts450_plus_1_dark.png`
- `plots/v52_lgbm_ender20_vs_torch_mlp_resid_latest_ender20_roundS3_m6_small_blend_w50_plus_1_dark.png`

## 2026-03-03 Round S5 (small, explicit random-seed averaging)

Objective:
- Verify whether the S4 `small` winner improves when averaged across random seeds.

Configs run:
- `torch_mlp_resid_latest_ender20_roundS5_std010_small_seed2031_ds_mts450`
- `torch_mlp_resid_latest_ender20_roundS5_std010_small_seed2032_ds_mts450`
- `torch_mlp_resid_latest_ender20_roundS5_std010_small_seed2033_ds_mts450`

Seed-average artifact:
- `torch_mlp_resid_latest_ender20_roundS5_std010_small_seedavg3_ds_mts450`
- Built via:
  - `blend_seed_runs.py`
- Components:
  - `seed2031`, `seed2032`, `seed2033` at equal weights.

Results:

| model | corr_mean | bmc_mean | bmc_last_200 | avg_corr_with_benchmark | proxy (0.75*corr + 2.25*bmc) |
| --- | ---:| ---:| ---:| ---:| ---:|
| `torch_mlp_resid_latest_ender20_roundS4_std010_small_era_rankinv_ds_mts450` | 0.004723 | 0.003652 | 0.003479 | 0.021313 | 0.011758 |
| `torch_mlp_resid_latest_ender20_roundS5_std010_small_seed2031_ds_mts450` | 0.003889 | 0.002650 | 0.002319 | 0.044054 | 0.008880 |
| `torch_mlp_resid_latest_ender20_roundS5_std010_small_seed2032_ds_mts450` | 0.004859 | 0.003362 | 0.002544 | 0.038800 | 0.011208 |
| `torch_mlp_resid_latest_ender20_roundS5_std010_small_seed2033_ds_mts450` | 0.005557 | 0.003925 | 0.004040 | 0.044516 | 0.013000 |
| `torch_mlp_resid_latest_ender20_roundS5_std010_small_seedavg3_ds_mts450` | 0.006844 | 0.004681 | 0.004313 | 0.056124 | 0.015664 |

Decision:
- Yes, this candidate is random-seed averaged.
- The 3-seed average outperformed S4 single-seed on all tracked metrics and improved payout proxy by ~33.2%.

Updated plot:
- `plots/v52_lgbm_ender20_vs_torch_mlp_resid_latest_ender20_roundS5_std010_small_seedavg3_ds_mts450_dark.png`

## 2026-03-03 Round S6 (promote seed-average via blend with prior best)

Objective:
- Check whether the new seed-averaged small model is better used as a standalone model or as a blend component.

Blend sweep:
- Blended `torch_mlp_resid_latest_ender20_roundS5_std010_small_seedavg3_ds_mts450` with
  `torch_mlp_resid_latest_ender20_roundS3_m6_small_blend_w50`.
- Proxy objective: `0.75*corr_mean + 2.25*bmc_mean`.
- Weight sweep showed best proxy at `w_seedavg3=0.50` and `w_s3m6=0.50`.

Promoted model:
- `torch_mlp_resid_latest_ender20_roundS6_small_seedavg3_w50_s3m6_w50_ds`
  - built from equal-weight blend of:
    - `torch_mlp_resid_latest_ender20_roundS5_std010_small_seedavg3_ds_mts450`
    - `torch_mlp_resid_latest_ender20_roundS3_m6_small_blend_w50`

Metrics:

| model | corr_mean | bmc_mean | bmc_last_200 | avg_corr_with_benchmark | proxy (0.75*corr + 2.25*bmc) |
| --- | ---:| ---:| ---:| ---:| ---:|
| `torch_mlp_resid_latest_ender20_roundS3_m6_small_blend_w50` | 0.009225 | 0.004729 | 0.004915 | 0.120591 | 0.017560 |
| `torch_mlp_resid_latest_ender20_roundS5_std010_small_seedavg3_ds_mts450` | 0.006844 | 0.004681 | 0.004313 | 0.056124 | 0.015664 |
| `torch_mlp_resid_latest_ender20_roundS6_small_seedavg3_w50_s3m6_w50_ds` | 0.010389 | 0.005872 | 0.005216 | 0.121552 | 0.021003 |

Decision:
- `roundS6_small_seedavg3_w50_s3m6_w50_ds` is the new best OOF candidate in this experiment folder by proxy payout, CORR, and BMC.

Updated plots:
- `plots/v52_lgbm_ender20_vs_torch_mlp_resid_latest_ender20_roundS6_small_seedavg3_w50_s3m6_w50_ds_dark.png`
- `plots/torch_mlp_resid_latest_ender20_roundS3_m6_small_blend_w50_vs_torch_mlp_resid_latest_ender20_roundS6_small_seedavg3_w50_s3m6_w50_ds_dark.png`

## 2026-03-03 Round S7 (small residual-scale scout around std010)

Objective:
- Test whether nudging residual benchmark scale above `0.010` improves scout performance.

Configs:
- `torch_mlp_resid_latest_ender20_roundS7_std011_small_seed2031_ds_mts450`
- `torch_mlp_resid_latest_ender20_roundS7_std012_small_seed2031_ds_mts450`

Results:

| model | corr_mean | bmc_mean | bmc_last_200 | avg_corr_with_benchmark | proxy (0.75*corr + 2.25*bmc) |
| --- | ---:| ---:| ---:| ---:| ---:|
| `torch_mlp_resid_latest_ender20_roundS7_std011_small_seed2031_ds_mts450` | 0.003925 | 0.002085 | 0.001526 | 0.055551 | 0.007634 |
| `torch_mlp_resid_latest_ender20_roundS7_std012_small_seed2031_ds_mts450` | 0.004510 | 0.003065 | 0.002454 | 0.044056 | 0.010279 |
| reference `torch_mlp_resid_latest_ender20_roundS5_std010_small_seed2033_ds_mts450` | 0.005557 | 0.003925 | 0.004040 | 0.044516 | 0.013000 |
| reference `torch_mlp_resid_latest_ender20_roundS5_std010_small_seedavg3_ds_mts450` | 0.006844 | 0.004681 | 0.004313 | 0.056124 | 0.015664 |
| reference `torch_mlp_resid_latest_ender20_roundS6_small_seedavg3_w50_s3m6_w50_ds` | 0.010389 | 0.005872 | 0.005216 | 0.121552 | 0.021003 |

Decision:
- `std011` and `std012` do not outperform existing `std010` seeds or the S6 blend.
- Keep S6 as best OOF candidate from this branch.


## 2026-03-03 Round S8 (small blend refinement on seed-averaged runs)

Objective:
- Continue improving the small-feature residual MLP branch without adding new folders or heavy retrains.
- Sweep light-weight blends over existing strong small models.

Blend components:
- `torch_mlp_resid_latest_ender20_roundS6_small_seedavg3_w50_s3m6_w50_ds`
- `torch_mlp_resid_latest_ender20_roundS4_std010_small_era_rankinv_ds_mts450`
- `torch_mlp_resid_latest_ender20_roundS5_std010_small_seedavg3_ds_mts450`

Weight sweep (S6, S4, S5):
- `0.90, 0.05, 0.05`
- `0.85, 0.10, 0.05`
- `0.80, 0.15, 0.05`
- `0.75, 0.20, 0.05`
- `0.70, 0.25, 0.05`

Results summary:

| model | corr_mean | bmc_mean | bmc_last_200 | proxy (0.75*corr + 2.25*bmc) |
|---|---:|---:|---:|---:|
| `torch_mlp_resid_latest_ender20_roundS6_small_seedavg3_w50_s3m6_w50_ds` (reference) | 0.010389 | 0.005872 | 0.005216 | 0.021003 |
| `torch_mlp_resid_latest_ender20_roundS8_blend_0.900.050.05` | 0.010448 | 0.006011 | 0.005362 | 0.021360 |
| `torch_mlp_resid_latest_ender20_roundS8_blend_0.850.100.05` | 0.010554 | 0.006124 | 0.005487 | 0.021694 |
| `torch_mlp_resid_latest_ender20_roundS8_blend_0.800.150.05` | 0.010602 | 0.006210 | 0.005596 | 0.021923 |
| **`torch_mlp_resid_latest_ender20_roundS8_blend_0.750.200.05`** | **0.010568** | **0.006257** | **0.005664** | **0.022003** |
| `torch_mlp_resid_latest_ender20_roundS8_blend_0.700.250.05` | 0.010448 | 0.006255 | 0.005681 | 0.021910 |

Decision:
- New best OOF candidate in this folder is now `torch_mlp_resid_latest_ender20_roundS8_blend_0.750.200.05` by payout proxy.
- Improvement vs previous S6 best proxy: `0.022003 - 0.021003 = +0.001000`.

Updated plot:
- `plots/v52_lgbm_ender20_vs_torch_mlp_resid_latest_ender20_roundS8_blend_0_750_200_05_dark.png`


## 2026-03-04 Round S9 (cross-blend small + medium residual NNs)

Objective:
- Continue from S8 and test whether adding medium-model diversity improves the payout proxy while keeping CORR/BMC additive.

Components used:
- `torch_mlp_resid_latest_ender20_roundS8_blend_0.750.200.05` (best small branch)
- `torch_mlp_resid_latest_ender20_roundM6_base_w80_m3_w20` (best medium branch)
- `torch_dense_resid_latest_ender20_roundM1_std008_medium_era_m3_mts450` (medium dense variant)

Sweeps run:
- 2-way S8+M6: weights `(S8, M6)` in
  - `(0.95,0.05)`, `(0.90,0.10)`, `(0.85,0.15)`, `(0.80,0.20)`, `(0.75,0.25)`, `(0.70,0.30)`
- 3-way S8+M6+D1: weights `(S8, M6, D1)` in
  - `(0.90,0.08,0.02)`, `(0.85,0.10,0.05)`, `(0.80,0.15,0.05)`, `(0.75,0.20,0.05)`

Results summary:

| model | corr_mean | bmc_mean | bmc_last_200 | proxy (0.75*corr + 2.25*bmc) |
|---|---:|---:|---:|---:|
| reference `torch_mlp_resid_latest_ender20_roundS8_blend_0.750.200.05` | 0.010568 | 0.006257 | 0.005664 | 0.022003 |
| `torch_mlp_resid_latest_ender20_roundS9_s8m6d1_0.900.080.02` | 0.011804 | 0.006306 | 0.005851 | 0.023042 |
| **`torch_mlp_resid_latest_ender20_roundS9_s8m6d1_0.850.100.05`** | **0.012509** | **0.006290** | **0.006034** | **0.023535** |
| `torch_mlp_resid_latest_ender20_roundS9_s8m6d1_0.800.150.05` | 0.012565 | 0.006158 | 0.005921 | 0.023280 |
| `torch_mlp_resid_latest_ender20_roundS9_s8m6d1_0.750.200.05` | 0.012531 | 0.005998 | 0.005788 | 0.022895 |
| best 2-way `torch_mlp_resid_latest_ender20_roundS9_s8m6_0.950.05` | 0.010856 | 0.006192 | 0.005610 | 0.022075 |

Decision:
- New best OOF model in this experiment branch is `torch_mlp_resid_latest_ender20_roundS9_s8m6d1_0.850.100.05`.
- Gain vs S8 best on proxy: `+0.001532` (`0.023535 - 0.022003`).
- BMC drawdown also improved in the benchmark plot summary (`0.055979` vs `0.062675`).

Updated plot:
- `plots/v52_lgbm_ender20_vs_torch_mlp_resid_latest_ender20_roundS8_blend_0_750_200_05_plus_1_dark.png`


## 2026-03-04 Upload Status (Model Upload Skill flow)

Using the Numerai model-upload workflow (runtime check -> upload auth -> PUT -> create -> validate -> assign):

- Slot `betasamurai` (`modelId=d80e59e3-b76f-4a09-b91e-86e52e77b44e`)
  - uploaded/validated pickle: `b9e7964c-62d0-4c52-bffb-f80b434a2b91`
  - filename: `torch_resid_latest_ender20_roundC_blend_noi12_live_py312_np126_20260303_220928-DGmDhZplb6Bh.pkl`
  - assignment: completed (`assignPickleToModel=true`)

- Slot `thrivaldi` (`modelId=1a336ddd-5158-4e48-b06c-03a92e15a8a9`)
  - validated pickle: `84170a91-d368-486f-9b2f-b9e573298c46`
  - assignment: completed (`assignPickleToModel=true`)


## 2026-03-04 Round S10 (fine sweep around S9 best blend)

Objective:
- Refine S9 best 3-way blend weights to maximize payout proxy with minimal extra complexity.

Base components:
- `torch_mlp_resid_latest_ender20_roundS8_blend_0.750.200.05`
- `torch_mlp_resid_latest_ender20_roundM6_base_w80_m3_w20`
- `torch_dense_resid_latest_ender20_roundM1_std008_medium_era_m3_mts450`

Weight candidates (S8, M6, D1):
- `(0.88,0.08,0.04)`
- `(0.87,0.08,0.05)`
- `(0.86,0.09,0.05)`
- `(0.85,0.09,0.06)`
- `(0.84,0.10,0.06)`
- `(0.83,0.11,0.06)`
- `(0.82,0.12,0.06)`

Results summary:

| model | corr_mean | bmc_mean | bmc_last_200 | proxy (0.75*corr + 2.25*bmc) |
|---|---:|---:|---:|---:|
| reference `torch_mlp_resid_latest_ender20_roundS9_s8m6d1_0.850.100.05` | 0.012509 | 0.006290 | 0.006034 | 0.023535 |
| **`torch_mlp_resid_latest_ender20_roundS10_s8m6d1_0.870.080.05`** | **0.012448** | **0.006325** | **0.006062** | **0.023567** |
| `torch_mlp_resid_latest_ender20_roundS10_s8m6d1_0.860.090.05` | 0.012472 | 0.006303 | 0.006044 | 0.023535 |
| `torch_mlp_resid_latest_ender20_roundS10_s8m6d1_0.850.090.06` | 0.012558 | 0.006262 | 0.006061 | 0.023508 |

Decision:
- New best OOF in this branch is `torch_mlp_resid_latest_ender20_roundS10_s8m6d1_0.870.080.05`.
- Incremental proxy gain vs S9 best: `+0.000032`.
- Compared to S9 in plot summary: slightly higher BMC mean and lower BMC drawdown.

Updated plot:
- `plots/v52_lgbm_ender20_vs_torch_mlp_resid_latest_ender20_roundS9_s8m6d1_0_850_100_05_plus_1_dark.png`


## 2026-03-04 Round S11 (tiny faith2 injection into S10 best)

Objective:
- Test whether a very small `faith2` contribution improves robustness/diversity without degrading proxy.

Components:
- `torch_mlp_resid_latest_ender20_roundS10_s8m6d1_0.870.080.05`
- `torch_mlp_resid_latest_ender20_roundS4_std008_faith2_era_rankinv_ds_mts450`

Weight sweep (S10, faith2):
- `(0.99,0.01)`, `(0.98,0.02)`, `(0.97,0.03)`

Results summary:

| model | corr_mean | bmc_mean | bmc_last_200 | proxy (0.75*corr + 2.25*bmc) |
|---|---:|---:|---:|---:|
| reference `torch_mlp_resid_latest_ender20_roundS10_s8m6d1_0.870.080.05` | 0.012448 | 0.006325 | 0.006062 | 0.023567 |
| `torch_mlp_resid_latest_ender20_roundS11_s10f2_0.990.01` | 0.012475 | 0.006340 | 0.006085 | 0.023621 |
| `torch_mlp_resid_latest_ender20_roundS11_s10f2_0.980.02` | 0.012497 | 0.006351 | 0.006104 | 0.023662 |
| **`torch_mlp_resid_latest_ender20_roundS11_s10f2_0.970.03`** | **0.012510** | **0.006358** | **0.006116** | **0.023688** |

Decision:
- New best OOF candidate in this branch: `torch_mlp_resid_latest_ender20_roundS11_s10f2_0.970.03`.
- Incremental proxy gain vs S10 best: `+0.000121`.
- Plot summary also shows improved BMC drawdown (`0.052161` vs `0.053995`).

Correlation sanity check:
- On overlap with validation example predictions, mean per-era rank correlation is `~0.0995` to both benchmark and example predictions.
- This is far below the previous high-correlation concern and indicates non-trivial diversification.

Updated plot:
- `plots/v52_lgbm_ender20_vs_torch_mlp_resid_latest_ender20_roundS10_s8m6d1_0_870_080_05_plus_1_dark.png`

## 2026-03-05 Round S12 (faith2 seed expansion + S11 micro-refinement)

Objective:
- Continue research on `small + faith2` branch with explicit seed expansion and a tighter blend search around S11.

### S12-A: New `faith2` std010 seed runs (downsampled, `max_train_samples=450k`)

Configs:
- `torch_mlp_resid_latest_ender20_roundS12_std010_faith2_seed2031_ds_mts450`
- `torch_mlp_resid_latest_ender20_roundS12_std010_faith2_seed2032_ds_mts450`
- `torch_mlp_resid_latest_ender20_roundS12_std010_faith2_seed2033_ds_mts450`

Results:

| model | corr_mean | bmc_mean | proxy (0.75*corr + 2.25*bmc) |
|---|---:|---:|---:|
| `...seed2031...` | -0.001179 | -0.001210 | -0.003608 |
| `...seed2032...` |  0.000299 | -0.000457 | -0.000805 |
| `...seed2033...` |  0.000593 |  0.000041 |  0.000537 |

Decision:
- New `faith2` std010 single-seed scouts are weak standalone; use them only as diversification candidates, not as promoted bases.

### S12-B: Blend sweep around S11 best

Sweep artifact:
- `results/roundS12_blend_sweep_summary.csv`

Reference model:
- `torch_mlp_resid_latest_ender20_roundS11_s10f2_0.970.03`
  - corr_mean `0.012510`
  - bmc_mean `0.006358`
  - proxy `0.023688`

Best sweep candidate:
- `A0.97_H0.03` where
  - `A = torch_mlp_resid_latest_ender20_roundS11_s10f2_0.970.03`
  - `H = torch_mlp_resid_latest_ender20_roundS4_std008_faith2_era_rankinv_ds_mts450`
- Promoted name:
  - `torch_mlp_resid_latest_ender20_roundS12_s11h_0.970.03`

Promoted metrics:
- corr_mean `0.012530`
- bmc_mean `0.006364`
- bmc_last_200_mean `0.006142`
- proxy `0.023716`

Incremental gain vs S11 reference:
- proxy `+0.000028`
- corr_mean `+0.000020`
- bmc_mean `+0.000006`

Plots:
- `plots/v52_lgbm_ender20_vs_torch_mlp_resid_latest_ender20_roundS12_s11h_0_970_03_dark.png`
- `plots/v52_lgbm_ender20_vs_torch_mlp_resid_latest_ender20_roundS11_s10f2_0_970_03_plus_1_dark.png`

### S12-C: Cloudpickle packaging and upload

Packaged cloudpickle:
- `models/torch_resid_latest_ender20_roundS12_s11h_overlay006_live_py312_20260305_2210.pkl`
- metadata:
  - `models/torch_resid_latest_ender20_roundS12_s11h_overlay006_live_py312_20260305_2210.metadata.json`

Upload (strict age guard, very-old criterion set to 30 days):
- slot: `alberteinstein_` (`8d3fe55b-517f-460f-ad5c-b33b77144a4b`)
- upload id: `c4907be8-da7a-4f1d-a933-c2300a338bf3`
- status: `validated` + `submission_succeeded`
- log:
  - `results/rotation_uploads_20260305_s12_alberteinstein.json`
- slot: `georgecarlin` (`a740de40-3bd0-45e0-ae2b-9e46cc4a6650`)
- upload id: `38a21fba-a5dd-48d0-99cd-7d28d9a698ac`
- status: `validated` + `submission_succeeded`
- log:
  - `results/rotation_uploads_20260305_s12_georgecarlin.json`

## 2026-03-06 Continuation (rotation uploads + S14/S15 fine sweep)

Objective:
- Continue strict slot rotation uploads for the current S12 live cloudpickle.
- Run finer local blend sweeps (`small + faith2`) around S12/S14 to test if we can find a material proxy uplift.

### Rotation uploads completed

Current deployed pickle:
- `models/torch_resid_latest_ender20_roundS12_s11h_overlay006_live_py312_20260305_2210.pkl`

Successful uploads (all `validated` + `submission_succeeded`):
- `sisterdog` (`f0a86984-d317-48f2-9978-a8d8983c71ee`)
  - log: `results/rotation_uploads_20260306_s12_sisterdog.json`
- `lfg` (`48c83eeb-fa49-45f2-8a6b-5f9ecc8eafe6`)
  - log: `results/rotation_uploads_20260306_s12_lfg.json`
- `alphadog` (`53f45b7d-59a6-4a34-8637-974b702de3b4`)
  - log: `results/rotation_uploads_20260306_s12_alphadog.json`
- `forsaken` (`d61ef8df-bdb3-4d90-9cde-54cbbfc64799`)
  - log: `results/rotation_uploads_20260306_s12_forsaken.json`
- `numeraiordie` (`622542fa-2b38-4364-8034-866d5a9b6cb7`)
  - log: `results/rotation_uploads_20260306_s12_numeraiordie.json`

Latest slot audit after these rotations:
- `results/slot_audit_enriched_overrides_explicit_20260306_continue5_summary.txt`
- remaining eligible replacement slots: `17`

### S14 targeted blends (manual compact sweep)

Artifact:
- `results/roundS14_manual_targeted_blends.csv`

Best S14 candidate:
- `S14_b96_f4_03_f2033_01`
- components:
  - `0.96 * torch_mlp_resid_latest_ender20_roundS11_s10f2_0.970.03`
  - `0.03 * torch_mlp_resid_latest_ender20_roundS4_std008_faith2_era_rankinv_ds_mts450`
  - `0.01 * torch_mlp_resid_latest_ender20_roundS12_std010_faith2_seed2033_ds_mts450`

Delta vs S12 ref (`torch_mlp_resid_latest_ender20_roundS12_s11h_0.970.03`):
- `corr_mean`: `+0.0000051`
- `bmc_mean`: `-0.00000037`
- `bmc_last200_mean`: `+0.0000035`
- proxy (`0.75*corr + 2.25*bmc`): `+0.0000030`

Interpretation:
- Positive but tiny uplift; likely noise-level.

### S15 fine blend sweep

Fine sweep artifact:
- `results/roundS15_fine_blend_sweep.csv`

Best by proxy:
- `fine4_b0.95_o0.02_n0.02_s0.01`
- promoted/stored as:
  - `torch_mlp_resid_latest_ender20_roundS15_b95_o02_n02_s01`
  - predictions: `predictions/torch_mlp_resid_latest_ender20_roundS15_b95_o02_n02_s01.parquet`
  - results: `results/torch_mlp_resid_latest_ender20_roundS15_b95_o02_n02_s01.json`

Delta vs S12 ref:
- `corr_mean`: `-0.0000083`
- `bmc_mean`: `+0.0000070`
- `bmc_last200_mean`: `+0.0000052`
- proxy: `+0.0000094`

Interpretation:
- Slightly larger proxy lift than S14 best, but still very small.
- No material edge established yet; kept S12 cloudpickle as the live rotation artifact.

Updated comparison plot:
- `plots/v52_lgbm_ender20_vs_s12_ref_vs_s15_b95_o02_n02_s01_dark.png`

## 2026-03-06 Round S16 (medium + faith2 hybrid feature experiments)

Objective:
- Test whether `medium` plus a subset of `faith2` features can outperform the current small/faith2 stack.

### Pipeline extension (feature-set composition)

Implemented deterministic composite feature-set support in:
- `agents/code/modeling/utils/data.py`

New accepted syntax:
- `set`
- `set:N` (first `N` features from that set)
- `setA+setB:N` (deduped ordered union)

Examples used in this round:
- `medium+faith2:32`
- `medium+faith2:64`
- `medium+faith2:128`

### Standalone S16 scout runs

Configs:
- `configs/torch_mlp_resid_latest_ender20_roundS16_std010_medfaith32_seed2031_ds_mts450.py`
- `configs/torch_mlp_resid_latest_ender20_roundS16_std010_medfaith64_seed2031_ds_mts450.py`
- `configs/torch_mlp_resid_latest_ender20_roundS16_std010_medfaith128_seed2031_ds_mts450.py`

Summary artifact:
- `results/roundS16_medfaith_summary.csv`

Result:
- All standalone S16 variants underperformed S12 on proxy.
- Best standalone was `medium+faith2:64`, but still below S12.

### Overlay sweep on S12

Given weak standalone performance, tested S16 models as low-weight overlays onto:
- `torch_mlp_resid_latest_ender20_roundS12_s11h_0.970.03`

Sweep artifact:
- `results/roundS16_overlay_on_s12_summary.csv`

Weight curve artifact (`medfaith64` only):
- `results/roundS16_medfaith64_weight_curve.csv`

Best region:
- `medfaith64` overlay around `0.10` to `0.15`
- peak proxy at `w=0.12`

### Promoted S16 blend candidate (OOF)

Model:
- `torch_mlp_resid_latest_ender20_roundS16_s12_medfaith64_w12`

Artifacts:
- predictions: `predictions/torch_mlp_resid_latest_ender20_roundS16_s12_medfaith64_w12.parquet`
- results: `results/torch_mlp_resid_latest_ender20_roundS16_s12_medfaith64_w12.json`
- plot: `plots/v52_lgbm_ender20_vs_s12_ref_vs_s16_s12_medfaith64_w12_dark.png`

Metrics vs S12 ref:
- S12 ref:
  - `corr_mean = 0.012530`
  - `bmc_mean = 0.006364`
  - `bmc_last200 = 0.006142`
  - `proxy = 0.023716`
- S16 w12:
  - `corr_mean = 0.012699`
  - `bmc_mean = 0.006606`
  - `bmc_last200 = 0.006404`
  - `proxy = 0.024388`

Delta (S16 w12 - S12 ref):
- `corr_mean: +0.000169`
- `bmc_mean: +0.000242`
- `bmc_last200: +0.000262`
- `proxy: +0.000672`

Decision:
- This is a materially stronger uplift than the prior S14/S15 noise-level gains.
- Keep this as the current best research candidate for next packaging/upload step.

## 2026-03-06 Round S17 (medium+faith2 follow-up, higher sample caps)

Objective:
- Continue the medium+faith2 line with larger per-fold row budgets and lower benchmark residual scales to test if we can improve CORR-delta consistency without giving up payout proxy.

Configs:
- `configs/torch_mlp_resid_latest_ender20_roundS17_std008_medfaith64_seed2031_ds_mts700.py`
- `configs/torch_mlp_resid_latest_ender20_roundS17_std008_medfaith64_seed2032_ds_mts700.py`
- `configs/torch_mlp_resid_latest_ender20_roundS17_std009_medfaith64_seed2031_ds_mts700.py`
- `configs/torch_mlp_resid_latest_ender20_roundS17_std008_medfaith64_seed2031_ds_mts900.py`
- `configs/torch_mlp_resid_latest_ender20_roundS17_std008_medfaith96_seed2031_ds_mts700_lr1e4.py`

Standalone summary:
- `results/roundS17_summary.csv`

Outcome:
- Standalone S17 runs remained below S12/S16 blend-level performance.
- Best standalone in S17 by proxy was `std008_medfaith64_seed2031_ds_mts900` (`proxy=0.007760`), still far below S12 reference-level blends.

Recomputed overlay sweeps against S12 reference:
- `results/roundS17_overlay_on_s12_summary.csv`
- `results/roundS17_vsS16_overlay_compare.csv`

Key findings:
- S17 best overlay (`S17 std008 mts900`, `w=0.10`) reached:
  - `corr_mean=0.012895`
  - `bmc_mean=0.006414`
  - `proxy=0.024104`
  - `proxy uplift vs S12 = +0.000388`
- S16 seed-average overlay remains stronger:
  - `S16 seedavg2`, `w=0.15`: `proxy=0.024537` (`+0.000820` vs S12)
- For stricter CORR-delta consistency (minimal cumulative drawdown in delta vs S12):
  - `S16 seedavg2`, `w=0.03`: `proxy=0.023962` (`+0.000245` vs S12), `delta_cumsum_min=+0.000323`

Materialized promoted S17 artifacts:
- `predictions/torch_mlp_resid_latest_ender20_roundS17_s12_plus_s16seedavg2_w03.parquet`
- `results/torch_mlp_resid_latest_ender20_roundS17_s12_plus_s16seedavg2_w03.json`
- `predictions/torch_mlp_resid_latest_ender20_roundS17_s12_plus_s16seedavg2_w15.parquet`
- `results/torch_mlp_resid_latest_ender20_roundS17_s12_plus_s16seedavg2_w15.json`

Updated comparison plot:
- `plots/v52_lgbm_ender20_vs_torch_mlp_resid_latest_ender20_roundS12_s11h_0_970_03_plus_2_dark.png`

Decision:
- Keep two deployment-ready choices:
  - Proxy-first: `...w15`
  - Consistency-first (cleaner CORR-delta): `...w03`

## 2026-03-06 Round S18 (fine weight tuning for consistency-constrained uplift)

Objective:
- Refine only the overlay weight on the strongest component (`S16 seedavg2`) to maximize payout proxy while enforcing stable CORR-delta behavior vs S12.

Fine sweep:
- `results/roundS18_s16seedavg2_fine_weights.csv`
- grid: `w_overlay` from `0.01` to `0.08` in `0.005` steps

Best consistency-constrained candidate (`delta_cumsum_min >= 0`):
- `w_overlay = 0.045`
- `corr_mean = 0.012701`
- `bmc_mean = 0.006468`
- `proxy = 0.024080`
- `proxy uplift vs S12 = +0.000363`
- `delta_cumsum_end = +0.041110`
- `delta_cumsum_min = +0.000069`

Materialized artifacts:
- `predictions/torch_mlp_resid_latest_ender20_roundS18_s12_plus_s16seedavg2_w045.parquet`
- `results/torch_mlp_resid_latest_ender20_roundS18_s12_plus_s16seedavg2_w045.json`

Interpretation:
- `w=0.045` improves on prior `w=0.03` (higher proxy uplift while keeping cumulative CORR-delta non-negative).
- `w=0.15` remains the max-proxy option overall but has a deeper temporary CORR-delta drawdown.

Updated plot:
- `plots/v52_lgbm_ender20_vs_s12_vs_s16seedavg2_w045_w15_dark.png`

## 2026-03-07 Round S19 (different idea: 3-way blend)

Objective:
- Try a different blend structure, not just 2-way weight tuning:
  - `S12 base` + `S16 seedavg2` + `S17 mts900`

Grid sweep artifact:
- `results/roundS19_three_way_blend_grid.csv`

Top proxy candidate:
- weights: `w_base=0.87`, `w_s16=0.10`, `w_s17=0.03`
- model artifact:
  - `results/torch_mlp_resid_latest_ender20_roundS19_s12_s16_s17_w870_100_030.json`
  - `predictions/torch_mlp_resid_latest_ender20_roundS19_s12_s16_s17_w870_100_030.parquet`
- metrics:
  - `corr_mean=0.012963`
  - `bmc_mean=0.006589`
  - `proxy=0.024546` (slightly above prior 2-way max)
- tradeoff:
  - better proxy than S18 `w045`
  - but temporary CORR-delta drawdown remains (`delta_cumsum_min < 0`)

Top consistency-constrained 3-way candidate (`delta_cumsum_min >= 0`):
- weights: `w_base=0.955`, `w_s16=0.04`, `w_s17=0.005`
- artifact:
  - `results/torch_mlp_resid_latest_ender20_roundS19_s12_s16_s17_w955_040_005.json`
  - `predictions/torch_mlp_resid_latest_ender20_roundS19_s12_s16_s17_w955_040_005.parquet`
- proxy `0.024078`, effectively tied with S18 `w045` consistency-first result.

Updated plot:
- `plots/v52_lgbm_ender20_vs_s12_vs_s18_w045_vs_s19_threeway_dark.png`

## 2026-03-07 Upload workflow update

Actions:
- Queried Numerai docker images; default runtime confirmed:
  - `numerai_predict_py_3_12:f1a3f48` (`Python 3.12`)
- Built dedicated env:
  - `example-scripts/.venv312`
- Rebuilt cloudpickle in Python 3.12:
  - `models/torch_resid_latest_ender20_roundS18_overlay045_medium_live_py312_20260307.pkl`
  - metadata:
    - `models/torch_resid_latest_ender20_roundS18_overlay045_medium_live_py312_20260307.metadata.json`

Strict slot-policy uploader fix:
- `upload_pickle_with_age_guard.py` now normalizes pickle filenames before duplicate detection, matching audit logic and allowing duplicate-family slot selection.

Upload result:
- selected slot: `trianglewoman`
- upload id: `2fe5883d-79bf-4510-ad52-db53e6b0c573`
- terminal status:
  - `validationStatus=validated`
  - `triggerStatus=submission_succeeded`
- log snapshot:
  - `results/rotation_uploads_20260307_s18_trianglewoman.json`

## 2026-03-07 Round S21 (broad model + auxiliary target sweep, main-target evaluation)

Objective:
- Run a larger scout matrix across model families and auxiliary targets.
- Always evaluate on `target_ender_20` with CORR/BMC/payout proxy (`0.75*CORR + 2.25*BMC`, clipped `+-0.05` per era).
- Add ensemble variants and a ridge-based feature neutralization sweep.

Matrix run:
- runner: `round_s21_matrix_run.py`
- attempted configs: 17
- completed: 15
- failed: 2 (`TabPFNRegressor` CPU practicality limits in this environment)
- summary: `results/roundS21_run_summary.json`

Completed model families:
- `LGBMRegressor`: 3 runs
- `XGBRegressor`: 3 runs
- `CatBoostRegressor`: 3 runs
- `RidgeRegressor`: 3 runs
- `TorchMLPRegressor`: 2 runs
- `TabNetRegressor`: 1 run

Main-target evaluation artifacts:
- model-only summary: `results/roundS21_main_target_summary_models_only.csv`
- with ensembles: `results/roundS21_main_target_summary_with_ensembles.csv`
- neutralization coarse: `results/roundS21_neutralization_coarse.csv`
- neutralization fine: `results/roundS21_neutralization_fine.csv`
- final summary: `results/roundS21_main_target_summary_final.csv`
- final summary json: `results/roundS21_main_target_summary_final.json`

Top source=`model` runs on main target (`target_ender_20`):
1. `roundS21_xgb_medium_tender60_std008_rank` (`trained_target=target_ender_60`)
   - `corr_mean=0.016370`, `bmc_mean=0.005538`, `payout_mean=0.019269`
2. `roundS21_cat_medium_tender60_std008_rank` (`trained_target=target_ender_60`)
   - `corr_mean=0.013052`, `bmc_mean=0.003909`, `payout_mean=0.015390`
3. `roundS21_lgbm_medium_tender60_std008_rank` (`trained_target=target_ender_60`)
   - `corr_mean=0.015322`, `bmc_mean=0.003599`, `payout_mean=0.015208`

Top overall (including ensembles/neutralization):
- `roundS21_ens_rankmean_topk__evalmain`
  - `corr_mean=0.018724`, `bmc_mean=0.007363`, `payout_mean=0.024634`
- best neutralized row tied the same metrics:
  - `roundS21_neutbest_roundS21_ens_rankmean_topk__evalmain_smallc64_a0p0_p00__evalmain`
  - neutralization setting effectively selected `p=0.00`, `alpha=0.0` (no-change optimum under this grid).

Interpretation:
- Auxiliary-target tree models (especially `target_ender_60`) were the strongest *single-model* direction in this batch.
- Ensemble payout proxy improved versus individual models, but CORR cumsum delta vs `v52_lgbm_ender20` remained negative across this window, so benchmark-level CORR was not exceeded.

S21 plots:
- ensemble-centric:
  - `plots/v52_lgbm_ender20_vs_roundS21_ens_rankmean_topk__evalmain_plus_3_dark.png`
- model-only top-4:
  - `plots/v52_lgbm_ender20_vs_roundS21_xgb_medium_tender60_std008_rank__evalmain_plus_3_dark.png`

## 2026-03-07 Round S22 (medium/small/faith2 + aux targets + seed expansion)

### Goal
- Continue broad search for additive signal vs `v52_lgbm_ender20` using medium/small/faith2 feature mixes and auxiliary targets.
- Keep output in existing experiment folder and generate updated OOF diagnostics.

### Artifacts
- Runner: `agents/experiments/nn_ender20_latest_residualized/round_s22_matrix_run.py`
- Evaluator: `agents/experiments/nn_ender20_latest_residualized/round_s22_evaluate_main_target.py`
- Run summary: `agents/experiments/nn_ender20_latest_residualized/results/roundS22_run_summary.json`
- Final summary: `agents/experiments/nn_ender20_latest_residualized/results/roundS22_main_target_summary_final.csv`
- Neutralization sweep:
  - `agents/experiments/nn_ender20_latest_residualized/results/roundS22_neutralization_coarse.csv`
  - `agents/experiments/nn_ender20_latest_residualized/results/roundS22_neutralization_fine.csv`
- Combo sweep (top 5 models):
  - `agents/experiments/nn_ender20_latest_residualized/results/roundS22_combo_search_top5.csv`

### Result
- S22 best payout proxy came from `roundS22_ens_rawmean_topk__evalmain`:
  - corr_mean `0.017508`
  - bmc_mean `0.005940`
  - payout_mean `0.021273`
- CORR delta vs benchmark remained negative:
  - delta_cumsum_end `-2.359580`
  - delta_cumsum_min `-2.359580`
- Local ridge-neutralization sweep selected `p=0.00` (no uplift beyond base ensemble).

### Decision
- S22 branch improved internal payout relative to its own candidates but did not beat benchmark on cumulative CORR delta.
- Existing historical candidate `xgb_full_offset0_medium_walkfwd_lam009_eval575plus` remains the strongest additive CORR-delta profile in this repo (`delta_cumsum_end > 0`).

## 2026-03-07 Round S23 (MikeP-style wider2 MLP replication)

### Setup
- `TorchMLPRegressor` with hidden sizes `(1536, 1024, 512)`, lr `1.5e-4`, dropout `0`, medium features, residual std `0.008`.
- Run summary file: `agents/experiments/nn_ender20_latest_residualized/results/roundS23_mikep_like_run_summary.json`

### Early read
- Seed `2301` was non-additive vs benchmark on this setup:
  - corr_mean `0.002917`
  - bmc_mean `0.001089`
  - payout_mean `0.004311`
  - delta_cumsum_end `-4.635772`
- Remaining S23 seeds were stopped to preserve compute budget.

## 2026-03-07 Deployment Packaging (strict GBT lam009)

### Built cloudpickles
- Live artifacts (all eras):
  - `agents/experiments/nn_ender20_latest_residualized/models/xgb_strict_resid008_lam009_live_artifacts`
  - `agents/experiments/nn_ender20_latest_residualized/models/xgb_strict_resid008_lam009_live.pkl`
- Diagnostics-matched artifacts (eras <= 560):
  - `agents/experiments/nn_ender20_latest_residualized/models/xgb_strict_resid008_lam009_diag_artifacts`
  - `agents/experiments/nn_ender20_latest_residualized/models/xgb_strict_resid008_lam009_diag560.pkl`

### Upload status
- Target slot: `mark_twain` (old + failed-history eligible slot, no recent overwrite).
- New compute pickle id: `eaf153c8-82da-4272-9d86-5e41117fe660`
- Status at write time: `validationStatus=validating`, `triggerStatus=queued`.

## 2026-03-07 Main-target strict continuation (XGB + LGBM cap-sensitivity)

### Goal
- Continue strict additive research on **main target** (`target`) with benchmark-correlation cap constraints.
- Extend strict sweep coverage to:
  - `medium+faith2:64`
  - `small+faith2:64`
  - `faith2:96`
  - additional tree variants under same walk-forward protocol.

### Code updates
- Enhanced strict runner:
  - `agents/experiments/nn_ender20_latest_residualized/strict_gbt_walkforward_research.py`
  - Added feature-set token parsing (`+` composition and `:cap`) with `faith2 -> faith` alias.
  - Added CatBoost model-family support in strict walker.
  - Added new strict specs for `medium+faith2`, `small+faith2`, and `faith2` slices.
  - Added robust raw-cache fallback: if cache columns do not match requested target, retrain automatically.

### Key artifacts
- Main strict summary (target=main, blend-only strict pass):
  - `agents/experiments/nn_ender20_latest_residualized/results/gbt_strict_walkforward_summary.json`
- Main-target strict scan over all `*_strict.parquet` predictions:
  - `agents/experiments/nn_ender20_latest_residualized/results/all_strict_preds_main_target_scan.csv`
- Corr-cap sensitivity scan (top completed strict XGB caches):
  - `agents/experiments/nn_ender20_latest_residualized/results/strict_xgb_cap_sensitivity_main_target.csv`
- LGBM lambda scan on raw strict cache:
  - `agents/experiments/nn_ender20_latest_residualized/results/lgbm_dart_resid008_medium_lambda_scan_main_target.csv`

### Main findings
- For completed strict XGB caches, **no positive-delta feasible model at cap=0.99**.
- Relaxing to `cap=0.999` yields feasible positive-delta XGB candidates, but not with consistently nonnegative cumulative delta.
- Best strict-model delta in main-target scan came from prior LGBM strict run:
  - `lgbm_dart_strict_resid008_medium_walkfwd_strict`
  - `delta_cumsum_end=0.012664`, `payout_mean=0.026506`, `max_corr=0.990993` (just above 0.99).

### Tuned 0.99-cap candidate (main target)
- Built tuned candidate from LGBM raw strict cache:
  - model stem: `lgbm_dart_strict_resid008_medium_walkfwd_lam013_main_cap099`
  - prediction artifact:
    - `agents/experiments/nn_ender20_latest_residualized/predictions/lgbm_dart_strict_resid008_medium_walkfwd_lam013_main_cap099.parquet`
  - metrics:
    - `corr_mean=0.032688`
    - `bmc_mean=0.001111`
    - `payout_mean=0.026584`
    - `delta_cumsum_end=0.008513`
    - `delta_cumsum_min=-0.041257`
    - `max_corr_cap_metric=0.989644`

Interpretation:
- At strict `0.99` cap, we can obtain **positive end-of-window additive delta** and improved payout proxy.
- We still do **not** have fully monotonic/always-nonnegative cumsum delta (`delta_cumsum_min < 0`), so "consistently upward with no drawdown" remains unmet.

### Latest plots
- Benchmark vs tuned LGBM candidate:
  - `agents/experiments/nn_ender20_latest_residualized/plots/v52_lgbm_ender20_vs_lgbm_dart_strict_resid008_medium_walkfwd_lam013_main_cap099_dark.png`
- Candidate with payout-proxy subplot:
  - `agents/experiments/nn_ender20_latest_residualized/plots/lgbm_lam013_main_cap099_with_payout_dark.png`
