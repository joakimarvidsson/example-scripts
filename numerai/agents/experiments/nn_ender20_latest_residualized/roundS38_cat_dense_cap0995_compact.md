# Round S38: Dense Capped Confirm for `cat_strict_resid008_medfaith64_walkfwd`

Goal: take the `Round S37` scout winner and test whether the same model remains additive on the denser cached OOF slice under the hard `0.995` cap.

## Settings

- script: `tune_corrcap_two_stage.py`
- model file: `cat_strict_resid008_medfaith64_walkfwd_raw_walkfwd_dense_e4_r700.parquet`
- hard corr cap: `0.995`
- objective: `delta_cumsum_end`
- compact search around the same lambda/neutralization regime as `Round S37`

Result file:

- `results/corrcap_two_stage_roundS38_cat_dense_cap0995_compact_2026-03-11.json`

## Best dense result

- status: `no_positive_delta_feasible`
- `delta_mean = -0.000142`
- `delta_cumsum_end = -0.022090`
- `bmc_mean = 0.000703`
- `payout_mean = 0.025661`
- `payout_sortino = 16.384320`
- `corr_cap_used = 0.994222`
- selected lambda: `0.10`
- benchmark neutralize: `0.00`
- example neutralize: `0.00`
- feature spec: `none`

## Decision

- The `Round S37` CatBoost med-faith winner did **not** survive dense confirmation.
- It remained strong on BMC and payout proxy, but it reverted to clearly negative cumulative CORR delta.
- Do not promote this model under the hard `0.995` cap.
