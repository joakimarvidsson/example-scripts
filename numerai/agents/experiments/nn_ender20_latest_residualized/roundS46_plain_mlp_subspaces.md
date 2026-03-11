# Round S46: Plain MLP Feature-Subspace Seed-Average Program

Goal: follow the recommended next step directly.

- keep the plain py3.12 MLP recipe fixed
- move variation into feature subspaces instead of target swaps or architecture changes
- run `3` scout seeds per subspace
- average predictions across seeds before scoring
- dense-confirm only if a subspace survives the scout gate

## Code changes

Updated:

- `strict_multitask_mlp_walkforward.py`

Changes:

- added deterministic custom medium feature masks:
  - `medium_mask20a`
  - `medium_mask40a`
  - `medium_mask40b`
  - `medium_mask60a`
  - `medium_mask60b`
- added plain-MLP subspace specs around `resid010`
- fixed a reproducibility bug:
  - torch model initialization now reseeds per spec / per block before building the network
  - before this fix, results depended on the order specs were run in the same process

## Broad subspace scout

Runtime:

- Python `3.12`
- `mps`

Scout setup:

- `eval-era-step = 8`
- `train-era-step = 4`
- `block-size = 26`
- `max_rows_per_era = 700`
- seeds: `1337`, `2021`, `2401`

Subspaces run:

- `medium:256+faith2:64`
- `medium:256`
- `faith2:128`
- `faith2:192`
- `medium:128+faith2:128`
- `medium_mask40a+faith2:64`
- `medium_mask40b+faith2:64`
- `medium_mask60a+faith2:64`
- `medium_mask60b+faith2:64`
- `medium_mask20a+faith2:192`

Raw-only summaries:

- `results/mtmlp_roundS46_subspaces_seed1337_rawonly_summary.json`
- `results/mtmlp_roundS46_subspaces_seed2021_rawonly_summary.json`
- `results/mtmlp_roundS46_subspaces_seed2401_rawonly_summary.json`

Initial broad seed-averaged result file:

- `results/corrcap_two_stage_roundS46_subspace_seedavg3_light.json`

Important caveat:

- the initial broad seed-averaged result is not reliable for promotion because it was generated before the per-spec torch reseeding bug was fixed
- it is still useful only as a shortlist source

Broad-pass shortlist chosen for rerun:

- `medium:256+faith2:64` baseline
- `medium_mask40b+faith2:64`
- `medium_mask60a+faith2:64`
- `medium_mask20a+faith2:192`

## Corrected shortlist rerun

Fixed-seed raw-only summaries:

- `results/mtmlp_roundS46_shortlistfix_seed1337_rawonly_summary.json`
- `results/mtmlp_roundS46_shortlistfix_seed2021_rawonly_summary.json`
- `results/mtmlp_roundS46_shortlistfix_seed2401_rawonly_summary.json`

Corrected seed-averaged result file:

- `results/corrcap_two_stage_roundS46_shortlistfix_seedavg3_light.json`

Corrected results:

### `medium:256+faith2:64`

- status: `no_positive_delta_feasible`
- `delta_mean = -0.000145`
- `delta_cumsum_end = -0.011273`
- `bmc_mean = 0.000328`
- `payout_mean = 0.024083`

### `medium_mask40b+faith2:64`

- status: `no_positive_delta_feasible`
- `delta_mean = -0.000058`
- `delta_cumsum_end = -0.004550`
- `bmc_mean = 0.000356`
- `payout_mean = 0.024342`

### `medium_mask60a+faith2:64`

- status: `no_positive_delta_feasible`
- `delta_mean = -0.000196`
- `delta_cumsum_end = -0.015309`
- `bmc_mean = 0.000314`
- `payout_mean = 0.024156`

### `medium_mask20a+faith2:192`

- status: `no_positive_delta_feasible`
- `delta_mean = -0.000075`
- `delta_cumsum_end = -0.005868`
- `bmc_mean = 0.000340`
- `payout_mean = 0.023993`

## Decision

- After fixing the seed bug, none of the shortlisted subspaces remained additive.
- The least-bad corrected subspace is `medium_mask40b+faith2:64`.
- It improves payout proxy slightly over the corrected baseline, but it is still negative on cumulative CORR delta.
- No dense confirmation, upload, or replacement is justified from this round.

## Takeaway

- Seed averaging is still worth keeping.
- Feature-subspace variation is directionally useful, but the best corrected subspace is only near-flat, not additive.
- The original positive broad-pass hit was largely seed-order noise.
