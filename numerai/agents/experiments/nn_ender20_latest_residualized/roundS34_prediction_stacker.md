# Round S34: Walk-forward prediction stacker

Goal: stack the strongest existing OOF models using a cheap walk-forward meta-model instead of another raw model family.

## stack_ridge_treepack_mlp_r006_a10

- base models: `cat_dense, xgb_hist, ridge_smallfaith, mlp_roundM6`
- feature mode: `raw_ranks`
- target mode: `residualized`
- merged eras: `78` (`577` to `1193`)
- predicted eras: `52` (`785` to `1193`)
- delta cumsum end: `0.007366`
- bmc mean: `0.000077`
- payout mean: `0.019961`

## stack_ridge_treepack_mlp_lgbm_r006_a10

- base models: `cat_dense, xgb_hist, ridge_smallfaith, lgbm_optuna_confirm, mlp_roundM6`
- feature mode: `raw_ranks`
- target mode: `residualized`
- merged eras: `78` (`577` to `1193`)
- predicted eras: `52` (`785` to `1193`)
- delta cumsum end: `0.008920`
- bmc mean: `0.000102`
- payout mean: `0.020042`

## stack_ridge_deltas_mlp_r006_a10

- base models: `cat_dense, xgb_hist, ridge_smallfaith, mlp_roundM6`
- feature mode: `tree_deltas_plus_mlp`
- target mode: `residualized`
- merged eras: `78` (`577` to `1193`)
- predicted eras: `52` (`785` to `1193`)
- delta cumsum end: `0.009999`
- bmc mean: `0.000102`
- payout mean: `0.020057`

## stack_ridge_cat_mlp_r006_a1

- base models: `cat_dense, mlp_roundM6`
- feature mode: `raw_ranks`
- target mode: `residualized`
- merged eras: `156` (`577` to `1197`)
- predicted eras: `130` (`681` to `1197`)
- delta cumsum end: `0.000009`
- bmc mean: `0.000000`
- payout mean: `0.023464`

## stack_ridge_treepackfull_mlp_r006_a10

- base models: `cat_dense, xgb_hist, lgbm_optuna_confirm, mlp_roundM6`
- feature mode: `raw_ranks`
- target mode: `residualized`
- merged eras: `156` (`577` to `1197`)
- predicted eras: `130` (`681` to `1197`)
- delta cumsum end: `0.002092`
- bmc mean: `0.000008`
- payout mean: `0.023493`

## stack_poslin_treepackfull_mlp_raw

- base models: `cat_dense, xgb_hist, lgbm_optuna_confirm, mlp_roundM6`
- feature mode: `raw_ranks`
- target mode: `raw`
- merged eras: `156` (`577` to `1197`)
- predicted eras: `130` (`681` to `1197`)
- delta cumsum end: `0.005823`
- bmc mean: `0.000020`
- payout mean: `0.023538`

## stack_poslin_treepack_mlp_raw

- base models: `cat_dense, xgb_hist, ridge_smallfaith, mlp_roundM6`
- feature mode: `raw_ranks`
- target mode: `raw`
- merged eras: `78` (`577` to `1193`)
- predicted eras: `52` (`785` to `1193`)
- delta cumsum end: `0.014516`
- bmc mean: `0.000138`
- payout mean: `0.020203`

## Decision

- The narrow winners that include `ridge_smallfaith` are not directly comparable to the dense incumbent because they only score `52` late eras (`785` to `1193`).
- The best comparable full-overlap stacker is `stack_poslin_treepackfull_mlp_raw`, which scores `130` eras (`681` to `1197`).
- On that comparable overlap, the stacker stays additive on CORR vs benchmark, but it does not beat the dense CatBoost incumbent on BMC or payout proxy.
- Summary artifact: `results/roundS34_stacker_vs_cat_overlap_summary.json`
- Plot artifact: `plots/v52_lgbm_ender20_vs_cat_dense_vs_stack_poslin_treepackfull_mlp_raw_overlap_dark.png`
