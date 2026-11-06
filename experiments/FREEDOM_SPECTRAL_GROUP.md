# Fixed spectral user groups on FREEDOM

This uses the ordinary TRAIN-graph spectral grouping selected after the local
diagnostic screen. It does not use timestamps, content features, numeric ID
distances, or complete-history anchor labels to form groups.

From binary TRAIN incidence `R`, compute `X = D_user^(-1/2) R D_item^(-1/2)`.
Take 64 left singular vectors of `X`, normalize each user row to unit length,
and run MiniBatchKMeans with **16 total user groups**. The clustering seed is
fixed at 2026. Groups are built once and stay fixed throughout recommendation
training; cold users receive no group contribution.

The local Baby screen found same-group rates of 15.55%, 13.05%, 10.71%, and
9.04% for user-ID offsets 1, 4, 16, and 64 respectively. The group-size-preserving
random-pair rate is 7.21%. The 16 groups have 349--2,279 users. These are
exploratory diagnostics, not a recommendation gain, and ID locality was used
to choose this rule from the structural candidates. It is not independent
validation of the rule or a reason to select recommender hyperparameters.

`FreedomSpectralGroup` constructs the input user representation as
`personal_user_embedding + beta * group_embedding[group[user]]`, before the
original UI propagation. The item branch and original FREEDOM objectives are
retained. The only optional new loss is `personal_reg_weight * mean_user
||personal_user_embedding||^2`. Group vectors use ordinary Xavier initialization.
This adds 16 * 64 = 1,024 learnable parameters at the default dimension.

The six experiments all use training seed 999, fixed groups and no baseline
rerun. They test beta 0.1/0.3/1.0 without the personal penalty, plus
`(beta, penalty) = (0.1, 0.01), (0.1, 0.1), (0.3, 0.1)`.

Run on the server from the repository root, with a fresh output directory:

```bash
git pull --ff-only
nohup bash experiments/run_freedom_spectral_group.sh /root/autodl-tmp/MMRec-Ours/data 0 \
  night_runs/freedom-spectral-group-1 baby > freedom-spectral-group-1.log 2>&1 &
```

View the queue and first experiment logs:

```bash
tail -F freedom-spectral-group-1.log \
  night_runs/freedom-spectral-group-1/freedom_spectral_b01_s999/console.log
```

The grouping hash and group sizes are written to every experiment log. Group
membership can change with clustering-library versions; the local verified
result used NumPy 1.24.3, SciPy 1.13.1, and scikit-learn 1.4.1.post1. The queue
does not save model checkpoints. Review `manifest.json`, `summary.json`, and
`summary.csv` after completion; model selection uses validation Recall@20.
