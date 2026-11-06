# FREEDOM with an explicit original-user-ID neighborhood prior

This experiment is authorized as an **explicit numeric user-ID-order prior**.
It does not use sinusoidal PE, group embeddings, correlated initialization,
new graph coordinates or additional learnable parameters. It DOES use the
original numeric user ordering. It must not be described as an order-free
method, TRAIN-graph-inferred structure or an independently discovered taste graph.
Any gains can depend on dataset preprocessing and numbering artifacts. Reaching
the previous PE accuracy is a target, not an established result.

Original FREEDOM forward, two UI layers, one MM layer, ID dimensions and
initialization remain unchanged. Let z_u be the normalized final UI user vector.
For each unique user appearing in the training batch, sample four active TRAIN
users v with 0 < abs(u-v) <= radius, with probability proportional to
exp(-abs(u-v)/bandwidth), normalized separately at boundaries. No dense user-user
graph is stored. Cold users are excluded from the prior.

Two regularizers are compared:

- **attract:** mean(1 - dot(z_u, z_v)), a weighted normalized Laplacian-style
  smoothness loss estimated by sampling. It can oversmooth, including across
  numbering-block boundaries; no equality or final-distance guarantee is claimed.
- **local_rank:** sample active users w uniformly subject to abs(u-w) >= far_gap;
  mean(tau * softplus((dot(z_u,z_w) - dot(z_u,z_v) + margin) / tau)).
  Margin=tau=0.1. This compares near and far similarity instead of only attracting
  near users; it does not guarantee avoiding collapse. Distant users may actually
  share tastes, so this is an explicit and potentially wrong numbering assumption.

The total objective is original FREEDOM item BPR + reg_weight times visual/textual
BPR + order_weight times the prior. The prior uses the same UI forward result;
no extra UI propagation is run to calculate the loss. Anchors without eligible
partners are skipped. Batch anchors are deduplicated, but inclusion frequency
still depends on the original interaction sampler. The eight-job screen uses
seed 999, validation Recall@20 for selection, no baseline reruns or checkpoints:

| Mode | Bandwidth | Radius | Far gap | Weights |
| --- | ---: | ---: | ---: | --- |
| attract | 16 | 64 | unused | 0.01 / 0.1 / 1.0 |
| attract | 64 | 256 | unused | 0.01 / 0.1 / 1.0 |
| local_rank | 32 | 128 | 256 | 0.1 / 1.0 |

Every training log explicitly names ORIGINAL_NUMERIC_USER_ID_ORDER and reports
both the raw and weighted order loss. Existing baseline results should be reused
only when their data, objective and evaluation protocol match this FREEDOM setup.
Do not present this as a fair order-free comparison against baselines without
this additional prior. No held-out edges or labels are read to build the prior,
but using IDs assigned before splitting can still convey full-data preprocessing
information. Authorization to test it does not establish benchmark fairness.

Server run, with a fresh output directory:

```bash
cd /root/autodl-tmp/MMRec-Ours
git pull --ff-only
nohup bash experiments/run_freedom_order_prior.sh /root/autodl-tmp/MMRec-Ours/data 0 \
  night_runs/freedom-order-prior-1 baby > freedom-order-prior-1.log 2>&1 &
```

Queue and first job logs:

```bash
tail -F freedom-order-prior-1.log \
  night_runs/freedom-order-prior-1/freedom_order_attract_b16_w0.01_s999/console.log
```

Return manifest.json, summary.json and summary.csv. Check that every job reports
state=complete; timeout/failed jobs cannot be compared as converged results.

CPU tests cover near/far sampling support, inactive/boundary masks, both explicit
loss formulas and gradients, zero-weight FREEDOM equivalence, parameter count,
prediction consistency and all eight one-epoch synthetic runs. An optional CUDA
test checks constructor/device handling on hosts with CUDA.
