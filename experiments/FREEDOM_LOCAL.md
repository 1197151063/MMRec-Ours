# Only local user-order loss on original FREEDOM

This experiment injects an explicitly disclosed original numeric-user-order
prior THROUGH TRAINING, without adding PE or any new representation branch.
It uses no variance/covariance, anti-collapse or disentanglement objective.
Original FREEDOM forward, graph layers, BPR/modality BPR, dimensions, Xavier
initialization and prediction formula are retained. No parameters are added.

For raw user ID table P, the only additional loss is:

    local = mean_{adjacent active TRAIN users u,u+1; dimensions k}
            (P[u+1,k] - rho * P[u,k])^2 / c0
    total = original_FREEDOM_loss + local_weight * local

Here c0=(1+rho^2)*mean(P_initial^2) is a fixed, detached scalar established at
ordinary initialization. It makes weight strengths interpretable across table
sizes; there is no dynamic normalization that can cancel gradients. This is
only a positive rescaling of the quadratic penalty, not an extra regularizer.
All active adjacent edges are included every batch, rather than sampling nearby
users or normalizing their vectors to cosine. Gaps containing cold users are
not bridged by renumbering. An empty edge set contributes zero.

The default rho=exp(-1/16)=0.939413. Three runs use local_weight .01/.1/1, seed999.
There is no baseline rerun, random-ID experiment or checkpoint saving. Full-table
cost is O(users * embedding_dim) per batch, with no U-by-U matrix.

This is an AR(1)-style local quadratic penalty. It omits the boundary norm and
other terms of a full stationary Gaussian prior, so it must not be presented
as guaranteeing covariance rho^abs(u-v) in the fitted model. rho=0 acts as
one-sided row shrinkage; rho=1 penalizes adjacent differences; intermediate rho
mixes shrinkage and coupling. Increasing its weight does not guarantee monotonic
final pairwise similarity, improved ranking, or freedom from collapse. Only raw
user inputs are directly constrained; UI propagation can change final geometry.
The raw and weighted loss plus user RMS are logged to help detect norm shrinkage.

Although inference contains only original FREEDOM operations, this model STILL
uses numeric ordering during training. Its source is logged as
ORIGINAL_NUMERIC_USER_ID_ORDER; it is not an independently learned collaborative
signal and not an anti-collapse module. Recommendation gains remain unmeasured.

CPU tests check exact disabled-weight FREEDOM equivalence, unchanged predictions
before training, the full regularized objective, local descent and empty edges,
and all three one-epoch queue jobs without saved models. An additional CUDA test
is conditional on CUDA hardware.

Server run with a new output directory:

```bash
cd /root/autodl-tmp/MMRec-Ours
git pull --ff-only
nohup bash experiments/run_freedom_local.sh /root/autodl-tmp/MMRec-Ours/data 0 \
  night_runs/freedom-local-1 baby > freedom-local-1.log 2>&1 &
```

View queue and first job logs:

```bash
tail -F freedom-local-1.log \
  night_runs/freedom-local-1/freedom_local_w0.01_s999/console.log
```

Return manifest.json, summary.json, summary.csv after completion.
