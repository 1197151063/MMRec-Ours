# Content item inputs, one SSM, item constraint, existing user local loss

New model: `FreedomLocalContent`. Original `FreedomLocal` remains available.

## Forward and learnable parameters

- User ID table P: ordinary Xavier uniform initialization, dimension 64.
- Visual/text feature tables: initialized from pretrained features, trainable as
  in FREEDOM (`freeze_features: False`). Thus there is no independent item-ID
  embedding table, but per-item feature parameters remain trainable. Set
  `freeze_features: True` to freeze them; this is not part of the default sweep.
- Two single Linear projectors: raw visual/text dimensions -> 64.
- Item input Q = alpha * image_projection + (1-alpha) * text_projection.
  Default alpha=.1 is the VISUAL weight; text weight=.9.
- Concatenate P and Q, apply the same two TRAIN UI LightGCN layers, and average
  layers 0/1/2. Preserve FREEDOM's degree-sensitive pruning (.8) for training;
  use the full TRAIN graph at evaluation. No item-item message passing or
  item-graph residual remains. This model is not graph-free.
- Prediction retains FREEDOM's raw dot product. SSM uses cosine scores during
  training, following the user's original SSM implementation.

## Objective

    L = SSM(U_UI, I_UI) + ii_weight * L_ii(Q) + local_weight * L_local(P)

There is one recommendation SSM, no BPR and no separate visual/text ranking
loss. SSM uses temperature .04 and 32 independently sampled uniform catalog
negatives per interaction. As in the supplied original SSM, sampling is
unfiltered (can include TRAIN positives, the anchor or repeated candidates),
and the denominator contains only negatives:

    mean[logsumexp(cos(u,negative)/tau) - cos(u,positive)/tau]

This is NOT standard positive-in-denominator InfoNCE; negative loss values are
possible and do not alone indicate an error. Evaluate downstream ranking to
judge results.

Static image/text KNNs are built separately from detached original pretrained
features using cosine similarity. Store only K indices/weights per item;
construction is blockwise, without retaining an I-by-I similarity matrix.
Default K=10 (same as FREEDOM), self-neighbors retained, raw signed cosine
weights retained. Q is the fused content BEFORE UI propagation:

    align_m = sum_{i in batch positives, j in KNN_m(i)}
              cosine_m(i,j) * softplus(-dot(Q[i], Q[j]))
    L_ii = alpha * align_visual + (1-alpha) * align_text

This corrects the supplied `[B,B,K]` broadcasting bug to `[B,K]` and replaces
`-log(sigmoid(score))` by its stable equivalent `softplus(-score)`. Repeated
positive anchors and SUM reduction are retained. Therefore ii_weight depends
on batch size and K; default .0005 is an experimental starting value. Signed
negative edge weights remain signed, rather than being silently clipped.
The constraint is an alternative training mechanism; it is not mathematically
equivalent to item message passing. With trainable feature tables it updates
both those tables and the projectors.

L_local is unchanged from FreedomLocal: full-table adjacent raw user quadratic
penalty, rho=exp(-1/16), fixed initial scale. It explicitly uses ORIGINAL
NUMERIC USER ID ORDER; replacing the item encoder does not remove that prior.

The loss logger prints raw SSM/ii/local, weighted contributions, total loss,
user RMS and content RMS. The generic trainer's legacy ID/modality diagnostics
may be NaN for scalar-return models; use the named model loss entries instead.

## Runs

Three new-model runs vary only local_weight=.01/.1/1, seed999. No old baseline
reruns, multiple seeds or model checkpoint saving. Recommendation quality has
not yet been measured. CPU checks cover exact fusion/UI/SSM/ii/total formulas,
gradients, pruning, cache invalidation and all three one-epoch queue jobs on
synthetic data. CUDA checks run only if CUDA hardware is available.

Use a fresh output directory (the timestamp also gives each launch a new log):

```bash
cd /root/autodl-tmp/MMRec-Ours
git pull --ff-only
run="content-ssm-$(date +%Y%m%d-%H%M%S)"
nohup bash experiments/run_freedom_local_content.sh \
  /root/autodl-tmp/MMRec-Ours/data 0 "night_runs/$run" baby > "$run.log" 2>&1 &
echo "Run: $run"
# View queue and the first training job; interrupt tail with Ctrl-C.
tail -F "$run.log" "night_runs/$run/content_ssm_local_w0.01_s999/console.log"
```

Return that run's manifest.json, summary.csv and summary.json for review.
