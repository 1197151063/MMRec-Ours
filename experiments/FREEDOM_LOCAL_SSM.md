# ID BPR, modality SSM, item-ID constraint

Model: `FreedomLocalSSM`. This implements the clarified FREEDOM modification.
Previous models/configurations remain available.

## Forward

1. Ordinary Xavier user-ID and item-ID embeddings, both 64-dimensional.
2. The UI branch takes ONLY those ID tables, applies two LightGCN layers on
   the TRAIN interaction graph, and averages layers 0/1/2. Retain FREEDOM's
   degree-sensitive .8 edge pruning for training; evaluate on full TRAIN.
3. No item-item message passing and no item-graph residual.
4. Image and text each use one Linear(raw_dimension,64) projector. Their
   pretrained feature tables remain trainable, as in original FREEDOM.
5. Full-sort scores = dot(user_UI, item_UI). Projected modalities are training
   targets; they are NOT fused with, or added to, the item-ID inputs/scores.

## Objective

    L = BPR(U_UI, I_UI)
        + modal_weight * [SSM(U_UI, image_projection) + SSM(U_UI, text_projection)]
        + ii_weight * [alpha * alignment_image(E_item)
                       + (1-alpha) * alignment_text(E_item)]
        + local_weight * local(E_user)

- BPR uses the batch positive and TRAIN-negative item IDs supplied by the loader.
- Each modality SSM uses user_UI, its positive projected feature, and 32 uniform
  random catalog negatives per interaction (same negative IDs for both modalities).
  tau=.04. This follows the user's original negative-only-denominator SSM:
  `mean(logsumexp(negative_cosines/tau) - positive_cosine/tau)`.
  Negatives are unfiltered (can include known positives or duplicates); the
  positive is not explicitly added to the denominator. This is not standard
  InfoNCE and can yield negative SSM values. No main ID SSM or modal BPR remains.
- Alignment operates on RAW item-ID embeddings E_item, not propagated item_UI
  or multimodal projections. Static separate image/text KNNs come from detached
  original features (K=10, self-neighbors retained, raw cosine weights).
  `sum_[i in batch positives,j in KNN_m(i)] weight_m(i,j)*softplus(-dot(E[i],E[j]))`.
  Correct weight indexing is [B,K], not [B,B,K]. Repeated positive anchors and
  sum reduction are retained; scale depends on batch size and K.
- alpha=.1 ONLY weights the visual vs textual ii constraints; it does not fuse
  features into the forward branch or weight the two modality SSMs.
- Default modal_weight=.01 and ii_weight=.0005 are experiment starting values,
  not tuned results. modal_weight is explicit, separate from FREEDOM's old
  reg_weight for modality BPR; no extra norm regularizer is silently introduced.
- User local loss is unchanged from FreedomLocal. It explicitly uses original
  numeric user-ID adjacency on raw E_user, rho=exp(-1/16), fixed initial scale.
  Thus this variant still uses the previously disclosed numeric-order prior.
  Set local_weight=0 to disable it; this is not part of the default sweep.

The model logs raw BPR, both SSMs, ii, local, total, weighted contributions and
user/item RMS. The generic trainer's legacy ID/mm diagnostic slots may be NaN
for scalar-return models; inspect the named `FreedomLocalSSM loss` entries.

## Run and logs

Only three new-model runs: local_weight=.01/.1/1, seed999. No old baseline rerun,
multiple training seeds or model checkpoint saving. Epoch selection uses the
existing validation protocol. Actual recommendation gains remain unmeasured.

```bash
cd /root/autodl-tmp/MMRec-Ours
git pull --ff-only
run="idbpr-modalssm-$(date +%Y%m%d-%H%M%S)"
nohup bash experiments/run_freedom_local_ssm.sh \
  /root/autodl-tmp/MMRec-Ours/data 0 "night_runs/$run" baby > "$run.log" 2>&1 &
echo "Run: $run"
# View queue and first training job; Ctrl-C stops tail, not training.
tail -F "$run.log" "night_runs/$run/idbpr_modalssm_local_w0.01_s999/console.log"
```

Return manifest.json, summary.csv and summary.json from that output directory.
CPU tests verify the exact forward/objective, isolated ii gradients on raw item
IDs, all trainable gradients, pruning/cache, pure BPR when auxiliaries are off,
and a three-job synthetic one-epoch queue without checkpoints. CUDA coverage is
conditional on hardware; CPU smoke runs do not establish recommendation quality.
