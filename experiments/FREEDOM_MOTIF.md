# Item-conditioned user groups and four-hop bridges

`FreedomMotif` adds two losses to original FREEDOM. It has the same learnable
parameters and prediction rule as FREEDOM. Graphs use TRAIN interactions only;
numeric ID distances and full-history minimum-item labels are not inputs.

For an observed `(u,i)`, sample another TRAIN consumer `v` of item `i`. This
preserves overlapping, item-level groups rather than merging users into broad
clusters. Let `h_u` be the normalized FREEDOM user representation and `q_i^m`
the detached, normalized existing modality projection. The semantic star loss is

`d * mean_m [((h_u - h_v) dot q_i^m)^2]`.

The loss compares normalized preference scores along the item's image/text
directions. It does not require the entire user vectors to agree. Normalization
still couples gradients across coordinates; this is not a guarantee that every
orthogonal coordinate is unchanged. Detaching the directions prevents the
projectors from changing them solely to evade the new constraint. Item weights
`degree(i)^(-p)` are normalized over the valid minibatch pairs. The `full`
ablation instead penalizes `||h_u-h_v||^2`. Singleton item groups contribute zero.

For bridges, build normalized shared-item user affinities `S`, downweight generic
bridge users by their UU degree, then compare their neighborhood vectors by
cosine. Keep 20 positive neighbors per user, excluding the user itself and
every user already sharing a TRAIN item. Independent tie keys break score ties.
This corresponds to `u-item-w-item-v` paths. The weak bridge loss is a weighted
mean of `||h_u-h_v||^2`. It does not add another propagation layer to prediction.

The objective is `FREEDOM_BPR + original_reg_weight * modality_BPR +
star_weight * star_loss + bridge_weight * bridge_loss`.

The eight-job screen uses seed 999 only and does not rerun the baseline:

| Variant | Star mode | Star weight | Popularity power | Bridge weight |
| --- | --- | ---: | ---: | ---: |
| star_sem_001 | semantic | 0.01 | 0.5 | 0 |
| star_sem_005 | semantic | 0.05 | 0.5 | 0 |
| star_sem_010 | semantic | 0.10 | 0.5 | 0 |
| star_full_005 | full | 0.05 | 0.5 | 0 |
| star_equal_items | semantic | 0.05 | 1.0 | 0 |
| bridge_001 | off | 0 | — | 0.01 |
| star_bridge_0003 | semantic | 0.05 | 0.5 | 0.003 |
| star_bridge_001 | semantic | 0.05 | 0.5 | 0.01 |

Run from the repository on the server, choosing a fresh output directory:

```bash
git pull --ff-only
nohup bash experiments/run_freedom_motif.sh /root/autodl-tmp/MMRec-Ours/data 0 \
  night_runs/freedom-motif-1 baby > freedom-motif-1.log 2>&1 &
```

View the queue log and the first experiment's training log:

```bash
tail -f freedom-motif-1.log
tail -f night_runs/freedom-motif-1/freedom_motif_star_sem_001_s999/console.log
```

The graph builder logs progress before bridge runs start training. Model
checkpoints are disabled by the queue. After completion, review `manifest.json`,
`summary.json`, and `summary.csv` in the output directory. Best epochs are chosen
using validation Recall@20. No recommendation improvement has yet been measured
for these new constraints.
