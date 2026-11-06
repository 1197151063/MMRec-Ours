# Explicit user-order history votes: directly exploiting the observed shared-item pattern

This is an explicitly numbering-aware diagnostic recommender. It uses original
numeric user-ID distances and TRAIN interactions, without PE, learned user
groups, embedding initialization, multimodal inputs or gradient training.
It is not an order-free method or an improvement in multimodal representation
learning. Its dependency on preprocessing-dependent numbering must be reported.

The earlier finding concerns a specific shared item, not necessarily similar
entire preference vectors. Here it is used directly in item recommendation:

    score(u,i) = sum_{v: 0 < |u-v| <= radius}
                 exp(-|u-v| / bandwidth) * R_train[v,i] / max(degree_train[v],1)^power

The target user's own row is excluded from voting. TRAIN items are masked at
evaluation. A fixed 1e-8 * normalized TRAIN item-popularity fallback handles
zero-score ties; this fallback is identical for every candidate. No timestamps,
full-history anchors or held-out labels are used to calculate scores. Numeric
IDs themselves may carry information from preprocessing before splitting, so
TRAIN-only edges do not make this a fair order-free benchmark comparison.

## Local Baby results

Fourteen settings were fixed before evaluation: bandwidths 1/2/4/8/16/32/64,
radius=4*bandwidth, and user-degree powers 0/1. Validation Recall@20 selected
bandwidth=16, radius=64, power=0. Test was evaluated only for that selected model.
Rechecking the same frozen winner with the actual RecDataset, EvalDataLoader
and TopKEvaluator reproduced the probe metrics. No test-driven setting changes
were made. There are 19,445 users, 7,050 items and 118,551 TRAIN interactions.

| Split | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.1480 | 0.1641 | 0.1244 | 0.1285 |
| Test | 0.1478 | 0.1633 | 0.1239 | 0.1279 |

These values exceed the previously supplied PE table's Baby values
(0.1326 / 0.1535 / 0.1068 / 0.1122). That is a comparison to a historical report,
not a fresh matched PE training rerun. They also exceed the recorded historical
reference_both_s999 test metrics (0.1292 / 0.1524 / 0.0999 / 0.1059).
Other datasets and server environments have not yet been evaluated here.

The selected vote matrix took about 0.47 seconds to construct locally and
contains 11,406,745 nonzeros (about 91 MB for float32 CSR data/indices before
object overhead). No matrix or model checkpoint is saved. Scores are densified
only in evaluation batches. All 14 validation candidates and the single selected
test result are in reports/baby_order_history_votes.json; native-loader checks
are in reports/baby_order_history_votes_native.json.

This directly demonstrates that original numbering can supply strong predictive
information without multimodal representation learning. It does not establish
the provenance or validity of the numbering, guarantee gains under renumbering,
or show that adding votes to FREEDOM improves on the standalone vote result.
The eight order-loss GPU experiments need not finish to reproduce this result;
their separate question is whether a learned model benefits from that loss.

## Reproduce on the server (CPU; no neural training)

Choose a fresh output filename. The wrapper runs the 14-candidate validation
screen, evaluates the selected winner on test, then checks that frozen winner
with the native loaders/evaluator:

```bash
cd /root/autodl-tmp/MMRec-Ours
git pull --ff-only
nohup bash experiments/run_order_history_votes.sh \
  /root/autodl-tmp/MMRec-Ours/data night_runs/baby-order-votes-1.json baby \
  > baby-order-votes-1.log 2>&1 &
```

View the log:

```bash
tail -F baby-order-votes-1.log
```

Return baby-order-votes-1.json and baby-order-votes-1.native.json from night_runs.
The native output should end with matches_probe=true and the metrics above if
the same input data is used (input SHA-256 is checked). Different libraries can
break exact ties differently, so numerical reproduction remains a check to run,
not a guarantee about an untested server.
