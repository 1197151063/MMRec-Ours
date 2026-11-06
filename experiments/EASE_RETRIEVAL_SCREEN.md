# TRAIN-only EASE retrieval screen: negative result

This CPU screen checks whether a direct item-history retrieval model is a useful
next route after group and continuous-coordinate experiments underperformed.
It implements the established EASE model (Steck, WWW 2019,
https://arxiv.org/abs/1905.03375), not a new architecture or a reconstruction of
the numeric-ID PE prior.

For binary TRAIN matrix R, calculate P = inverse(R.T R + lambda I),
B_ij = -P_ij / P_jj with B_jj = 0, then score R B. No numeric index distances,
timestamps, grouping labels, or held-out edges enter fitting.

The local Baby screen used 19,445 users, 7,050 items and 118,551 TRAIN edges.
Full-catalog evaluation masks TRAIN positives only and uses the repository's
macro Recall/NDCG functions. Users with no TRAIN history are excluded. The
regularization is selected solely by validation Recall@20. No test results
were evaluated; no train+validation refit, checkpoint or score table was saved.

| Ridge | Validation Recall@20 | Validation NDCG@20 |
| ---: | ---: | ---: |
| 10 | 0.061616 | 0.031667 |
| 100 | 0.079377 | 0.039200 |
| 1000 | 0.081650 | 0.038761 |

Fitting took about 9.1--9.3 seconds per setting on the local CPU; fitting plus
validation took about 19 seconds each. These results do not justify claiming
that EASE matches PE, or launching a new training suite around it. Weak
standalone scores do not rule out complementarity in an ensemble, but no such
gain has been established. Historical neural results are not a fresh matched
benchmark comparison. Full measurements and the input hash are in
reports/baby_ease_retrieval.json.

The tests verify the closed-form coefficients against independently solved
columnwise constrained ridge problems, TRAIN masking, metric values and
consistent item relabeling. They do not establish a recommendation gain.

To reproduce on a CPU, choose a fresh output filename:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -u experiments/probe_ease_retrieval.py \
  --data-path /root/autodl-tmp/MMRec-Ours/data --dataset baby \
  --output night_runs/baby-ease-screen-1.json > baby-ease-screen-1.log 2>&1 &
```

View the log:

```bash
tail -F baby-ease-screen-1.log
```

This dense implementation is intentionally restricted to <=10,000 items; it
uses quadratic memory and cubic solve time. The optional --evaluate-test flag
evaluates test exactly once, after the validation winner is selected.
