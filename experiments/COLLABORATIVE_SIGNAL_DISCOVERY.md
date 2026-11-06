# Can a TRAIN-only collaborative signal recover the order-vote gain?

## Objective and boundary

The objective is to find independently available TRAIN-interaction/content
information that predicts the particular shared item revealed by numeric-ID
neighborhood votes. Merely transforming an ID-distance graph or its teacher
scores would still inject the numbering prior; that is not the target here.

Every candidate below is constructed from TRAIN and, for the semantic screen,
pretrained catalog features. User and item numbers are categorical indices only:
no ordering, index distance, full-history anchor, timestamp or order-generated
supervision is used. Independent random keys handle tied neighbor weights, and
permutation tests carry those keys with the corresponding entities. All model
selection is based on VALID Recall@20; no new TEST metrics were evaluated.

The post-hoc coverage audit is deliberately separate: it uses the previously
fixed order rule to label diagnostic subsets, never to construct, train or tune
any of the collaborative candidates.

## Actual local Baby results

All screens use 19,445 users, 7,050 items and 118,551 TRAIN edges, input SHA-256
e0abb033ea5cc538bb2becd8c3dc50b619f28f7974ba61f5ed11ce27cf405940.

| Independent signal | Screen | Best VALID Recall@20 | VALID NDCG@20 |
| --- | --- | ---: | ---: |
| Direct user-neighbor history votes | 13 settings | .07920 | .03731 |
| Candidate-item conditional/lift/Bayes evidence | 42 settings | .07168 | .03516 |
| Untruncated paths and joint shared-history support | 12 settings | .08252 | .03789 |
| Candidate-conditioned multi-interest semantic votes | 12 settings | .05990 | .02658 |
| Shared history denoising MLP | 2 settings, validation early stopping | .07735 | .03415 |
| Earlier explicit numeric-order votes, reference only | Uses ordering | .16415 | .12850 |

The 79 retrieval settings and two converged MLP settings are exploratory screens,
not a benchmark leaderboard or proof of an upper bound. The MLP initially ran
20 epochs; because both curves were still increasing, the same two settings
were rerun with max200, checking VALID every5 and patience6. Dropout .3 stopped
at50 with best20; dropout .6 stopped at70 with best40. Thus the extended result
is not explained by a 20-epoch time cutoff. It is only one hidden64 shared
architecture (909,514 parameters), not a claim about every neural model.

### Concrete signals and distinctions from earlier work

- User-neighbor votes directly aggregate neighbor TRAIN histories instead of
  applying an attraction loss to whole user embeddings. Cosine, rarity-weighted
  cosine and Jaccard were tested, with top20/80/256 and limited power/degree
  controls. Best was ordinary cosine at256.
- Item-conditioned scoring asks which missing item is supported by the history:
  conditional probability, shrunken lift, sum/max/top2/top3, geometric support
  and a smoothed Bayes latent-anchor likelihood. Extra nonlinearity did not
  improve on the best linear pooling here.
- Removing user-neighbor truncation slightly improved the score. The best was
  exactly R R.T R. Degree corrections did not improve it. Additional support
  choose(|H_u intersect H_v|, 2) also failed to improve it.
- Semantic retrieval preserves individual historical interests using max or
  sum over candidate-conditioned edges. Co-consumer confidence improves pure
  semantic scores, but max did not recover the order-vote advantage.
- The MLP is a shared sparse Linear(I,64)+ReLU+Linear(64,I), implemented with
  EmbeddingBag. It reconstructs TRAIN interactions from corrupted normalized
  histories, with no user table, order teacher or ID distances. It learned
  ordinary collaborative prediction but did not reproduce the strong scores.

## What information is missing from simple paths?

The diagnostic audit examines actual VALID edges, not just near-user similarity:

- All VALID positive edges: 20,559.
- Positive edges reachable by some TRAIN three-hop path: 11,898 (57.87%).
- VALID edges hit by the frozen order-vote top20: 3,255.
- Of those order hits, 1,041 (31.98%) have no TRAIN three-hop path.

This rules out explaining all order hits as merely recovering missed weights
on existing three-hop edges. It does NOT show that higher-order, semantic or
other learned inference is impossible. The remaining 68% still requires correct
ranking; coverage is not recommendation accuracy. Earlier exact-match controls
also found little residual preference similarity after the specific common item
was removed, weakening the assumption that entire nearby users have similar tastes.

## Conclusion of this round

The requested equally strong, independently constructible collaborative signal
has NOT been found. The experiments do not justify packaging an ID-derived
intermediate as a new collaborative signal, nor adding these weak screens to
FREEDOM and declaring the order advantage recovered. No new main-model training
queue is launched by this work. Useful next evidence would be a TRAIN-only
missing-item predictor that actually improves held-out recommendation, or a
trace of the original numbering/preprocessing that changes what information is
legitimately available. The current evidence is compatible with numbering
retaining pre-split relationships that TRAIN no longer exposes, but exact provenance remains
untraced and no general impossibility claim is made.

## Artifacts and verification

Reports in experiments/reports:

- baby_collaborative_neighbors.json
- baby_collaborative_anchors.json
- baby_untruncated_paths.json
- baby_semantic_history_retrieval_with_reference.json
- baby_history_mlp.json (initial budget)
- baby_history_mlp_convergence.json (early-stopped extension)
- baby_order_signal_coverage.json (POST-HOC diagnostic only)

Six focused test files, 17 tests in total, pass. They check sparse/dense
formula agreement, entity relabeling, gradients, TRAIN/TEST isolation, validation
early stopping, short-budget exits without fabricated winners, and reachable/unreachable audit examples. No checkpoints are saved.

All probes already ran locally. Optional CPU reproduction on the server, using
a fresh directory, does not rerun FREEDOM or the PE baseline:

```bash
cd /root/autodl-tmp/MMRec-Ours
git pull --ff-only
nohup bash experiments/run_collaborative_discovery.sh \
  /root/autodl-tmp/MMRec-Ours/data night_runs/collaborative-discovery-1 baby \
  > collaborative-discovery-1.log 2>&1 &
```

View the queue and individual logs:

```bash
tail -F collaborative-discovery-1.log
tail -F night_runs/collaborative-discovery-1/history_mlp.log
```
