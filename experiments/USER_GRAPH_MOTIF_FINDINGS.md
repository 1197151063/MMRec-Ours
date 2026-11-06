# What the Baby ID-locality pattern corresponds to in the graph

This is a CPU structural investigation, with no recommender baseline training or
multi-seed evaluation. All candidate graph scores use TRAIN interactions and
pretrained item text. Numeric IDs define diagnostic pairs only.

## The strongest explanatory observation

For every user, take the minimum item index in their **complete** interaction
history. These minima are nondecreasing along the entire user-ID sequence.
86.83% of adjacent users have the same minimum item, and 87.32% share some item
in complete histories. Only 44.44% share an observed TRAIN item.

Among the 10,804 adjacent pairs without a common TRAIN item, 77.18% share an
item in complete histories and 76.74% share the same full-history minimum item.
The latter gives 2,562 contiguous forensic groups: group sizes have median 2,
90th percentile 16, and maximum 315.

The repository preprocessing notebook `0rating2inter.ipynb` enumerates users in
`pd.unique` first-appearance order before splitting. This code and the data
pattern are consistent with item-grouped input producing item-centered user
index groups. The exact original Baby raw-file processing has not been traced.

**Full-history minima and their group memberships are forensic labels only.**
They must not be used as training features, graph edges, or supervision. Minimum
TRAIN item index would also encode arbitrary item-index order, so it is not a
proposed model input.

The finding narrows the meaning of "similar taste": much of the local pattern
is a specific common-item relationship, rather than evidence that the users'
entire preference vectors should coincide. A random split hides many of these
common-item edges. A TRAIN-only method must infer the missing relations.

## TRAIN graph signals that retain part of the trend

On 2,498 adjacent pairs with no TRAIN item intersection, versus controls with
the same target-user degree, similar consumed-item popularity, and no TRAIN
intersection:

| Structure | Near mean | Matched far mean |
| --- | ---: | ---: |
| Weighted four-hop common-consumer cosine | 0.00667 | 0.00434 |
| Four-hop cosine with generic bridge users downweighted | 0.00408 | 0.00201 |
| Best co-consumption match between the two item histories | 0.02609 | 0.01949 |
| Shared inferred-item profile, top 32 missing-item candidates | 0.00875 | 0.00463 |
| Symmetric best text match between histories | 0.38372 | 0.37236 |

The four-hop motif is `user - item - bridge-user - item - user`. A bridge user
with too many different co-consumers is less specific, so it is downweighted.
The six-hop motif connects users through a common predicted missing item. Its
signal is sparse: positive profile overlap occurs in 18.3% of near pairs and
13.2% of controls, with many score ties. Ratios across different score types
are not directly comparable effect sizes.

For direct shared-item graphs, a fixed sample of 4,096 query users gives:

| TRAIN-only edge score | Adjacent target in top 20 | In top 80 |
| --- | ---: | ---: |
| Shared-history cosine | 6.79% | 20.53% |
| Rare-item-weighted cosine | 9.18% | 21.78% |
| Log-popularity-weighted cosine | 5.96% | 17.70% |
| Popularity-weighted cosine | 5.32% | 15.38% |

Thus emphasizing popular stars is rejected by this diagnostic, despite shared
near-pair items having higher median TRAIN degree (66 versus 31 across TRAIN
interactions). All rank ties use independent entity keys, not ID order.

Four-hop bridge scores also retain weak neighbor-selection signal without
shared TRAIN items: on 512 diagnostic query pairs, top-20 contains 3 near
targets and 0 matched far targets. This count is too small for a strong graph
recovery claim. The six-hop profile gives 1 versus 1 at top-20.

## Modeling implication for lightweight FREEDOM

Prioritize the union of **item-induced overlapping user groups**, rather than
64 broad user cohorts or a global user-vector attraction. Every TRAIN item
defines one group, without selecting or ordering items by their numeric IDs.
Retain multiple memberships, use degree-corrected group weights, and constrain
the shared-interest component of users in the same group. Four-hop bridge
scores can provide a separate, weak relation for users without direct overlap.

This is a concrete next hypothesis, not a demonstrated recommendation gain.
The graph motifs are standard structural quantities; novelty would have to
come from an effective, well-supported use in the recommendation method.

## Reproduce on CPU

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /opt/anaconda3/envs/CYJ/bin/python \
  experiments/mine_user_graph_motifs.py \
  --data-path /Users/chenyijun/Downloads/LightMMRec/data --dataset baby \
  --pairs 2500 --output experiments/reports/baby_user_graph_motifs.json \
  2>&1 | tee experiments/reports/baby_user_graph_motifs.log
```

View the log:

```bash
tail -f experiments/reports/baby_user_graph_motifs.log
```
