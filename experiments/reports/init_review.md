# Baby initialization study review

Inputs: user-supplied summary (1).csv, summary (2).json, manifest (1).json,
and baby-init-1-neighbors.json. Manifest revision 8c5ad6e, seed 999, 1000-epoch cap.
All 11 jobs complete; CSV and JSON fields agree, names match manifest. Reported aggregate
job time is 1.227 hours. This verifies summary consistency, not raw logs/data provenance.

| Run | Validation R@20 | Test R@20 | Test N@20 |
| --- | ---: | ---: | ---: |
| Original no PE | .0796 | .0797 | .0360 |
| Original user PE forward | .1577 | .1542 | .1018 |
| Original user PE initialization | .1580 | .1549 | .1080 |
| Clean baseline | .0875 | .0880 | .0386 |
| Random init .1 / 1 | .0871 / .0859 | .0878 / .0861 | .0385 / .0380 |
| Interaction init .1 / 1 | .0865 / .0857 | .0868 / .0844 | .0378 / .0376 |
| Content init .1 / 1 | .0852 / .0855 | .0853 / .0865 | .0375 / .0383 |
| Mismatched interaction init 1 | .0865 | .0842 | .0369 |

No history initializer beats the clean baseline on validation. Interaction init at strength 1
is also below its mismatched control on validation. Single-seed differences do not establish
statistical significance, but do not justify scaling the same initialization-only idea blindly.
Moving user PE to initialization retains the original-order gain; it does not remove the
arbitrary-ID dependency. Compare within model families: original and clean differ in more than PE.

Training-only neighbor sample: 10,000 pairs, 7,812 unique anchors, 118,551 binary train pairs.

| Statistic | Adjacent numeric users | Random | Degree-matched random |
| --- | ---: | ---: | ---: |
| Any shared training item | 44.23% | 1.80% | 1.85% |
| Mean Jaccard | .046997 | .001468 | .001426 |
| Popularity-weighted Jaccard | .039822 | .001098 | .001077 |
| Visual history cosine | .619308 | .577214 | .576540 |
| Text history cosine | .653332 | .607877 | .606444 |

Adjacent mean Jaccard is roughly 33× the degree-matched control; overlap probability roughly
24×. Similar average degree and popularity-downweighted results make degree alone an inadequate
explanation for this descriptive contrast. Reused users/pairs are not independent samples.
The index mapping contains strong training-cohort structure. This supports the order-dependence
interpretation, but does not establish how IDs were created or prove held-out leakage. Inspecting
original preprocessing provenance remains necessary before making such a claim in the paper.

Next experiment: persistently share projected content history through the existing linear
projectors; do not just add a one-time offset. Enforce leave-one-out training; compare own versus
shuffled histories and beta=0 under matched hyperparameter searches. See the README profile suite
for 70 configurations and three interleaved training seeds. No SOTA claim until actual runs finish.
