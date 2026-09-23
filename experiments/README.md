# Overnight hypothesis audit

Use the existing server environment, from the repository root:

```bash
git pull --ff-only
nohup python -u experiments/run_night.py \
  --data-path /root/autodl-tmp/MMRec-Ours/data --dataset baby --gpu-id 0 \
  --hours 8 --output night_runs/baby-night-1 \
  > baby-night-1.log 2>&1 &
```

Default: 50 predeclared configurations, training seed 999, 1000 epoch ceiling,
45 minute per-job timeout, sequential GPU use, 8-hour total budget. No checkpoints.
This is a queue budget, **not a promise that all 50 finish in eight hours**.
An active job is terminated if its timeout or the total budget is exhausted.
Failed jobs print their last 35 log lines and exit code is recorded. The queue stops after
3 consecutive failures/timeouts (`--max-consecutive-failures` adjusts this threshold).
Successful jobs are skipped when rerunning the exact same command. Failed/timed-out
jobs retry from the beginning; there are intentionally no optimizer checkpoints.
To extend time, use the same output directory and e.g. `--hours 12 --job-minutes 90`.
Changed configurations/code require a new output directory. Do not change data files
in place between resumes: the manifest records the path, not a full data checksum.
Linux/macOS file locking prevents concurrent queues sharing one output directory.

Preview (no training/data access):

```bash
python experiments/run_night.py --data-path /root/autodl-tmp/MMRec-Ours/data --dry-run
```

Short real-data smoke run, isolated output:

```bash
python experiments/run_night.py --data-path /root/autodl-tmp/MMRec-Ours/data \
  --limit 2 --epochs 2 --hours 0.5 --output night_runs/smoke
```

Use `--seeds 999 2024 2025` in a **new** output directory for 150 configurations,
with a proportionally larger time budget. This loops the entire 50-job plan for each
seed, not just the best-performing configuration. `--cpu` is for small smoke tests.

## Reading the results

- `manifest.json`: hypotheses, complete overrides, source fingerprint and git revision.
- `summary.csv` / `summary.json`: every job, status, runtime, best epoch (1-based),
  validation metrics and corresponding test metrics. Rows stay in planned order;
  results are never sorted or selected by test performance.
- `<job>/override.yaml`, `console.log`, `result.json`, `csv/`: complete per-job evidence.
- Outer `baby-night-1.log`: queue progress. Existing legacy logs are not overwritten.

Return `summary.csv`, `manifest.json` and logs of failures/unusual results for review.
There is no need to send all per-epoch logs initially. Background launching is your
server command; this desktop agent has not started a remote server process.

## Hypotheses and limits

| ID | Controlled comparison | What it can support |
|---|---|---|
| H1 | no PE vs original both-side PE | Reproduce historical gain before explaining it |
| H2 | user-only, item-only, both | Locate the responsible side and interaction |
| H3 | random both/user/item, training-file order, shuffled-training order, train-degree order | Distinguish index coupling, row-order artifacts and observable popularity |
| H4 | sinusoidal vs independent Gaussian vectors vs shared constant vector, equal norm | Separate coordinate geometry, fixed identity cues and common-offset effects |
| H5 | PE scale 0.01/0.1/0.3/1/3 | Identify whether amplitude dominates content/user initialization |
| H6 | item coordinates offset by 1000/10000 | Probe cross-type coordinate coupling while retaining within-item PE distances |
| H7 | raw-dot vs cosine evaluation, both/no PE | Probe whether item norms contribute to reported ranking gains |
| H8 | frozen vs trainable modality lookup tables, both/no PE | Probe dependence on per-item feature updates |
| H9 | independent order seeds 2026/2027/2028 | Check stochastic order construction sensitivity |
| H10 | content/ID with temperature 0.2/0.3/0.5 | Continue the previous boundary-optimum search fairly |
| H11 | text weights 0.25/0.5/0.75 across temperatures | Check visual-heavy vs text-heavy fusion |
| H12 | small item residual and stronger L2 at temperature 0.2 | Revisit residual after correcting the previous low-temperature setting |

Original-family experiments preserve the supplied MLP, trainable features (except H8),
32 unfiltered negatives, decoupled loss, tau=0.04, and raw-dot evaluation (except H7).
Clean SIMMRec experiments use frozen content, sampled softmax, filtered 128 negatives,
cosine train/eval and explicit L2. Do not treat comparisons **across** these families
as single-factor causal estimates. Alpha=0.5 and learning rate=0.001 for the original
family still require checking against the historical PE run.

`original` uses the existing numeric IDs. `train_file` means the current filtered
training-file row order, which may already be sorted by user ID or carry full-data
provenance; it cannot reconstruct pre-split raw-file order. Training degree uses only
training counts and independent random tie-breaking. These are diagnostic controls,
not proof of leakage, causal effects, or legitimate semantics of arbitrary identifiers.
Even an effective train-degree coordinate should be disclosed as a popularity-derived
structural feature. An item-coordinate shift also changes its relationship to learned
content axes, so H6 alone does not isolate the user/item cross-term.

Do not use PE-dependent gains as evidence of stronger semantic learning. Select
hyperparameters on validation, report matched-seed results, and validate any mechanism
across datasets before revising the paper's claims. No accuracy improvement is promised.

## Minimal user initialization suite (11 configurations)

```bash
nohup bash experiments/run_init.sh /root/autodl-tmp/MMRec-Ours/data 0 night_runs/baby-init-1 > baby-init-1.log 2>&1 &
```

First computes training-only adjacency statistics (numeric user neighbors, random partners,
log2-degree-matched partners) with shared anchors, binary-history Jaccard, popularity-downweighted
Jaccard and normalized visual/text history cosine. Writes `night_runs/baby-init-1-neighbors.json`.
Sampled pairs repeat users: descriptive means are not significance tests. Existing numeric IDs
and file provenance remain a confound; this diagnostic does not establish legitimate ordering.

Then runs `--suite init`: three original-family mechanism controls (no PE, user PE in forward,
user PE only at initialization), plus eight clean SIMMInit variants. Clean models freeze content,
use one linear projector per modality, alpha=0.75, tau=0.2, 128 negatives, L2=0.0001, no item ID.
Variants: unchanged user initialization; added unit random user directions at strengths 0.1/1;
training-history random-item-code averages at 0.1/1; initial projected-content history averages
at 0.1/1; random reassignment of the interaction-derived directions at strength 1. Histories are
binary, training-only; missing history gives zero direction. No history aggregator persists
in forward/inference, no extra loss is introduced. Initializer seed 2026 is independent of training
RNG; the random item code table is discarded. Entity relabeling tests must permute these codes
with their entities; regenerating codes by numeric row under the same seed is not that test.
The mismatch control uses a random permutation, which can retain a few fixed points.

Original user-PE initialization still uses arbitrary user IDs and is a diagnostic, not the proposed
order-free method. Algebraic equivalence assumes the unchanged original loss and no user weight
decay/regularizer; finite precision can cause trajectories to diverge (Adam can amplify nearly-zero BatchNorm bias gradient differences). The algebraic test uses float64; full server runs measure float32 behavior rather than assuming identical results. The clean-family L2
regularizes the final user parameter after initialization, not its displacement from initialization.
Feature preprocessing and one-time initialization add training startup work, not inference work.

Queue timeout/resume/no-save behavior is unchanged. For three training seeds use `run_night.py
--suite init --seeds 999 2024 2025 ...` with a new output directory (33 configurations).
Return the neighbor JSON, manifest.json and summary.csv. Success means beating matched-strength
random controls, not merely beating the unchanged initializer. Select by validation only.

## Persistent shared history profiles (70 configurations × 3 seeds)

```bash
nohup bash experiments/run_profile.sh /root/autodl-tmp/MMRec-Ours/data 0 night_runs/baby-profile-1 48 baby > baby-profile-1.log 2>&1 &
```

The wrapper recomputes neighbor diagnostics and runs 210 jobs sequentially, with a 48-hour
queue budget, 45-minute per-job cap, 1000-epoch maximum and existing validation early stopping.
No checkpoints are saved. Runtime is not guaranteed: pending/timed-out jobs can be resumed
with exactly the same command and unchanged source. Completed jobs are skipped. Never run two
queues on the same output directory. Use a new output directory after changing source/configs.
For Sports/Clothing pass the corresponding dataset name and a distinct output directory;
run sequentially on a single GPU. To screen one seed use `run_night.py --suite profile --seeds 999 ...`.

SIMMProfile precomputes each user's weighted mean of frozen, normalized raw modality features
from binary TRAIN interactions. It reuses the item's two linear projectors to produce a user
history representation on every update. The representation is normalized and added to the
learnable user vector with fixed weight beta. No added trainable parameters, no item IDs,
no arbitrary numeric-order encoding, no iterative graph propagation, no auxiliary loss.
This is explicit history aggregation and must be described as such; it is not structure-free.
There is additional startup work, U×(visual_dim+text_dim) buffer storage, and per-batch user
projection work. At evaluation, projected user/item representations use the existing cache.

Training removes the current positive item from the history mean (leave one out), including
its contribution to the denominator. All negatives in that row share the same resulting user
representation. Empty histories yield zero profile. Validation/test use full TRAIN history only.
Shuffled controls reassign histories across users; removal is conditional on the target actually
being present in the reassigned history. These random mappings must co-permute in relabel tests.
The no-user-ID variant freezes and ignores the lookup, leaving only shared projector learning.

| Block | Configs per seed | Purpose |
| --- | ---: | --- |
| Priority controls | 12 | beta=0 baseline; own/shuffled beta=.1/.3/1/3; no user ID; detached history gradient; target-included diagnostic |
| Matched search | 44 | tau=.1/.2/.3/.5, alpha=.5/.75/1, beta=0/.3/1/3; excludes 4 previously covered settings |
| Popularity weighting | 6 | train item degree exponent .5/1 × beta=.3/1/3 |
| Residual regularization | 8 | reg=0/.001 × beta=0/.3/1/3 |

Seeds 999/2024/2025 are interleaved per configuration; the first 36 jobs cover all priority
controls. Select by mean validation Recall@20 across completed seeds, never best test result.
The target-included run is a shortcut diagnostic and excluded from candidate selection.
Tune the no-history baseline under the same temperature/fusion/regularization ranges before
claiming a profile gain. Random controls have fixed profile_seed=2026 across training seeds;
this checks optimization variability, not robustness across shuffle realizations. Wide search
can overfit validation; independently validate a locked choice on other datasets/seeds.

```bash
python experiments/review_profile.py night_runs/baby-profile-1 > night_runs/baby-profile-1/review.json
tail -n 30 baby-profile-1.log
```

Return `summary.csv`, `summary.json`, `manifest.json`, and the sibling `baby-profile-1-neighbors.json`.
For failed jobs include their `console.log`. Do not infer model quality from unfinished jobs.

## Correlated initialization and components (109 jobs per seed)

```bash
nohup bash experiments/run_corr.sh /root/autodl-tmp/MMRec-Ours/data 0 night_runs/baby-corr-1 48 baby > baby-corr-1.log 2>&1 &
```

Default is training seed 999, 109 configurations, 48-hour queue budget, 45-minute/job cap,
1000-epoch maximum, validation early stopping, no checkpoints. Existing queue resume semantics
apply. For three interleaved seeds (327 jobs), invoke `run_night.py --suite corr --seeds 999 2024
2025 --hours 96 --data-path ... --output night_runs/baby-corr-3seed` instead. Keep output names
separate and avoid simultaneous queues on one GPU. Real runtime depends on graph construction.

### Initializer

CorrRec starts from LightMRecNoPE (original MLP/BatchNorm, trainable raw features, raw-dot scoring,
unfiltered sampled negatives, decoupled contrastive objective). Only the user table is replaced:

`e[u] = std * (sqrt(1-rho) * epsilon[position[u]] + sqrt(rho) * z[position[u]])`

In implementation the complete mixed process is built in position order, then assigned to users;
therefore epsilon rows co-permute with z in the random-order control. With independent Gaussian
innovations delta: `z[0]=delta[0]`, `z[p]=eta*z[p-1]+sqrt(1-eta^2)*delta[p]`.
Each coordinate has marginal variance std² and lag-k covariance std²*rho*eta^k (k>0).
No rowwise normalization or post-hoc sample centering is applied. rho=0 is IID. All generation
uses a local RNG, not training/dropout/sampler RNG. Initializer seed is training seed + 1027;
order shuffle seed is 2026. Multiple training seeds also vary the process realization.

Shared-noise controls implement the literal epsilon-reused formula. Their marginal variance
is NOT std²: away from the first row it is multiplied by
`1+2*sqrt(rho*(1-rho))*sqrt(1-eta^2)`; the first row uses factor
`1+2*sqrt(rho*(1-rho))`. Treat these as separate controls, not matched-variance estimates.
Original numeric ordering remains an explicit experimental dependency, even though it is used
only at initialization. A successful run does not make the method invariant to ID remapping.

### Queue order and components

First 6 jobs: unchanged Xavier no-PE reference; IID Gaussian std .01/.1/sqrt(.5); correlated
anchor rho=.75, eta=.99, std=sqrt(.5) in original and random user order.
Next 30 jobs: 10 component settings, each with original correlated, random-order correlated,
and IID at the same std. Components are:

1. modality-specific SSM auxiliary weight .01;
2. modality-specific SSM auxiliary weight .1;
3. BPR primary loss;
4. BPR + modality SSM .1;
5. item-ID residual weight .1;
6. one user-item propagation layer, mean including layer zero;
7. semantic item KNN residual .1;
8. semantic KNN / finite PPR mixture residual .1;
9. user dropout .1 before normalized SSM;
10. combined item-ID 1 + UI layer 1 + item graph 1 + BPR + auxiliary .1 + L2 .0001.

Remaining 70 grid jobs cover rho=.25/.75/1 × eta=.5/.9/.99/.999 × std=.01/.1/sqrt(.5)
× original/random ordering, excluding the two anchor jobs. Final 3 controls reuse epsilon at
eta=.5/.9/.99. Hyperparameters are predeclared; select by validation, not by the test metric.
Shared-noise controls and combined models should be reported separately from one-factor effects.

### Adaptation details versus supplied FREEDOM snippet

This is a component adaptation onto the content baseline, not an exact FREEDOM reproduction.
The content MLP stays in the scoring path; optional IDs are additive. There is one shared forward
per loss evaluation. SSM uses stable logsumexp; BPR averages over sampled negatives. Auxiliary
text/image losses share the same candidates, with visual weight 1. The inherited sampler can draw
known positives as negatives, deliberately retaining the original protocol for these ablations.
Adam uses explicit zero weight decay; optional sampled embedding L2 is active. The snippet's
AdamW default weight decay and commented-out L2 are not silently carried over.

Semantic KNN uses initial detached raw features, normalized cosine, binary positive edges,
visual/text mixture .1/.9, k=10, no self-neighbors. Cosines are computed in 256-row blocks.
PPR uses TRAIN binary interactions, alpha=.15 as propagation probability, restart .85, 20 finite
iterations, no transition-diagonal renormalization. Exact finite-polynomial rows are computed
in blocks of 128 and top-k is applied at the end. Unlike the snippet, the final self diagonal
is removed, so top-k counts other items. Mixture is .9 semantic + .1 PPR where enabled.
The resulting graph is explicitly symmetrized and degree-normalized. No complete dense I×I
array is constructed. The sparse item transition can still become large on dense datasets;
blockwise dense workspace does not imply overall linear sparse-graph memory. No torch_sparse
or new library dependency is required. Graphs are built once per job and reused in forward.

All early-stopping/best-test reporting stays tied to the best validation epoch. The reference
snippet updates `best_result` directly using test metrics; that selection behavior is excluded.
Graph/auxiliary/ID components add computation or parameters and must not be called structure-free.
Return summary.csv, summary.json and manifest.json; add console.log for failed jobs.

## Group-count sensitivity with two-layer LightGCN and BPR

```bash
nohup bash experiments/run_group.sh /root/autodl-tmp/MMRec-Ours/data 0 night_runs/baby-group-count-1 6 baby > baby-group-count-1.log 2>&1 &
```

The queue runs THREE configurations, seed999, with 32, 64 and 128 total contiguous user groups,
group strength1. Membership is floor(user_id * num_groups / n_users), giving balanced group sizes
that differ by at most one when all numeric user IDs are present. The table has num_groups rows; tiny synthetic datasets with fewer users than groups necessarily
leave some groups empty. No random grouping experiments are run. Only num_groups changes
between jobs; compare against the previously completed 16-group run. The standalone model
default remains 16 groups. Group-vector RNG is local, so changing table size does not change
personal/item initialization or training RNG. Original numeric order remains an explicit input to grouping.

User/item ID embeddings (64-dimensional, Xavier initialization) are concatenated and propagated
through two LightGCN layers on binary TRAIN interactions only. The adjacency is symmetric
D^-1/2 A D^-1/2, with no self loops. Layer0, layer1 and layer2 are averaged. The learnable group
vector is then added to the propagated USER vector; it is not inserted into graph propagation.
Group vectors retain Normal(std=.125) initialization. Image/text projectors remain single
Linear(input_dim,64) layers and raw feature tables remain trainable.

Loss = BPR(grouped propagated user, propagated item ID) + alpha*SSM(user, projected image)
+ (1-alpha)*SSM(user, projected text). Alpha=.5 (image weight). BPR uses raw-dot scores and
averages softplus(negative-positive) across the same 32 unfiltered uniform negative candidates.
Both modality SSM terms retain cosine normalization, tau=.04 and the negatives-only denominator.
No new regularization, dropout or graph auxiliary loss is added. All terms reuse one graph forward.
Inference uses grouped propagated users and propagated item IDs with raw-dot scoring.

The queue retains validation-based early stopping, 1000-epoch maximum, 6-hour total budget,
and no checkpoint saving. Use the new output directory above; old manifests are incompatible.
Return summary.csv, summary.json and manifest.json. Local tests use synthetic data only.

## LinkProp-inspired weighted BPR (three runs)

Source: Fu et al., *Revisiting Neighborhood-based Link Prediction for Collaborative Filtering*,
arXiv:2203.15789v1, §3.1 Eqs.11–15, §3.3. The paper proposes degree-aware three-hop link scores
and directly searches parameters using validation NDCG. It does NOT propose the weighted BPR
objective below. This experiment is an adaptation, not a reproduction of LinkProp/LinkProp-Multi.

For each observed binary TRAIN edge (u,i), compute structural support:

`q_ui = sum_{x != i, v != u} R_ux R_vx R_vi / (d_x^beta d_v^gamma d_i^delta)`.

We fix beta=gamma=delta=.5, and the focal user exponent to zero, following the score form of
Eq.12. These are experimental choices, not the paper's tuned optimum. All degrees are fixed on
the full binary TRAIN graph. We remove walks that reuse the scored edge (x=i or v=u), so an
observed edge does not support itself via a trivial backtrack. This is not a full leave-one-edge-out
recomputation of degrees. No validation/test edges or scores enter these weights. No pseudo-links
or iterative degree updates are added. Triples are computed in 32-user sparse row blocks and only
observed-edge scores are retained, avoiding a full U×I score table. Dense interaction graphs can
still create large intermediate sparse blocks; startup time is logged.

Define `c_ui=1+log(1+q_ui)` and `w_ui=(1-lambda)+lambda*c_ui/mean_train_edges(c)`.
The +1 keeps zero-support positives trainable. Weights are positive, have mean1 over unique
training edges, and are fixed buffers without gradients. This normalization is global, not
per minibatch. The hypothesis is that independently supported positives deserve greater BPR
weight; this does not establish that they are less noisy or guarantee better rankings.

`L = mean_{u,i,j}[w_ui * softplus(s_uj-s_ui)] + alpha*SSM_image + (1-alpha)*SSM_text`.

Only BPR is weighted. Both modality SSM terms, their coefficients/temperature, the shared
unfiltered 32-negative sampler, and inference are inherited unchanged. Keep 16 total contiguous
user groups, two-layer LightGCN and all original default optimizer/initialization settings.
The suite compares lambda=0/.5/1, seed999. Lambda0 skips support preprocessing and is exactly
ordinary BPR. This queue is separate from the group-count sensitivity queue.

```bash
nohup bash experiments/run_wbpr.sh /root/autodl-tmp/MMRec-Ours/data 0 night_runs/baby-wbpr-1 6 baby > baby-wbpr-1.log 2>&1 &
```

Three runs, six-hour queue budget, existing validation early stopping, no model saving. Return
summary.csv, summary.json and manifest.json; console.log includes weight statistics and startup
time. Rank configs by validation, never by test. Local synthetic tests verify the weighted loss,
its gradients, unchanged SSM/inference, and exact path enumeration; actual gains require server data.

## RelationRec: order-free homogeneous relations as losses

Reference: https://github.com/MrShouxingMa/REARM/blob/main/model.py (inspected 2026-09-23).
REARM propagates item/user ID and modality channels on homogeneous graphs before further UI
propagation. This is a simplified adaptation, not an exact reproduction or an algebraic equivalence
to those propagations. No REARM attention, meta-network or orthogonality loss is introduced yet.

**No group embedding, numeric block, positional code, correlated initializer, or ID-distance
feature is instantiated.** IDs only index entities. Historical order-dependent models remain
in the repository for reproducibility; this new queue never launches them.

Architecture: 64-dimensional learnable user/item IDs, two single linear modality projectors,
and trainable raw image/text feature tables. Forward uses bare ID lookups by default. Optional
UI LightGCN has two layers, symmetric binary TRAIN adjacency and layer0/1/2 mean. No user-user
or item-item propagation occurs in training or inference. Final scoring is raw ID dot product;
modality and neighbor branches provide training supervision only. Evaluation embeddings cache
until training resumes.

Neighbors are computed once before training:
- Separate image/text top-k lists from initial normalized raw features, positive cosine only.
- User top-k lists ranked by shared-item count in binary TRAIN interactions only.
- Self-neighbors excluded. Nonzero row weights normalized to sum1. Empty rows use zero weights.
- Equal scores use independent random entity tie keys (local seed2026), never ID distance.
  Exact relabeling tests must co-permute those keys with entities, like other random initial states;
  regenerating tie keys under a new row order is not an exact equivariance test. Tests cover this.
- K=10. Sparse user products and blockwise feature similarities avoid full dense NxN storage.
  Top-k search still has pairwise-comparison startup cost; sparse overlaps can be large.
  Graphs are rebuilt per process. Fixed feature neighbors do not track subsequent feature updates.

For a positive (u,i), define the UltraGCN-inspired constraints:

`L_II = mean_(u,i) [alpha sum_(j in N_v(i)) a_ij softplus(-s_uj)
                  +(1-alpha) sum_(j in N_t(i)) b_ij softplus(-s_uj)]`

`L_UU = mean_(u,i) sum_(v in N_u(u)) c_uv softplus(-s_vi)`.

These transfer preference to neighboring items/users; they do not directly regress an embedding
to the average of its neighbors. They follow the supplied weighted -log(sigmoid) form but use
stable softplus, normalized neighbor weights and a batch mean instead of an unnormalized sum.
Zero-neighbor rows contribute zero. No added negative samples are used inside these positive
constraints. The primary objective supplies discrimination; explicit sampled ID L2=1e-4 helps
control norms. Collapse/over-smoothing prevention is an empirical question, not guaranteed.
User overlap can include the current positive item, as in TRAIN-graph supervision; this is not
leave-one-out construction of the homogeneous graphs.

Total loss:
`L = L_primary + 1.0*(alpha*SSM_image+(1-alpha)*SSM_text)
                + lambda_II*L_II + lambda_UU*L_UU + 1e-4*sampled_ID_L2`.

Alpha=.5, 32 negatives, SSM tau=.04; SSM is the existing negatives-only logsumexp objective.
BPR uses raw-dot softplus differences; primary SSM uses cosine scores. Unlike historical order
experiments, negative sampling excludes ALL TRAIN positives for both primary and modality losses.
All new ablations share this policy, optimizer and initialization; old group runs are not matched
baselines. Validation/test labels never construct graphs, weights or the negative rejection table.

Optional LinkProp weight multiplies ONLY each positive's primary loss (BPR or SSM), not modality,
II, UU or L2 terms. It reuses the documented fixed training-only three-hop support with backtrack
removal, log1p compression and global mean1 normalization; it is not an original LinkProp loss.
No pseudo-label propagation, parameter tuning on test or structural inference blending is added.

### Twelve-run queue

For EACH of UI layers0 and2, run these six settings (seed999):

| Setting | Primary | II coefficient | UU coefficient | LinkProp mix |
| --- | --- | ---: | ---: | ---: |
| baseline | BPR | 0 | 0 | 0 |
| ii | BPR | .1 | 0 | 0 |
| uu | BPR | 0 | .1 | 0 |
| both | BPR | .1 | .1 | 0 |
| weighted_both | BPR | .1 | .1 | .5 |
| ssm_both | SSM | .1 | .1 | 0 |

Layer0 configurations run first. Compare both against ii/uu/baseline, weighted_both against both,
and layer0 against its layer2 counterpart. This is a small first pass at fixed coefficients, not
an exhaustive tuning budget. Choose by validation and confirm across seeds before claiming gains.

```bash
nohup bash experiments/run_relation.sh /root/autodl-tmp/MMRec-Ours/data 0 night_runs/baby-relation-1 12 baby > baby-relation-1.log 2>&1 &
```

Default: 12-hour total cap, 45-minute per-job cap, 1000-epoch maximum with validation early stopping,
no checkpoints. Resume by the same command with unchanged source. Old queues are not stopped.
Return summary.csv, summary.json and manifest.json, plus console.log for failures. Logs include
neighbor coverage and unscaled component losses every200 batches. Runtime and accuracy on real
data are not established by local CPU synthetic tests.

## 完整 REARM 官方基线（当前阶段）

停止在 RelationRec 上叠加组件，先运行固定官方源码和论文配置。
详见 [REARM_BASELINE.md](REARM_BASELINE.md)，含数据/协议说明、运行和日志查看命令。

### 第一步替换：item 同构传播 → item alignment loss

[REARM_ITEM_LOSS.md](REARM_ITEM_LOSS.md) 定义单模块替换、5 组小规模对照和带日志的运行命令。
