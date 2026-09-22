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
