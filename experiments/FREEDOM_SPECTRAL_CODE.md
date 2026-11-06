# Continuous TRAIN-graph coordinates on FREEDOM

This experiment keeps continuous spectral coordinates instead of clustering them
into groups. The sole structural input is the dataloader's binary TRAIN incidence
matrix R. No timestamps, user/item index distances, held-out edges, or ID-locality
labels enter the model. No additional regularizer or change in loss is introduced.

Compute X = Du^(-1/2) R Di^(-1/2) and its truncated SVD U S V^T, keeping up to
64 singular directions (min(64, min(shape)-1) on small graphs). Codes are U S and
V S with one common scalar normalization giving unit entry RMS over active
entities. Degree-related leading directions are retained. Cold entity codes are
zero. Coordinates are fixed buffers, computed once per run, without clustering.

A bias-free Linear(rank, 64), shared by users and items, projects the codes.
It uses Xavier initialization. One fixed scalar calibrates the joint projected
code RMS to the joint ID embedding RMS at initialization, using active TRAIN
entities on both sides even for the user-only variant. This avoids a user-only
versus both-side scaling confound. It is not an ongoing norm constraint or a
guarantee that final embeddings remain close. Beta is fixed at 1 for this screen.
The default rank/dimension adds 4,096 trainable parameters.

Inputs are p_u + beta * scale * W(code_u) and, for the both-side variants,
q_i + beta * scale * W(code_i). Optional UI propagation follows these inputs.
The original FREEDOM item-item residual always operates on raw q_i, so the new
item code is not also propagated on the item graph. The objective remains the
original FREEDOM item BPR plus reg_weight times the two modality BPR terms.
No graph attraction loss, SSM replacement, or free group table is added.

Three configurations use training seed 999 and SVD seed 2026:

| Name | Coordinate injection | UI layers |
| --- | --- | ---: |
| freedom_code_user_noui_s999 | user only | 0 |
| freedom_code_both_noui_s999 | user and item | 0 |
| freedom_code_both_ui2_s999 | user and item | 2 |

The item-item graph remains enabled in all three: this is not graph-free.
No-UI variants skip UI edge pruning; inherited FREEDOM still constructs an
unused UI adjacency at initialization. There are no baseline reruns or saved
model checkpoints. Epoch selection uses validation Recall@20. SVD preprocessing
time is logged separately from epoch time and must be included in cost reporting.

Spectral signs/bases can vary across numerical libraries, particularly in
degenerate eigenspaces. Coordinate Gram matrices, rather than raw coordinates,
are the appropriate permutation-equivalence check. Training trajectories from
fresh initializations need not be identical after arbitrary remapping. This is a
standard graph representation method, not a demonstrated new algorithm or an
assurance of reproducing ID-based PE gains.

Server run (fresh output directory):

```bash
cd /root/autodl-tmp/MMRec-Ours
git pull --ff-only
nohup bash experiments/run_freedom_spectral_code.sh /root/autodl-tmp/MMRec-Ours/data 0 \
  night_runs/freedom-spectral-code-1 baby > freedom-spectral-code-1.log 2>&1 &
```

View queue and first experiment logs:

```bash
tail -F freedom-spectral-code-1.log \
  night_runs/freedom-spectral-code-1/freedom_code_user_noui_s999/console.log
```

Return manifest.json, summary.json and summary.csv from the output directory.
CPU tests validate geometry, gradients, the disabled-branch FREEDOM regression,
entity-remapped predictions and a one-epoch queue. They do not measure benchmark
recommendation gains.

The local Baby coordinate-only check used 19,445 users, 7,050 items and 118,551
TRAIN edges. Rank-64 construction took about 0.39 seconds on the local CPU;
this excludes FREEDOM initialization and model training. The input hash and
coordinate summary are in reports/baby_spectral_code_verified.json.
