# Soft item cohorts on original FREEDOM

This experiment adds one parameter-free user residual after FREEDOM's existing
UI propagation. The original BPR and two modality losses, item graph, dimensions,
and training protocol remain intact.

Build item features from TRAIN co-consumption, pretrained item text, or both.
Each item joins its two nearest learned item cohorts. A user's TRAIN history
defines a soft mixture `P[u,c]`. The module calculates cohort peer means by
`P @ diag(1 / P.sum(0)) @ P.T @ user_embeddings`, removes each user's self
contribution, and applies `user + beta * (peers - user)`. The U x U graph is never
materialized. Numeric user IDs are not inputs to clustering or training.

The local CPU structural screen on Baby used 2,498 matched adjacent user pairs
with no common TRAIN item. For joint item cohorts, their average membership
inner product was 0.0286 versus 0.0163 for far pairs matched on target-user
degree and consumed-item popularity. Same-primary-cohort rates were 3.84%
versus 1.44%. These are descriptive diagnostics, not recommendation gains.

Run the five-job screen on a fresh output directory:

```bash
bash experiments/run_freedom_soft_cohort.sh /root/autodl-tmp/MMRec-Ours/data 0 night_runs/freedom-soft-1 baby \
  > freedom-soft-1.log 2>&1 &
```

View queue and per-job logs:

```bash
tail -f freedom-soft-1.log
tail -f night_runs/freedom-soft-1/freedom_soft_joint_01_s999/console.log
```

The first job is an exact original-FREEDOM control (`beta=0`). The remaining
jobs compare text-only, collaborative-only, and joint cohorts at `beta=0.1`,
plus joint cohorts at `beta=0.3`. Select by validation Recall@20. The queue
disables model checkpoint saving. Use a new output directory for a new run.
