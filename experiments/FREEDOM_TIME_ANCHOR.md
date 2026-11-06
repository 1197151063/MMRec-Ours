# Item-time anchor groups on FREEDOM

This is the raw anchor rule with 25.11% adjacent-user same-group rate in the
local Baby diagnostic, separate from the 16-group spectral rule.

For each item, calculate its earliest observed review timestamp over TRAIN
interactions: `time[i] = min(t[u,i] : (u,i) in TRAIN)`. For each user, choose the
consumed TRAIN item with the smallest such `time[i]` as the anchor. Users with
the same anchor share a group. Independent random item keys break equal-time
ties; numeric user/item order does not select the anchor. The timestamp is a
first observation in the available TRAIN sample, not an item release date and
not necessarily that user's own earliest interaction.

The model rereads the timestamp column from `.inter`, first filters to TRAIN
rows, then restricts further to actual dataloader TRAIN edges. Validation/test
timestamps and internally removed training edges cannot affect group times.
No full-history minimum-item labels are used. The timestamp field defaults to
`timestamp` and can be changed through `anchor_time_field`.

Baby produces 2,808 groups, with sizes min/median/p90/max = 1/2/15/312. Group
count is automatic; no further merging into 16 groups is performed. Locality
is an exploratory structural diagnostic, not a measured Recall improvement.
This rule was selected after examining ID-locality diagnostics and should be
described as such in research reporting.

The representation and objective match the spectral-group experiment:
`user_input = personal_embedding + beta * group_embedding[anchor_group]`,
followed by the original FREEDOM UI propagation. The optional personal penalty
is `lambda * mean_user ||personal_embedding||^2`. The group table uses Xavier
initialization. At dimension 64 this table has 179,712 learnable parameters,
versus 1,024 for 16 spectral groups; group count and initialization scale differ
between the two methods, so their results do not isolate grouping quality alone.

Six variants use training seed 999: beta 0.1/0.3/1.0 without the personal
penalty, plus `(beta, penalty) = (0.1, 0.01), (0.1, 0.1), (0.3, 0.1)`. There
is no baseline rerun or model-checkpoint saving.

Run from the repository root on the server, using a fresh output directory:

```bash
git pull --ff-only
nohup bash experiments/run_freedom_time_anchor.sh /root/autodl-tmp/MMRec-Ours/data 0 \
  night_runs/freedom-time-anchor-1 baby > freedom-time-anchor-1.log 2>&1 &
```

View queue and first experiment logs:

```bash
tail -F freedom-time-anchor-1.log \
  night_runs/freedom-time-anchor-1/freedom_anchor_b01_s999/console.log
```

The first experiment log reports the group count, group sizes, and anchor hash.
After completion, collect `manifest.json`, `summary.json`, and `summary.csv`
from the output directory. Epoch selection uses validation Recall@20.

## Direct user embedding without UI propagation

The additional `freedom_time_anchor_no_ui` suite keeps the same TRAIN anchors,
ordinary initialization, loss and item branch. It sets `n_ui_layers: 0`, so
the final user representation is directly
`user_id_embedding + beta * group_embedding[group[user]]`. No neighborhood
aggregation or degree-sensitive edge pruning runs on the user-item graph.
The inherited FREEDOM constructor still prepares its UI adjacency; it is unused
in this mode. This is not a fully graph-free model: the item representation is
`item_id_embedding + MM_adj^n_mm_layers @ item_id_embedding`.

The objective remains the original FREEDOM item BPR plus `reg_weight` times
visual/textual BPR. Three runs use beta 0.1/0.3/1.0, training seed 999 and no
extra personal penalty, matching the three unregularized UI-enabled settings
above. Beta 1.0 is the literal `user_id + group` sum. No baseline or UI-enabled
configuration is rerun; compare against the corresponding previous results.
No recommendation improvement has been measured by the CPU smoke test.

Run in a fresh output directory:

```bash
git pull --ff-only
nohup bash experiments/run_freedom_time_anchor_no_ui.sh /root/autodl-tmp/MMRec-Ours/data 0 \
  night_runs/freedom-time-anchor-no-ui-1 baby > freedom-time-anchor-no-ui-1.log 2>&1 &
```

View the queue and first experiment logs:

```bash
tail -F freedom-time-anchor-no-ui-1.log \
  night_runs/freedom-time-anchor-no-ui-1/freedom_anchor_noui_b01_s999/console.log
```
