"""CPU-only, TRAIN-only probe of item-induced overlapping user cohorts.

Numeric user IDs are used only to form the final diagnostic pairs. They never
enter item clustering, cohort construction, or the matched-control selection.
This is a structural diagnostic, not a recommendation accuracy experiment.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.cluster import MiniBatchKMeans


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def history(r, u):
    return r.indices[r.indptr[u]:r.indptr[u + 1]]


def overlap(r, u, v):
    return np.intersect1d(history(r, u), history(r, v), assume_unique=True).size


def paired_controls(r, sample, seed):
    """Match user degree and mean consumed-item popularity; exclude overlap."""
    rng = np.random.default_rng(seed)
    degree = np.diff(r.indptr)
    item_degree = np.asarray(r.sum(axis=0)).ravel()
    pop = np.asarray(r @ np.log1p(item_degree)).ravel() / np.maximum(degree, 1)
    buckets = {int(d): np.flatnonzero(degree == d) for d in np.unique(degree)}
    near = [u for u in range(len(degree) - 1)
            if degree[u] and degree[u + 1] and not overlap(r, u, u + 1)]
    selected = rng.choice(near, size=min(sample, len(near)), replace=False)
    triples = []
    for u in selected:
        v = u + 1
        candidates = buckets[int(degree[v])]
        if len(candidates) < 2:
            continue
        # Sample before matching so the CPU cost stays small and predictable.
        pool = rng.choice(candidates, size=min(128, len(candidates)), replace=False)
        pool = pool[np.abs(pool - u) > 256]
        if not len(pool):
            continue
        pool = pool[np.argsort(np.abs(pop[pool] - pop[v]))]
        far = next((int(w) for w in pool if not overlap(r, u, int(w))), None)
        if far is not None:
            triples.append((int(u), int(v), far))
    return np.asarray(triples, dtype=np.int32).reshape(-1, 3), {
        "eligible_adjacent_no_shared_item": len(near),
        "sampled": len(selected), "matched": len(triples),
        "control": "same target degree, closest mean log item popularity in a 128-user random pool; no shared item; ID separation >256",
    }


def neighbor_retrieval(weighted, r, triples, block=64):
    """Probe a UU graph restricted to users with no directly shared item."""
    anchors, near, far = triples.T
    n = len(anchors)
    counts = {"near_top20": 0, "matched_far_top20": 0,
              "near_top80": 0, "matched_far_top80": 0}
    for start in range(0, n, block):
        end = min(start + block, n)
        users = anchors[start:end]
        scores = weighted[users] @ weighted.T
        shared = (r[users] @ r.T).tocoo()
        scores[shared.row, shared.col] = -np.inf
        scores[np.arange(len(users)), users] = -np.inf
        for label, targets in (("near", near[start:end]),
                               ("matched_far", far[start:end])):
            target_scores = scores[np.arange(len(users)), targets]
            # Strict rank avoids arbitrary ID-dependent tie breaking.
            better = np.sum(scores > target_scores[:, None], axis=1)
            positive = target_scores > 1e-8
            for k in (20, 80):
                counts[f"{label}_top{k}"] += int(np.sum(positive & (better < k)))
    return {key: value / n for key, value in counts.items()}


def pair_scores(weighted, triples, seed, r=None):
    weighted = unit(weighted)
    u, v, far = triples.T
    near = np.sum(weighted[u] * weighted[v], axis=1)
    control = np.sum(weighted[u] * weighted[far], axis=1)
    delta = near - control
    rng = np.random.default_rng(seed + 700)
    boot = np.mean(delta[rng.integers(len(delta), size=(500, len(delta)))], axis=1)
    result = {
        "near_mean": float(near.mean()), "matched_far_mean": float(control.mean()),
        "paired_difference": float(delta.mean()),
        "bootstrap_95_descriptive": np.quantile(boot, [.025, .975]).tolist(),
        "near_wins_fraction": float(np.mean(delta > 0)),
        "non_tie_fraction": float(np.mean(np.abs(delta) > 1e-8)),
    }
    if r is not None:
        result["no_shared_item_neighbor_retrieval"] = neighbor_retrieval(weighted, r, triples)
    return result


def cohort_scores(r, item_vectors, clusters, seed, triples, rank_probe=False):
    model = MiniBatchKMeans(n_clusters=clusters, random_state=seed,
                            batch_size=1024, n_init=3, max_iter=100)
    model.fit(item_vectors)
    centers = unit(model.cluster_centers_)
    similarities = item_vectors @ centers.T
    top2 = np.argpartition(similarities, -2, axis=1)[:, -2:]
    top_scores = np.take_along_axis(similarities, top2, axis=1)
    top_weights = np.exp((top_scores - top_scores.max(axis=1, keepdims=True)) / .1)
    top_weights /= top_weights.sum(axis=1, keepdims=True)
    item_cohorts = sp.csr_matrix((top_weights.ravel(),
                                  (np.repeat(np.arange(r.shape[1]), 2), top2.ravel())),
                                 shape=(r.shape[1], clusters), dtype=np.float32)
    # Divide out broad cohorts, which otherwise connect nearly every user.
    mass = (r @ item_cohorts).toarray().astype(np.float32)
    cohort_pop = mass.sum(axis=0)
    result = pair_scores(mass / np.sqrt(np.maximum(cohort_pop, 1)), triples, seed,
                         r=r if rank_probe else None)
    anchors, near, far = triples.T
    primary = np.argmax(mass, axis=1)
    result["same_primary_cohort"] = {
        "near_fraction": float(np.mean(primary[anchors] == primary[near])),
        "matched_far_fraction": float(np.mean(primary[anchors] == primary[far])),
    }
    result["cohort_occupancy_min_max"] = [float(cohort_pop.min()), float(cohort_pop.max())]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--dataset", default="baby")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rank-probe", action="store_true")
    args = parser.parse_args()
    if args.sample < 1:
        parser.error("sample must be positive")
    folder = Path(args.data_path) / args.dataset
    paths = [folder / f"{args.dataset}.inter", folder / "text_feat.npy"]
    data = pd.read_csv(paths[0], sep="\t", usecols=["userID", "itemID", "x_label"])
    train = data.loc[data.x_label == 0, ["userID", "itemID"]].drop_duplicates()
    text = unit(np.load(paths[1], allow_pickle=False))
    r = sp.csr_matrix((np.ones(len(train), np.float32),
                       (train.userID.to_numpy(), train.itemID.to_numpy())),
                      shape=(int(data.userID.max()) + 1, len(text)))
    r.data[:] = 1
    r.sort_indices()
    triples, control_info = paired_controls(r, args.sample, args.seed)
    if not len(triples):
        raise RuntimeError("No matched adjacent pairs")
    report = {"arguments": vars(args), "input_sha256": {str(p): sha256(p) for p in paths},
              "train_edges": int(r.nnz), "pairs": control_info, "results": {}}
    # An item co-consumer profile is derived solely from TRAIN interactions.
    degree = np.diff(r.indptr).astype(np.float32)
    profile = r.T @ sp.diags(1 / np.sqrt(np.maximum(degree, 1)))
    latent, singular, _ = svds(profile.astype(np.float32), k=32,
                               random_state=args.seed)
    collaboration = unit(latent * singular)
    vectors = {
        "text": text,
        "collaborative": collaboration,
        "joint": unit(np.concatenate([text, collaboration], axis=1)),
    }
    for name, item_vectors in vectors.items():
        report["results"][f"{name}_mean"] = pair_scores(
            r @ item_vectors, triples, args.seed,
            r=r if args.rank_probe and name in ("collaborative", "joint") else None)
        for k in (64, 128):
            key = f"{name}_c{k}"
            print("Probing", key, flush=True)
            report["results"][key] = cohort_scores(
                r, item_vectors, k, args.seed, triples,
                rank_probe=args.rank_probe and name == "joint" and k == 64)
        # Preserve the exact item-vector distribution but sever its item identity.
        shuffled = item_vectors[np.random.default_rng(args.seed + 99).permutation(len(item_vectors))]
        key = f"{name}_shuffled_c64"
        print("Probing", key, flush=True)
        report["results"][key] = cohort_scores(r, shuffled, 64, args.seed, triples)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pairs": control_info, "differences":
                      {name: value["paired_difference"] for name, value in report["results"].items()}}, indent=2))
    print("Saved", output)


if __name__ == "__main__":
    main()
