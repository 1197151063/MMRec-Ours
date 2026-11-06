"""CPU structural mining; ID proximity and held-out edges are diagnostic labels only.

Candidate scores use TRAIN edges and pretrained item text. Full-history minimum
item labels are isolated forensic annotations and never enter a score or graph.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cpu_item_cohort_probe import paired_controls, sha256


def binary(r):
    r = r.astype(np.float32).tocsr(copy=True)
    r.sum_duplicates(); r.eliminate_zeros(); r.data[:] = 1; r.sort_indices()
    return r


def normalize_sparse(r):
    norms = np.sqrt(np.asarray(r.multiply(r).sum(1)).ravel())
    return (sp.diags(1 / np.maximum(norms, 1e-12)) @ r).tocsr()


def pair_dot(x, pairs):
    u, v, far = pairs.T
    return (np.asarray(x[u].multiply(x[v]).sum(1)).ravel(),
            np.asarray(x[u].multiply(x[far]).sum(1)).ravel())


def summary(near, far, masks):
    out = {}
    for label, mask in masks.items():
        a, b = np.asarray(near)[mask], np.asarray(far)[mask]
        if not len(a):
            continue
        delta = a - b
        out[label] = dict(pairs=len(a), near_mean=float(a.mean()), far_mean=float(b.mean()),
                          difference=float(delta.mean()), near_wins=float(np.mean(delta > 1e-8)),
                          ties=float(np.mean(np.abs(delta) <= 1e-8)),
                          positive_near=float(np.mean(a > 1e-8)), positive_far=float(np.mean(b > 1e-8)))
    return out


def provenance(frame, r, full):
    n = r.shape[0]
    anchor = frame.groupby('userID').itemID.min().reindex(range(n)).to_numpy()
    a, b = np.arange(n - 1), np.arange(1, n)
    train_shared = np.asarray(r[a].multiply(r[b]).sum(1)).ravel() > 0
    full_shared = np.asarray(full[a].multiply(full[b]).sum(1)).ravel() > 0
    same = anchor[a] == anchor[b]
    _, group_sizes = np.unique(anchor, return_counts=True)
    return anchor, dict(
        note='Full-history item minima are forensics ONLY, not training features. Numeric item order is not used in candidate graphs.',
        full_minimum_item_nondecreasing_fraction=float(np.mean(np.diff(anchor) >= 0)),
        distinct_full_minimum_items=int(len(np.unique(anchor))),
        full_minimum_item_group_size_quantiles=np.quantile(group_sizes, [0, .5, .9, .95, 1]).tolist(),
        adjacent_train_shared_fraction=float(train_shared.mean()),
        adjacent_full_shared_fraction=float(full_shared.mean()),
        adjacent_same_full_minimum_item_fraction=float(same.mean()),
        among_adjacent_without_train_overlap=dict(
            pairs=int((~train_shared).sum()), full_shared_fraction=float(full_shared[~train_shared].mean()),
            same_full_minimum_fraction=float(same[~train_shared].mean())))


def score_max_item_pairs(matrix, r, pairs, dense=False, symmetric_mean=False):
    def score(u, v):
        left = r.indices[r.indptr[u]:r.indptr[u + 1]]
        right = r.indices[r.indptr[v]:r.indptr[v + 1]]
        block = matrix[left] @ matrix[right].T if dense else matrix[left][:, right].toarray()
        if symmetric_mean:
            return .5 * (block.max(0).mean() + block.max(1).mean())
        return block.max()
    return (np.array([score(u, v) for u, v, _ in pairs]),
            np.array([score(u, v) for u, _, v in pairs]))


def inferred_item_profiles(r, ii, users, k=32, seed=2026):
    """A TRAIN-only six-hop motif via shared predicted item anchors."""
    rng = np.random.default_rng(seed)
    keys = rng.random(r.shape[1])
    rows, cols, values = [], [], []
    for start in range(0, len(users), 64):
        batch = users[start:start + 64]
        scores = (r[batch] @ ii).tocsr()
        for local, u in enumerate(batch):
            a, b = scores.indptr[local:local + 2]
            ids, weights = scores.indices[a:b], scores.data[a:b]
            seen = r.indices[r.indptr[u]:r.indptr[u + 1]]
            valid = (weights > 0) & ~np.isin(ids, seen)
            ids, weights = ids[valid], weights[valid]
            order = np.lexsort((keys[ids], -weights))[:k]
            rows.extend([int(u)] * len(order)); cols.extend(ids[order]); values.extend(weights[order])
    return normalize_sparse(sp.csr_matrix((values, (rows, cols)), shape=r.shape, dtype=np.float32))


def direct_graph_audit(r, count=4096, seed=2026):
    """Compare TRAIN-only star weights at fixed graph sizes; ID is evaluation only."""
    rng = np.random.default_rng(seed)
    queries = rng.choice(r.shape[0] - 1, min(count, r.shape[0] - 1), replace=False)
    keys = rng.random(r.shape[0])
    degree = np.asarray(r.sum(0)).ravel()
    result = {}
    for name, weight in (
            ('shared_cosine', np.ones_like(degree)),
            ('rare_item_cosine', 1 / np.sqrt(np.maximum(degree, 1))),
            ('log_popular_item_cosine', np.sqrt(np.log1p(degree))),
            ('popular_item_cosine', np.sqrt(degree))):
        x = normalize_sparse(r @ sp.diags(weight))
        counts = {20: 0, 80: 0}
        for start in range(0, len(queries), 128):
            batch = queries[start:start + 128]
            scores = (x[batch] @ x.T).tocsr()
            for local, u in enumerate(batch):
                a, b = scores.indptr[local:local + 2]
                ids, values = scores.indices[a:b], scores.data[a:b]
                ok = (ids != u) & (values > 0)
                ids, values = ids[ok], values[ok]
                ordered = ids[np.lexsort((keys[ids], -values))]
                for k in counts:
                    counts[k] += int(u + 1 in ordered[:k])
        result[name] = dict(query_users=len(queries),
                            **{f'adjacent_pair_top{k}_coverage': value / len(queries) for k, value in counts.items()})
    return result


def bridge_graph_audit(r, vectors, pairs, count=512, seed=2026):
    """Rank non-overlapping neighbors by TRAIN-only motifs, with independent tie keys."""
    pairs = pairs[:count]
    keys = np.random.default_rng(seed).random(r.shape[0])
    counts = {f'{label}_top{k}': 0 for label in ('near', 'matched_far') for k in (20, 80)}
    for start in range(0, len(pairs), 32):
        batch = pairs[start:start + 32]
        scores = (vectors[batch[:, 0]] @ vectors.T).tocsr()
        shared = (r[batch[:, 0]] @ r.T).tocsr()
        for local, (u, v, far) in enumerate(batch):
            a, b = scores.indptr[local:local + 2]
            ids, values = scores.indices[a:b], scores.data[a:b]
            seen = shared.indices[shared.indptr[local]:shared.indptr[local + 1]]
            ok = (ids != u) & (values > 1e-8) & ~np.isin(ids, seen)
            ids, values = ids[ok], values[ok]
            ranked = ids[np.lexsort((keys[ids], -values))]
            for label, target in (('near', v), ('matched_far', far)):
                for k in (20, 80):
                    counts[f'{label}_top{k}'] += int(target in ranked[:k])
    return dict(pairs=len(pairs), **{key: value / len(pairs) for key, value in counts.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-path', required=True)
    p.add_argument('--dataset', default='baby')
    p.add_argument('--output', required=True)
    p.add_argument('--pairs', type=int, default=2500)
    args = p.parse_args()
    started = time.monotonic()
    folder = Path(args.data_path) / args.dataset
    paths = [folder / f'{args.dataset}.inter', folder / 'text_feat.npy']
    frame = pd.read_csv(paths[0], sep='\t', usecols=['userID', 'itemID', 'x_label'])
    text = np.load(paths[1], allow_pickle=False).astype(np.float32)
    n, m = int(frame.userID.max()) + 1, len(text)
    matrix = lambda f: binary(sp.csr_matrix((np.ones(len(f), np.float32), (f.userID, f.itemID)), shape=(n, m)))
    r = matrix(frame[frame.x_label == 0])
    full = matrix(frame)
    # No held-out object is passed to any candidate scorer below.
    anchor, forensic = provenance(frame, r, full)
    del full, frame
    pairs, controls = paired_controls(r, args.pairs, 2026)
    masks = {'all_no_train_overlap': np.ones(len(pairs), dtype=bool),
             'forensic_same_full_anchor': anchor[pairs[:, 0]] == anchor[pairs[:, 1]],
             'forensic_different_full_anchor': anchor[pairs[:, 0]] != anchor[pairs[:, 1]]}
    report = dict(arguments=vars(args), input_sha256={str(path): sha256(path) for path in paths},
                  forensics=forensic, controls=controls, scores={})
    du = np.diff(r.indptr).astype(np.float32)
    di = np.asarray(r.sum(0)).ravel()
    print('Mining TRAIN-only four-hop common-consumer motifs', flush=True)
    # User neighborhoods around items: weight all paths, rather than Boolean reachability.
    x = (sp.diags(1 / np.sqrt(np.maximum(du, 1))) @ r @ sp.diags(1 / np.sqrt(np.maximum(di, 1)))).tocsr()
    uu = (x @ x.T).tocsr(); uu.setdiag(0); uu.eliminate_zeros()
    degree_uu = np.diff(uu.indptr)
    for name, vectors in (
            ('fourhop_weighted', normalize_sparse(uu)),
            ('fourhop_binary_common_consumers', normalize_sparse(binary(uu))),
            ('fourhop_rare_bridge', normalize_sparse(uu @ sp.diags(1 / np.sqrt(np.maximum(degree_uu, 1)))))):
        near, far = pair_dot(vectors, pairs)
        report['scores'][name] = summary(near, far, masks)
        if name == 'fourhop_rare_bridge':
            print('Auditing rare-bridge non-overlap graph neighbors', flush=True)
            report['fourhop_rare_bridge_graph'] = bridge_graph_audit(r, vectors, pairs)
    def strongest_bridge(a, b):
        product = uu[a].multiply(uu[b]).tocsr()
        return np.asarray(product.max(axis=1).toarray()).ravel()
    report['scores']['fourhop_strongest_bridge'] = summary(
        strongest_bridge(pairs[:, 0], pairs[:, 1]), strongest_bridge(pairs[:, 0], pairs[:, 2]), masks)
    print('Mining item-side max-match and inferred-anchor motifs', flush=True)
    ii = (r.T @ sp.diags(1 / np.maximum(du, 1)) @ r).tocsr()
    scale = 1 / np.sqrt(np.maximum(ii.diagonal(), 1e-12))
    ii = (sp.diags(scale) @ ii @ sp.diags(scale)).tocsr()
    ii.setdiag(0); ii.eliminate_zeros()
    for name, dense, vectors in (('collaborative', False, ii),
                                  ('text', True, text / np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-12))):
        for mode in (False, True):
            values = score_max_item_pairs(vectors, r, pairs, dense=dense, symmetric_mean=mode)
            key = name + ('_symmetric_best_match' if mode else '_single_best_item_pair')
            report['scores'][key] = summary(*values, masks)
    predicted = inferred_item_profiles(r, ii, np.arange(n), k=32)
    report['scores']['sixhop_shared_inferred_item'] = summary(*pair_dot(predicted, pairs), masks)
    report['sixhop_inferred_item_graph'] = bridge_graph_audit(r, predicted, pairs)
    # Popularity of the TRAIN shared item: a separate direct-edge structural audit.
    adjacent = np.arange(n - 1)
    common = r[adjacent].multiply(r[adjacent + 1]).tocsr()
    row, col = common.nonzero()
    report['direct_shared_item_structure'] = dict(
        adjacent_shared_edges=len(col), shared_item_train_degree_quantiles=np.quantile(di[col], [.1, .5, .9]).tolist(),
        all_interaction_item_degree_quantiles=np.quantile(di[r.indices], [.1, .5, .9]).tolist(),
        shared_is_users_most_popular_item_fraction=float(np.mean([
            di[i] == max(di[r.indices[r.indptr[u]:r.indptr[u + 1]]]) for u, i in zip(row, col)])))
    print('Auditing direct star weights at fixed graph sizes', flush=True)
    report['direct_graph_audit'] = direct_graph_audit(r)
    report['seconds'] = time.monotonic() - started
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print('Forensics:', json.dumps(forensic), flush=True)
    for name, values in report['scores'].items():
        print(name, json.dumps(values['all_no_train_overlap']), flush=True)
    print('Saved', out, flush=True)


if __name__ == '__main__':
    main()
