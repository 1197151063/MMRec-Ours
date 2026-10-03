"""Same-item matched controls and TRAIN-internal prediction of residual tastes.

ID distances define diagnostic pairs ONLY. Neighbor scores and masked-interaction
prediction never receive a distance, ordering feature, or diagnostic label.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from build_semantic_user_graphs import normalize, sha

METHODS = ('shared_cosine', 'shared_random', 'residual_semantic',
           'residual_collaborative', 'residual_joint')


def binary(r):
    r = r.astype(np.float32).tocsr(copy=True)
    r.sum_duplicates()
    r.eliminate_zeros()
    r.data[:] = 1
    r.sort_indices()
    return r


def history(r, u):
    return r.indices[r.indptr[u]:r.indptr[u + 1]]


def projected_content(features, dim, seed):
    """Fixed independent Gaussian projection, not learned from ID locality."""
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(features.shape[1], dim)).astype(np.float32)
    projection /= np.sqrt(dim)
    return normalize(normalize(features) @ projection)


def collaborative_features(r, dim, keys):
    """Projected item consumer profiles; all degrees/interactions come from r.

    keys is an IID user probe matrix, co-permuted in a relabeling test. This is
    a stochastic sketch of structure, never a function of numeric ID distance.
    """
    degree = np.diff(r.indptr)
    return normalize(r.T @ (keys[:, :dim] / np.sqrt(np.maximum(degree, 1))[:, None]))


class ResidualScores:
    def __init__(self, r, features):
        self.r = r
        self.degree = np.diff(r.indptr)
        self.features = features
        self.totals = {name: r @ x for name, x in features.items()}

    def compare(self, u, candidates):
        """Subtract ALL common items from BOTH users before comparing them."""
        candidates = np.asarray(candidates, dtype=np.int64)
        common = self.r[candidates].multiply(self.r[u]).tocsr()
        counts = np.diff(common.indptr)
        valid = (self.degree[u] > counts) & (self.degree[candidates] > counts)
        values = {}
        for name, features in self.features.items():
            shared = common @ features
            left = self.totals[name][u] - shared
            right = self.totals[name][candidates] - shared
            norms = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
            ok = valid & (norms > 1e-10)
            sim = np.full(len(candidates), np.nan)
            sim[ok] = np.clip(np.sum(left[ok] * right[ok], axis=1) / norms[ok], -1, 1)
            values[name] = sim
        return values, counts


def matched_pairs(r, offset, max_pairs, far_distance, rng):
    """Match the SAME u and anchor item, exact degree(v), exact overlap(u,v).

    One uniformly selected common anchor per sampled near pair; unmatched pairs
    are dropped, counted and never relaxed to a weaker matching criterion.
    """
    degree = np.diff(r.indptr)
    rt = r.T.tocsr()
    eligible = np.flatnonzero((degree[:-offset] > 0) & (degree[offset:] > 0)) if offset < len(degree) else np.array([], dtype=int)
    shared_pairs = []
    for u in eligible:
        common = np.intersect1d(history(r, u), history(r, u + offset), assume_unique=True)
        if len(common):
            shared_pairs.append((int(u), int(u + offset), common))
    chosen = rng.permutation(len(shared_pairs))[:max_pairs]
    records = []
    for index in chosen:
        u, v, common = shared_pairs[index]
        anchor = int(rng.choice(common))
        pool = history(rt, anchor)
        pool = pool[(np.abs(pool - u) > far_distance) & (degree[pool] == degree[v])]
        if len(pool):
            overlaps = np.asarray(r[pool].multiply(r[u]).sum(1)).ravel()
            pool = pool[overlaps == len(common)]
        if len(pool):
            records.append((u, v, int(rng.choice(pool)), anchor, len(common)))
    return np.asarray(records, dtype=np.int64).reshape(-1, 5), dict(
        eligible_near_pairs=len(eligible), shared_near_pairs=len(shared_pairs),
        sampled_shared_pairs=len(chosen), exact_matched_pairs=len(records),
        unmatched_pairs=len(chosen) - len(records))


def paired_summary(near, far, users, rng):
    near, far = np.asarray(near), np.asarray(far)
    mask = np.isfinite(near) & np.isfinite(far)
    a, b, users = near[mask], far[mask], np.asarray(users)[mask]
    result = dict(valid_pairs=len(a), missing_pairs=int((~mask).sum()))
    if not len(a):
        return result
    delta = a - b
    # Resample anchor-user means. Other endpoints/items can still be dependent;
    # these are descriptive intervals, not a formal significance test.
    _, inverse = np.unique(users, return_inverse=True)
    means = np.bincount(inverse, weights=delta) / np.bincount(inverse)
    boots = [float(np.mean(rng.choice(means, len(means), replace=True))) for _ in range(300)]
    result.update(near_mean=float(a.mean()), far_mean=float(b.mean()),
                  mean_paired_difference=float(delta.mean()), near_wins=float(np.mean(a > b)),
                  anchor_mean_difference=float(means.mean()),
                  descriptive_anchor_bootstrap_95=np.quantile(boots, [.025, .975]).tolist())
    return result


def diagnose(r, features, offsets, max_pairs, far_distance, seed, out):
    scores = ResidualScores(r, features)
    results = {}
    for offset in offsets:
        pairs, summary = matched_pairs(r, offset, max_pairs, far_distance,
                                      np.random.default_rng(seed + offset))
        np.savez_compressed(out / f'diagnostic_pairs_offset{offset}.npz',
                            columns=np.array(['u', 'near', 'far', 'anchor', 'common_count']), pairs=pairs)
        near = {key: [] for key in features}
        far = {key: [] for key in features}
        for u, v, w, _, _ in pairs:
            values, _ = scores.compare(u, [v, w])
            for key, value in values.items():
                near[key].append(value[0]); far[key].append(value[1])
        summary['residual_similarity'] = {
            key: paired_summary(near[key], far[key], pairs[:, 0], np.random.default_rng(seed))
            for key in features}
        summary['by_shared_count'] = {}
        for label, mask in [('one', pairs[:, 4] == 1), ('multiple', pairs[:, 4] > 1)]:
            summary['by_shared_count'][label] = {
                key: paired_summary(np.asarray(near[key])[mask], np.asarray(far[key])[mask],
                                    pairs[mask, 0], np.random.default_rng(seed)) for key in features}
        results[str(offset)] = summary
        print(f'offset={offset}: {len(pairs)} exact-matched pairs', flush=True)
        for key, result in summary['residual_similarity'].items():
            print(f'  {key}: {json.dumps(result)}', flush=True)
    return results


def mask_training(r, seed):
    """Hide one TRAIN edge for EVERY degree>=3 user, before any CF feature."""
    rng = np.random.default_rng(seed)
    fit = r.copy()
    users = np.flatnonzero(np.diff(r.indptr) >= 3)
    targets = np.full(r.shape[0], -1, dtype=np.int64)
    for u in users:
        index = int(rng.integers(r.indptr[u], r.indptr[u + 1]))
        targets[u] = r.indices[index]
        fit.data[index] = 0
    fit.eliminate_zeros()
    return fit, targets


def neighbor_scores(scorer, u, candidates):
    values, common = scorer.compare(u, candidates)
    base = common / np.sqrt(scorer.degree[u] * scorer.degree[candidates])
    semantic = (values['visual'] + values['text']) / 2
    cf = values['collaborative']
    # Missing residual histories get no residual edge, not an invented similarity.
    def positive(x):
        return np.maximum(np.nan_to_num(x, nan=0.), 0.)
    return dict(shared_cosine=base, shared_random=np.ones(len(candidates)),
                residual_semantic=base * positive(semantic),
                residual_collaborative=base * positive(cf),
                residual_joint=base * positive((semantic + cf) / 2))


def build_neighbor_rows(r, content, probes, queries, ks, user_ties, dim):
    """No locality input; neighbor graphs only see the fit interaction matrix."""
    features = dict(content, collaborative=collaborative_features(r, dim, probes))
    scorer = ResidualScores(r, features)
    buffers = {(method, k): ([], []) for method in METHODS for k in ks}
    candidate_lists = []
    for row, u in enumerate(queries):
        overlaps = (r[u] @ r.T).tocsr()
        candidates = overlaps.indices[overlaps.indices != u]
        candidate_lists.append(candidates.copy())
        if len(candidates):
            scores = neighbor_scores(scorer, u, candidates)
            for method, score in scores.items():
                active = np.flatnonzero(score > 0)
                order = active[np.lexsort((user_ties[candidates[active]], -score[active]))]
                for k in ks:
                    selected = candidates[order[:k]]
                    rows, cols = buffers[(method, k)]
                    rows.extend([row] * len(selected)); cols.extend(selected.tolist())
        if row % 256 == 0:
            print(f'  neighbors: {row}/{len(queries)}', flush=True)
    graphs = {}
    for key, (rows, cols) in buffers.items():
        graph = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(queries), r.shape[0]))
        mass = np.asarray(graph.sum(1)).ravel()
        graphs[key] = (sp.diags(1 / np.maximum(mass, 1)) @ graph).tocsr()
    return graphs, candidate_lists


def vote_metrics(graph, r, queries, targets, item_ties):
    """Uniform neighbor mean of degree-normalized histories; full catalog rank.

    Items with no positive vote are NOT filled with random recommendations.
    Unreachable targets and users with no edges stay in the denominator.
    """
    degree = np.diff(r.indptr)
    votes = (graph @ sp.diags(1 / np.maximum(degree, 1)) @ r).tocsr()
    metrics = {f'{metric}@{k}': [] for k in (10, 20) for metric in ('recall', 'ndcg')}
    for row, u in enumerate(queries):
        items = history(votes, row)
        values = votes.data[votes.indptr[row]:votes.indptr[row + 1]]
        mask = (values > 0) & ~np.isin(items, history(r, u))
        items, values = items[mask], values[mask]
        order = np.lexsort((item_ties[items], -values))[:20]
        selected = items[order]
        positions = np.flatnonzero(selected == targets[u])
        rank = int(positions[0]) + 1 if len(positions) else 1000000
        for k in (10, 20):
            hit = rank <= k
            metrics[f'recall@{k}'].append(float(hit))
            metrics[f'ndcg@{k}'].append(float(1 / np.log2(rank + 1)) if hit else 0.)
    return {key: np.asarray(value) for key, value in metrics.items()}


def prediction_study(r, content, dim, seed, queries_limit, ks, out):
    fit, targets = mask_training(r, seed)
    rng = np.random.default_rng(seed + 10000)
    queries = rng.permutation(np.flatnonzero(targets >= 0))[:queries_limit]
    if not len(queries):
        raise ValueError('No degree>=3 users for inner masked-interaction prediction')
    user_ties = rng.random(r.shape[0]); item_ties = rng.random(r.shape[1])
    probes = rng.normal(size=(r.shape[0], dim)).astype(np.float32) / np.sqrt(dim)
    graphs, candidates = build_neighbor_rows(fit, content, probes, queries, ks, user_ties, dim)
    coverage = [bool(len(c) and fit[c, targets[u]].nnz) for u, c in zip(queries, candidates)]
    report = dict(mask_seed=seed, fit_edges=int(fit.nnz), hidden_edges=int((targets >= 0).sum()),
                  query_users=len(queries), candidate_target_coverage=float(np.mean(coverage)), methods={})
    arrays = dict(queries=queries, targets=targets[queries])
    baseline = {}
    for (method, k), graph in graphs.items():
        name = f'{method}_k{k}'
        metrics = vote_metrics(graph, fit, queries, targets, item_ties)
        if method == 'shared_cosine':
            baseline[k] = metrics
        result = {metric: float(values.mean()) for metric, values in metrics.items()}
        result.update(mean_neighbors=float(np.diff(graph.indptr).mean()),
                      users_without_neighbors=int(np.count_nonzero(np.diff(graph.indptr) == 0)))
        rows, cols = graph.nonzero()
        result['diagnostic_only_edge_fraction_within_id_window'] = {
            str(window): float(np.mean(np.abs(queries[rows] - cols) <= window)) if len(rows) else None
            for window in (1, 4, 16, 64)}
        if method != 'shared_cosine':
            delta = metrics['recall@20'] - baseline[k]['recall@20']
            result['paired_vs_shared_cosine'] = dict(recall20_delta=float(delta.mean()),
                                                       wins=int((delta > 0).sum()), losses=int((delta < 0).sum()))
        report['methods'][name] = result
        print(f'  {name}: R@20={result["recall@20"]:.6f} '
              f'N@20={result["ndcg@20"]:.6f} neighbors={result["mean_neighbors"]:.2f}', flush=True)
        for metric, values in metrics.items():
            arrays[f'{name}_{metric}'] = values
    np.savez_compressed(out / f'prediction_seed{seed}.npz', **arrays)
    return report


def write_json(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(path)


def load_train(folder, dataset):
    paths = [folder / f'{dataset}.inter', folder / 'image_feat.npy', folder / 'text_feat.npy']
    df = pd.read_csv(paths[0], sep='\t', usecols=['userID', 'itemID', 'x_label'])
    edges = df.loc[df.x_label == 0, ['userID', 'itemID']].drop_duplicates()
    ids = edges.to_numpy()
    if not len(ids) or not np.isfinite(ids).all() or (ids < 0).any() or not (ids == ids.astype(np.int64)).all():
        raise ValueError('Invalid TRAIN identifiers')
    raw = {name: np.load(path, allow_pickle=False).astype(np.float32)
           for name, path in zip(('visual', 'text'), paths[1:])}
    if any(x.ndim != 2 or not np.isfinite(x).all() for x in raw.values()):
        raise ValueError('Invalid modality features')
    n_items = len(raw['visual'])
    if len(raw['text']) != n_items or ids[:, 1].max() >= n_items:
        raise ValueError('Feature/interaction indexing mismatch')
    r = binary(sp.csr_matrix((np.ones(len(ids)), (ids[:, 0].astype(int), ids[:, 1].astype(int))),
                            shape=(int(ids[:, 0].max()) + 1, n_items)))
    return r, raw, paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-path', required=True); p.add_argument('--dataset', default='baby')
    p.add_argument('--output', required=True)
    p.add_argument('--pairs', type=int, default=4000)
    p.add_argument('--queries', type=int, default=2048)
    p.add_argument('--offsets', type=int, nargs='+', default=[1, 4, 16, 64])
    p.add_argument('--far-distance', type=int, default=256)
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--ks', type=int, nargs='+', default=[20, 80])
    p.add_argument('--mask-seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()
    if min(args.pairs, args.queries, args.dim, *args.offsets, *args.ks) < 1 or args.far_distance < max(args.offsets):
        p.error('Sizes must be positive and far-distance >= all offsets')
    if min(args.seed, *args.mask_seeds) < 0:
        p.error('Seeds must be nonnegative')
    out = Path(args.output)
    if out.exists():
        p.error('Output already exists; choose a new output directory')
    out.mkdir(parents=True)
    started = time.monotonic()
    write_json(out / 'status.json', dict(state='running', stage='loading'))
    try:
        r, raw, paths = load_train(Path(args.data_path) / args.dataset, args.dataset)
        content = {name: projected_content(x, args.dim, args.seed + j)
                   for j, (name, x) in enumerate(raw.items())}
        probes = np.random.default_rng(args.seed + 2).normal(size=(r.shape[0], args.dim)).astype(np.float32) / np.sqrt(args.dim)
        # Exact original semantic dimensions for the central diagnostic. The
        # smaller fixed projections are only used to screen many candidate edges.
        features = {name: normalize(x) for name, x in raw.items()}
        features['collaborative'] = collaborative_features(r, args.dim, probes)
        report = dict(arguments=vars(args), training_edges=int(r.nnz),
                      input_sha256={str(path): sha(path) for path in paths},
                      source_sha256={str(Path(__file__).name): sha(Path(__file__)),
                                     'build_semantic_user_graphs.py': sha(Path(__file__).with_name('build_semantic_user_graphs.py'))},
                      notes=[
                          'Official validation/test interactions are never used in matching, fitting or targets.',
                          'ID distance only selects diagnostic near/far pairs; never neighbor scores or prediction labels.',
                          'Exact controls match u, one sampled common item, degree(v), and total shared-item count; no fallback.',
                          'Residual similarities remove ALL common items. Empty residuals are missing, not zero similarity.',
                          'Diagnostic visual/text similarities use original feature dimensions; prediction uses fixed Gaussian projections.',
                          'CF features are stochastic sketches of consumer profiles, rebuilt after each TRAIN masking.',
                          'Bootstrap intervals are descriptive; reused endpoints/items can remain dependent.',
                          'Inner trials reuse one dataset and content projection; not independent benchmark replications.',
                          'Prediction uses uniform selected-neighbor votes divided by neighbor degree; no neural model is trained.',
                          'Do not select a graph by ID proximity. Require inner prediction benefit, then validate the main model.'],
                      diagnostics={}, prediction=[])
        write_json(out / 'manifest.json', {key: report[key] for key in
                                          ('arguments', 'training_edges', 'input_sha256', 'source_sha256', 'notes')})
        write_json(out / 'status.json', dict(state='running', stage='matched_diagnostics'))
        report['diagnostics'] = diagnose(r, features, args.offsets, args.pairs, args.far_distance, args.seed, out)
        del features, raw
        write_json(out / 'report.json', report)
        for seed in args.mask_seeds:
            print(f'Inner masked-interaction prediction seed={seed}', flush=True)
            write_json(out / 'status.json', dict(state='running', stage='inner_prediction', seed=seed))
            report['prediction'].append(prediction_study(r, content, args.dim, seed, args.queries, args.ks, out))
            write_json(out / 'report.json', report)
        report['seconds'] = time.monotonic() - started
        write_json(out / 'report.json', report)
        write_json(out / 'status.json', dict(state='complete', seconds=report['seconds']))
        print(f'Finished: {out / "report.json"}', flush=True)
    except Exception as exc:
        write_json(out / 'status.json', dict(state='failed', error=repr(exc), seconds=time.monotonic() - started))
        raise


if __name__ == '__main__':
    main()
