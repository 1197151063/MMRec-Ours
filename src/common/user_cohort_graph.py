"""Frozen TRAIN-only user neighbors, with no numeric ID-order features."""
import numpy as np
import scipy.sparse as sp


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def project_content(x, dim, seed):
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(x.shape[1], dim)).astype(np.float32) / np.sqrt(dim)
    return normalize(normalize(x) @ projection)


def build_cohort_graph(r, visual, text, kind, k=80, dim=64, seed=2026,
                       ties=None, progress=None):
    """Top-k shared-item candidates, then uniform row-normalized edge weights.

    Residual semantic score = shared cosine * positive mean residual modality
    cosine. ALL shared items are subtracted from both histories. Missing residuals
    do not create edges. Ties use independent entity keys; relabel those keys when
    checking exact permutation equivariance. Never materialize a dense U x U graph.
    """
    if kind not in ('shared_cosine', 'shared_random', 'residual_semantic'):
        raise ValueError('Unknown user cohort graph kind')
    if k < 1 or dim < 1 or seed < 0:
        raise ValueError('Invalid cohort graph dimensions or seed')
    r = r.astype(np.float32).tocsr(copy=True)
    r.sum_duplicates(); r.eliminate_zeros(); r.data[:] = 1; r.sort_indices()
    degree = np.diff(r.indptr)
    rt = r.T.tocsr()
    ties = np.random.default_rng(seed + 10000).random(r.shape[0]) if ties is None else np.asarray(ties)
    if ties.shape != (r.shape[0],) or not np.isfinite(ties).all() or len(np.unique(ties)) != len(ties):
        raise ValueError('Tie keys must be finite and unique for every user')
    features, totals = [], []
    if kind == 'residual_semantic':
        for j, x in enumerate((visual, text)):
            if x is None or x.ndim != 2 or len(x) != r.shape[1] or not np.isfinite(x).all():
                raise ValueError('Invalid cohort modality features')
            f = project_content(np.asarray(x, dtype=np.float32), dim, seed + j)
            features.append(f); totals.append(r @ f)
    rows, cols = [], []
    for u in range(r.shape[0]):
        shared_counts = (r[u] @ rt).tocsr()
        active = shared_counts.indices != u
        candidates = shared_counts.indices[active]
        counts = shared_counts.data[active]
        if len(candidates):
            score = counts / np.sqrt(degree[u] * degree[candidates])
            if kind == 'shared_random':
                score = np.ones(len(candidates))
            elif kind == 'residual_semantic':
                common = r[candidates].multiply(r[u]).tocsr()
                valid = (degree[u] > counts) & (degree[candidates] > counts)
                similarities = []
                for f, total in zip(features, totals):
                    shared = common @ f
                    left, right = total[u] - shared, total[candidates] - shared
                    norm = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
                    ok = valid & (norm > 1e-10)
                    similarity = np.full(len(candidates), np.nan)
                    similarity[ok] = np.clip((left[ok] * right[ok]).sum(1) / norm[ok], -1, 1)
                    similarities.append(similarity)
                semantic = (similarities[0] + similarities[1]) / 2
                score *= np.maximum(np.nan_to_num(semantic, nan=0.), 0.)
            valid = np.flatnonzero(score > 0)
            selected = valid[np.lexsort((ties[candidates[valid]], -score[valid]))[:k]]
            cols.extend(candidates[selected].tolist()); rows.extend([u] * len(selected))
        if progress is not None and (u % 2048 == 0 or u + 1 == r.shape[0]):
            progress(u + 1, r.shape[0])
    graph = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                          shape=(r.shape[0], r.shape[0]))
    count = np.diff(graph.indptr)
    return (sp.diags(1. / np.maximum(count, 1)) @ graph).astype(np.float32).tocsr()
