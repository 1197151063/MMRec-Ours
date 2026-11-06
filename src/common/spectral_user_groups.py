"""Fixed user groups from the normalized TRAIN bipartite graph."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.cluster import MiniBatchKMeans


def build_spectral_user_groups(interactions, groups=16, rank=64, seed=2026):
    r = interactions.tocsr().astype(np.float32).copy()
    r.sum_duplicates(); r.eliminate_zeros(); r.data[:] = 1; r.sort_indices()
    degree_u = np.diff(r.indptr)
    degree_i = np.asarray(r.sum(0)).ravel()
    active = degree_u > 0
    if groups < 2 or groups > int(active.sum()) or rank < 1:
        raise ValueError('Require 2 <= groups <= active users and positive spectral rank')
    x = (sp.diags(1 / np.sqrt(np.maximum(degree_u, 1))) @ r @
         sp.diags(1 / np.sqrt(np.maximum(degree_i, 1)))).tocsr()
    used_rank = min(rank, min(x.shape) - 1)
    if used_rank < 1:
        raise ValueError('Interaction graph too small for spectral grouping')
    users, _, _ = svds(x, k=used_rank, random_state=seed)
    users = users.astype(np.float32)
    users /= np.maximum(np.linalg.norm(users, axis=1, keepdims=True), 1e-12)
    model = MiniBatchKMeans(n_clusters=groups, random_state=seed, n_init=3,
                            batch_size=1024, max_iter=100).fit(users[active])
    labels = np.full(r.shape[0], -1, dtype=np.int64)
    labels[active] = model.labels_
    return labels
