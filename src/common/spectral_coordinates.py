"""Continuous coordinates from a binary TRAIN interaction matrix, without clustering."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds


def build_spectral_coordinates(interactions, rank=64, seed=2026):
    r = interactions.tocsr().astype(np.float32).copy()
    r.sum_duplicates(); r.eliminate_zeros(); r.sort_indices()
    if not np.isfinite(r.data).all() or (r.data < 0).any():
        raise ValueError('Interaction values must be finite and nonnegative')
    r.data[:] = 1.
    used_rank = min(int(rank), min(r.shape) - 1)
    if used_rank < 1 or r.nnz == 0:
        raise ValueError('Require a positive rank and a nonempty graph with both dimensions >= 2')
    du = np.asarray(r.sum(1)).ravel()
    di = np.asarray(r.sum(0)).ravel()
    x = (sp.diags(1 / np.sqrt(np.maximum(du, 1))) @ r @
         sp.diags(1 / np.sqrt(np.maximum(di, 1)))).tocsr()
    u, singular, vt = svds(x, k=used_rank, random_state=seed)
    order = np.argsort(singular)[::-1]
    singular = singular[order]
    users = u[:, order] * singular
    items = vt[order].T * singular
    users[du == 0] = 0.; items[di == 0] = 0.
    # One shared scalar preserves user/item relative geometry; no row normalization.
    energy = float(np.square(users).sum(dtype=np.float64) + np.square(items).sum(dtype=np.float64))
    scale = np.sqrt(energy / ((np.count_nonzero(du) + np.count_nonzero(di)) * used_rank))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Degenerate spectral coordinates')
    return (users / scale).astype(np.float32), (items / scale).astype(np.float32), singular
