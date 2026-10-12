"""Train-only item cohorts with implicit, parameter-free user diffusion."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import torch
from torch import nn


def _unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def build_membership(interactions, text_features, kind='joint', count=64, seed=2026):
    """Build U x C memberships from TRAIN interactions and item features only."""
    if kind not in ('text', 'collaborative', 'joint') or count < 2:
        raise ValueError('Invalid cohort kind or count')
    r = interactions.tocsr().astype(np.float32).copy()
    r.sum_duplicates(); r.eliminate_zeros(); r.data[:] = 1
    text = np.asarray(text_features, dtype=np.float32)
    if text.ndim != 2 or len(text) != r.shape[1] or not np.isfinite(text).all():
        raise ValueError('Invalid item features')
    if count >= r.shape[1]:
        raise ValueError('Cohort count must be smaller than item count')
    vectors = _unit(text)
    if kind != 'text':
        user_degree = np.diff(r.indptr).astype(np.float32)
        profile = r.T @ sp.diags(1 / np.sqrt(np.maximum(user_degree, 1)))
        rank = min(32, min(profile.shape) - 1)
        item_latent, singular, _ = svds(profile.astype(np.float32), k=rank,
                                       random_state=seed)
        collaborative = _unit(item_latent * singular)
        vectors = collaborative if kind == 'collaborative' else _unit(
            np.concatenate((vectors, collaborative), axis=1))
    from sklearn.cluster import MiniBatchKMeans
    clusters = MiniBatchKMeans(n_clusters=count, batch_size=1024,
                               n_init=3, max_iter=100, random_state=seed).fit(vectors)
    scores = vectors @ _unit(clusters.cluster_centers_).T
    indices = np.argpartition(scores, -2, axis=1)[:, -2:]
    selected = np.take_along_axis(scores, indices, axis=1)
    weights = np.exp((selected - selected.max(axis=1, keepdims=True)) / .1)
    weights /= weights.sum(axis=1, keepdims=True)
    item_cohorts = sp.csr_matrix(
        (weights.ravel(), (np.repeat(np.arange(r.shape[1]), 2), indices.ravel())),
        shape=(r.shape[1], count), dtype=np.float32)
    mass = (r @ item_cohorts).toarray().astype(np.float32)
    mass /= np.sqrt(np.maximum(mass.sum(axis=0, keepdims=True), 1))
    row_sum = mass.sum(axis=1, keepdims=True)
    return np.divide(mass, row_sum, out=np.zeros_like(mass), where=row_sum > 0)


class SoftItemCohort(nn.Module):
    """Peer average through overlapping item cohorts, excluding each user's self edge."""

    def __init__(self, membership):
        super().__init__()
        p = torch.as_tensor(membership, dtype=torch.float32)
        if p.ndim != 2 or not torch.isfinite(p).all() or (p < 0).any():
            raise ValueError('Invalid membership')
        self.register_buffer('membership', p)
        mass = p.sum(0).clamp_min(1e-12)
        self.register_buffer('cohort_mass', mass)
        self.register_buffer('self_weight', (p.square() / mass).sum(1, keepdim=True))
        self.register_buffer('active', p.sum(1, keepdim=True) > 0)

    def forward(self, users):
        p = self.membership
        centers = (p.T @ users) / self.cohort_mass[:, None]
        mixed = p @ centers
        remaining = 1 - self.self_weight
        peers = (mixed - self.self_weight * users) / remaining.clamp_min(1e-12)
        return torch.where(self.active & (remaining > 1e-6), peers, users)
