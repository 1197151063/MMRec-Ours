"""TRAIN-only item groups and degree-corrected four-hop user relations."""
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F


def binary(interactions):
    r = interactions.tocsr().astype(np.float32).copy()
    r.sum_duplicates(); r.eliminate_zeros(); r.data[:] = 1; r.sort_indices()
    return r


class ItemGroupPeers(nn.Module):
    """Uniform other-consumer sampling within each TRAIN item hyperedge."""
    def __init__(self, interactions):
        super().__init__()
        r = binary(interactions)
        csc = r.tocsc(); csc.sort_indices()
        item_ids = np.repeat(np.arange(r.shape[1]), np.diff(csc.indptr))
        keys = csc.indices.astype(np.int64) * r.shape[1] + item_ids
        order = np.argsort(keys)
        self.n_items = r.shape[1]
        for name, array in dict(item_ptr=csc.indptr, consumers=csc.indices,
                                keys=keys[order], positions=np.arange(r.nnz)[order]).items():
            self.register_buffer(name, torch.as_tensor(array, dtype=torch.long))

    def sample(self, users, items):
        key = users * self.n_items + items
        loc = torch.searchsorted(self.keys, key)
        if torch.any(loc >= len(self.keys)) or torch.any(self.keys[loc] != key):
            raise ValueError('Item-group anchors must be observed TRAIN interactions')
        start = self.item_ptr[items]
        degree = self.item_ptr[items + 1] - start
        own = self.positions[loc] - start
        offset = (torch.rand(len(users), device=users.device) * (degree - 1).clamp_min(1)).long()
        offset = offset + (offset >= own).long()
        valid = degree > 1
        offset = torch.where(valid, offset, torch.zeros_like(offset))
        peer = self.consumers[start + offset]
        return peer, valid, degree


def interest_distance(users, peers, directions=None):
    """Full cosine attraction or consistency along detached modality directions."""
    diff = F.normalize(users, dim=-1) - F.normalize(peers, dim=-1)
    if directions is None:
        return diff.square().sum(-1)
    # Unit modality directions cannot shrink or rotate to evade this constraint.
    values = [(diff * F.normalize(q.detach(), dim=-1)).sum(-1).square()
              for q in directions]
    return users.shape[-1] * torch.stack(values).mean(0)


def build_bridges(interactions, k=20, block=64, seed=2026, ties=None, progress=None):
    """Cosine of UU-neighborhoods, downweighting generic bridge users.

    Only positive four-hop scores without direct shared TRAIN items are kept.
    Never materializes the full dense U x U graph. Independent keys break ties.
    """
    r = binary(interactions)
    du = np.asarray(r.sum(1)).ravel()
    di = np.asarray(r.sum(0)).ravel()
    if k < 1 or block < 1:
        raise ValueError('Bridge k/block must be positive')
    x = sp.diags(1 / np.sqrt(np.maximum(du, 1))) @ r @ sp.diags(1 / np.sqrt(np.maximum(di, 1)))
    shared = (x @ x.T).tocsr(); shared.setdiag(0); shared.eliminate_zeros()
    q = shared @ sp.diags(1 / np.sqrt(np.maximum(np.diff(shared.indptr), 1)))
    norm = np.sqrt(np.asarray(q.multiply(q).sum(1)).ravel())
    q = (sp.diags(1 / np.maximum(norm, 1e-12)) @ q).tocsr()
    ties = np.random.default_rng(seed).random(r.shape[0]) if ties is None else np.asarray(ties)
    if ties.shape != (r.shape[0],) or not np.isfinite(ties).all() or len(np.unique(ties)) != len(ties):
        raise ValueError('Bridge tie keys must be finite and unique')
    neighbors = np.repeat(np.arange(r.shape[0])[:, None], k, axis=1)
    weights = np.zeros(neighbors.shape, dtype=np.float32)
    for start in range(0, r.shape[0], block):
        scores = (q[start:start + block] @ q.T).tocsr()
        for local in range(scores.shape[0]):
            u = start + local
            a, b = scores.indptr[local:local + 2]
            ids, values = scores.indices[a:b], scores.data[a:b]
            direct = shared.indices[shared.indptr[u]:shared.indptr[u + 1]]
            valid = (ids != u) & (values > 1e-8) & ~np.isin(ids, direct)
            ids, values = ids[valid], values[valid]
            if len(values) > k:
                threshold = np.partition(values, -k)[-k]
                keep = values >= threshold
                ids, values = ids[keep], values[keep]
            selected = np.lexsort((ties[ids], -values))[:k]
            ids, values = ids[selected], values[selected]
            neighbors[u, :len(ids)] = ids
            if len(values):
                weights[u, :len(ids)] = values / values.sum()
        if progress is not None:
            progress(min(start + block, r.shape[0]), r.shape[0])
    return neighbors, weights
