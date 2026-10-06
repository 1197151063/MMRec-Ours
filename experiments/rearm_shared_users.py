"""User representations tied to current TRAIN item bases, without ID-order inputs.

ID offsets appear only in detached diagnostic pair selection, never the forward,
loss, graph, regularization or validation-based model selection.
"""
import weakref
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from rearm_cf_graphs import tensor


def history_weights(r, pop_power=0.):
    r = r.tocsr().astype(np.float32).copy()
    r.sum_duplicates(); r.eliminate_zeros(); r.data[:] = 1
    degree_i = np.asarray(r.sum(0)).ravel()
    a = r @ sp.diags((degree_i + 1).astype(np.float32) ** (-pop_power))
    total = np.asarray(a.sum(1)).ravel()
    inv = np.divide(1., total, out=np.zeros_like(total), where=total > 0)
    return (sp.diags(inv) @ a).tocsr()


def bounded_residual(shared, delta, cap):
    # No independent direction can exceed cap[u] times the shared row norm.
    # The detached norm sets scale; shared item bases still get aggregation gradients.
    direction = delta / (1 + delta.norm(dim=-1, keepdim=True))
    return shared + cap * shared.norm(dim=-1, keepdim=True).detach() * direction


class SharedUserID(nn.Module):
    def __init__(self, source, pooling, cap):
        super().__init__()
        self._source = weakref.ref(source)
        self.register_buffer('pooling', pooling)
        self.register_buffer('cap', cap)
        delta = source.weight.new_zeros(pooling.shape[0], source.embedding_dim)
        if torch.any(cap > 0):
            self.residual = nn.Parameter(delta)
        else:
            self.register_buffer('residual', delta)

    @property
    def weight(self):
        return bounded_residual(torch.sparse.mm(self.pooling, self._source().weight), self.residual, self.cap)

    def forward(self, indices):
        return self.weight[indices]


class SharedUserModal(nn.Module):
    def __init__(self, source, encoder, pooling, cap):
        super().__init__()
        self._source = weakref.ref(source)
        self.encoder = encoder
        self.register_buffer('pooling', pooling)
        self.register_buffer('cap', cap)

    def forward(self, delta):
        item_basis = self.encoder(self._source().weight)
        return bounded_residual(torch.sparse.mm(self.pooling, item_basis), delta, self.cap)


def configure_shared_users(model):
    c = model.cf
    pooling = tensor(history_weights(model.r, c['shared_pop_power']), model.device)
    degree = np.asarray(model.r.sum(1)).ravel()
    gate = np.sqrt(np.divide(degree, degree + c['shared_degree_c'],
                            out=np.zeros_like(degree), where=(degree + c['shared_degree_c']) > 0))
    cap = torch.tensor(gate[:, None] * c['shared_residual'], device=model.device)
    for table, projector, source in (('user_v_prefer', 'image_u_trs', 'image_embedding'),
                                      ('user_t_prefer', 'text_u_trs', 'text_embedding')):
        old = getattr(model, table)
        delta = old.new_zeros((model.n_users, model.embedding_dim))
        delattr(model, table)
        if c['shared_residual']:
            setattr(model, table, nn.Parameter(delta))
        else:
            model.register_buffer(table, delta)
        encoder = getattr(model, projector)
        setattr(model, projector, SharedUserModal(getattr(model, source), encoder, pooling, cap))
    if c['shared_users'] == 'all':
        model.user_id_embedding = SharedUserID(model.item_id_embedding, pooling, cap)
    # Separate diagnostic-only objects. They are not touched by forward/loss.
    model.geometry_pairs = diagnostic_pairs(model.r)


def diagnostic_pairs(r, count=1024):
    rng = np.random.default_rng(2026)
    degree = np.diff(r.indptr)
    n = r.shape[0]
    active = np.flatnonzero(degree > 0)
    pools = {d: np.flatnonzero(degree == d) for d in np.unique(degree[active])}
    pairs = {}
    for offset in (1, 4, 16, 64):
        valid = active[active + offset < n]
        valid = valid[degree[valid + offset] > 0]
        anchors = rng.choice(valid, min(count, len(valid)), replace=False)
        near, far = [], []
        for u in anchors:
            v = u + offset
            candidates = pools[degree[v]]
            candidates = candidates[np.abs(candidates-u) > 256]
            if not len(candidates):
                continue  # No approximate/fallback degree match.
            near.append((u, v)); far.append((u, int(rng.choice(candidates))))
        pairs[f'near_{offset}'] = np.asarray(near, dtype=np.int64).reshape(-1, 2)
        pairs[f'degree_far_{offset}'] = np.asarray(far, dtype=np.int64).reshape(-1, 2)
    # Shared-item pairs have no ID-distance filter; they measure the actual prior.
    csc = r.tocsc()
    shared = []
    for u in rng.choice(active, min(count, len(active)), replace=False):
        items = r.indices[r.indptr[u]:r.indptr[u+1]].copy()
        rng.shuffle(items)
        for i in items:
            peers = csc.indices[csc.indptr[i]:csc.indptr[i+1]]
            peers = peers[peers != u]
            if len(peers):
                shared.append((u, int(rng.choice(peers))))
                break
    pairs['shared_item'] = np.asarray(shared, dtype=np.int64).reshape(-1, 2)
    if len(active) > 1:
        anchors = rng.choice(active, min(count, len(active)), replace=False)
        others = []
        for u in anchors:
            others.append(int(rng.choice(active[active != u])))
        pairs['random'] = np.column_stack([anchors, others])
    return pairs


@torch.no_grad()
def geometry(model):
    stages = dict(user_modal_input=torch.cat([model._projected['image_u_trs'], model._projected['text_u_trs']], 1),
                  final_user=model.cf_out[:model.n_users])
    result = dict(note='Diagnostic only: exact degree-matched far controls; not used to train or select models',
                  pair_counts={k: len(p) for k, p in model.geometry_pairs.items()})
    for stage, h in stages.items():
        h = h.detach()
        unit = F.normalize(h, dim=-1)
        stats = dict(mean_norm=float(h.norm(dim=-1).mean()),
                     mean_coordinate_variance=float(h.var(dim=0, unbiased=False).mean()))
        for key, pair in model.geometry_pairs.items():
            if len(pair):
                idx = torch.as_tensor(pair, device=h.device)
                cosine = (unit[idx[:, 0]] * unit[idx[:, 1]]).sum(-1)
                stats[key] = dict(mean_cosine=float(cosine.mean()),
                                 mean_squared_distance=float((h[idx[:, 0]] - h[idx[:, 1]]).square().sum(-1).mean()))
            else:
                stats[key] = None
        for offset in (1, 4, 16, 64):
            near, far = stats[f'near_{offset}'], stats[f'degree_far_{offset}']
            stats[f'near_minus_far_cosine_{offset}'] = near['mean_cosine']-far['mean_cosine'] if near and far else None
        result[stage] = stats
    return result
