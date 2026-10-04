"""Train-only CF graph operators. No numeric-ID distance or held-out labels."""
import numpy as np
import scipy.sparse as sp
import torch


def tensor(a, device):
    a = a.tocoo().astype(np.float32)
    return torch.sparse_coo_tensor(torch.from_numpy(np.stack([a.row, a.col])),
                                   torch.from_numpy(a.data), a.shape, device=device).coalesce()


def normalize(a):
    degree = np.asarray(a.sum(1)).ravel()
    inv = np.zeros_like(degree, dtype=np.float32)
    np.power(degree, -.5, out=inv, where=degree > 0)
    return (sp.diags(inv) @ a @ sp.diags(inv)).tocsr()


def ultra_neighbors(r, k=20, block=256):
    """UltraGCN Eq. 16: top-k omega, not top-k raw co-occurrence.

    Full G degrees include the diagonal; self-neighbors are excluded from the loss.
    Padded neighbors have zero weight, including isolated items.
    """
    r = r.tocsr().astype(np.float32)
    g = np.asarray(r.T @ np.asarray(r.sum(1)).ravel()).ravel()
    diag = np.asarray(r.sum(0)).ravel()
    n = r.shape[1]
    k = min(k, max(1, n - 1))
    ids, weights = np.zeros((n, k), np.int64), np.zeros((n, k), np.float32)
    for start in range(0, n, block):
        counts = (r[:, start:start + block].T @ r).tocsr()
        for b in range(counts.shape[0]):
            i = start + b
            lo, hi = counts.indptr[b:b + 2]
            col, val = counts.indices[lo:hi], counts.data[lo:hi]
            keep = (col != i) & (g[col] > 0)
            col, val = col[keep], val[keep]
            if g[i] <= diag[i] or not len(col):
                continue
            omega = val / (g[i] - diag[i]) * np.sqrt(g[i] / g[col])
            top = np.argsort(-omega, kind='stable')[:k]
            ids[i, :len(top)], weights[i, :len(top)] = col[top], omega[top]
    return ids, weights


def cir_direction(incidence, similarity='jc', block=256):
    """Return phi[v,w] on existing edges: mean similarity of w to N(v).

    incidence rows are centers v, columns are neighbors w. Similarities include
    self, as in the CIR neighborhood sum. Dense workspace is only block x V.
    Both orientations are computed separately; phi need not be symmetric.
    """
    r = incidence.tocsr().astype(np.float32)
    c = r.T.tocsr()
    degree = np.asarray(c.sum(1)).ravel()
    rows, cols, values = [], [], []
    center_degree = np.asarray(r.sum(1)).ravel().clip(1)
    for start in range(0, c.shape[0], block):
        counts = (c[start:start + block] @ c.T).tocoo()
        a, b = degree[start + counts.row], degree[counts.col]
        den = a + b - counts.data if similarity == 'jc' else a * b
        counts.data /= np.maximum(den, 1e-12)
        accumulated = (counts.tocsr() @ c).tocsr()
        observed = c[start:start + block].tocoo()
        val = np.asarray(accumulated[observed.row, observed.col]).ravel()
        rows.append(observed.col); cols.append(start + observed.row)
        values.append(val / center_degree[observed.col])
    return sp.csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=r.shape)


def cagcn(r, similarity='jc', mix=.5, block=256):
    """One-hop CIR; CAGCN row-mass rescaling (Eq.25), mixed with LightGCN."""
    base = normalize(sp.bmat([[None, r], [r.T, None]], format='csr'))
    phi = sp.bmat([[None, cir_direction(r, similarity, block)],
                  [cir_direction(r.T, similarity, block), None]], format='csr')
    mass = np.asarray(base.sum(1)).ravel()
    sums = np.asarray(phi.sum(1)).ravel()
    scale = np.divide(mass, sums, out=np.zeros_like(mass), where=sums > 0)
    adjusted = sp.diags(scale) @ phi
    return ((1 - mix) * base + mix * adjusted).tocsr()


@torch.no_grad()
def top_dot(x, y, k, block=256, same=False):
    """Binary top-k graph by raw learned dot products; blockwise, detached."""
    k = min(k, len(y) - int(same))
    if k <= 0:
        return sp.csr_matrix((len(x), len(y)), dtype=np.float32)
    rows, cols = [], []
    for start in range(0, len(x), block):
        scores = x[start:start + block] @ y.T
        if same:
            scores[torch.arange(len(scores), device=x.device),
                   torch.arange(start, start + len(scores), device=x.device)] = -torch.inf
        ids = scores.topk(k, dim=1).indices.cpu().numpy()
        rows.append(np.repeat(np.arange(start, start + len(ids)), k)); cols.append(ids.ravel())
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(x), len(y)))


def graphda(embeddings, nusers, r, k=10, homogeneous_k=0, mix=1., block=256):
    """GraphDA-inspired REARM adapter: UI union, optional symmetric UU/II.

    Retains REARM's original homogeneous stages. mix=1 uses reconstructed UI;
    mix<1 is an explicitly separate residual variant. No score thresholding.
    """
    u, i = embeddings[:nusers], embeddings[nusers:]
    enhanced = top_dot(u, i, k, block).maximum(top_dot(i, u, k, block).T)
    uu = top_dot(u, u, homogeneous_k, block, same=True)
    ii = top_dot(i, i, homogeneous_k, block, same=True)
    uu, ii = uu.maximum(uu.T), ii.maximum(ii.T)
    graph = normalize(sp.bmat([[uu, enhanced], [enhanced.T, ii]], format='csr'))
    base = normalize(sp.bmat([[None, r], [r.T, None]], format='csr'))
    stats = dict(ui_edges=enhanced.nnz, uu_edges=uu.nnz, ii_edges=ii.nnz,
                 retained_train_edges=enhanced.multiply(r).nnz,
                 new_edges=enhanced.nnz - enhanced.multiply(r).nnz,
                 teacher_uses='training interactions only', mix=mix)
    return ((1 - mix) * base + mix * graph).tocsr(), stats
