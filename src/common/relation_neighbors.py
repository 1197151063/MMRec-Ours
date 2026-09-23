"""Fixed top-k neighbor tables without dense full graph storage or ID-distance rules."""
import numpy as np
import torch
from torch.nn import functional as F


def select_row(indices, values, row, k, tie_keys):
    valid=(indices!=row)&(values>0)&np.isfinite(values)
    indices,values=indices[valid],values[valid]
    # Random entity keys resolve equal similarity; never prefer nearby numeric IDs.
    if len(values)>k:
        threshold=np.partition(values,len(values)-k)[len(values)-k]
        keep=values>=threshold
        indices,values=indices[keep],values[keep]
    order=np.lexsort((tie_keys[indices],-values))[:k]
    indices,values=indices[order],values[order]
    if len(values):values=values/values.sum()
    return indices,values


def neighbor_tables(size,k,rows,tie_keys):
    if k<1:raise ValueError('neighbor_k must be positive')
    ids=np.zeros((size,k),dtype=np.int64); weights=np.zeros((size,k),dtype=np.float32)
    for row,indices,values in rows:
        indices,values=select_row(indices,values,row,k,tie_keys)
        ids[row,:len(indices)]=indices;weights[row,:len(values)]=values
    return torch.from_numpy(ids),torch.from_numpy(weights)


def feature_neighbors(features,k,seed=2026,block=256,tie_keys=None):
    if not torch.isfinite(features).all():raise ValueError('Features must be finite')
    size=len(features)
    if tie_keys is None:tie_keys=np.random.default_rng(seed).random(size)
    features=F.normalize(features.detach(),dim=-1)
    def rows():
        for start in range(0,size,block):
            scores=(features[start:start+block]@features.T).cpu().numpy()
            for offset,score in enumerate(scores):
                yield start+offset,np.arange(size),score
    return neighbor_tables(size,k,rows(),tie_keys)


def user_neighbors(binary,k,seed=2026,block=256,tie_keys=None):
    r=binary.tocsr().astype(np.float32)
    r.eliminate_zeros();r.sum_duplicates();r.data[:]=1
    if tie_keys is None:tie_keys=np.random.default_rng(seed).random(r.shape[0])
    def rows():
        for start in range(0,r.shape[0],block):
            co=(r[start:start+block]@r.T).tocsr()
            for offset in range(co.shape[0]):
                lo,hi=co.indptr[offset:offset+2]
                yield start+offset,co.indices[lo:hi],co.data[lo:hi]
    return neighbor_tables(r.shape[0],k,rows(),tie_keys)
