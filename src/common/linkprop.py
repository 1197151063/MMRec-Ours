"""Training-only LinkProp-inspired support, shared by independent models."""
import numpy as np


def positive_support(matrix, beta=.5, gamma=.5, delta=.5, block=32):
    """Eq.12 three-hop scores, excluding walks that reuse the scored positive edge.

    Degrees are fixed on the complete binary TRAIN graph (not recomputed per edge).
    No full user-item score matrix is retained; return scores on observed edges only.
    """
    if block < 1 or any(not np.isfinite(x) or x < 0 for x in (beta,gamma,delta)):
        raise ValueError('Invalid LinkProp exponent/block size')
    r=matrix.tocsr().astype(np.float64)
    r.eliminate_zeros(); r.sum_duplicates(); r.data[:]=1.; r.sort_indices()
    du=np.asarray(r.sum(1)).ravel().clip(1)
    di=np.asarray(r.sum(0)).ravel().clip(1)
    ib,ug,id_=di**-beta,du**-gamma,di**-delta
    right=r.multiply(ug[:,None]).multiply(id_[None,:]).tocsr()
    rows=np.repeat(np.arange(r.shape[0]),np.diff(r.indptr));cols=r.indices.copy()
    support=np.empty(len(cols),dtype=np.float64)
    back_item=ib*np.asarray(r.T@ug).ravel()
    back_user=ug*np.asarray(r@ib).ravel()
    for start in range(0,r.shape[0],block):
        stop=min(start+block,r.shape[0])
        lo,hi=r.indptr[start],r.indptr[stop]
        if lo==hi:continue
        score=(r[start:stop].multiply(ib[None,:])@r.T)@right
        u,i=rows[lo:hi],cols[lo:hi]
        full=np.asarray(score.tocsr()[u-start,i]).ravel()
        # Union of ix=i and vx=u, with their overlap added back once.
        back=(back_item[i]+back_user[u]-ib[i]*ug[u])*id_[i]
        support[lo:hi]=np.maximum(full-back,0.)
    keys=rows*r.shape[1]+cols
    return keys,support

