"""Factorized TRAIN-only UU distributions; never materializes a U x U graph."""
import numpy as np
import scipy.sparse as sp


class RowSampler:
    def __init__(self, matrix, weighted=False):
        matrix=matrix.tocsr().astype(np.float64)
        matrix.sum_duplicates();matrix.eliminate_zeros();matrix.sort_indices()
        if not np.isfinite(matrix.data).all() or (matrix.data<=0).any():
            raise ValueError('Sampling weights must be positive and finite')
        self.ptr=matrix.indptr.copy();self.ids=matrix.indices.copy()
        self.cdf=np.cumsum(matrix.data,dtype=np.float64) if weighted else None

    def draw(self, rows, rng):
        rows=np.asarray(rows,dtype=np.int64)
        start=self.ptr[rows];end=self.ptr[rows+1];valid=end>start
        result=np.zeros(rows.shape,dtype=np.int64)
        if not valid.any():return result,valid
        first,last=start[valid],end[valid]
        q=rng.random(first.shape)
        if self.cdf is None:
            idx=first+np.floor(q*(last-first)).astype(np.int64)
        else:
            lo=np.where(first>0,self.cdf[np.maximum(first-1,0)],0.)
            hi=self.cdf[last-1]
            idx=np.searchsorted(self.cdf,lo+q*(hi-lo),side='right')
            idx=np.minimum(np.maximum(idx,first),last-1)
        result[valid]=self.ids[idx]
        return result,valid


class UserWalks:
    def __init__(self, interactions, semantic=None, seed=2026):
        r=interactions.tocsr().astype(np.float32)
        r.sum_duplicates();r.eliminate_zeros();r.data[:]=1
        self.ui=RowSampler(r);self.iu=RowSampler(r.T)
        self.semantic=RowSampler(semantic,weighted=True) if semantic is not None else None
        # Independent streams keep shared walks identical with/without bridges.
        self.shared_rng=np.random.default_rng(seed)
        self.bridge_rng=np.random.default_rng(seed+1)

    def sample(self, users, count, bridge=False):
        if count<1:raise ValueError('Positive sample count required')
        users=np.asarray(users,dtype=np.int64)
        anchors=np.broadcast_to(users[:,None],(len(users),count))
        rng=self.bridge_rng if bridge else self.shared_rng
        items,valid=self.ui.draw(anchors,rng)
        if bridge:
            if self.semantic is None:raise ValueError('Semantic sampler not built')
            items,ok=self.semantic.draw(items,rng);valid &= ok
        neighbors,ok=self.iu.draw(items,rng)
        # Zero self/dead-end paths without rejection. This is an unbiased sample
        # of the explicitly defined off-diagonal, sub-stochastic walk operator.
        valid &= ok & (neighbors!=anchors)
        return neighbors,valid
