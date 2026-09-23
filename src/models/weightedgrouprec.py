"""LinkProp-inspired fixed structural-support weights for BPR, unchanged modality SSM."""
import logging
import time
import numpy as np
import torch
from torch.nn import functional as F
from models.grouprec import GroupRec


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


class WeightedGroupRec(GroupRec):
    def __init__(self,config,dataloader):
        super().__init__(config,dataloader)
        self.weight_mix=float(config['wbpr_mix'])
        if not 0 <= self.weight_mix <= 1:
            raise ValueError('wbpr_mix must be in [0,1]')
        if self.weight_mix:
            start=time.perf_counter()
            keys,support=positive_support(dataloader.inter_matrix(form='coo'),
                float(config['link_beta']),float(config['link_gamma']),float(config['link_delta']))
            if not len(keys):raise ValueError('Weighted BPR requires training edges')
            confidence=1+np.log1p(support)
            weights=(1-self.weight_mix)+self.weight_mix*confidence/confidence.mean()
            self.register_buffer('support_keys',torch.from_numpy(keys).long())
            self.register_buffer('support_weights',torch.from_numpy(weights.astype(np.float32)))
            logging.getLogger().info('WBPR mix=%s TRAIN edges=%s weight[min,mean,max]=%s,%s,%s startup_s=%.3f; fixed weights; edge-reusing paths removed',
                self.weight_mix,len(keys),weights.min(),weights.mean(),weights.max(),time.perf_counter()-start)

    def bpr_objective(self,users,positives,pos_scores,neg_scores):
        if not self.weight_mix:
            return super().bpr_objective(users,positives,pos_scores,neg_scores)
        keys=users*self.n_items+positives
        positions=torch.searchsorted(self.support_keys,keys)
        safe=positions.clamp_max(self.support_keys.numel()-1)
        if ((positions>=self.support_keys.numel()) | (self.support_keys[safe]!=keys)).any():
            raise ValueError('BPR positive is absent from the training-only support table')
        weights=self.support_weights[positions]
        return (weights[:,None]*F.softplus(neg_scores-pos_scores[:,None])).mean()
