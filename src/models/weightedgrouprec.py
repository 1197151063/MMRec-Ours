"""LinkProp-inspired fixed structural-support weights for BPR, unchanged modality SSM."""
import logging
import time
import numpy as np
import torch
from torch.nn import functional as F
from models.grouprec import GroupRec


from common.linkprop import positive_support


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
