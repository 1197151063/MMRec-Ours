"""FreedomAlign plus complete shared-item and optional semantic-bridge UU loss."""
import math
from logging import getLogger
import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F
from models.freedomalign import FreedomAlign
from common.user_walks import UserWalks


class FreedomAlignUU(FreedomAlign):
    def __init__(self,config,dataset):
        super().__init__(config,dataset)
        self.uu_weight=float(config['uu_weight'])
        self.uu_bridge_weight=float(config['uu_bridge_weight'])
        self.uu_samples=int(config['uu_samples'])
        if any(not math.isfinite(x) or x<0 for x in (self.uu_weight,self.uu_bridge_weight)) or self.uu_samples<1:
            raise ValueError('Invalid UU weights/sample count')
        self.walks=None
        if self.uu_weight:
            r=dataset.inter_matrix(form='csr').astype(np.float32).tocsr()
            semantic=None
            if self.uu_bridge_weight:
                with torch.no_grad():
                    features=F.normalize(torch.cat((F.normalize(self.v_feat,dim=1),F.normalize(self.t_feat,dim=1)),dim=1),dim=1)
                    ids,values=self.content_knn(features,int(config['uu_item_k']),int(config['knn_block']),False)
                ids=ids.cpu().numpy();values=values.cpu().numpy()
                rows=np.repeat(np.arange(self.n_items),ids.shape[1]);cols=ids.ravel();weights=values.ravel()
                keep=weights>0
                semantic=sp.csr_matrix((weights[keep],(rows[keep],cols[keep])),shape=(self.n_items,self.n_items))
                semantic=semantic.maximum(semantic.T).tocsr()
            self.walks=UserWalks(r,semantic,seed=int(config['uu_sampling_seed']))

    @staticmethod
    def sampled_constraint(embedding,users,neighbors,valid):
        scores=(embedding[users,None]*embedding[neighbors]).sum(-1)
        # Fixed denominator B*M: self and dead-end draws contribute zero.
        return (F.softplus(-scores)*valid.to(scores.dtype)).mean()

    def calculate_loss(self,interaction):
        base=super().calculate_loss(interaction)
        if not self.uu_weight:return base
        users=interaction[0]
        cpu_users=users.detach().cpu().numpy()
        def branch(bridge):
            neighbors,valid=self.walks.sample(cpu_users,self.uu_samples,bridge)
            neighbors=torch.from_numpy(neighbors).to(users.device)
            valid=torch.from_numpy(valid).to(users.device)
            return self.sampled_constraint(self.user_embedding.weight,users,neighbors,valid),valid.float().mean()
        shared,shared_fraction=branch(False)
        bridge=base.new_zeros(());bridge_fraction=base.new_zeros(())
        if self.uu_bridge_weight:bridge,bridge_fraction=branch(True)
        uu=self.uu_weight*(shared+self.uu_bridge_weight*bridge)
        total=base+uu
        self.last_loss_terms.update(uu_shared=shared.detach().item(),uu_bridge=bridge.detach().item(),
                                   uu_weighted=uu.detach().item(),total=total.detach().item())
        if self._batches==1 or self._batches%200==0:
            getLogger().info('FreedomAlignUU: base=%.6f shared=%.6f bridge=%.6f weighted_uu=%.6f total=%.6f valid_walks=(%.3f,%.3f)',
                base.detach().item(),shared.detach().item(),bridge.detach().item(),uu.detach().item(),total.detach().item(),
                shared_fraction.item(),bridge_fraction.item())
        return total
