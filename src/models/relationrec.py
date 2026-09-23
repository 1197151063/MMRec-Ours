"""Order-free ID recommendation with modality and homogeneous-neighbor losses."""
import logging
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models.lightmrecnope import LightMRecNoPE
from models.simmrec import SIMMRec
from common.linkprop import positive_support
from common.relation_neighbors import feature_neighbors,user_neighbors


class RelationRec(LightMRecNoPE):
    sample_negatives=SIMMRec.sample_negatives

    def make_projectors(self,dim):
        return nn.Linear(self.v_feat.shape[1],dim),nn.Linear(self.t_feat.shape[1],dim)

    def __init__(self,config,dataloader):
        super().__init__(config,dataloader)
        self.item_embedding=nn.Embedding(self.n_items,config['embedding_size'])
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.layers=int(config['ui_layers']); self.primary=config['primary_loss']
        self.ii_weight=float(config['ii_weight']);self.uu_weight=float(config['uu_weight'])
        self.modality_weight=float(config['modality_weight']);self.reg=float(config['relation_reg'])
        self.link_mix=float(config['link_mix'])
        if self.layers not in (0,1,2) or self.primary not in ('bpr','ssm'):
            raise ValueError('Invalid propagation/loss setting')
        if not 0<=self.alpha<=1 or not 0<=self.link_mix<=1 or self.temperature<=0 or self.num_negatives<1:
            raise ValueError('Invalid loss setting')
        if any(not math.isfinite(v) or v<0 for v in (self.ii_weight,self.uu_weight,self.modality_weight,self.reg)):
            raise ValueError('Loss weights must be finite and nonnegative')
        r=dataloader.inter_matrix(form='coo').tocsr().astype(np.float32)
        r.eliminate_zeros();r.sum_duplicates();r.data[:]=1;r.sort_indices()
        co=r.tocoo()
        keys=torch.from_numpy(co.row.astype(np.int64))*self.n_items+torch.from_numpy(co.col.astype(np.int64))
        keys=keys.unique(sorted=True)
        if not len(keys) or (torch.bincount(keys//self.n_items,minlength=self.n_users)>=self.n_items).any():
            raise ValueError('Need training positives and unobserved negative candidates')
        self.register_buffer('train_positive_keys',keys)
        if self.layers:
            u,i=keys//self.n_items,keys%self.n_items+self.n_users
            row,col=torch.cat((u,i)),torch.cat((i,u))
            n=self.n_users+self.n_items
            degree=torch.bincount(row,minlength=n).float().clamp_min(1)
            self.register_buffer('ui_adj',torch.sparse_coo_tensor(torch.stack((row,col)),
                (degree[row]*degree[col]).rsqrt(),(n,n)).coalesce())
        k=int(config['neighbor_k']);seed=int(config['neighbor_seed'])
        def register(prefix,pair):
            self.register_buffer(prefix+'_ids',pair[0]);self.register_buffer(prefix+'_weights',pair[1])
            logging.getLogger().info('%s neighbor coverage=%.4f',prefix,(pair[1].sum(-1)>0).float().mean().item())
        if self.ii_weight:
            register('visual_nn',feature_neighbors(self.image_embedding.weight,k,seed))
            register('text_nn',feature_neighbors(self.text_embedding.weight,k,seed))
        if self.uu_weight:register('user_nn',user_neighbors(r,k,seed))
        if self.link_mix:
            support_keys,support=positive_support(r)
            if not np.array_equal(support_keys,keys.numpy()):raise ValueError('Support key mismatch')
            confidence=1+np.log1p(support)
            weights=(1-self.link_mix)+self.link_mix*confidence/confidence.mean()
            self.register_buffer('link_weights',torch.from_numpy(weights.astype(np.float32)))
        self._eval_embeddings=None
        self.last_loss_terms={}
        self._loss_steps=0
        logging.getLogger().info('RELATION ui_layers=%s primary=%s ii=%s uu=%s link_mix=%s; no groups/order encodings; TRAIN-only user relations',
            self.layers,self.primary,self.ii_weight,self.uu_weight,self.link_mix)

    def forward(self):
        if not self.layers:return self.user_embedding.weight,self.item_embedding.weight
        x=torch.cat((self.user_embedding.weight,self.item_embedding.weight));out=x
        for _ in range(self.layers):
            x=torch.sparse.mm(self.ui_adj,x);out=out+x
        return (out/(self.layers+1)).split((self.n_users,self.n_items))

    def ssm_rows(self,user,candidates):
        logits=(F.normalize(user,dim=-1)[:,None]*F.normalize(candidates,dim=-1)).sum(-1)/self.temperature
        return torch.logsumexp(logits[:,1:],dim=-1)-logits[:,0]

    @staticmethod
    def neighbor_loss(anchors,neighbors,weights):
        return (weights*F.softplus(-(anchors[:,None]*neighbors).sum(-1))).sum(-1).mean()

    def calculate_loss(self,interaction):
        self._eval_embeddings=None
        users,positives=interaction[:2];u,i=self.forward()
        negatives=self.sample_negatives(users)
        candidates=torch.cat((positives[:,None],negatives),dim=1)
        if self.primary=='bpr':
            pos=(u[users]*i[positives]).sum(-1)
            neg=(u[users,None]*i[negatives]).sum(-1)
            main=F.softplus(neg-pos[:,None]).mean(-1)
        else:main=self.ssm_rows(u[users],i[candidates])
        if self.link_mix:
            keys=users*self.n_items+positives
            positions=torch.searchsorted(self.train_positive_keys,keys)
            if (positions>=len(self.train_positive_keys)).any() or not torch.equal(self.train_positive_keys[positions],keys):
                raise ValueError('Weighted positive not in TRAIN graph')
            main=main*self.link_weights[positions]
        main=main.mean();modal=main.new_zeros(());ii=modal;uu=modal
        if self.modality_weight:
            unique,inverse=candidates.unique(return_inverse=True)
            if self.alpha:
                v=self.image_trs(self.image_embedding(unique))[inverse]
                modal=modal+self.alpha*self.ssm_rows(u[users],v).mean()
            if self.alpha<1:
                t=self.text_trs(self.text_embedding(unique))[inverse]
                modal=modal+(1-self.alpha)*self.ssm_rows(u[users],t).mean()
        if self.ii_weight:
            for prefix,coefficient in (('visual_nn',self.alpha),('text_nn',1-self.alpha)):
                ids=getattr(self,prefix+'_ids')[positives];weights=getattr(self,prefix+'_weights')[positives]
                ii=ii+coefficient*self.neighbor_loss(u[users],i[ids],weights)
        if self.uu_weight:
            ids=self.user_nn_ids[users];weights=self.user_nn_weights[users]
            uu=self.neighbor_loss(i[positives],u[ids],weights)
        reg=self.user_embedding(users).square().sum(-1).mean()+self.item_embedding(candidates).square().sum(-1).mean()
        loss=main+self.modality_weight*modal+self.ii_weight*ii+self.uu_weight*uu+self.reg*reg
        self.last_loss_terms={name:val.detach() for name,val in dict(main=main,modality=modal,ii=ii,uu=uu,reg=reg).items()}
        if self._loss_steps % 200 == 0:
            logging.getLogger().info('RELATION unscaled loss terms: %s',
                {name:round(value.item(),6) for name,value in self.last_loss_terms.items()})
        self._loss_steps+=1
        return loss

    def train(self,mode=True):
        self._eval_embeddings=None
        return super().train(mode)

    @torch.no_grad()
    def full_sort_predict(self,interaction):
        users=interaction[0] if isinstance(interaction,(tuple,list)) else interaction
        if self.training:u,i=self.forward()
        else:
            if self._eval_embeddings is None:self._eval_embeddings=self.forward()
            u,i=self._eval_embeddings
        return u[users]@i.T
