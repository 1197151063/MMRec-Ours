"""Learnable group preference residual with single-layer 64-dimensional modality projectors."""
import hashlib
import logging
import math
import torch
from torch import nn
from models.lightmrecnope import LightMRecNoPE


def user_groups(n_users, group_size, mode, seed):
    if n_users < 1 or group_size < 1 or mode not in ('contiguous', 'random'):
        raise ValueError('Require positive sizes and contiguous/random grouping')
    positions = torch.arange(n_users)
    if mode == 'random':
        positions = torch.randperm(n_users, generator=torch.Generator().manual_seed(seed))
    return positions // group_size


class GroupRec(LightMRecNoPE):
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        dim = config['embedding_size']
        self.group_strength = float(config['group_strength'])
        self.use_personal = bool(config['use_personal'])
        group_size = int(config['group_size'])
        mode = config['group_mode']
        group_init = config['group_init']
        std = float(config['group_std'])
        if not math.isfinite(self.group_strength) or self.group_strength < 0 or not math.isfinite(std) or std < 0:
            raise ValueError('Group strength/std must be finite and nonnegative')
        if group_init not in ('normal', 'zero'):
            raise ValueError('group_init must be normal or zero')
        if not self.use_personal and self.group_strength == 0:
            raise ValueError('At least one user preference path must be active')
        groups = user_groups(self.n_users, group_size, mode, int(config['group_seed']))
        self.register_buffer('user_group', groups)
        count = int(groups.max())+1
        generator = torch.Generator().manual_seed(int(config['group_seed'])+1)
        weights = torch.randn(count, dim, generator=generator)*std
        if group_init == 'zero':
            weights.zero_()
        self.group_embedding = nn.Embedding.from_pretrained(weights, freeze=not bool(config['group_trainable']))
        self.user_embedding.weight.requires_grad_(self.use_personal)
        digest = hashlib.sha256(groups.numpy().tobytes()).hexdigest()[:16]
        logging.getLogger().info('GROUP mode=%s size=%s count=%s strength=%s init=%s std=%s trainable=%s map=%s; contiguous uses original numeric user IDs',
                                mode,group_size,count,self.group_strength,group_init,std,
                                config['group_trainable'],digest)

    def make_projectors(self, dim):
        self.feat_embed_dim = 64
        if dim != self.feat_embed_dim:
            raise ValueError('GroupRec requires embedding_size=64 to match modality projections')
        return (nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim))

    def forward(self):
        user,item = super().forward()
        if not self.use_personal:
            user = torch.zeros_like(user)
        if self.group_strength:
            user = user + self.group_strength*self.group_embedding(self.user_group)
        return user,item
