"""Two-layer ID LightGCN, user group residual, BPR and modality SSM."""
import hashlib
import logging
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.lightmrecnope import LightMRecNoPE


def user_groups(n_users, num_groups, mode, seed):
    if n_users < 1 or num_groups < 1 or mode not in ('contiguous', 'random'):
        raise ValueError('Require positive sizes and contiguous/random grouping')
    positions = torch.arange(n_users)
    if mode == 'random':
        positions = torch.randperm(n_users, generator=torch.Generator().manual_seed(seed))
    return positions * num_groups // n_users


class GroupRec(LightMRecNoPE):
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        dim = config['embedding_size']
        if not 0 <= self.alpha <= 1:
            raise ValueError('alpha must be in [0,1] (image loss weight)')
        self.item_embedding = nn.Embedding(self.n_items, dim)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.n_layers = 2
        # Binary TRAIN interactions only, symmetric D^-1/2 A D^-1/2,
        # without self loops. Layer zero is included in the final average.
        matrix = dataloader.inter_matrix(form='coo')
        keys = (torch.as_tensor(matrix.row, dtype=torch.long)*self.n_items
                + torch.as_tensor(matrix.col, dtype=torch.long)).unique()
        users, items = keys // self.n_items, keys % self.n_items + self.n_users
        row, col = torch.cat((users, items)), torch.cat((items, users))
        total = self.n_users + self.n_items
        degree = torch.bincount(row, minlength=total).float().clamp_min(1)
        values = (degree[row]*degree[col]).rsqrt()
        self.register_buffer('ui_adj', torch.sparse_coo_tensor(
            torch.stack((row, col)), values, (total, total)).coalesce())
        self.group_strength = float(config['group_strength'])
        self.use_personal = bool(config['use_personal'])
        num_groups = int(config['num_groups'])
        mode = config['group_mode']
        group_init = config['group_init']
        std = float(config['group_std'])
        if not math.isfinite(self.group_strength) or self.group_strength < 0 or not math.isfinite(std) or std < 0:
            raise ValueError('Group strength/std must be finite and nonnegative')
        if group_init not in ('normal', 'zero'):
            raise ValueError('group_init must be normal or zero')
        if not self.use_personal and self.group_strength == 0:
            raise ValueError('At least one user preference path must be active')
        groups = user_groups(self.n_users, num_groups, mode, int(config['group_seed']))
        self.register_buffer('user_group', groups)
        count = num_groups
        generator = torch.Generator().manual_seed(int(config['group_seed'])+1)
        weights = torch.randn(count, dim, generator=generator)*std
        if group_init == 'zero':
            weights.zero_()
        self.group_embedding = nn.Embedding.from_pretrained(weights, freeze=not bool(config['group_trainable']))
        self.user_embedding.weight.requires_grad_(self.use_personal)
        digest = hashlib.sha256(groups.numpy().tobytes()).hexdigest()[:16]
        logging.getLogger().info('GROUP mode=%s requested_groups=%s count=%s strength=%s init=%s std=%s trainable=%s map=%s; contiguous uses original numeric user IDs',
                                mode,num_groups,count,self.group_strength,group_init,std,
                                config['group_trainable'],digest)

    def make_projectors(self, dim):
        self.feat_embed_dim = 64
        if dim != self.feat_embed_dim:
            raise ValueError('GroupRec requires embedding_size=64 to match modality projections')
        return (nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim))

    def forward(self):
        x = torch.cat((self.user_embedding.weight, self.item_embedding.weight))
        layers = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.ui_adj, x)
            layers.append(x)
        user, item = torch.stack(layers).mean(0).split((self.n_users, self.n_items))
        # Add the group preference after propagating user/item ID embeddings.
        if not self.use_personal:
            user = torch.zeros_like(user)
        if self.group_strength:
            user = user + self.group_strength*self.group_embedding(self.user_group)
        return user,item

    def ssm(self, user, candidates):
        user = F.normalize(user, dim=-1)
        candidates = F.normalize(candidates, dim=-1)
        logits = (candidates * user[:, None]).sum(-1) / self.temperature
        return (torch.logsumexp(logits[:, 1:], dim=-1) - logits[:, 0]).mean()

    def bpr_objective(self, users, positives, pos_scores, neg_scores):
        return F.softplus(neg_scores - pos_scores[:, None]).mean()

    def calculate_loss(self, interaction):
        users, positives = interaction[0], interaction[1]
        user_table, item_table = self.forward()
        negatives = torch.randint(self.n_items, (users.numel(), self.num_negatives), device=users.device)
        candidates = torch.cat((positives[:, None], negatives), dim=1)
        user = user_table[users]
        pos_scores = (user * item_table[positives]).sum(-1)
        neg_scores = (user[:, None] * item_table[negatives]).sum(-1)
        loss = self.bpr_objective(users, positives, pos_scores, neg_scores)
        # Use the same sampled candidates for all three objectives. With linear
        # projectors, projecting unique sampled rows is exactly sufficient.
        unique, inverse = candidates.unique(return_inverse=True)
        if self.alpha > 0:
            image = self.image_trs(self.image_embedding(unique))[inverse]
            loss = loss + self.alpha * self.ssm(user, image)
        if self.alpha < 1:
            text = self.text_trs(self.text_embedding(unique))[inverse]
            loss = loss + (1-self.alpha) * self.ssm(user, text)
        return loss
