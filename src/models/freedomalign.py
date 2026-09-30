"""User-provided FREEDOM variant: UI LightGCN + modality SSM + item attraction.

Corrects the pairwise weight indexing; default keeps raw cosine weights, self
neighbors, repeated positive anchors, sum reduction, and Xavier gain=2.
"""
import math
from logging import getLogger
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from common.abstract_recommender import GeneralRecommender


class FreedomAlign(GeneralRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.embedding_dim = int(config['embedding_size'])
        self.feat_embed_dim = int(config['feat_embed_dim'])
        self.n_ui_layers = int(config['n_ui_layers'])
        self.knn_k = int(config['knn_k'])
        self.temperature = float(config['temperature'])
        self.modal_weight = float(config['modal_weight'])
        self.ii_weight = float(config['ii_weight'])
        self.image_weight = float(config['mm_image_weight'])
        self.reduction = config['alignment_reduction']
        self.keep_self = bool(config['knn_keep_self'])
        self.clip_weights = bool(config['knn_positive_only'])
        if self.embedding_dim != self.feat_embed_dim:
            raise ValueError('User and projected modality dimensions must match')
        if self.n_ui_layers < 0 or self.knn_k < 1 or self.reduction not in ('sum', 'mean'):
            raise ValueError('Invalid layers, knn_k or alignment reduction')
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('temperature must be positive and finite')
        if not 0 <= self.image_weight <= 1 or any(not math.isfinite(w) or w < 0 for w in (self.modal_weight, self.ii_weight)):
            raise ValueError('Invalid loss weights')
        if self.v_feat is None or self.t_feat is None:
            raise ValueError('FreedomAlign requires both modalities')
        for features in (self.v_feat, self.t_feat):
            if features.shape[0] != self.n_items or not torch.isfinite(features).all():
                raise ValueError('Invalid content features')
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=2.)
        nn.init.xavier_uniform_(self.item_id_embedding.weight, gain=2.)
        self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
        r = dataset.inter_matrix(form='csr').astype(np.float32).tocsr()
        r.sum_duplicates(); r.eliminate_zeros(); r.data[:] = 1
        a = sp.bmat([[None, r], [r.T, None]], format='csr')
        scale = np.power(np.asarray(a.sum(1)).ravel() + 1e-7, -.5)
        a = (sp.diags(scale) @ a @ sp.diags(scale)).tocoo()
        self.register_buffer('norm_adj', torch.sparse_coo_tensor(
            torch.from_numpy(np.stack((a.row, a.col))).long(), torch.from_numpy(a.data),
            (self.n_users + self.n_items,)*2).coalesce())
        # Build both even for ablations: parameter/RNG initialization stays matched.
        for name, features in (('image', self.v_feat), ('text', self.t_feat)):
            ids, weights = self.content_knn(features.detach(), self.knn_k,
                                           int(config['knn_block']), self.keep_self)
            if self.clip_weights:
                weights = weights.clamp_min(0)
            self.register_buffer(name + '_knn_idx', ids)
            self.register_buffer(name + '_knn_weight', weights)
        self._eval_embeddings = None
        self._batches = 0

    @staticmethod
    @torch.no_grad()
    def content_knn(features, k, block=256, keep_self=True):
        if block < 1 or k > len(features) - int(not keep_self):
            raise ValueError('Invalid KNN block size or k exceeds candidate count')
        x = F.normalize(features, dim=-1)
        indices, weights = [], []
        for start in range(0, len(x), block):
            similarity = x[start:start+block] @ x.T
            if not keep_self:
                local = torch.arange(len(similarity), device=x.device)
                similarity[local, local + start] = -torch.inf
            values, ids = similarity.topk(k, dim=-1, sorted=False)
            indices.append(ids); weights.append(values)
        return torch.cat(indices), torch.cat(weights)

    def train(self, mode=True):
        self._eval_embeddings = None
        return super().train(mode)

    def forward(self, adj=None):
        adj = self.norm_adj if adj is None else adj
        x = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight))
        representations = [x]
        for _ in range(self.n_ui_layers):
            x = torch.sparse.mm(adj, x)
            representations.append(x)
        result = torch.stack(representations, 1).mean(1)
        return result.split((self.n_users, self.n_items))

    @staticmethod
    def modality_ssm(users, positives, negatives, temperature):
        u, p, n = (F.normalize(x, dim=-1) for x in (users, positives, negatives))
        positive = (u*p).sum(-1)/temperature
        # Batch negative pool; duplicates and false negatives are retained as supplied.
        return (torch.logsumexp((u @ n.T)/temperature, dim=1) - positive).mean()

    def item_alignment(self, items, neighbor_ids, neighbor_weights):
        e = self.item_id_embedding.weight
        scores = (e[items, None] * e[neighbor_ids[items]]).sum(-1)
        weighted = neighbor_weights[items] * F.softplus(-scores)
        return weighted.sum() if self.reduction == 'sum' else weighted.mean()

    def calculate_loss(self, interaction):
        self._eval_embeddings = None
        users, pos, neg = interaction[:3]
        u, i = self.forward()
        bpr = F.softplus((u[users] * (i[neg] - i[pos])).sum(-1)).mean()
        modal = bpr.new_zeros(())
        alignment = bpr.new_zeros(())
        if self.modal_weight:
            # Linear projection is rowwise: unique rows are algebraically identical to full-table projection.
            selected, inverse = torch.unique(torch.cat((pos, neg)), return_inverse=True)
            for embedding, projector in ((self.image_embedding, self.image_trs), (self.text_embedding, self.text_trs)):
                features = projector(embedding(selected))[inverse]
                modal = modal + self.modality_ssm(u[users], features[:len(pos)], features[len(pos):], self.temperature)
        if self.ii_weight:
            visual = self.item_alignment(pos, self.image_knn_idx, self.image_knn_weight)
            textual = self.item_alignment(pos, self.text_knn_idx, self.text_knn_weight)
            alignment = self.image_weight * visual + (1-self.image_weight) * textual
        total = bpr + self.modal_weight*modal + self.ii_weight*alignment
        self.last_loss_terms = dict(bpr=bpr.detach().item(), modality=modal.detach().item(),
                                   alignment=alignment.detach().item(), total=total.detach().item())
        self._batches += 1
        if self._batches == 1 or self._batches % 200 == 0:
            getLogger().info('FreedomAlign loss: %s; weighted modality=%.6f alignment=%.6f',
                             self.last_loss_terms, self.modal_weight*modal.detach().item(), self.ii_weight*alignment.detach().item())
        return total

    def full_sort_predict(self, interaction):
        users = interaction[0] if isinstance(interaction, (tuple, list)) else interaction
        if self.training:
            u, i = self.forward()
        else:
            if self._eval_embeddings is None:
                with torch.no_grad():
                    self._eval_embeddings = self.forward()
            u, i = self._eval_embeddings
        return u[users] @ i.T
