"""Content item inputs + UI LightGCN + one SSM + item attraction + user local loss."""
import math
from logging import getLogger

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common.abstract_recommender import GeneralRecommender
from models.freedom import FREEDOM
from models.freedomalign import FreedomAlign
from models.freedomlocal import FreedomLocal


class FreedomLocalContent(GeneralRecommender):
    # Reuse only graph preprocessing and the existing local formula, not parent
    # constructors: no item ID table or item-message-passing graph is allocated.
    get_norm_adj_mat = FREEDOM.get_norm_adj_mat
    get_edge_info = FREEDOM.get_edge_info
    _normalize_adj_m = FREEDOM._normalize_adj_m
    pre_epoch_processing = FREEDOM.pre_epoch_processing
    local_loss = FreedomLocal.local_loss

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.embedding_dim = int(config['embedding_size'])
        self.feat_embed_dim = int(config['feat_embed_dim'])
        self.n_ui_layers = int(config['n_ui_layers'])
        self.dropout = float(config['dropout'])
        self.alpha = float(config['alpha'])  # Visual weight, not textual weight.
        self.temperature = float(config['temperature'])
        self.num_negatives = int(config['num_negatives'])
        self.ii_weight = float(config['ii_weight'])
        self.local_weight = float(config['local_weight'])
        self.local_rho = float(config['local_rho'])
        self.knn_k = int(config['knn_k'])
        if self.embedding_dim < 1 or self.embedding_dim != self.feat_embed_dim or self.n_ui_layers < 0:
            raise ValueError('Positive matching user/content dimensions and nonnegative UI layers required')
        if not 0 <= self.alpha <= 1 or not 0 <= self.dropout < 1 or not 0 <= self.local_rho <= 1:
            raise ValueError('Invalid alpha, dropout or local_rho')
        if not math.isfinite(self.temperature) or self.temperature <= 0 or self.num_negatives < 1:
            raise ValueError('Positive finite temperature and positive negative count required')
        if any(not math.isfinite(w) or w < 0 for w in (self.ii_weight, self.local_weight)) or self.knn_k < 1:
            raise ValueError('Nonnegative finite loss weights and positive KNN size required')
        if self.v_feat is None or self.t_feat is None:
            raise ValueError('FreedomLocalContent requires both modalities')
        for features in (self.v_feat, self.t_feat):
            if features.shape[0] != self.n_items or not torch.isfinite(features).all():
                raise ValueError('Invalid content features')

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        self._init_item_input(config)
        freeze = bool(config['freeze_features'])
        self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=freeze)
        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=freeze)
        self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        self.n_nodes = self.n_users + self.n_items
        r = dataset.inter_matrix(form='csr').astype(np.float32).tocsr()
        r.sum_duplicates(); r.eliminate_zeros(); r.data[:] = 1
        self.interaction_matrix = r.tocoo()
        self.register_buffer('norm_adj', self.get_norm_adj_mat().coalesce())
        edges, values = self.get_edge_info()
        self.register_buffer('edge_indices', edges)
        self.register_buffer('edge_values', values)
        self.register_buffer('masked_adj', None)
        for name, features in (('image', self.v_feat), ('text', self.t_feat)):
            ids, weights = FreedomAlign.content_knn(
                features.detach(), self.knn_k, int(config['knn_block']), bool(config['knn_keep_self']))
            # Retain raw cosine weights, as in the requested formula.
            self.register_buffer(name + '_knn_idx', ids)
            self.register_buffer(name + '_knn_weight', weights)

        active = np.asarray(r.sum(1)).ravel() > 0
        self.register_buffer('local_pair_mask', torch.as_tensor(active[:-1] & active[1:]))
        moment = float(self.user_embedding.weight.detach().square().mean())
        self.register_buffer('local_scale', torch.tensor(max((1 + self.local_rho ** 2) * moment, 1e-12)))
        self._eval_embeddings = None
        self._batches = 0
        self._log_architecture()

    def _init_item_input(self, config):
        # The content-only model has no independent item-ID parameters.
        pass

    def _log_architecture(self):
        freeze = not self.image_embedding.weight.requires_grad
        getLogger().info('FreedomLocalContent: NO item-ID table or item-item propagation; '
                         'item=alpha*image+(1-alpha)*text alpha=%g; UI layers=%d; '
                         'one negative-denominator SSM tau=%g uniform_negatives=%d (unfiltered); '
                         'ii=sum raw-dot softplus on pre-UI fused content, weight=%g; '
                         'features_trainable=%s; EXPLICIT local source=ORIGINAL_NUMERIC_USER_ID_ORDER '
                         'rho=%g weight=%g target=raw_user_ID_table',
                         self.alpha, self.n_ui_layers, self.temperature, self.num_negatives,
                         self.ii_weight, not freeze, self.local_rho, self.local_weight)

    def train(self, mode=True):
        self._eval_embeddings = None
        return super().train(mode)

    def item_representations(self):
        return (self.alpha * self.image_trs(self.image_embedding.weight) +
                (1 - self.alpha) * self.text_trs(self.text_embedding.weight))

    def forward(self, adj=None, content=None):
        adj = self.norm_adj if adj is None else adj
        content = self.item_representations() if content is None else content
        x = torch.cat((self.user_embedding.weight, content))
        result = x
        for _ in range(self.n_ui_layers):
            x = torch.sparse.mm(adj, x)
            result = result + x
        return (result / (self.n_ui_layers + 1)).split((self.n_users, self.n_items))

    @staticmethod
    def ssm_loss(users, positives, negatives, temperature):
        u, p, n = (F.normalize(x, dim=-1) for x in (users, positives, negatives))
        positive = (u * p).sum(-1) / temperature
        negative = (u[:, None] * n).sum(-1) / temperature
        # User's SSM: positive is NOT part of the denominator. Can be negative.
        return (torch.logsumexp(negative, dim=1) - positive).mean()

    @staticmethod
    def item_alignment(items, neighbor_ids, neighbor_weights, embeddings):
        # Each anchor uses its own K weights: [B,K], never [B,B,K].
        scores = (embeddings[items, None] * embeddings[neighbor_ids[items]]).sum(-1)
        return (neighbor_weights[items] * F.softplus(-scores)).sum()

    def calculate_loss(self, interaction):
        self._eval_embeddings = None
        users, pos = interaction[:2]
        content = self.item_representations()
        adj = self.masked_adj if self.masked_adj is not None else self.norm_adj
        u, i = self.forward(adj, content)
        neg = torch.randint(self.n_items, (users.numel(), self.num_negatives), device=users.device)
        ssm = self.ssm_loss(u[users], i[pos], i[neg], self.temperature)
        ii = ssm.new_zeros(())
        if self.ii_weight:
            visual = self.item_alignment(pos, self.image_knn_idx, self.image_knn_weight, content)
            textual = self.item_alignment(pos, self.text_knn_idx, self.text_knn_weight, content)
            ii = self.alpha * visual + (1 - self.alpha) * textual
        local = self.local_loss() if self.local_weight else ssm.new_zeros(())
        total = ssm + self.ii_weight * ii + self.local_weight * local
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite FreedomLocalContent loss')
        self.last_loss_terms = {key: value.detach().item() for key, value in
                                dict(ssm=ssm, ii=ii, local=local, total=total).items()}
        self._batches += 1
        if self._batches == 1 or self._batches % 200 == 0:
            getLogger().info('FreedomLocalContent loss: %s; weighted_ii=%g weighted_local=%g '
                             'user_rms=%g content_rms=%g', self.last_loss_terms,
                             self.ii_weight * ii.detach(), self.local_weight * local.detach(),
                             self.user_embedding.weight.detach().square().mean().sqrt(),
                             content.detach().square().mean().sqrt())
        return total

    @torch.no_grad()
    def full_sort_predict(self, interaction):
        users = interaction[0] if isinstance(interaction, (list, tuple)) else interaction
        if self.training:
            u, i = self.forward()
        else:
            if self._eval_embeddings is None:
                self._eval_embeddings = self.forward()
            u, i = self._eval_embeddings
        # Retain FREEDOM's raw dot-product ranking, with the full TRAIN graph.
        return u[users] @ i.T
