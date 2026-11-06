"""FREEDOM with TRAIN-graph spectral groups and free personal user residuals."""
import hashlib
import math
import time
from logging import getLogger

import numpy as np
import torch
from torch import nn

from common.spectral_user_groups import build_spectral_user_groups
from models.freedom import FREEDOM


class FreedomSpectralGroup(FREEDOM):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.num_groups = int(config['num_groups'])
        self.group_strength = float(config['group_strength'])
        self.personal_reg_weight = float(config['personal_reg_weight'])
        if any(not math.isfinite(x) or x < 0 for x in (self.group_strength, self.personal_reg_weight)):
            raise ValueError('Group strength and personal regularization must be finite and nonnegative')
        r = dataset.inter_matrix(form='csr').tocsr()
        self.register_buffer('group_active_users', torch.as_tensor(np.asarray(r.sum(1)).ravel() > 0, device=self.device))
        self.register_buffer('spectral_group_ids', None)
        self.group_embedding = None
        if self.group_strength:
            started = time.monotonic()
            getLogger().info('Building fixed TRAIN spectral user groups: groups=%d rank=%s',
                             self.num_groups, config['spectral_rank'])
            labels = build_spectral_user_groups(r, self.num_groups, int(config['spectral_rank']),
                                               int(config['group_seed']))
            self.spectral_group_ids = torch.as_tensor(labels, device=self.device)
            self.group_embedding = nn.Embedding(self.num_groups, self.embedding_dim).to(self.device)
            nn.init.xavier_uniform_(self.group_embedding.weight)
            sizes = np.bincount(labels[labels >= 0], minlength=self.num_groups)
            getLogger().info('Spectral groups: source=TRAIN_GRAPH_ONLY used_groups=%d sizes_min_median_max=%s '
                             'seconds=%.2f labels_sha256=%s beta=%g personal_reg=%g; injection=before_UI_propagation',
                             int(np.count_nonzero(sizes)), [int(sizes.min()), float(np.median(sizes)), int(sizes.max())],
                             time.monotonic() - started, hashlib.sha256(labels.tobytes()).hexdigest(),
                             self.group_strength, self.personal_reg_weight)
        self._group_batches = 0

    def user_input(self):
        personal = self.user_embedding.weight
        if self.group_embedding is None:
            return personal
        shared = self.group_embedding(self.spectral_group_ids.clamp_min(0))
        return personal + self.group_strength * shared * self.group_active_users[:, None]

    def forward(self, adj):
        h = self.item_id_embedding.weight
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        x = torch.cat((self.user_input(), self.item_id_embedding.weight), dim=0)
        layers = [x]
        for _ in range(self.n_ui_layers):
            x = torch.sparse.mm(adj, x)
            layers.append(x)
        users, items = torch.stack(layers, dim=1).mean(dim=1).split((self.n_users, self.n_items))
        return users, items + h

    def calculate_loss(self, interaction):
        base, bpr, modality = super().calculate_loss(interaction)
        personal_l2 = self.user_embedding.weight[self.group_active_users].square().sum(-1).mean()
        total = base + self.personal_reg_weight * personal_l2
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite spectral-group loss')
        self._group_batches += 1
        if self._group_batches == 1 or self._group_batches % 200 == 0:
            getLogger().info('FreedomSpectralGroup loss: base=%g personal_l2=%g weighted_personal_l2=%g total=%g',
                             base.detach(), personal_l2.detach(),
                             (self.personal_reg_weight * personal_l2).detach(), total.detach())
        return total, bpr, modality
