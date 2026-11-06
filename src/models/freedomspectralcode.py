"""FREEDOM with shared, continuous TRAIN-graph coordinates on users and items."""
import hashlib
import math
import time
from logging import getLogger

import numpy as np
import torch
from torch import nn

from common.spectral_coordinates import build_spectral_coordinates
from models.freedom import FREEDOM


class FreedomSpectralCode(FREEDOM):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.graph_strength = float(config['graph_strength'])
        self.graph_sides = config['graph_sides']
        if self.graph_sides not in ('user', 'both'):
            raise ValueError('graph_sides must be user or both')
        if not math.isfinite(self.graph_strength) or self.graph_strength < 0:
            raise ValueError('graph_strength must be finite and nonnegative')
        self.register_buffer('user_codes', None)
        self.register_buffer('item_codes', None)
        self.register_buffer('code_scale', torch.tensor(1., device=self.device))
        self.code_projector = None
        if self.graph_strength:
            # quick_start calls model.to(device) only AFTER this constructor.
            # FREEDOM creates ID tables on CPU, while codes/masks already use
            # config.device. Align them before the initialization calibration.
            self.user_embedding.to(self.device)
            self.item_id_embedding.to(self.device)
            started = time.monotonic()
            r = dataset.inter_matrix(form='csr').tocsr()
            user_codes, item_codes, singular = build_spectral_coordinates(
                r, rank=int(config['spectral_rank']), seed=int(config['spectral_seed']))
            self.user_codes = torch.as_tensor(user_codes, device=self.device)
            self.item_codes = torch.as_tensor(item_codes, device=self.device)
            self.code_projector = nn.Linear(user_codes.shape[1], self.embedding_dim, bias=False).to(self.device)
            nn.init.xavier_uniform_(self.code_projector.weight)
            # Match the INITIAL joint graph-branch RMS to the ID-branch RMS.
            # This scalar stays fixed; it is not an ongoing norm constraint.
            with torch.no_grad():
                active_u = torch.as_tensor(np.asarray(r.sum(1)).ravel() > 0, device=self.device)
                active_i = torch.as_tensor(np.asarray(r.sum(0)).ravel() > 0, device=self.device)
                ids = torch.cat((self.user_embedding.weight[active_u], self.item_id_embedding.weight[active_i]))
                projected = self.code_projector(torch.cat((self.user_codes[active_u], self.item_codes[active_i])))
                id_rms = ids.square().mean().sqrt()
                graph_rms = projected.square().mean().sqrt()
                self.code_scale.copy_(id_rms / graph_rms.clamp_min(1e-12))
            digest = hashlib.sha256(user_codes.tobytes() + item_codes.tobytes()).hexdigest()
            getLogger().info('SpectralCode: source=TRAIN_GRAPH_ONLY rank=%d sides=%s beta=%g '
                             'UI_layers=%d MM_layers=%d shared_projection_params=%d code_scale=%g '
                             'initial_joint_id_rms=%g singular_max_min=%s codes_sha256=%s seconds=%.2f; '
                             'no timestamps, groups or numeric-ID distances',
                             user_codes.shape[1], self.graph_sides, self.graph_strength,
                             self.n_ui_layers, self.n_layers, self.code_projector.weight.numel(),
                             self.code_scale.item(), id_rms.item(), [float(singular.max()), float(singular.min())],
                             digest, time.monotonic() - started)

    def structural_inputs(self):
        users, items = self.user_embedding.weight, self.item_id_embedding.weight
        if self.code_projector is not None:
            factor = self.graph_strength * self.code_scale
            users = users + factor * self.code_projector(self.user_codes)
            if self.graph_sides == 'both':
                items = items + factor * self.code_projector(self.item_codes)
        return users, items

    def pre_epoch_processing(self):
        if self.n_ui_layers == 0:
            self.masked_adj = None
            return
        super().pre_epoch_processing()

    def forward(self, adj):
        # Keep the original item-item branch on raw item IDs in every variant.
        h = self.item_id_embedding.weight
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        users, items = self.structural_inputs()
        if self.n_ui_layers == 0:
            return users, items + h
        x = torch.cat((users, items))
        layers = [x]
        for _ in range(self.n_ui_layers):
            x = torch.sparse.mm(adj, x)
            layers.append(x)
        users, items = torch.stack(layers, dim=1).mean(dim=1).split((self.n_users, self.n_items))
        return users, items + h
