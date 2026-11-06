"""Original FREEDOM plus an explicitly numeric-user-order local training penalty."""
import math
from logging import getLogger

import numpy as np
import torch

from models.freedom import FREEDOM


class FreedomLocal(FREEDOM):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.local_weight = float(config['local_weight'])
        self.local_rho = float(config['local_rho'])
        if not math.isfinite(self.local_weight) or self.local_weight < 0:
            raise ValueError('local_weight must be finite and nonnegative')
        if not math.isfinite(self.local_rho) or not 0 <= self.local_rho <= 1:
            raise ValueError('local_rho must be in [0, 1]')
        active = np.asarray(dataset.inter_matrix(form='csr').sum(1)).ravel() > 0
        self.register_buffer('local_pair_mask', torch.as_tensor(active[:-1] & active[1:], device=self.device))
        # ID tables may still be on CPU during __init__. Read a scalar there;
        # no GPU mask indexes CPU weights before quick_start's outer .to().
        initial_moment = float(self.user_embedding.weight.detach().square().mean())
        scale = max((1 + self.local_rho ** 2) * initial_moment, 1e-12)
        self.register_buffer('local_scale', torch.tensor(scale, device=self.device))
        self._local_batches = 0
        getLogger().info('EXPLICIT_LOCAL_PRIOR: source=ORIGINAL_NUMERIC_USER_ID_ORDER '
                         'rho=%g weight=%g active_adjacent_pairs=%d initial_fixed_scale=%g; '
                         'target=raw_user_ID_table; unchanged FREEDOM forward; no PE, extra branch, '
                         'variance/covariance loss or new learnable parameters',
                         self.local_rho, self.local_weight, int(self.local_pair_mask.sum()), scale)

    def local_loss(self):
        p = self.user_embedding.weight
        delta = p[1:] - self.local_rho * p[:-1]
        per_pair = delta.square().mean(-1)
        weights = self.local_pair_mask.to(per_pair.dtype)
        return (per_pair * weights).sum() / weights.sum().clamp_min(1) / self.local_scale

    def calculate_loss(self, interaction):
        base, bpr, modal = super().calculate_loss(interaction)
        local = self.local_loss() if self.local_weight else base.new_zeros(())
        total = base + self.local_weight * local
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite FREEDOM local loss')
        self._local_batches += 1
        if self._local_batches == 1 or self._local_batches % 200 == 0:
            getLogger().info('FreedomLocal loss: base=%g local=%g weighted_local=%g total=%g user_rms=%g',
                             base.detach(), local.detach(), (self.local_weight * local).detach(),
                             total.detach(), self.user_embedding.weight.detach().square().mean().sqrt())
        return total, bpr, modal
