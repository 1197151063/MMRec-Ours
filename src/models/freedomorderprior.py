"""FREEDOM + explicitly disclosed numeric user-ID neighborhood regularization.

No sinusoidal PE is used, but the model DOES use original numeric ID ordering.
"""
import math
from logging import getLogger

import numpy as np

from common.user_order_prior import UserOrderPrior
from models.freedom import FREEDOM


class FreedomOrderPrior(FREEDOM):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.order_weight = float(config['order_weight'])
        self.order_mode = config['order_mode']
        self.order_margin = float(config['order_margin'])
        self.order_temperature = float(config['order_temperature'])
        if not math.isfinite(self.order_weight) or self.order_weight < 0:
            raise ValueError('order_weight must be finite and nonnegative')
        if self.order_mode not in ('attract', 'local_rank'):
            raise ValueError('Invalid order_mode')
        active = np.asarray(dataset.inter_matrix(form='csr').sum(1)).ravel() > 0
        self.order_prior = UserOrderPrior(active, float(config['order_bandwidth']),
            int(config['order_radius']), int(config['order_samples']), int(config['order_far_gap'])).to(self.device)
        self._order_batches = 0
        getLogger().info('EXPLICIT_ORDER_PRIOR: source=ORIGINAL_NUMERIC_USER_ID_ORDER '
                         'NOT_TRAIN_GRAPH_INFERRED mode=%s weight=%g bandwidth=%s radius=%s '
                         'samples=%s far_gap=%s; no PE; original FREEDOM forward retained',
                         self.order_mode, self.order_weight, config['order_bandwidth'],
                         config['order_radius'], config['order_samples'], config['order_far_gap'])

    def calculate_loss(self, interaction):
        users, pos, neg = interaction[:3]
        u, i = self.forward(self.masked_adj)
        self.build_item_graph = False
        bpr = self.bpr_loss(u[users], i[pos], i[neg])
        modal = bpr.new_zeros(())
        for source, projector in (('text_embedding', 'text_trs'), ('image_embedding', 'image_trs')):
            if hasattr(self, source):
                features = getattr(self, projector)(getattr(self, source).weight)
                modal = modal + self.bpr_loss(u[users], features[pos], features[neg])
        base = bpr + self.reg_weight * modal
        order = self.order_prior(u, users, self.order_mode, self.order_margin,
                                 self.order_temperature) if self.order_weight else base.new_zeros(())
        total = base + self.order_weight * order
        self._order_batches += 1
        if self._order_batches == 1 or self._order_batches % 200 == 0:
            getLogger().info('FreedomOrderPrior loss: base=%g order=%g weighted_order=%g total=%g',
                             base.detach(), order.detach(), (self.order_weight * order).detach(), total.detach())
        if not total.isfinite():
            raise FloatingPointError('Nonfinite explicit order-prior loss')
        return total, bpr, modal
