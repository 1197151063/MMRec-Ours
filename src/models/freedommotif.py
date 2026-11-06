"""Original FREEDOM plus item-conditioned user groups and weak bridge losses."""
import math
from logging import getLogger
import torch

from common.user_motifs import ItemGroupPeers, interest_distance, build_bridges
from models.freedom import FREEDOM


class FreedomMotif(FREEDOM):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.star_weight = float(config['star_weight'])
        self.bridge_weight = float(config['bridge_weight'])
        self.star_pop_power = float(config['star_pop_power'])
        self.star_mode = config['star_mode']
        if self.star_mode not in ('semantic', 'full'):
            raise ValueError('Unknown star_mode')
        if any(not math.isfinite(x) or x < 0 for x in (self.star_weight, self.bridge_weight, self.star_pop_power)):
            raise ValueError('Motif weights must be finite and nonnegative')
        self.item_groups = ItemGroupPeers(dataset.inter_matrix(form='csr')).to(self.device) if self.star_weight else None
        self.register_buffer('bridge_ids', None)
        self.register_buffer('bridge_weights', None)
        if self.bridge_weight:
            def progress(done, total):
                if done % 2048 == 0 or done == total:
                    getLogger().info('Building TRAIN four-hop bridge graph: %d/%d', done, total)
            ids, weights = build_bridges(dataset.inter_matrix(form='csr'),
                                         k=int(config['bridge_k']), block=int(config['bridge_block']),
                                         seed=int(config['motif_graph_seed']), progress=progress)
            self.bridge_ids = torch.as_tensor(ids, device=self.device)
            self.bridge_weights = torch.as_tensor(weights, device=self.device)
        self._motif_batches = 0

    def calculate_loss(self, interaction):
        users, pos, neg = interaction[:3]
        u, i = self.forward(self.masked_adj)
        self.build_item_graph = False
        bpr = self.bpr_loss(u[users], i[pos], i[neg])
        modal = bpr.new_zeros(())
        directions = []
        for source, projector in (('text_embedding', 'text_trs'), ('image_embedding', 'image_trs')):
            if hasattr(self, source):
                features = getattr(self, projector)(getattr(self, source).weight)
                modal = modal + self.bpr_loss(u[users], features[pos], features[neg])
                directions.append(features[pos])
        star = bpr.new_zeros(())
        if self.star_weight:
            peers, valid, degree = self.item_groups.sample(users, pos)
            losses = interest_distance(u[users], u[peers], directions if self.star_mode == 'semantic' else None)
            weights = valid.to(losses.dtype) * degree.to(losses.dtype).clamp_min(1).pow(-self.star_pop_power)
            star = (weights * losses).sum() / weights.sum().clamp_min(1e-12)
        bridge = bpr.new_zeros(())
        if self.bridge_weight:
            anchors = users.unique()
            weights = self.bridge_weights[anchors]
            active = weights.sum(1) > 0
            difference = interest_distance(u[anchors, None], u[self.bridge_ids[anchors]])
            bridge = (weights * difference).sum() / active.sum().clamp_min(1)
        total = bpr + self.reg_weight * modal + self.star_weight * star + self.bridge_weight * bridge
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite FREEDOM motif loss')
        self.last_motif_terms = {name: float(value.detach()) for name, value in
                                dict(bpr=bpr, modal=modal, star=star, bridge=bridge, total=total).items()}
        self._motif_batches += 1
        if self._motif_batches == 1 or self._motif_batches % 200 == 0:
            getLogger().info('FreedomMotif losses: %s; lambda_star=%g lambda_bridge=%g',
                             self.last_motif_terms, self.star_weight, self.bridge_weight)
        return total, bpr, modal
