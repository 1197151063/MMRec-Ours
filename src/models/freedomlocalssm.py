"""FREEDOM ID/UI BPR + two modality SSMs + raw item-ID attraction + user local loss."""
import math
from logging import getLogger

import torch
from torch import nn
from torch.nn import functional as F

from models.freedomlocalcontent import FreedomLocalContent


class FreedomLocalSSM(FreedomLocalContent):
    def _init_ssm_config(self, config):
        self.tau = float(config['tau'])
        if not math.isfinite(self.tau) or self.tau <= 0:
            raise ValueError('tau must be finite and positive')

    def _init_item_input(self, config):
        self.ssm_weight = float(config['lambda'])
        self.ii_image_weight = float(config['ii_image_weight'])
        if not math.isfinite(self.ssm_weight) or self.ssm_weight < 0:
            raise ValueError('lambda must be finite and nonnegative')
        if not 0 <= self.ii_image_weight <= 1:
            raise ValueError('ii_image_weight must be in [0, 1]')
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

    def _log_architecture(self):
        getLogger().info('FreedomLocalSSM: item input=ONLY item-ID embedding; UI layers=%d; '
                         'ID recommendation=BPR; modality=lambda*(alpha*text_SSM+(1-alpha)*image_SSM) '
                         'lambda=%g alpha=%g tau=%g; '
                         'SSM negatives=off-diagonal batch positives only (no sampling); '
                         'NO item-item message passing; ii=raw item-ID attraction, weight=%g '
                         'visual_weight=%g; features_trainable=%s; '
                         'EXPLICIT local source=ORIGINAL_NUMERIC_USER_ID_ORDER '
                         'rho=%g weight=%g target=raw_user_ID_table',
                         self.n_ui_layers, self.ssm_weight, self.alpha, self.tau,
                         self.ii_weight, self.ii_image_weight, self.image_embedding.weight.requires_grad,
                         self.local_rho, self.local_weight)

    def item_representations(self):
        # Multimodal vectors are auxiliary SSM targets, not added to ID inputs.
        return self.item_id_embedding.weight

    def ssm_loss(self, u, i):
        if u.ndim != 2 or i.ndim != 2 or u.shape != i.shape:
            raise ValueError('SSM requires matching [batch, dimension] user/item tensors')
        # A singleton batch has no off-diagonal negatives. Skip only its SSM,
        # keeping a differentiable zero; BPR/ii/local still train normally.
        if u.size(0) < 2:
            return (u.sum() + i.sum()) * 0
        scores = F.normalize(u, dim=-1) @ F.normalize(i, dim=-1).t()
        scores = scores / self.tau
        positive = scores.diagonal()
        diagonal = torch.eye(u.size(0), dtype=torch.bool, device=u.device)
        negative = scores.masked_fill(diagonal, float('-inf'))
        instance_loss = torch.logsumexp(negative, dim=1) - positive
        return instance_loss.mean()

    def calculate_loss(self, interaction):
        self._eval_embeddings = None
        users, pos, neg = interaction[:3]
        adj = self.masked_adj if self.masked_adj is not None else self.norm_adj
        u, i = self.forward(adj)
        bpr = F.softplus((u[users] * (i[neg] - i[pos])).sum(-1)).mean()
        visual_ssm = textual_ssm = bpr.new_zeros(())
        if self.ssm_weight:
            # Preserve the full batch, including repeated users/items: only
            # diagonal positions are positives, exactly as in the supplied SSM.
            visual = self.image_trs(self.image_embedding(pos))
            textual = self.text_trs(self.text_embedding(pos))
            visual_ssm = self.ssm_loss(u[users], visual)
            textual_ssm = self.ssm_loss(u[users], textual)
        modal = self.alpha * textual_ssm + (1 - self.alpha) * visual_ssm
        ii = bpr.new_zeros(())
        if self.ii_weight:
            raw_items = self.item_id_embedding.weight
            visual_ii = self.item_alignment(pos, self.image_knn_idx, self.image_knn_weight, raw_items)
            textual_ii = self.item_alignment(pos, self.text_knn_idx, self.text_knn_weight, raw_items)
            ii = self.ii_image_weight * visual_ii + (1 - self.ii_image_weight) * textual_ii
        local = self.local_loss() if self.local_weight else bpr.new_zeros(())
        total = bpr + self.ssm_weight * modal + self.ii_weight * ii + self.local_weight * local
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite FreedomLocalSSM loss')
        self.last_loss_terms = {key: value.detach().item() for key, value in
                                dict(bpr=bpr, visual_ssm=visual_ssm, textual_ssm=textual_ssm,
                                     ii=ii, local=local, total=total).items()}
        self._batches += 1
        if self._batches == 1 or self._batches % 200 == 0:
            getLogger().info('FreedomLocalSSM loss: %s; weighted_modal=%g weighted_ii=%g '
                             'weighted_local=%g user_rms=%g item_id_rms=%g', self.last_loss_terms,
                             self.ssm_weight * modal.detach(), self.ii_weight * ii.detach(),
                             self.local_weight * local.detach(),
                             self.user_embedding.weight.detach().square().mean().sqrt(),
                             self.item_id_embedding.weight.detach().square().mean().sqrt())
        return total
