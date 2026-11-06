"""FREEDOM ID/UI BPR + two modality SSMs + raw item-ID attraction + user local loss."""
import math
from logging import getLogger

import torch
from torch import nn
from torch.nn import functional as F

from models.freedomlocalcontent import FreedomLocalContent


class FreedomLocalSSM(FreedomLocalContent):
    def _init_item_input(self, config):
        self.modal_weight = float(config['modal_weight'])
        if not math.isfinite(self.modal_weight) or self.modal_weight < 0:
            raise ValueError('modal_weight must be finite and nonnegative')
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

    def _log_architecture(self):
        getLogger().info('FreedomLocalSSM: item input=ONLY item-ID embedding; UI layers=%d; '
                         'ID recommendation=BPR; modality terms=SSM(user_UI,image_projection) '
                         '+SSM(user_UI,text_projection), weight=%g; tau=%g uniform_negatives=%d '
                         '(unfiltered, same samples for both modalities, negative-only denominator); '
                         'NO item-item message passing; ii=raw item-ID attraction, weight=%g '
                         'visual_weight=%g; features_trainable=%s; '
                         'EXPLICIT local source=ORIGINAL_NUMERIC_USER_ID_ORDER '
                         'rho=%g weight=%g target=raw_user_ID_table',
                         self.n_ui_layers, self.modal_weight, self.temperature, self.num_negatives,
                         self.ii_weight, self.alpha, self.image_embedding.weight.requires_grad,
                         self.local_rho, self.local_weight)

    def item_representations(self):
        # Multimodal vectors are auxiliary SSM targets, not added to ID inputs.
        return self.item_id_embedding.weight

    def calculate_loss(self, interaction):
        self._eval_embeddings = None
        users, pos, neg = interaction[:3]
        adj = self.masked_adj if self.masked_adj is not None else self.norm_adj
        u, i = self.forward(adj)
        bpr = F.softplus((u[users] * (i[neg] - i[pos])).sum(-1)).mean()
        visual_ssm = textual_ssm = bpr.new_zeros(())
        if self.modal_weight:
            sampled = torch.randint(self.n_items, (users.numel(), self.num_negatives), device=users.device)
            # A linear projector is rowwise: project each sampled row once,
            # rather than projecting the full catalog for every modality SSM.
            candidates = torch.cat((pos[:, None], sampled), dim=1)
            selected, inverse = torch.unique(candidates, return_inverse=True)
            visual = self.image_trs(self.image_embedding(selected))[inverse]
            textual = self.text_trs(self.text_embedding(selected))[inverse]
            visual_ssm = self.ssm_loss(u[users], visual[:, 0], visual[:, 1:], self.temperature)
            textual_ssm = self.ssm_loss(u[users], textual[:, 0], textual[:, 1:], self.temperature)
        modal = visual_ssm + textual_ssm
        ii = bpr.new_zeros(())
        if self.ii_weight:
            raw_items = self.item_id_embedding.weight
            visual_ii = self.item_alignment(pos, self.image_knn_idx, self.image_knn_weight, raw_items)
            textual_ii = self.item_alignment(pos, self.text_knn_idx, self.text_knn_weight, raw_items)
            ii = self.alpha * visual_ii + (1 - self.alpha) * textual_ii
        local = self.local_loss() if self.local_weight else bpr.new_zeros(())
        total = bpr + self.modal_weight * modal + self.ii_weight * ii + self.local_weight * local
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite FreedomLocalSSM loss')
        self.last_loss_terms = {key: value.detach().item() for key, value in
                                dict(bpr=bpr, visual_ssm=visual_ssm, textual_ssm=textual_ssm,
                                     ii=ii, local=local, total=total).items()}
        self._batches += 1
        if self._batches == 1 or self._batches % 200 == 0:
            getLogger().info('FreedomLocalSSM loss: %s; weighted_modal=%g weighted_ii=%g '
                             'weighted_local=%g user_rms=%g item_id_rms=%g', self.last_loss_terms,
                             self.modal_weight * modal.detach(), self.ii_weight * ii.detach(),
                             self.local_weight * local.detach(),
                             self.user_embedding.weight.detach().square().mean().sqrt(),
                             self.item_id_embedding.weight.detach().square().mean().sqrt())
        return total
