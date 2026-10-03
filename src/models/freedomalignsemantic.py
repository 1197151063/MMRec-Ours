"""Factorial test: frozen raw modalities and a direct semantic ranking residual.

No UU graph. The unchanged FreedomAlign forward supplies CF representations;
the same CF-dot + weighted modality-cosine score is used by BPR and inference.
"""
import math
from logging import getLogger

import torch
from torch.nn import functional as F

from models.freedomalign import FreedomAlign


class FreedomAlignSemantic(FreedomAlign):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.semantic_beta = float(config['semantic_beta'])
        self.semantic_image_weight = float(config['semantic_image_weight'])
        self.freeze_modal_features = config['freeze_modal_features']
        if not isinstance(self.freeze_modal_features, bool):
            raise ValueError('freeze_modal_features must be a YAML boolean')
        if not math.isfinite(self.semantic_beta) or self.semantic_beta < 0:
            raise ValueError('semantic_beta must be finite and nonnegative')
        if not math.isfinite(self.semantic_image_weight) or not 0 <= self.semantic_image_weight <= 1:
            raise ValueError('semantic_image_weight must be in [0, 1]')
        for embedding in (self.image_embedding, self.text_embedding):
            embedding.weight.requires_grad_(not self.freeze_modal_features)
        self._eval_semantic = None
        self._semantic_epochs = 0
        generator = torch.Generator(device='cpu').manual_seed(2026)
        probe = torch.randperm(self.n_items, generator=generator)[:min(256, self.n_items)]
        probe = probe.to(self.image_embedding.weight.device)
        self.register_buffer('_feature_probe', probe, persistent=False)
        for name, embedding in (('visual', self.image_embedding), ('text', self.text_embedding)):
            self.register_buffer('_initial_' + name, embedding.weight.detach()[probe].clone(), persistent=False)
        getLogger().info('FreedomAlignSemantic freeze_modal_features=%s beta=%g image_weight=%g '
                         'trainable_parameters=%d; no UU module', self.freeze_modal_features,
                         self.semantic_beta, self.semantic_image_weight,
                         sum(p.numel() for p in self.parameters() if p.requires_grad))

    @torch.no_grad()
    def feature_drift(self):
        values = {}
        for name, embedding in (('visual', self.image_embedding), ('text', self.text_embedding)):
            initial = getattr(self, '_initial_' + name)
            current = embedding.weight[self._feature_probe]
            values[name + '_initial_cosine'] = F.cosine_similarity(initial, current, dim=-1).mean().item()
            values[name + '_relative_l2'] = ((current - initial).norm() / initial.norm().clamp_min(1e-12)).item()
        return values

    def post_epoch_processing(self):
        self._semantic_epochs += 1
        if self._semantic_epochs == 1 or self._semantic_epochs % 10 == 0:
            getLogger().info('FreedomAlignSemantic feature drift (fixed <=256 item probe), epoch=%d: %s',
                             self._semantic_epochs, self.feature_drift())

    def train(self, mode=True):
        self._eval_semantic = None
        return super().train(mode)

    def semantic_items(self, visual, textual):
        # Do not normalize again after fusion: this is a weighted sum of cosines.
        alpha = self.semantic_image_weight
        return alpha * F.normalize(visual, dim=-1) + (1 - alpha) * F.normalize(textual, dim=-1)

    def calculate_loss(self, interaction):
        self._eval_semantic = None
        if self.semantic_beta == 0:
            # Exact reference arithmetic and unchanged losses for both beta=0 arms.
            return super().calculate_loss(interaction)
        self._eval_embeddings = None
        users, pos, neg = interaction[:3]
        u, i = self.forward()
        selected, inverse = torch.unique(torch.cat((pos, neg)), return_inverse=True)
        visual = self.image_trs(self.image_embedding(selected))[inverse]
        textual = self.text_trs(self.text_embedding(selected))[inverse]
        semantics = self.semantic_items(visual, textual)
        semantic_user = F.normalize(u[users], dim=-1)
        pos_semantic = (semantic_user * semantics[:len(pos)]).sum(-1)
        neg_semantic = (semantic_user * semantics[len(pos):]).sum(-1)
        pos_cf = (u[users] * i[pos]).sum(-1)
        neg_cf = (u[users] * i[neg]).sum(-1)
        bpr = F.softplus(neg_cf - pos_cf + self.semantic_beta * (neg_semantic - pos_semantic)).mean()
        modal = bpr.new_zeros(())
        if self.modal_weight:
            for features in (visual, textual):
                modal = modal + self.modality_ssm(u[users], features[:len(pos)], features[len(pos):], self.temperature)
        alignment = bpr.new_zeros(())
        if self.ii_weight:
            v_align = self.item_alignment(pos, self.image_knn_idx, self.image_knn_weight)
            t_align = self.item_alignment(pos, self.text_knn_idx, self.text_knn_weight)
            alignment = self.image_weight * v_align + (1 - self.image_weight) * t_align
        total = bpr + self.modal_weight * modal + self.ii_weight * alignment
        self.last_loss_terms = dict(bpr=bpr.detach().item(), modality=modal.detach().item(),
                                   alignment=alignment.detach().item(), total=total.detach().item())
        self._batches += 1
        if self._batches == 1 or self._batches % 200 == 0:
            getLogger().info('FreedomAlignSemantic loss=%s; cf_score_std=%.6f '
                             'weighted_semantic_score_std=%.6f cf_abs_gap=%.6f weighted_semantic_abs_gap=%.6f',
                             self.last_loss_terms,
                             torch.cat((pos_cf, neg_cf)).detach().std(unbiased=False).item(),
                             self.semantic_beta * torch.cat((pos_semantic, neg_semantic)).detach().std(unbiased=False).item(),
                             (pos_cf - neg_cf).detach().abs().mean().item(),
                             self.semantic_beta * (pos_semantic - neg_semantic).detach().abs().mean().item())
        return total

    def full_sort_predict(self, interaction):
        if self.semantic_beta == 0:
            return super().full_sort_predict(interaction)
        users = interaction[0] if isinstance(interaction, (tuple, list)) else interaction
        if self.training:
            u, i = self.forward()
            semantic = self.semantic_items(self.image_trs(self.image_embedding.weight),
                                           self.text_trs(self.text_embedding.weight))
        else:
            with torch.no_grad():
                if self._eval_embeddings is None:
                    self._eval_embeddings = self.forward()
                if self._eval_semantic is None:
                    self._eval_semantic = self.semantic_items(self.image_trs(self.image_embedding.weight),
                                                               self.text_trs(self.text_embedding.weight))
            u, i = self._eval_embeddings
            semantic = self._eval_semantic
        return u[users] @ i.T + self.semantic_beta * (F.normalize(u[users], dim=-1) @ semantic.T)
