"""EXPLICIT numeric user-ID ordering prior. Not a TRAIN-graph-derived similarity."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class UserOrderPrior(nn.Module):
    def __init__(self, active_users, bandwidth=16., radius=64, samples=4, far_gap=256):
        super().__init__()
        if not math.isfinite(bandwidth) or bandwidth <= 0 or min(radius, samples, far_gap) < 1:
            raise ValueError('Positive bandwidth, radius, samples and far_gap required')
        active = torch.as_tensor(active_users, dtype=torch.bool)
        offsets = torch.arange(-int(radius), int(radius) + 1, device=active.device)
        offsets = offsets[offsets != 0]
        self.register_buffer('active', active)
        self.register_buffer('active_ids', torch.nonzero(active, as_tuple=True)[0])
        self.register_buffer('offsets', offsets)
        self.register_buffer('kernel', torch.exp(-offsets.abs().float() / bandwidth))
        self.samples, self.far_gap = int(samples), int(far_gap)

    @torch.no_grad()
    def sample_near(self, anchors):
        candidates = anchors[:, None] + self.offsets
        in_bounds = (candidates >= 0) & (candidates < len(self.active))
        candidates = candidates.clamp(0, len(self.active) - 1)
        weights = self.kernel[None] * (in_bounds & self.active[candidates])
        valid = weights.sum(-1) > 0
        # A dummy row is sampled only to keep multinomial well-defined; its loss is masked.
        weights[~valid, 0] = 1.
        selected = torch.multinomial(weights, self.samples, replacement=True)
        return candidates.gather(1, selected), valid

    @torch.no_grad()
    def sample_far(self, anchors):
        if not len(self.active_ids):
            return anchors[:, None].expand(-1, self.samples), torch.zeros_like(anchors, dtype=torch.bool)
        left = torch.searchsorted(self.active_ids, anchors - self.far_gap, right=True)
        right = torch.searchsorted(self.active_ids, anchors + self.far_gap)
        counts = left + len(self.active_ids) - right
        draw = (torch.rand((len(anchors), self.samples), device=anchors.device) * counts[:, None]).long()
        indices = torch.where(draw < left[:, None], draw, right[:, None] + draw - left[:, None])
        return self.active_ids[indices.clamp(0, len(self.active_ids) - 1)], counts > 0

    def forward(self, embeddings, anchors, mode='attract', margin=.1, temperature=.1):
        anchors = anchors.unique()
        anchors = anchors[self.active[anchors]]
        if not len(anchors):
            return embeddings.sum() * 0
        near, valid = self.sample_near(anchors)
        a = F.normalize(embeddings[anchors], dim=-1)[:, None]
        near_similarity = (a * F.normalize(embeddings[near], dim=-1)).sum(-1)
        if mode == 'attract':
            per_pair = 1 - near_similarity
        elif mode == 'local_rank':
            if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(margin) or margin < 0:
                raise ValueError('Positive temperature and nonnegative margin required')
            far, far_valid = self.sample_far(anchors)
            valid = valid & far_valid
            far_similarity = (a * F.normalize(embeddings[far], dim=-1)).sum(-1)
            per_pair = temperature * F.softplus((far_similarity - near_similarity + margin) / temperature)
        else:
            raise ValueError('Order prior mode must be attract or local_rank')
        return (per_pair.mean(-1) * valid).sum() / valid.sum().clamp_min(1)
