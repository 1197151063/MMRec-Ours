"""Explicit preference capacity: history residuals and candidate-dependent interests.

No ID ordering. Histories/graphs are TRAIN-only upstream inputs. Multiple interest
scores are combined after matching an item, never by averaging user vectors.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class HistoryResidual(nn.Module):
    """Keep history as an input throughout training; learn only a d-dimensional delta."""
    def __init__(self, history, encoder):
        super().__init__()
        self.register_buffer('history', history.detach().clone())
        self.encoder = encoder

    def forward(self, delta):
        return self.encoder(self.history) + delta


class InterestBank(nn.Module):
    def __init__(self, users, dim, heads, tau, device):
        super().__init__()
        self.heads, self.tau = heads, tau
        # Independent perturbations prevent identical heads from staying identical.
        # Local CPU RNG preserves the base model/training RNG; first heads match
        # across K, so K comparisons do not reinitialize the common heads.
        seed = torch.initial_seed()
        values = []
        with torch.random.fork_rng(devices=[]):
            for modality in range(2):
                slots = []
                for head in range(heads):
                    generator = torch.Generator().manual_seed((seed + 1009 * modality + 7919 * (head + 1)) % (2**63 - 1))
                    slots.append(torch.randn(users, dim, generator=generator) * .1)
                values.append(torch.stack(slots, 1).to(device))
        self.visual_codes = nn.Parameter(values[0])
        self.text_codes = nn.Parameter(values[1])

    def context(self, features, users, dim):
        result = []
        for modal, codes in zip(features.split(dim, dim=1), (self.visual_codes, self.text_codes)):
            anchor = F.normalize(modal[:users], dim=-1)
            query = F.normalize(anchor[:, None] + codes, dim=-1)
            item = F.normalize(modal[users:], dim=-1)
            result.append((query, item))
        return result

    def pool(self, scores, dim):
        # log-mean-exp: identical/duplicated heads do not add a tau*log(K) offset.
        return self.tau * (torch.logsumexp(scores / self.tau, dim=dim) - math.log(self.heads))

    def paired(self, context, users, items):
        return sum(self.pool((q[users] * item[items, None]).sum(-1), 1)
                   for q, item in context) * .5

    def block(self, context, users, begin, end):
        return sum(self.pool(torch.einsum('bkd,id->bki', q[users], item[begin:end]), 1)
                   for q, item in context) * .5


def configure_preferences(model, options):
    mode = options['preference_mode']
    for table, projector in (('user_v_prefer', 'image_u_trs'), ('user_t_prefer', 'text_u_trs')):
        history, encoder = getattr(model, table), getattr(model, projector)
        if mode == 'frozen':
            history.requires_grad_(False)
        elif mode == 'projected':
            with torch.no_grad():
                projected = encoder(history).detach().clone()
            setattr(model, table, nn.Parameter(projected))
            setattr(model, projector, nn.Identity())
        elif mode == 'residual':
            setattr(model, table, nn.Parameter(history.new_zeros((model.n_users, model.embedding_dim))))
            setattr(model, projector, HistoryResidual(history, encoder))
    if options['freeze_item_features']:
        model.image_embedding.weight.requires_grad_(False)
        model.text_embedding.weight.requires_grad_(False)
    if options['activation'] != 'original':
        model.prl = nn.GELU() if options['activation'] == 'gelu' else nn.ReLU()
    if options['interest_heads']:
        model.interest_bank = InterestBank(model.n_users, model.embedding_dim,
                                          options['interest_heads'], options['interest_tau'], model.device)
