"""Capacity ablations without modifying the pinned REARM forward or losses."""
import torch
from torch import nn

ATTENTION = ('self_i_attn1', 'self_i_attn2', 'mutual_i_attn1', 'mutual_i_attn2')
PROJECTORS = ('image_i_trs', 'text_i_trs', 'image_u_trs', 'text_u_trs')
META = ('mlp_u1', 'mlp_u2', 'mlp_i1', 'mlp_i2', 'meta_netu', 'meta_neti')


class ZeroAttention(nn.Module):
    """Remove attention while retaining upstream residual/LN/PReLU.

    Returning the query would double the residual, so return zero. This module
    has no parameters, attention computation or dropout.
    """
    def forward(self, query, key, value, **kwargs):
        return torch.zeros_like(query), None


def configure(model, options):
    if options['attention'] == 'none':
        for name in ATTENTION:
            setattr(model, name, ZeroAttention())
    hidden = options['projector_hidden']
    if hidden:
        names = PROJECTORS if options['projector_scope'] == 'all' else PROJECTORS[:2]
        # CPU initialization in a private RNG context leaves shared upstream
        # parameters and the subsequent training RNG unchanged.
        with torch.random.fork_rng(devices=[]):
            for name in names:
                old = getattr(model, name)
                layers = nn.Sequential(nn.Linear(old.in_features, hidden), nn.ReLU(),
                                       nn.Linear(hidden, old.out_features))
                for layer in layers:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_normal_(layer.weight)
                        nn.init.zeros_(layer.bias)
                setattr(model, name, layers.to(model.device))
    if options['meta'] == 'none':
        for name in META:
            delattr(model, name)
    from rearm_preferences import configure_preferences
    configure_preferences(model, options)


def parameter_breakdown(model):
    """Count registered parameters exactly once, including trainable feature tables."""
    groups = {k: dict(total=0, trainable=0) for k in
              ('feature_tables', 'id_embeddings', 'projectors', 'attention', 'meta', 'interests', 'other')}
    for name, parameter in model.named_parameters():
        prefix = name.split('.')[0]
        if prefix in ('image_embedding', 'text_embedding', 'user_v_prefer', 'user_t_prefer'):
            group = 'feature_tables'
        elif prefix in ('user_id_embedding', 'item_id_embedding'):
            group = 'id_embeddings'
        elif prefix in PROJECTORS:
            group = 'projectors'
        elif prefix in ATTENTION:
            group = 'attention'
        elif prefix in META:
            group = 'meta'
        elif prefix == 'interest_bank':
            group = 'interests'
        else:
            group = 'other'
        groups[group]['total'] += parameter.numel()
        if parameter.requires_grad:
            groups[group]['trainable'] += parameter.numel()
    return groups
