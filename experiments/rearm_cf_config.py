"""Serializable, validated CF experiment options; importable without PyTorch."""
import math

DEFAULTS = dict(aux='ssm', target='projected', ssm_weight=.01, tau=.1, negatives=32,
                image_weight=.5, include_positive=False, exclude_train=True,
                main='bpr', degree_power=0., ultra_ui=0., ultra_ii=0., ultra_k=20,
                cir='none', cir_mix=.5, negative_pool=1, negative_rank=0,
                negative_popularity=0., denoise='none', drop_rate=.1, warmup=30,
                ramp=30, ema_decay=.9, freeze_features=False,
                nt_weight=0., nt_user=1., nt_item=1., nt_bidirectional=False,
                graphda_epochs=0, graphda_k=10, graphda_homogeneous_k=0, graphda_mix=1.,
                attention='original', projector_hidden=0, projector_scope='item', meta='original')
OVERRIDES = dict(l_r=(1e-7, .1), reg_weight=(0., 1.), num_layer=(0, 10),
                 n_ii_layers=(0, 5), n_uu_layers=(0, 5), diff_loss_weight=(0., 1.),
                 cl_loss_weight=(0., 1.), s_drop=(0., 1.), m_drop=(0., 1.))


def validate(spec):
    if set(spec) - {'options', 'overrides'}:
        raise ValueError('Unknown experiment config sections')
    options = dict(DEFAULTS)
    if set(spec.get('options', {})) - set(DEFAULTS):
        raise ValueError('Unknown CF options: ' + str(set(spec['options']) - set(DEFAULTS)))
    options.update(spec.get('options', {}))
    for k, v in options.items():
        if isinstance(DEFAULTS[k], bool):
            if not isinstance(v, bool):
                raise ValueError(k + ' must be boolean')
        elif isinstance(DEFAULTS[k], (int, float)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(k + ' must be finite and nonnegative')
            if isinstance(DEFAULTS[k], int) and not isinstance(v, int):
                raise ValueError(k + ' must be an integer')
    for key, allowed in dict(aux=('official', 'none', 'ssm', 'both'), target=('projected', 'propagated'),
                             main=('bpr', 'ssm', 'bce'), cir=('none', 'jc', 'lhn'),
                             denoise=('none', 'trim', 'ema'), attention=('original', 'none'),
                             projector_scope=('item', 'all'), meta=('original', 'none')).items():
        if options[key] not in allowed:
            raise ValueError('Invalid ' + key)
    for key in ('image_weight', 'cir_mix', 'graphda_mix'):
        if options[key] > 1:
            raise ValueError(key + ' must be <= 1')
    if options['drop_rate'] >= 1 or options['ema_decay'] >= 1:
        raise ValueError('drop_rate and ema_decay must be < 1')
    if not options['tau'] or min(options[k] for k in ('negatives', 'ultra_k', 'negative_pool', 'ramp')) < 1:
        raise ValueError('Positive tau, negatives, ultra_k, negative_pool and ramp required')
    if options['negative_rank'] >= options['negative_pool']:
        raise ValueError('negative_rank must be smaller than negative_pool')
    if options['graphda_epochs'] and options['cir'] != 'none':
        raise ValueError('GraphDA and CIR change the same adjacency; test them separately')
    if options['graphda_epochs'] and (options['attention'] != 'original' or options['meta'] != 'original' or options['projector_hidden']):
        raise ValueError('Capacity ablations must run separately from GraphDA experiments')
    overrides = dict(spec.get('overrides', {}))
    for k, v in overrides.items():
        if k not in OVERRIDES or isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v):
            raise ValueError('Invalid upstream override: ' + k)
        lo, hi = OVERRIDES[k]
        if not lo <= v <= hi or ('layers' in k or k == 'num_layer') and not isinstance(v, int):
            raise ValueError('Out-of-range upstream override: ' + k)
    return dict(options=options, overrides=overrides)
