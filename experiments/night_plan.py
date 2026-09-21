"""Predeclared single-factor diagnostics; no configurations selected using test scores."""

def build_plan(seeds=(999,)):
    jobs = []
    def add(name, hypothesis, model='LightMRecOrder', **overrides):
        base = dict(alpha=0.5, temperature=0.04, num_negatives=32,
                    order_source='original', pe_side='both', order_seed=2026)
        if model == 'SIMMRec':
            base = dict(alpha=0.5, temperature=0.2, num_negatives=128,
                        item_id_weight=0.0, reg_weight=0.0001, representation='content')
        base.update(overrides)
        jobs.append(dict(name=name, hypothesis=hypothesis, model=model, overrides=base))

    # Put the most diagnostic reference runs first.
    add('reference_none', 'H1: reproduce no-PE control', pe_side='none')
    add('reference_both', 'H1: reproduce historical original-coordinate effect')
    add('original_item', 'H2: item PE alone explains gains', pe_side='item')
    add('original_user', 'H2: user PE alone explains gains', pe_side='user')
    for source in ('random', 'random_user', 'random_item', 'train_file', 'train_shuffled', 'train_degree'):
        add(source + '_both', 'H3: distinguish coordinate coupling, row order and train-degree bias', order_source=source)
    for source in ('random', 'train_file', 'train_shuffled', 'train_degree'):
        add(source + '_item', 'H3: item-side order effect without user PE', order_source=source, pe_side='item')
    for kind in ('gaussian', 'constant'):
        for side in ('both', 'item'):
            add(kind + '_' + side, 'H4: fixed identity vectors versus common offset at matched norm', pe_kind=kind, pe_side=side)
    for scale in (0.01, 0.1, 0.3, 3.0):
        add('scale_' + str(scale), 'H5: original PE benefit depends on amplitude', pe_scale=scale)
    for offset in (1000, 10000):
        add('item_offset_' + str(offset), 'H6: same user/item coordinate origin is important', item_pe_offset=offset)
    for side in ('none', 'both'):
        add('cosine_' + side, 'H7: raw item norms contribute to observed PE gains', pe_side=side, eval_cosine=True)
        add('frozen_' + side, 'H8: per-item feature fine-tuning is necessary', pe_side=side, freeze_features=True)
    for order_seed in (2027, 2028):
        for source in ('random', 'train_shuffled'):
            add(source + '_repeat_' + str(order_seed), 'H9: ordering effect survives independent order seeds', order_source=source, order_seed=order_seed)
    for tau in (0.2, 0.3, 0.5):
        add('content_tau_' + str(tau), 'H10: improve order-free generalization by temperature', model='SIMMRec', temperature=tau)
        add('id_tau_' + str(tau), 'H10: matched pure-ID control', model='SIMMRec', temperature=tau, representation='id', is_multimodal_model=False)
    for alpha in (0.25, 0.75):
        for tau in (0.2, 0.3, 0.5):
            add('fusion_' + str(alpha) + '_' + str(tau), 'H11: modality weighting interacts with temperature', model='SIMMRec', alpha=alpha, temperature=tau)
    for gamma in (0.03, 0.1, 0.3):
        for reg in (0.0001, 0.001):
            add('residual_' + str(gamma) + '_' + str(reg), 'H12: regularized identity residual at improved temperature', model='SIMMRec', item_id_weight=gamma, reg_weight=reg)
    return [dict(job, name=job['name'] + '_s' + str(seed), overrides=dict(job['overrides'], seed=seed))
            for seed in seeds for job in jobs]
