"""Three mechanism checks, eight minimal initialization configurations."""
def build_plan(seeds=(999,)):
    jobs = []
    for name, side, application in [('mechanism_none', 'none', 'forward'),
                                     ('mechanism_user_forward', 'user', 'forward'),
                                     ('mechanism_user_init', 'user', 'init')]:
        jobs.append(dict(name=name, model='LightMRecOrder', hypothesis='M1: user PE as translated initialization',
                         overrides=dict(alpha=0.5, temperature=0.04, num_negatives=32, order_source='original',
                                        pe_side=side, pe_application=application, order_seed=2026)))
    variants = [('none', 0.0)] + [(kind, strength) for kind in ('random', 'interaction', 'content')
                               for strength in (0.1, 1.0)] + [('mismatched', 1.0)]
    for kind, strength in variants:
        jobs.append(dict(name='init_' + kind + '_' + str(strength), model='SIMMInit',
                         hypothesis='M2: relationship-aware initialization versus matched-scale controls',
                         overrides=dict(alpha=0.75, temperature=0.2, num_negatives=128, item_id_weight=0.0,
                                        reg_weight=0.0001, user_init=kind, init_strength=strength, init_seed=2026)))
    return [dict(job, name=job['name']+'_s'+str(seed), overrides=dict(job['overrides'], seed=seed))
            for seed in seeds for job in jobs]
