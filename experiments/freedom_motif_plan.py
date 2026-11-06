"""Single-seed motif constraints on FREEDOM; no baseline rerun."""


def build_plan(seeds=(999,)):
    variants = [
        ('star_sem_001', dict(star_weight=.01), 'Weak item-conditioned user agreement'),
        ('star_sem_005', dict(star_weight=.05), 'Item-conditioned user agreement'),
        ('star_sem_010', dict(star_weight=.1), 'Stronger item-conditioned user agreement'),
        ('star_full_005', dict(star_weight=.05, star_mode='full'), 'Mechanism ablation: whole user vector attraction'),
        ('star_equal_items', dict(star_weight=.05, star_pop_power=1.), 'Give rare item groups greater influence'),
        ('bridge_001', dict(star_weight=0., bridge_weight=.01), 'Weak four-hop relations without direct TRAIN overlap'),
        ('star_bridge_0003', dict(star_weight=.05, bridge_weight=.003), 'Item groups plus a weak bridge constraint'),
        ('star_bridge_001', dict(star_weight=.05, bridge_weight=.01), 'Item groups plus a stronger bridge constraint'),
    ]
    return [dict(name=f'freedom_motif_{name}_s{seed}', model='FreedomMotif',
                 hypothesis=question, overrides=dict(seed=seed, **overrides))
            for seed in seeds for name, overrides, question in variants]
