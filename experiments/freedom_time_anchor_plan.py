"""TRAIN-time item anchors; automatic group count, one seed, no baseline rerun."""


def build_plan(seeds=(999,)):
    settings = [('b01', .1, 0.), ('b03', .3, 0.), ('b10', 1., 0.),
                ('b01_r001', .1, .01), ('b01_r010', .1, .1), ('b03_r010', .3, .1)]
    return [dict(name=f'freedom_anchor_{name}_s{seed}', model='FreedomTimeAnchor',
                 hypothesis='Group by earliest TRAIN-observed consumed item; uses TRAIN timestamps',
                 overrides=dict(seed=seed, group_seed=2026, group_strength=beta, personal_reg_weight=reg))
            for seed in seeds for name, beta, reg in settings]
