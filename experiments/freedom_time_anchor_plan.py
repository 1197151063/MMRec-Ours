"""TRAIN-time item anchors; automatic group count, one seed, no baseline rerun."""


def build_plan(seeds=(999,)):
    settings = [('b01', .1, 0.), ('b03', .3, 0.), ('b10', 1., 0.),
                ('b01_r001', .1, .01), ('b01_r010', .1, .1), ('b03_r010', .3, .1)]
    return [dict(name=f'freedom_anchor_{name}_s{seed}', model='FreedomTimeAnchor',
                 hypothesis='Group by earliest TRAIN-observed consumed item; uses TRAIN timestamps',
                 overrides=dict(seed=seed, group_seed=2026, group_strength=beta, personal_reg_weight=reg))
            for seed in seeds for name, beta, reg in settings]


def build_no_ui_plan(seeds=(999,)):
    """Match the three unregularized anchor settings, removing UI propagation only."""
    jobs = [job for job in build_plan(seeds) if job['overrides']['personal_reg_weight'] == 0.]
    for job in jobs:
        job['name'] = job['name'].replace('freedom_anchor_', 'freedom_anchor_noui_')
        job['hypothesis'] = 'Direct user_id + beta * TRAIN-time anchor group; no UI propagation; retain item graph'
        job['overrides']['n_ui_layers'] = 0
    return jobs
