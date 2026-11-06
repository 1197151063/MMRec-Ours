"""Fixed TRAIN spectral groups; one training seed, no baseline rerun."""


def build_plan(seeds=(999,)):
    settings = [('b01', .1, 0.), ('b03', .3, 0.), ('b10', 1., 0.),
                ('b01_r001', .1, .01), ('b01_r010', .1, .1), ('b03_r010', .3, .1)]
    return [dict(name=f'freedom_spectral_{name}_s{seed}', model='FreedomSpectralGroup',
                 hypothesis='Shared group preference from TRAIN graph spectral clustering (16 groups)',
                 overrides=dict(seed=seed, num_groups=16, spectral_rank=64, group_seed=2026,
                                group_strength=beta, personal_reg_weight=reg))
            for seed in seeds for name, beta, reg in settings]
