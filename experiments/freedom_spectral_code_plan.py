"""Three continuous-coordinate variants, without baseline reruns."""


def build_plan(seeds=(999,)):
    settings = [('user_noui', 'user', 0), ('both_noui', 'both', 0), ('both_ui2', 'both', 2)]
    return [dict(name=f'freedom_code_{name}_s{seed}', model='FreedomSpectralCode',
                 hypothesis=f'Continuous TRAIN spectral coordinates: sides={sides}, UI layers={layers}',
                 overrides=dict(seed=seed, spectral_seed=2026, spectral_rank=64,
                                graph_strength=1., graph_sides=sides, n_ui_layers=layers))
            for seed in seeds for name, sides, layers in settings]
