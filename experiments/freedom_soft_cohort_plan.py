"""Small FREEDOM-only screen for implicit overlapping user cohorts."""


def build_plan(seeds=(999,)):
    variants = (
        ('baseline', 'text', 0., 'Exact original FREEDOM control'),
        ('text_01', 'text', .1, 'Text-derived item cohorts'),
        ('collaborative_01', 'collaborative', .1, 'Train co-consumption item cohorts'),
        ('joint_01', 'joint', .1, 'Joint text and co-consumption cohorts'),
        ('joint_03', 'joint', .3, 'Stronger joint-cohort residual'),
    )
    return [dict(name=f'freedom_soft_{name}_s{seed}', model='FreedomSoftCohort',
                 hypothesis=hypothesis,
                 overrides=dict(seed=seed, cohort_kind=kind, cohort_beta=beta,
                                cohort_count=64, cohort_seed=2026))
            for seed in seeds for name, kind, beta, hypothesis in variants]
