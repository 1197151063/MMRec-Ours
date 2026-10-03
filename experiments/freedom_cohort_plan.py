"""Four matched forward-pass controls; fixed beta and no UU attraction loss."""


def build_plan(seeds=(999,)):
    return [dict(name=f'freedom_cohort_{kind}_s{seed}', model='FreedomAlignCohort',
                 hypothesis='One user residual after UI LightGCN; compare neighbor selection only',
                 overrides=dict(seed=seed, modal_weight=.01, ii_weight=.0005,
                                cohort_kind=kind, cohort_beta=.1, cohort_k=80,
                                cohort_dim=64, cohort_seed=2026))
            for kind in ('none', 'shared_cosine', 'shared_random', 'residual_semantic')
            for seed in seeds]
