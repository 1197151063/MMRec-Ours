"""Two-by-two test of freezing raw features and direct semantic BPR scores."""


def build_plan(seeds=(999,)):
    variants = [('baseline', False, 0.), ('freeze', True, 0.),
                ('score', False, .1), ('freeze_score', True, .1)]
    return [dict(name=f'freedom_semantic_{name}_s{seed}', model='FreedomAlignSemantic',
                 hypothesis='Frozen content x direct semantic scoring; no UU aggregation or UU loss',
                 overrides=dict(seed=seed, modal_weight=.01, ii_weight=.0005,
                                freeze_modal_features=freeze, semantic_beta=beta, semantic_image_weight=.5))
            for name, freeze, beta in variants for seed in seeds]
