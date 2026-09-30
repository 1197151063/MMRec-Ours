"""Matched ablations of the supplied FREEDOM variant, after weight-index correction."""
def build_plan(seeds=(999,)):
    jobs=[]
    for name, modal, ii in [('full',.01,.0005), ('no_ii',.01,0.), ('no_modal',0.,.0005), ('ui_only',0.,0.)]:
        for seed in seeds:
            jobs.append(dict(name=f'freedom_align_{name}_s{seed}', model='FreedomAlign',
                             hypothesis='Factorial ablation of modality SSM and raw-cosine item alignment',
                             overrides=dict(seed=seed,modal_weight=modal,ii_weight=ii)))
    return jobs
