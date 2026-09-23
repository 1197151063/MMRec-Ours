"""Twelve matched runs without user-order information."""

def build_plan(seeds=(999,)):
    variants=[('baseline',dict(ii_weight=0.,uu_weight=0.)),
              ('ii',dict(ii_weight=.1,uu_weight=0.)),
              ('uu',dict(ii_weight=0.,uu_weight=.1)),
              ('both',dict(ii_weight=.1,uu_weight=.1)),
              ('weighted_both',dict(ii_weight=.1,uu_weight=.1,link_mix=.5)),
              ('ssm_both',dict(ii_weight=.1,uu_weight=.1,primary_loss='ssm'))]
    jobs=[]
    for layers in (0,2):
        for name,overrides in variants:
            for seed in seeds:
                config=dict(ui_layers=layers,primary_loss='bpr',link_mix=0.,seed=seed)
                config.update(overrides)
                jobs.append(dict(name=f'relation_l{layers}_{name}_s{seed}',model='RelationRec',
                                 hypothesis='Loss-based homogeneous relations; matched UI propagation and primary objective',overrides=config))
    return jobs
