"""Three weights under identical group/encoder/SSM settings."""

def build_plan(seeds=(999,)):
    return [dict(name=f'wbpr_mix{mix}_s{seed}',model='WeightedGroupRec',
                 hypothesis='LinkProp-inspired positive structural support weighting',
                 overrides=dict(seed=seed,num_groups=16,group_mode='contiguous',group_strength=1.,
                                wbpr_mix=mix,link_beta=.5,link_gamma=.5,link_delta=.5))
            for mix in (0.,.5,1.) for seed in seeds]
