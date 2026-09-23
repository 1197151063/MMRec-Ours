"""One fixed 16-group LightGCN/BPR configuration; no random grouping jobs."""


def build_plan(seeds=(999,)):
    return [dict(name='groups16_lgcn2_bpr_s'+str(seed), model='GroupRec',
                 hypothesis='Fixed contiguous groups, two-layer ID LightGCN, BPR plus modality SSM',
                 overrides=dict(seed=seed, num_groups=16, group_mode='contiguous', group_strength=1.))
            for seed in seeds]
