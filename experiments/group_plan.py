"""Sensitivity to total group count; all other settings remain fixed."""


def build_plan(seeds=(999,)):
    return [dict(name=f'groups{count}_lgcn2_bpr_s{seed}', model='GroupRec',
                 hypothesis='Total group count sensitivity: fixed two-layer LightGCN, BPR plus modality SSM',
                 overrides=dict(seed=seed, num_groups=count, group_mode='contiguous', group_strength=1.))
            for count in (32, 64, 128) for seed in seeds]
