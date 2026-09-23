"""Five-run check with the existing default initialization and loss settings."""


def build_plan(seeds=(999,)):
    jobs=[dict(name='no_group',model='GroupRec',hypothesis='G0: item-ID plus modality SSM, no group',
               overrides=dict(group_strength=0.))]
    for size in (16,64):
        for mode in ('contiguous','random'):
            jobs.append(dict(name=f'block{size}_{mode}',model='GroupRec',
                             hypothesis='G1: matched-size contiguous/random group sharing',
                             overrides=dict(group_size=size,group_mode=mode)))
    return [dict(j,name=j['name']+'_s'+str(s),overrides=dict(j['overrides'],seed=s))
            for j in jobs for s in seeds]
