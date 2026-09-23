"""Matched-size grouping tests; no graph or correlated initializer."""
from itertools import product


def build_plan(seeds=(999,)):
    jobs=[]
    def add(name, hypothesis, **overrides):
        jobs.append(dict(name=name,model='GroupRec',hypothesis=hypothesis,overrides=overrides))
    add('no_group','G0: exact original no-PE control',group_strength=0.)
    jobs.append(dict(name='user_pe_reference',model='LightMRecOrder',hypothesis='G0: user PE reference',
                     overrides=dict(order_source='original',order_seed=2026,pe_side='user',pe_application='forward')))
    for size,strength,mode in product((2,4,8,16,32,64,128),(.3,1.,3.),('contiguous','random')):
        add(f'block{size}_w{strength}_{mode}','G1: same-size contiguous/random group sharing',
            group_size=size,group_strength=strength,group_mode=mode)
    for mode in ('contiguous','random'):
        add('zero_'+mode,'G2: sharing without initial group offset',group_init='zero',group_mode=mode)
        add('frozen_'+mode,'G3: fixed offset versus learned shared preference',group_trainable=False,group_mode=mode)
        add('group_only_'+mode,'G4: group preference without personal lookup',use_personal=False,group_mode=mode)
        add('large_std_'+mode,'G5: PE-scale group offset',group_std=2**-.5,group_mode=mode)
    add('singleton','G6: extra individual parameterization',group_size=1)
    add('global','G6: one preference shared by all users',group_size=1000000000)
    return [dict(j,name=j['name']+'_s'+str(s),overrides=dict(j['overrides'],seed=s,group_seed=s+1027))
            for j in jobs for s in seeds]
