"""Correlated initialization first, then matched component ablations."""
from itertools import product
import math


def build_plan(seeds=(999,)):
    jobs=[]
    base=dict(rho=.75, eta=.99, init_std=math.sqrt(.5), order_source='original',
              corr_init=True, shared_noise=False, ui_layers=0, item_id_weight=0., mm_weight=0.,
              aux_weight=0., user_dropout=0., loss_kind='ssm', corr_reg=0.)
    def add(name,hypothesis,**kw):
        jobs.append(dict(name=name,model='CorrRec',hypothesis=hypothesis,overrides=dict(base,**kw)))
    add('legacy_nope','C0: exact original no-PE reconstruction',corr_init=False)
    for scale in (.01,.1,math.sqrt(.5)):
        add(f'iid_std{scale:g}','C1: matched marginal-scale IID Gaussian',rho=0.,init_std=scale)
    for order in ('original','random_user'):
        add('anchor_'+order,'C2: correlated anchor and random-order control',order_source=order)
    components=[('aux001',dict(aux_weight=.01)),('aux01',dict(aux_weight=.1)),
                ('bpr',dict(loss_kind='bpr')),('bpr_aux',dict(loss_kind='bpr',aux_weight=.1)),
                ('itemid',dict(item_id_weight=.1)),('ui1',dict(ui_layers=1)),
                ('knn',dict(mm_weight=.1)),('knn_ppr',dict(mm_weight=.1,ppr_mix=.1)),
                ('dropout',dict(user_dropout=.1)),
                ('combined',dict(item_id_weight=1.,ui_layers=1,mm_weight=1.,ppr_mix=.1,
                                 loss_kind='bpr',aux_weight=.1,corr_reg=.0001))]
    for component,settings in components:
        for control in ('original','random_user','iid'):
            overrides=dict(settings,order_source='random_user' if control=='random_user' else 'original')
            if control=='iid': overrides['rho']=0.
            add(f'component_{component}_{control}','C3: matched component increment',**overrides)
    for rho,eta,scale,order in product((.25,.75,1.),(.5,.9,.99,.999),(.01,.1,math.sqrt(.5)),('original','random_user')):
        if (rho,eta,scale)==(.75,.99,math.sqrt(.5)): continue
        add(f'grid_r{rho}_e{eta}_s{scale:g}_{order}','C4: correlation/length/scale sweep',
            rho=rho,eta=eta,init_std=scale,order_source=order)
    for eta in (.5,.9,.99):
        add(f'shared_noise_e{eta}','C5: literal reused-innovation formula; variance differs',eta=eta,shared_noise=True)
    return [dict(j,name=j['name']+'_s'+str(s),overrides=dict(j['overrides'],seed=s,init_seed=s+1027))
            for j in jobs for s in seeds]
