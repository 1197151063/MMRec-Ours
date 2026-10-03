"""One matched baseline and six UU candidates; select with validation only."""
def build_plan(seeds=(999,)):
    variants=[('baseline',0.,0.)]
    variants += [(f'shared_w{w}',w,0.) for w in (.01,.1,1.)]
    variants += [(f'bridge_w{w}',w,.1) for w in (.01,.1,1.)]
    return [dict(name=f'freedom_uu_{name}_s{seed}',model='FreedomAlignUU',
                 hypothesis='No UU top-k: sample complete shared-item paths, with weak semantic bridges',
                 overrides=dict(seed=seed,modal_weight=.01,ii_weight=.0005,uu_weight=w,uu_bridge_weight=beta,
                                uu_samples=16,uu_item_k=5,uu_sampling_seed=2026))
            for name,w,beta in variants for seed in seeds]
