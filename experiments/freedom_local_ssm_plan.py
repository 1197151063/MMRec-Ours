"""ID-BPR/modality-SSM plans; local-only sweep and separate later grids."""
import math


LOCAL_WEIGHTS = (.1, .01, .001)


def build_plan(seeds=(999,)):
    # Sports current stage: vary ONLY local. These are fixed starting settings,
    # not claims about Sports-optimal tau/lambda or item constraints.
    return [dict(name=f'idbpr_modalssm_local_w{weight:g}_s{seed}', model='FreedomLocalSSM',
                 hypothesis='Explicit numeric-user-order local-only sweep (.1/.01/.001); tau=.2 lambda=.01 ii=.0005 K=10 fixed; validation Recall@20 selection',
                 overrides=dict(seed=seed, local_rho=math.exp(-1 / 16), local_weight=weight,
                                tau=.2, **{'lambda': .01}, alpha=.5, ii_weight=.0005,
                                knn_k=10, ii_image_weight=.1, knn_keep_self=True))
            for seed in seeds for weight in LOCAL_WEIGHTS]


TAUS = (.04, .07, .1, .2)
LAMBDAS = (.001, .005, .01, .05, .1)


def build_grid_plan(seeds=(999,)):
    return [dict(name=f'idbpr_batchssm_tau{tau:g}_lambda{weight:g}_s{seed}', model='FreedomLocalSSM',
                 hypothesis='In-batch weighted modality SSM grid; local=.01, alpha=.5; select by validation Recall@20',
                 overrides=dict(seed=seed, tau=tau, **{'lambda': weight}, alpha=.5,
                                local_rho=math.exp(-1 / 16), local_weight=.01))
            for seed in seeds for tau in TAUS for weight in LAMBDAS]


II_WEIGHTS = (.0001, .0005, .001, .005)
II_KS = (5, 10, 20, 40)


def build_item_grid_plan(seeds=(999,)):
    return [dict(name=f'idbpr_batchssm_ii{weight:g}_k{k}_s{seed}', model='FreedomLocalSSM',
                 hypothesis='Item-ID constraint weight/K grid; tau=.2, lambda=.01, local=.01, alpha=.5; validation Recall@20 selection',
                 overrides=dict(seed=seed, tau=.2, **{'lambda': .01}, alpha=.5,
                                local_rho=math.exp(-1 / 16), local_weight=.01,
                                ii_weight=weight, knn_k=k, ii_image_weight=.1,
                                knn_keep_self=True))
            for seed in seeds for weight in II_WEIGHTS for k in II_KS]
