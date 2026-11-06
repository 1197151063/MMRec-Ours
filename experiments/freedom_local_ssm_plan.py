"""Requested ID-BPR/modality-SSM model only; one seed and three local strengths."""
import math


def build_plan(seeds=(999,)):
    return [dict(name=f'idbpr_modalssm_local_w{weight:g}_s{seed}', model='FreedomLocalSSM',
                 hypothesis='ID/UI BPR + image/text SSM + raw item-ID constraint + explicit numeric-user-order local loss',
                 overrides=dict(seed=seed, local_rho=math.exp(-1 / 16), local_weight=weight))
            for seed in seeds for weight in (.01, .1, 1.)]


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
