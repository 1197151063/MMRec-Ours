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
