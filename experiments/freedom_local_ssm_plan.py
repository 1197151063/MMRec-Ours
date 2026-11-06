"""Requested ID-BPR/modality-SSM model only; one seed and three local strengths."""
import math


def build_plan(seeds=(999,)):
    return [dict(name=f'idbpr_modalssm_local_w{weight:g}_s{seed}', model='FreedomLocalSSM',
                 hypothesis='ID/UI BPR + image/text SSM + raw item-ID constraint + explicit numeric-user-order local loss',
                 overrides=dict(seed=seed, local_rho=math.exp(-1 / 16), local_weight=weight))
            for seed in seeds for weight in (.01, .1, 1.)]
