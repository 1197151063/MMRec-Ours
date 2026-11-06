"""Three new content/SSM models, no old baseline rerun, seed 999 by default."""
import math


def build_plan(seeds=(999,)):
    return [dict(name=f'content_ssm_local_w{weight:g}_s{seed}', model='FreedomLocalContent',
                 hypothesis='Content item inputs + UI2 + one SSM + ii attraction + explicit numeric-user-order local loss',
                 overrides=dict(seed=seed, local_rho=math.exp(-1 / 16), local_weight=weight))
            for seed in seeds for weight in (.01, .1, 1.)]
