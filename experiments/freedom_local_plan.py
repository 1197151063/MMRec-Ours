"""Only local regularization; three strengths, one seed, no baseline rerun."""
import math


def build_plan(seeds=(999,)):
    return [dict(name=f'freedom_local_w{weight:g}_s{seed}', model='FreedomLocal',
                 hypothesis='EXPLICIT numeric user-ID adjacent local loss on raw ID embeddings; no PE',
                 overrides=dict(seed=seed, local_rho=math.exp(-1 / 16), local_weight=weight))
            for seed in seeds for weight in (.01, .1, 1.)]
