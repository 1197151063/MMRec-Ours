"""Eight explicitly numeric-ID-order-aware configurations; one seed, no baseline rerun."""


def build_plan(seeds=(999,)):
    variants = [('attract', bandwidth, weight) for bandwidth in (16, 64) for weight in (.01, .1, 1.)]
    variants += [('local_rank', 32, weight) for weight in (.1, 1.)]
    return [dict(name=f'freedom_order_{mode}_b{bandwidth}_w{weight:g}_s{seed}', model='FreedomOrderPrior',
                 hypothesis='EXPLICIT original numeric user-ID neighborhood prior; not graph-derived or order-free',
                 overrides=dict(seed=seed, order_mode=mode, order_bandwidth=float(bandwidth),
                                order_radius=4 * bandwidth, order_weight=weight, order_samples=4,
                                order_far_gap=8 * bandwidth))
            for seed in seeds for mode, bandwidth, weight in variants]
