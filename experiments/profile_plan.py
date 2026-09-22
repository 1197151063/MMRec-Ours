"""Prioritized minimal profile study. Every hyperparameter choice is predeclared."""
from itertools import product


def build_plan(seeds=(999,)):
    jobs = []
    base = dict(alpha=.75, item_id_weight=0., temperature=.2, num_negatives=128,
                reg_weight=.0001, profile_weight=1., profile_mode='own', history_power=0.,
                leave_one_out=True, use_user_id=True, detach_profile=False, profile_seed=2026)
    def add(name, hypothesis, **overrides):
        jobs.append(dict(name=name, model='SIMMProfile', hypothesis=hypothesis,
                         overrides=dict(base, **overrides)))
    # First 12: interpretable matched controls, before wider exploration.
    add('baseline', 'P0: exact clean SIMMRec control', profile_weight=0.)
    for strength in (.1, .3, 1., 3.):
        add('own_b'+str(strength), 'P1: persistent shared content profile', profile_weight=strength)
        add('shuffled_b'+str(strength), 'P1: matched-strength wrong-user control',
            profile_weight=strength, profile_mode='shuffled')
    add('no_user_id', 'P2: can shared history replace the user lookup?', use_user_id=False)
    add('detach', 'P3: gradient through history projectors', detach_profile=True)
    add('include_target_DIAGNOSTIC', 'P4: self-target training shortcut; diagnostic only', leave_one_out=False)
    # Equal temperature/fusion search for baseline and three profile strengths.
    for temperature, alpha, strength in product((.1, .2, .3, .5), (.5, .75, 1.), (0., .3, 1., 3.)):
        if temperature == .2 and alpha == .75:
            continue  # already covered by first controls
        add(f'grid_t{temperature}_a{alpha}_b{strength}', 'P5: matched baseline/profile tuning',
            temperature=temperature, alpha=alpha, profile_weight=strength)
    for power, strength in product((.5, 1.), (.3, 1., 3.)):
        add(f'pop_p{power}_b{strength}', 'P6: downweight frequent training items',
            history_power=power, profile_weight=strength)
    for reg, strength in product((0., .001), (0., .3, 1., 3.)):
        add(f'reg_r{reg}_b{strength}', 'P7: user residual regularization',
            reg_weight=reg, profile_weight=strength)
    # Interleave seeds per configuration so a time cap retains matched-seed evidence.
    return [dict(job, name=job['name']+'_s'+str(seed), overrides=dict(job['overrides'], seed=seed))
            for job in jobs for seed in seeds]
