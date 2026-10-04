"""Fixed, hypothesis-driven experiments, never selected using test metrics."""
from rearm_cf_config import validate


def experiments():
    jobs = []
    def add(name, family, options=None, overrides=None, question=''):
        jobs.append(dict(name=name, family=family, question=question,
                         config=validate(dict(options=options or {}, overrides=overrides or {}))))
    add('reference', 'control', dict(aux='official'), question='Full original REARM, train-only protocol')
    add('without_cl', 'control', dict(aux='none'), question='Does removing original CL alone help?')
    add('ssm_base', 'ssm', question='Replace cross-modal CL by user-ID to projected modality negative-only SSM')
    add('ssm_legacy_sampling', 'ssm', dict(exclude_train=False), question='Exact user-style uniform negatives, including possible training positives')
    add('ssm_propagated', 'ssm', dict(target='propagated'), question='Supervise propagated instead of raw projected modalities')
    add('ssm_positive_denominator', 'ssm', dict(include_positive=True), question='Positive-included sampled softmax versus negative-only SSM')
    add('cl_plus_ssm', 'ssm', dict(aux='both'), question='Addition rather than replacement')
    for weight in (.0001, .001, .003, .03):
        add('ssm_weight_' + str(weight), 'ssm', dict(ssm_weight=weight), question='SSM gradient scale')
    for tau in (.04, .2):
        add('ssm_tau_' + str(tau), 'ssm', dict(tau=tau), question='SSM temperature')
    add('ssm_neg128', 'ssm', dict(negatives=128), question='More sampled negatives')
    add('ssm_text90', 'ssm', dict(image_weight=.1), question='Text-dominant auxiliary objective')
    add('ssm_without_diff', 'ssm', overrides=dict(diff_loss_weight=0.), question='Does orthogonality conflict with user-content alignment?')
    add('main_ssm', 'main', dict(main='ssm', aux='none'), question='Negative-only SSM on final prediction representation')
    add('main_sampled_softmax', 'main', dict(main='ssm', aux='none', include_positive=True), question='Positive-included softmax on final representation')
    for power in (.5, 1.):
        add('degree_bpr_' + str(power), 'ultra', dict(degree_power=power), question='Degree-derived weights on BPR')
    for weight in (.01, .1, 1.):
        add('ultra_ui_' + str(weight), 'ultra', dict(ultra_ui=weight), question='Degree-weighted UI logistic constraint')
        add('ultra_ii_' + str(weight), 'ultra', dict(ultra_ii=weight), question='User attraction to co-occurrence neighbors of positive items')
    for k in (10, 40):
        add('ultra_ii_k' + str(k), 'ultra', dict(ultra_ii=.1, ultra_k=k), question='II neighborhood coverage')
    add('ultra_bce', 'ultra', dict(main='bce', ultra_ui=.1, ultra_ii=.1), question='UltraGCN-style logistic main objective within REARM')
    add('ultra_no_ui_propagation', 'ultra', dict(ultra_ui=.1, ultra_ii=.1), dict(num_layer=0),
        'Can constraints compensate for removing UI propagation? Paired control: combo_ultra')
    for sim in ('jc', 'lhn'):
        for mix in (.25, .5, 1.):
            add(f'cir_{sim}_{mix}', 'cagcn', dict(cir=sim, cir_mix=mix), question='One-hop CIR weighting at fixed row message mass')
    add('nt_control', 'nt', dict(nt_weight=.01), question='Ordinary positive-included SSM on linear UI ID branch')
    add('nt_bidirectional_control', 'nt', dict(nt_weight=.01, nt_bidirectional=True), question='Directionality control before changing type coefficients')
    for au, ai in ((.5, 1.), (1., .5), (1.5, 1.), (1., 1.5)):
        add(f'nt_u{au}_i{ai}', 'nt', dict(nt_weight=.01, nt_user=au, nt_item=ai), question='Reweight starting user/item contributions to negative ID-branch scores')
    add('nt_bidirectional_u05', 'nt', dict(nt_weight=.01, nt_user=.5, nt_bidirectional=True), question='Type weighting plus both directions')
    for rate in (.1, .2):
        add('trim_' + str(rate), 'denoise', dict(denoise='trim', drop_rate=rate), question='Warmup then ramped high-loss exclusion, BPR adaptation of ADT')
    for rate in (.5, .8):
        add('ema_reweight_' + str(rate), 'denoise', dict(denoise='ema', drop_rate=rate), question='Soften persistent high-loss training edges; difficulty is not a noise label')
    add('trim_late', 'denoise', dict(denoise='trim', drop_rate=.1, warmup=80), question='Later intervention against memorization')
    add('hard4', 'sampling', dict(negative_pool=4), question='Highest scored of four eligible training negatives')
    add('hard8', 'sampling', dict(negative_pool=8), question='Harder ranking negatives')
    add('semihard8', 'sampling', dict(negative_pool=8, negative_rank=3), question='Fourth hardest of eight reduces extreme false-negative pressure')
    add('popularity05', 'sampling', dict(negative_popularity=.5), question='Frequency-biased negatives from train degrees')
    for lr in (.0001, .0002):
        add('lr_' + str(lr), 'optimization', overrides=dict(l_r=lr), question='Learning rate after objective replacement')
    for decay in (.0001, .001):
        add('decay_' + str(decay), 'optimization', overrides=dict(reg_weight=decay), question='Embedding memorization capacity via AdamW decay')
    add('freeze_features', 'optimization', dict(freeze_features=True), question='Freeze pretrained item features and averaged user-interest tables')
    add('ui_layers2', 'optimization', overrides=dict(num_layer=2), question='Reduce UI smoothing depth')
    # These original-CL controls prevent attributing a failed SSM transplant to a CF module.
    add('official_ultra_ii', 'bridge', dict(aux='official', ultra_ii=.1), question='II constraint without replacing CL')
    add('official_cir', 'bridge', dict(aux='official', cir='jc'), question='CIR without replacing CL')
    add('official_trim', 'bridge', dict(aux='official', denoise='trim'), question='Denoising without replacing CL')
    add('combo_ultra', 'combination', dict(ultra_ui=.1, ultra_ii=.1), question='UI plus II constraints')
    add('combo_cir_ii', 'combination', dict(cir='jc', ultra_ii=.1), question='Structural propagation weights plus neighbor supervision')
    add('combo_cir_trim', 'combination', dict(cir='jc', denoise='trim'), question='Topology and training-edge reliability')
    add('combo_ii_semihard', 'combination', dict(ultra_ii=.1, negative_pool=8, negative_rank=3), question='Neighbor attraction plus moderate negative pressure')
    add('combo_ssm_lr', 'combination', dict(ssm_weight=.003, image_weight=.1), dict(l_r=.0001), 'Conservative text-heavy SSM with faster optimization')
    # Expensive two-stage jobs come last; independently pretrained teachers, no test selection.
    for k in (5, 10, 20):
        add('graphda_k' + str(k), 'graphda', dict(graphda_epochs=120, graphda_k=k), question='Reconstruct UI graph from fixed-epoch teacher; fresh student')
    add('graphda_residual', 'graphda', dict(graphda_epochs=120, graphda_mix=.25), question='Conservative residual graph variant')
    add('graphda_homogeneous', 'graphda', dict(graphda_epochs=120, graphda_homogeneous_k=10), question='Include teacher-derived symmetric UU/II edges in joint propagation')
    add('graphda_teacher40', 'graphda', dict(graphda_epochs=40), question='Teacher convergence sensitivity against graphda_k10')
    assert len({j['name'] for j in jobs}) == len(jobs)
    return jobs


QUICK = ('reference', 'without_cl', 'ssm_base', 'ssm_propagated', 'ssm_weight_0.001',
         'ssm_positive_denominator', 'ultra_ii_0.1', 'cir_jc_0.5', 'nt_control',
         'nt_u0.5_i1.0', 'trim_0.1', 'semihard8', 'official_ultra_ii', 'graphda_residual')
