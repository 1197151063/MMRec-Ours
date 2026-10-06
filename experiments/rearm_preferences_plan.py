"""Localize useful capacity, then test a compact nonlinear multi-interest scorer."""
from rearm_cf_config import validate


def experiments(extended=False):
    clean = dict(preference_mode='residual', attention='none', meta='none', activation='gelu')
    mlp = dict(clean, projector_hidden=256)
    specs = [
        ('cap_original', {}, 'Historical original reference; opt-in rerun'),
        ('pref_freeze_users', dict(preference_mode='frozen'), 'Are independently trainable user preference tables necessary?'),
        ('pref_freeze_items', dict(freeze_item_features=True), 'Is item-feature fine-tuning necessary?'),
        ('pref_projected64', dict(preference_mode='projected'), 'Same initial forward, replace high-dimensional user tables and linear projections by direct 64-d tables'),
        ('pref_history_residual64', dict(preference_mode='residual'), 'Same initial forward, fixed history input plus learned 64-d residual'),
        ('pref_clean_linear', clean, 'Residual preferences, zero attention, no meta transform, shared GELU; retain LN/graphs/192-d output'),
        ('pref_clean_mlp256', mlp, 'Enlarge item projectors to input->256->64 on the clean scaffold'),
        ('pref_clean_mlp512', dict(clean, projector_hidden=512), 'Enlarge item projectors to input->512->64'),
    ]
    for k in (1, 2, 4, 8):
        specs.append((f'pref_interests{k}', dict(mlp, interest_heads=k),
                      f'Add {k} modality-specific user interests; candidate-dependent log-mean-exp score'))
    specs.append(('pref_interests4_w01', dict(mlp, interest_heads=4, interest_weight=.1),
                  'Four interests at 0.1 residual score weight; separate scale from head count'))
    if extended:
        specs += [
            ('pref_freeze_both', dict(preference_mode='frozen', freeze_item_features=True), 'Freeze both user history and item feature tables'),
            ('pref_residual_no_attention', dict(preference_mode='residual', attention='none'), 'Isolate attention removal on residual preference scaffold'),
            ('pref_clean_relu', dict(mlp, activation='relu'), 'Replace shared GELU by fixed ReLU; other settings unchanged'),
        ]
    return [dict(name=name, family='preferences', question=question,
                 config=validate(dict(options=dict(aux='official', **options))))
            for name, options, question in specs]
