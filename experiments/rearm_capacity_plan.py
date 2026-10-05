"""Architecture-only experiment: original BPR + CL + diff throughout."""
from rearm_cf_config import validate


def experiments(extended=False):
    specs = [
        ('cap_original', {}, 'Original attention and single-layer projectors'),
        ('cap_no_attention', dict(attention='none'), 'Remove attention; retain residual normalization'),
        ('cap_mlp_item256', dict(projector_hidden=256), 'Only enlarge item visual/text projectors'),
        ('cap_no_attention_mlp_item256', dict(attention='none', projector_hidden=256),
         'Replace attention with item input->256->64 MLPs'),
    ]
    if extended:
        specs += [
            ('cap_mlp_all256', dict(projector_hidden=256, projector_scope='all'), 'Enlarge all four projectors'),
            ('cap_no_attention_mlp_all256', dict(attention='none', projector_hidden=256, projector_scope='all'),
             'Remove attention and enlarge user/item projectors'),
            ('cap_no_attention_no_meta', dict(attention='none', meta='none'),
             'Remove attention and meta; preserve third ID residual and output size'),
            ('cap_no_attention_no_meta_mlp_item256', dict(attention='none', meta='none', projector_hidden=256),
             'Remove attention/meta and enlarge item projectors'),
        ]
    return [dict(name=name, family='capacity', question=question,
                 config=validate(dict(options=dict(aux='official', **options))))
            for name, options, question in specs]
