"""Prioritize shared user structure; preserve REARM attention/meta/losses as controls."""
from rearm_cf_config import validate


def experiments(extended=False):
    specs = [
        ('cap_original', {}, {}, 'Historical original REARM; no default rerun'),
        ('shared_modal', dict(shared_users='modal'), {}, 'Tie modal user preferences to current TRAIN item features; keep free user ID'),
        ('shared_all', dict(shared_users='all'), {}, 'Tie both ID and modality preferences to current TRAIN item bases'),
        ('shared_all_res01', dict(shared_users='all', shared_residual=.1), {}, 'Add bounded independent residuals, norm <= 10% of shared representation'),
        ('shared_all_res01_degree', dict(shared_users='all', shared_residual=.1, shared_degree_c=5.), {}, 'Reduce residual freedom for sparse users'),
    ]
    if extended:
        specs += [
            ('shared_all_res03', dict(shared_users='all', shared_residual=.3), {}, '30% residual norm cap'),
            ('shared_all_res01_no_uu', dict(shared_users='all', shared_residual=.1), dict(n_uu_layers=0), 'Avoid additional UU smoothing while retaining shared item aggregation'),
            ('shared_all_res01_pop05', dict(shared_users='all', shared_residual=.1, shared_pop_power=.5), {}, 'Downweight popular items before row-normalized history pooling'),
        ]
    return [dict(name=name, family='shared_users', question=question,
                 config=validate(dict(options=dict(aux='official', **options), overrides=overrides)))
            for name, options, overrides, question in specs]
