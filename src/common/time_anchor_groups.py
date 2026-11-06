"""Item-time anchors formed from exactly the supplied TRAIN interaction graph."""
import numpy as np
import pandas as pd


def train_item_times(path, interactions, user_field='userID', item_field='itemID',
                     time_field='timestamp', split_field='x_label', separator='\t'):
    """Read TRAIN timestamps and restrict them to actual dataloader TRAIN keys.

    Held-out rows never contribute to first-observed times. This is an observed
    review-time statistic, not a true item release date.
    """
    r = interactions.tocsr().copy()
    r.sum_duplicates(); r.eliminate_zeros(); r.sort_indices()
    rows, cols = r.nonzero()
    keys = np.sort(rows.astype(np.int64) * r.shape[1] + cols)
    df = pd.read_csv(path, sep=separator,
                     usecols=[user_field, item_field, time_field, split_field])
    df = df.loc[df[split_field] == 0].copy()
    df = df.loc[(df[user_field] >= 0) & (df[user_field] < r.shape[0]) &
                (df[item_field] >= 0) & (df[item_field] < r.shape[1])]
    candidates = df[user_field].to_numpy(dtype=np.int64) * r.shape[1] + df[item_field].to_numpy(dtype=np.int64)
    loc = np.searchsorted(keys, candidates)
    if not len(keys):
        raise ValueError('Empty TRAIN graph for time-anchor grouping')
    keep = (loc < len(keys)) & (keys[np.minimum(loc, len(keys) - 1)] == candidates)
    df = df.loc[keep].copy()
    seen = np.unique(candidates[keep])
    if not np.array_equal(seen, keys):
        raise ValueError('Timestamp source does not cover the dataloader TRAIN graph')
    df[time_field] = pd.to_numeric(df[time_field], errors='raise')
    if not np.isfinite(df[time_field].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite TRAIN timestamps')
    return df.groupby(item_field)[time_field].min().reindex(range(r.shape[1]), fill_value=np.inf).to_numpy(dtype=float)


def build_time_anchor_groups(interactions, item_times, seed=2026, ties=None):
    """A user's anchor is its consumed item with earliest TRAIN first observation."""
    r = interactions.tocsr().copy()
    r.sum_duplicates(); r.eliminate_zeros(); r.sort_indices()
    times = np.asarray(item_times)
    if times.shape != (r.shape[1],):
        raise ValueError('Item time shape mismatch')
    keys = np.random.default_rng(seed).random(r.shape[1]) if ties is None else np.asarray(ties)
    if keys.shape != times.shape or not np.isfinite(keys).all() or len(np.unique(keys)) != len(keys):
        raise ValueError('Tie keys must be finite and unique')
    anchors = np.full(r.shape[0], -1, dtype=np.int64)
    for u in range(r.shape[0]):
        items = r.indices[r.indptr[u]:r.indptr[u + 1]]
        if len(items):
            if not np.isfinite(times[items]).all():
                raise ValueError('TRAIN item lacks a finite observed timestamp')
            anchors[u] = items[np.lexsort((keys[items], times[items]))[0]]
    labels = np.full(r.shape[0], -1, dtype=np.int64)
    active = anchors >= 0
    _, labels[active] = np.unique(anchors[active], return_inverse=True)
    return labels, anchors
