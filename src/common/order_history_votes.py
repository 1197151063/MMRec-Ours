"""Explicit numeric user-order history votes, using TRAIN edges only."""
import math
import numpy as np
import scipy.sparse as sp


def build_order_history_votes(train, bandwidth=4., radius=None, user_degree_power=0.):
    if not math.isfinite(bandwidth) or bandwidth <= 0 or not math.isfinite(user_degree_power) or user_degree_power < 0:
        raise ValueError('Positive bandwidth and nonnegative user_degree_power required')
    r = train.tocsr().astype(np.float32).copy()
    r.sum_duplicates(); r.eliminate_zeros(); r.data[:] = 1.
    n = r.shape[0]
    radius = min(n - 1, int(math.ceil(4 * bandwidth)) if radius is None else int(radius))
    if radius < 1:
        raise ValueError('Require at least two users and positive radius')
    offsets = np.arange(1, radius + 1)
    upper = sp.diags([np.full(n - d, np.exp(-d / bandwidth), dtype=np.float32) for d in offsets],
                     offsets, shape=(n, n), format='csr')
    kernel = upper + upper.T  # Zero diagonal: never vote with the target user's own history.
    degree = np.diff(r.indptr)
    weighted_history = sp.diags(np.maximum(degree, 1).astype(np.float32) ** (-user_degree_power)) @ r
    votes = (kernel @ weighted_history).tocsr()
    votes.eliminate_zeros(); votes.sort_indices()
    return votes
