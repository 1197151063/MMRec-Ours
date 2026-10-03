"""FreedomAlign with one frozen user-neighbor residual AFTER UI LightGCN."""
import hashlib
import math
import time
from logging import getLogger

import numpy as np
import torch

from common.user_cohort_graph import build_cohort_graph
from models.freedomalign import FreedomAlign


class FreedomAlignCohort(FreedomAlign):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.cohort_kind = config['cohort_kind']
        self.cohort_beta = float(config['cohort_beta'])
        if self.cohort_kind not in ('none', 'shared_cosine', 'shared_random', 'residual_semantic'):
            raise ValueError('Unknown cohort_kind')
        if not math.isfinite(self.cohort_beta) or self.cohort_beta < 0:
            raise ValueError('cohort_beta must be finite and nonnegative')
        self.register_buffer('cohort_adj', None)
        if self.cohort_kind == 'none' or self.cohort_beta == 0:
            getLogger().info('FreedomAlignCohort: exact FreedomAlign baseline (no user residual)')
            return
        started = time.monotonic()
        graph = build_cohort_graph(
            dataset.inter_matrix(form='csr'), self.v_feat.detach().cpu().numpy(),
            self.t_feat.detach().cpu().numpy(), self.cohort_kind,
            k=int(config['cohort_k']), dim=int(config['cohort_dim']), seed=int(config['cohort_seed']),
            progress=lambda n, total: getLogger().info('Cohort graph %s: %d/%d users', self.cohort_kind, n, total))
        degree = np.diff(graph.indptr)
        digest = hashlib.sha256()
        for array in (graph.indptr, graph.indices, graph.data):
            digest.update(array.tobytes())
        coo = graph.tocoo()
        self.cohort_adj = torch.sparse_coo_tensor(
            torch.from_numpy(np.stack((coo.row, coo.col))).long(),
            torch.from_numpy(coo.data), graph.shape).coalesce()
        getLogger().info('FreedomAlignCohort kind=%s beta=%g edges=%d mean_neighbors=%.3f '
                         'isolated_users=%d graph_seconds=%.2f graph_sha256=%s',
                         self.cohort_kind, self.cohort_beta, graph.nnz, degree.mean(),
                         np.count_nonzero(degree == 0), time.monotonic() - started, digest.hexdigest())

    def forward(self, adj=None):
        users, items = super().forward(adj)
        if self.cohort_adj is not None:
            users = users + self.cohort_beta * torch.sparse.mm(self.cohort_adj, users)
        return users, items
