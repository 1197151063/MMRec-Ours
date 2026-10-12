"""Original FREEDOM with one low-rank, train-only user-cohort residual."""

import math

from common.soft_item_cohorts import build_membership, SoftItemCohort
from models.freedom import FREEDOM


class FreedomSoftCohort(FREEDOM):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.cohort_kind = config['cohort_kind']
        self.cohort_beta = float(config['cohort_beta'])
        if not math.isfinite(self.cohort_beta) or not 0 <= self.cohort_beta <= 1:
            raise ValueError('cohort_beta must be between 0 and 1')
        self.cohort = None
        if self.cohort_beta:
            if self.t_feat is None:
                raise ValueError('Soft item cohorts require item text features')
            membership = build_membership(
                dataset.inter_matrix(form='csr'),
                self.t_feat.detach().cpu().numpy(), kind=self.cohort_kind,
                count=int(config['cohort_count']), seed=int(config['cohort_seed']))
            self.cohort = SoftItemCohort(membership).to(self.device)

    def forward(self, adj):
        users, items = super().forward(adj)
        if self.cohort is not None:
            peers = self.cohort(users)
            users = users + self.cohort_beta * (peers - users)
        return users, items
