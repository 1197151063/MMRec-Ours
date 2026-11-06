#!/usr/bin/env python3
"""TRAIN-only item conditional evidence retrieval; VALID-only model selection.

No user or item ordering, timestamp, ID teacher, semantic feature, or held-out
edge enters model construction. Candidate IDs only index matrix rows/columns.
The nonlinear variants pool independent history-item evidence per candidate,
not a user-user similarity loss or embedding smoothing operation.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_ease_retrieval import load_splits
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from utils.metrics import recall_, ndcg_


def item_evidence(train, mode='cosine', shrink=0., power=.5):
    """Rows are observed history items; columns are candidate hidden anchors."""
    train = train.astype(np.float32)
    degree = np.asarray(train.sum(axis=0)).ravel()
    co = (train.T @ train).tocsr()
    co.setdiag(0); co.eliminate_zeros()
    row, col = co.nonzero()
    values = co.data.copy()
    if mode == 'cosine':
        values /= np.sqrt(np.maximum(degree[row] * degree[col], 1.))
    elif mode == 'conditional':
        # Conditional support for candidate c given history j, with optional
        # popularity correction on c. No numerical item order is involved.
        values /= np.maximum(degree[row], 1.) * np.maximum(degree[col], 1.) ** power
    elif mode == 'lift':
        values = np.log1p(values * train.shape[0] /
                          np.maximum(degree[row] * degree[col], 1.))
    else:
        raise ValueError(mode)
    if shrink:
        values *= co.data / (co.data + shrink)
    result = sp.csr_matrix((values, (row, col)), shape=co.shape)
    return result, degree


def pool_evidence(train, evidence, users, pool='sum', support_mix=0.):
    """Nonlinear row pooling. Only user histories and item evidence are read."""
    n_items = train.shape[1]
    result = np.empty((len(users), n_items), dtype=np.float32)
    for out, user in enumerate(users):
        history = train.indices[train.indptr[user]:train.indptr[user + 1]]
        rows = evidence[history].toarray()
        if not len(history):
            result[out] = 0.
            continue
        if pool == 'sum':
            score = rows.sum(axis=0)
        elif pool == 'max':
            score = rows.max(axis=0)
        elif pool in ('top2', 'top3', 'geomean2', 'and2'):
            k = min(int(pool[-1]) if pool.startswith('top') else 2, len(history))
            largest = np.partition(rows, -k, axis=0)[-k:]
            if pool.startswith('top'):
                score = largest.sum(axis=0)
            elif pool == 'geomean2':
                score = np.prod(largest, axis=0) ** (1. / k)
            else:
                score = largest.min(axis=0)
        elif pool == 'noisy_or':
            score = -np.expm1(np.log1p(-np.minimum(rows, .999)).sum(axis=0))
        else:
            raise ValueError(pool)
        if support_mix:
            score = score + support_mix * rows.sum(axis=0)
        result[out] = score
    return result


def bayes_evidence(train, concentration):
    """Multinomial candidate-anchor likelihood with popularity smoothing.

P(history j | anchor c)=(co[j,c]+alpha*p(j))/(degree[c]+alpha).
After dropping query-only constants the score is
sum_j log1p(co[j,c]/(alpha*p(j))) - |history|*log1p(degree[c]/alpha).
This is intentionally a direct test of the linear recoverability hypothesis.
"""
    degree = np.asarray(train.sum(axis=0)).ravel().astype(np.float32)
    probability = degree / max(float(degree.sum()), 1.)
    co = (train.T @ train).tocsr().astype(np.float32)
    co.setdiag(0); co.eliminate_zeros()
    row, col = co.nonzero()
    values = np.log1p(co.data / np.maximum(concentration * probability[row], 1e-12))
    return sp.csr_matrix((values, (row, col)), shape=co.shape), np.log1p(degree / concentration)


def evaluate_scores(train, valid, scorer, batch=128):
    users = np.flatnonzero((np.diff(train.indptr) > 0) & (np.diff(valid.indptr) > 0))
    hits, counts = [], []
    for start in range(0, len(users), batch):
        selected = users[start:start + batch]
        scores = scorer(selected)
        a, b = train[selected].nonzero()
        scores[a, b] = -1e10
        rec = torch.from_numpy(scores).topk(20, dim=-1).indices.numpy()
        actual = valid[selected[:, None], rec]
        hits.append((actual.toarray() if sp.issparse(actual) else np.asarray(actual)) > 0)
        counts.append(np.diff(valid.indptr)[selected])
    hit_matrix, lengths = np.concatenate(hits), np.concatenate(counts)
    metrics = {'users': int(len(users))}
    for name, function in [('recall', recall_), ('ndcg', ndcg_)]:
        values = function(hit_matrix, lengths)
        metrics.update({f'{name}@{k}': float(values[k - 1]) for k in (10, 20)})
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', required=True)
    p.add_argument('--dataset', default='baby')
    p.add_argument('--output', required=True)
    p.add_argument('--family', choices=['all', 'pooled', 'bayes'], default='all')
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        p.error('Output exists; choose a new filename')
    data = Path(args.data_path) / args.dataset / f'{args.dataset}.inter'
    train, valid, _ = load_splits(data)
    report = {'source': 'TRAIN_ONLY', 'selection': 'VALID_RECALL20_ONLY',
              'test_evaluated': False, 'dataset': args.dataset,
              'shape': list(train.shape), 'train_edges': train.nnz,
              'input_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
              'candidates': [], 'selected': None}
    output.parent.mkdir(parents=True, exist_ok=True)
    def record(config, scorer):
        start = time.monotonic()
        metrics = evaluate_scores(train, valid, scorer)
        row = dict(config=config, valid=metrics, eval_seconds=time.monotonic() - start)
        report['candidates'].append(row)
        if report['selected'] is None or metrics['recall@20'] > report['selected']['valid']['recall@20']:
            report['selected'] = row
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    if args.family in ('all', 'pooled'):
        for mode, shrink, power in [('cosine', 0., .5), ('cosine', 5., .5),
                                    ('conditional', 0., 0.), ('conditional', 0., .5),
                                    ('lift', 5., .5)]:
            evidence, degree = item_evidence(train, mode, shrink, power)
            for pool in ('sum', 'max', 'top2', 'top3', 'geomean2', 'and2'):
                config = dict(family='conditional_pool', evidence=mode, shrink=shrink,
                              popularity_power=power, pool=pool)
                record(config, lambda users, e=evidence, op=pool: pool_evidence(train, e, users, op))
    if args.family in ('all', 'bayes'):
        for concentration in (10., 100., 1000., 10000.):
            evidence, correction = bayes_evidence(train, concentration)
            for correction_weight in (0., .25, 1.):
                def scorer(users, e=evidence, c=correction, w=correction_weight):
                    score = (train[users] @ e).toarray()
                    score -= np.diff(train.indptr)[users, None] * c[None, :] * w
                    return score.astype(np.float32)
                record(dict(family='bayes_anchor', concentration=concentration,
                            missing_evidence_weight=correction_weight), scorer)
    print('VALIDATION WINNER ' + json.dumps(report['selected']), flush=True)

if __name__ == '__main__':
    main()
