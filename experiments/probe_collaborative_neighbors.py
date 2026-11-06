#!/usr/bin/env python3
"""TRAIN-only direct neighbor history votes; numeric ID distances never enter scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp

from probe_ease_retrieval import load_splits
from probe_order_history_votes import evaluate_votes


def similarity(train, kind='cosine'):
    r = train.tocsr().astype(np.float32).copy()
    r.sum_duplicates(); r.eliminate_zeros(); r.data[:] = 1.
    if kind == 'rare_cosine':
        degree_i = np.asarray(r.sum(0)).ravel()
        r = (r @ sp.diags(1 / np.sqrt(np.maximum(degree_i, 1)))).tocsr()
    overlap = (r @ r.T).tocoo()
    if kind in ('cosine', 'rare_cosine'):
        norm = np.sqrt(np.asarray(r.multiply(r).sum(1)).ravel())
        overlap.data /= np.maximum(norm[overlap.row] * norm[overlap.col], 1e-12)
    elif kind == 'jaccard':
        degree = np.asarray(r.sum(1)).ravel()
        overlap.data /= np.maximum(degree[overlap.row] + degree[overlap.col] - overlap.data, 1e-12)
    else:
        raise ValueError('Unknown similarity')
    out = overlap.tocsr()
    out.setdiag(0); out.eliminate_zeros(); out.sort_indices()
    return out


def neighbor_votes(train, scores, k=80, power=1., vote_degree_power=0., ties=None):
    if k < 1 or power <= 0 or vote_degree_power < 0:
        raise ValueError('Invalid neighbor setting')
    if ties is None:
        ties = np.random.default_rng(2026).random(train.shape[0])
    ties = np.asarray(ties)
    rows, columns, values = [], [], []
    for u in range(scores.shape[0]):
        a, b = scores.indptr[u:u + 2]
        ids, weights = scores.indices[a:b], scores.data[a:b]
        keep = (ids != u) & (weights > 0)
        ids, weights = ids[keep], weights[keep]
        if len(weights) > k:
            threshold = np.partition(weights, -k)[-k]
            keep = weights >= threshold
            ids, weights = ids[keep], weights[keep]
        selected = np.lexsort((ties[ids], -weights))[:k]
        ids, weights = ids[selected], weights[selected] ** power
        rows.extend([u] * len(ids)); columns.extend(ids); values.extend(weights)
    graph = sp.csr_matrix((values, (rows, columns)), shape=scores.shape, dtype=np.float32)
    degree = np.diff(train.indptr).astype(np.float32)
    history = sp.diags(np.maximum(degree, 1) ** (-vote_degree_power)) @ train
    return (graph @ history).tocsr()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--dataset', default='baby')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Use a fresh output file')
    path = Path(args.data_path) / args.dataset / f'{args.dataset}.inter'
    train, valid, _ = load_splits(path)
    report = dict(source='TRAIN_INTERACTIONS_ONLY', dataset=args.dataset, input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  note='No numeric ID ordering, timestamps, order-teacher targets or test evaluation. All candidates recorded.',
                  shape=list(train.shape), train_edges=train.nnz, candidates=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    for kind in ('cosine', 'rare_cosine', 'jaccard'):
        print('Building TRAIN similarity: ' + kind, flush=True)
        pair_scores = similarity(train, kind)
        settings = [(k, 1., 0.) for k in (20, 80, 256)]
        if kind == 'cosine':
            settings += [(k, 3., 0.) for k in (20, 80, 256)] + [(80, 1., 1.)]
        for k, power, degree_power in settings:
            started = time.monotonic()
            votes = neighbor_votes(train, pair_scores, k=k, power=power, vote_degree_power=degree_power)
            row = dict(kind=kind, k=k, similarity_power=power, vote_degree_power=degree_power,
                       vote_nnz=votes.nnz, valid=evaluate_votes(train, valid, votes), seconds=time.monotonic() - started)
            report['candidates'].append(row)
            report['selected'] = max(report['candidates'], key=lambda x:x['valid']['recall@20'])
            output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(row), flush=True)
    print('Finished: ' + json.dumps(report['selected']), flush=True)


if __name__ == '__main__':
    main()
