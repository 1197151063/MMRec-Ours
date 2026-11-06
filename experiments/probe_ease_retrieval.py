#!/usr/bin/env python3
"""TRAIN-only EASE retrieval screen; select on validation before any optional test evaluation.

Reference: Steck, Embarrassingly Shallow Autoencoders for Sparse Data, WWW 2019.
This is an established CF method, not a new architecture or a PE reconstruction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from utils.metrics import recall_, ndcg_


def fit_ease(train, ridge):
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('ridge must be finite and positive')
    x = train.astype(np.float64)
    gram = (x.T @ x).toarray(order='F')
    gram.flat[::len(gram) + 1] += ridge
    factor, lower = la.cho_factor(gram, overwrite_a=True, check_finite=False)
    precision = la.cho_solve((factor, lower), np.eye(len(gram), order='F'),
                             overwrite_b=True, check_finite=False)
    precision /= -precision.diagonal().copy()[None, :]
    np.fill_diagonal(precision, 0.)
    return precision.astype(np.float32)


def evaluate(train, target, coefficients, batch=256, topk=(10, 20)):
    active_train = np.diff(train.indptr) > 0
    users = np.flatnonzero((np.diff(target.indptr) > 0) & active_train)
    if not len(users) or max(topk) > train.shape[1]:
        raise ValueError('No eligible evaluation users or too few item candidates')
    hits = []
    counts = []
    for start in range(0, len(users), batch):
        selected = users[start:start + batch]
        history = train[selected]
        scores = np.asarray(history @ coefficients, dtype=np.float32)
        rows, items = history.nonzero()
        scores[rows, items] = -1e10
        recommendations = torch.from_numpy(scores).topk(max(topk), dim=-1).indices.numpy()
        selected_targets = target[selected[:, None], recommendations]
        hits.append((selected_targets.toarray() if sp.issparse(selected_targets)
                     else np.asarray(selected_targets)) > 0)
        counts.append(np.diff(target.indptr)[selected])
    hit_matrix = np.concatenate(hits)
    lengths = np.concatenate(counts)
    metrics = {}
    for name, function in [('recall', recall_), ('ndcg', ndcg_)]:
        values = function(hit_matrix, lengths)
        metrics.update({f'{name}@{k}': float(values[k - 1]) for k in topk})
    return dict(users=len(users), **metrics)


def load_splits(path):
    df = pd.read_csv(path, sep='\t', usecols=['userID', 'itemID', 'x_label'])
    shape = (int(df.userID.max()) + 1, int(df.itemID.max()) + 1)
    matrices = []
    for label in range(3):
        part = df[df.x_label == label]
        r = sp.csr_matrix((np.ones(len(part), dtype=np.float32), (part.userID, part.itemID)), shape=shape)
        if r.nnz != len(part):
            raise ValueError('Duplicate interactions within a split require explicit evaluation handling')
        matrices.append(r)
    if matrices[0].multiply(matrices[1] + matrices[2]).nnz:
        raise ValueError('TRAIN and held-out edges overlap')
    return matrices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--dataset', default='baby')
    parser.add_argument('--ridge', nargs='+', type=float, default=[10., 100., 1000.])
    parser.add_argument('--output', required=True)
    parser.add_argument('--evaluate-test', action='store_true', help='Evaluate ONLY the validation winner on test')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Output already exists; use a new filename')
    path = Path(args.data_path) / args.dataset / f'{args.dataset}.inter'
    train, valid, test = load_splits(path)
    if train.shape[1] > 10000:
        parser.error('Dense EASE screen restricted to <=10000 items for local memory use')
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(method='EASE (Steck, WWW 2019)', source='TRAIN_ONLY', dataset=args.dataset,
                  train_edges=train.nnz, shape=list(train.shape), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  protocol='Full catalog; mask TRAIN only; macro user Recall/NDCG; validation selects ridge; no train+valid refit',
                  candidates=[], test=None)
    best_score = -np.inf
    best_coefficients = None
    for ridge in args.ridge:
        started = time.monotonic()
        print(f'Fitting TRAIN-only EASE ridge={ridge:g}', flush=True)
        coefficients = fit_ease(train, ridge)
        fit_seconds = time.monotonic() - started
        metrics = evaluate(train, valid, coefficients)
        row = dict(ridge=ridge, fit_seconds=fit_seconds, valid=metrics,
                   total_seconds=time.monotonic() - started)
        report['candidates'].append(row)
        if metrics['recall@20'] > best_score:
            best_score = metrics['recall@20']
            best_coefficients = coefficients
            report['selected'] = row
        del coefficients
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    if args.evaluate_test:
        report['test'] = evaluate(train, test, best_coefficients)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('Finished; validation winner: ' + json.dumps(report['selected']), flush=True)
    if report['test'] is not None:
        print('Selected-model test: ' + json.dumps(report['test']), flush=True)


if __name__ == '__main__':
    main()
