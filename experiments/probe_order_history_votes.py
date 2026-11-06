#!/usr/bin/env python3
"""Direct numeric user-order votes. Select settings by validation, test winner once."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common.order_history_votes import build_order_history_votes
from utils.metrics import recall_, ndcg_
from probe_ease_retrieval import load_splits


def evaluate_votes(train, target, votes, batch=256, topk=(10, 20)):
    users = np.flatnonzero((np.diff(target.indptr) > 0) & (np.diff(train.indptr) > 0))
    if not len(users) or max(topk) > train.shape[1]:
        raise ValueError('No eligible evaluation users or too few item candidates')
    # A small TRAIN-popularity fallback breaks zero-score ties without using item index order.
    popularity = np.asarray(train.sum(0)).ravel().astype(np.float32)
    fallback = 1e-8 * popularity / max(float(popularity.max()), 1.)
    hit_parts, lengths = [], []
    for start in range(0, len(users), batch):
        selected = users[start:start + batch]
        scores = votes[selected].toarray().astype(np.float32)
        scores += fallback
        rows, items = train[selected].nonzero()
        scores[rows, items] = -1e10
        recommendations = torch.from_numpy(scores).topk(max(topk), dim=-1).indices.numpy()
        values = target[selected[:, None], recommendations]
        hit_parts.append((values.toarray() if sp.issparse(values) else np.asarray(values)) > 0)
        lengths.append(np.diff(target.indptr)[selected])
    hits, counts = np.concatenate(hit_parts), np.concatenate(lengths)
    metrics = dict(users=len(users))
    for name, function in [('recall', recall_), ('ndcg', ndcg_)]:
        values = function(hits, counts)
        metrics.update({f'{name}@{k}': float(values[k - 1]) for k in topk})
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--dataset', default='baby')
    parser.add_argument('--bandwidths', nargs='+', type=float, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument('--user-degree-powers', nargs='+', type=float, default=[0, 1])
    parser.add_argument('--output', required=True)
    parser.add_argument('--evaluate-test', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Output already exists; choose a fresh filename')
    path = Path(args.data_path) / args.dataset / f'{args.dataset}.inter'
    train, valid, test = load_splits(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(method='Explicit original-numeric-user-order TRAIN history votes',
                  source='ORIGINAL_NUMERIC_USER_ID_ORDER_AND_TRAIN_EDGES',
                  shape=list(train.shape), train_edges=train.nnz,
                  input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), candidates=[], test=None,
                  protocol='Full catalog; mask TRAIN only; macro Recall/NDCG; validation selects settings; no train+valid refit',
                  fallback='1e-8 * normalized TRAIN item popularity, fixed for all settings')
    best_score = -np.inf
    best_votes = None
    for bandwidth in args.bandwidths:
        for power in args.user_degree_powers:
            started = time.monotonic()
            print(f'Building EXPLICIT ORDER votes: bandwidth={bandwidth:g}, user_degree_power={power:g}', flush=True)
            votes = build_order_history_votes(train, bandwidth, user_degree_power=power)
            build_seconds = time.monotonic() - started
            row = dict(bandwidth=bandwidth, radius=min(train.shape[0] - 1, int(np.ceil(4 * bandwidth))),
                       user_degree_power=power, vote_nnz=votes.nnz, build_seconds=build_seconds,
                       valid=evaluate_votes(train, valid, votes), total_seconds=time.monotonic() - started)
            report['candidates'].append(row)
            if row['valid']['recall@20'] > best_score:
                best_score = row['valid']['recall@20']
                best_votes = votes
                report['selected'] = row
            del votes
            output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(row), flush=True)
    if args.evaluate_test:
        report['test'] = evaluate_votes(train, test, best_votes)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('Finished: ' + json.dumps(dict(selected=report['selected'], test=report['test'])), flush=True)


if __name__ == '__main__':
    main()
