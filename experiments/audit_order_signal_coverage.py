#!/usr/bin/env python3
"""POST-HOC audit, not a recommender: which validation order-vote hits have TRAIN paths?

Order labels are isolated diagnostics. They do not build or fit collaborative candidates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common.order_history_votes import build_order_history_votes
from probe_ease_retrieval import load_splits


def audit(train, valid, bandwidth=16., radius=64, batch=256):
    teacher = build_order_history_votes(train, bandwidth=bandwidth, radius=radius)
    gram = (train.T @ train).tocsr()
    users = np.flatnonzero((np.diff(train.indptr) > 0) & (np.diff(valid.indptr) > 0))
    popularity = np.asarray(train.sum(0)).ravel().astype(np.float32)
    fallback = 1e-8 * popularity / max(float(popularity.max()), 1.)
    counts = dict(valid_edges=0, reachable_threehop_edges=0, order_hit_edges=0,
                  order_hit_reachable_edges=0, order_hit_no_threehop_edges=0)
    degree_stats = {name: [] for name in ('order_hit_reachable', 'order_hit_no_threehop')}
    degree = np.diff(train.indptr)
    for start in range(0, len(users), batch):
        selected = users[start:start + batch]
        scores = teacher[selected].toarray() + fallback
        a, b = train[selected].nonzero(); scores[a, b] = -1e10
        top = torch.from_numpy(scores).topk(min(20, train.shape[1]), dim=-1).indices.numpy()
        targets = valid[selected].tocoo()
        paths = (train[selected] @ gram).tocsr()
        reachable = np.asarray(paths[targets.row, targets.col]).reshape(-1) > 0
        hit = np.any(top[targets.row] == targets.col[:, None], axis=1)
        counts['valid_edges'] += len(targets.data)
        counts['reachable_threehop_edges'] += int(reachable.sum())
        counts['order_hit_edges'] += int(hit.sum())
        counts['order_hit_reachable_edges'] += int((hit & reachable).sum())
        counts['order_hit_no_threehop_edges'] += int((hit & ~reachable).sum())
        for name, mask in [('order_hit_reachable', hit & reachable), ('order_hit_no_threehop', hit & ~reachable)]:
            degree_stats[name].extend(degree[selected[targets.row[mask]]].tolist())
    fractions = dict(threehop_coverage=counts['reachable_threehop_edges'] / max(counts['valid_edges'], 1),
                     order_hit_without_threehop_fraction=counts['order_hit_no_threehop_edges'] / max(counts['order_hit_edges'], 1))
    return dict(counts=counts, fractions=fractions,
                user_degree_quantiles={key:np.quantile(value, [0, .5, .9, 1]).tolist() if value else None
                                       for key, value in degree_stats.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--dataset', default='baby')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Choose a fresh output file')
    path = Path(args.data_path) / args.dataset / f'{args.dataset}.inter'
    train, valid, _ = load_splits(path)
    result = audit(train, valid)
    result.update(role='POST_HOC_ORDER_DIAGNOSTIC_ONLY_NOT_A_MODEL_OR_TRAINING_SIGNAL',
                  input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  note='No test labels evaluated. No candidate tuning by order-hit subsets. '
                       'No-three-hop does not prove impossibility for all higher-order or semantic models.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
