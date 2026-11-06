#!/usr/bin/env python3
"""Untruncated TRAIN path votes, plus combinatorial common-history support.

Exactly eight predeclared three-hop degree-weighted variants and four common-
item-pair variants. No order features, learned ID embeddings, teachers, top-k
neighbor truncation, timestamps or test metrics. Validation selects the winner.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
from probe_ease_retrieval import load_splits
from probe_collaborative_anchors import evaluate_scores


def item_gram(train, user_degree_power=0.):
    degree = np.maximum(np.asarray(train.sum(axis=1)).ravel(), 1.)
    weighted = sp.diags(degree ** (-user_degree_power)) @ train
    return (train.T @ weighted).tocsr().astype(np.float32)


def untruncated_paths(train, users, a=0., b=0., c=0., gram=None):
    """R Di^-a R.T Du^-b R Di^-c, with no neighbor or item pruning."""
    if gram is None:
        gram = item_gram(train, b)
    degree = np.maximum(np.asarray(train.sum(axis=0)).ravel(), 1.)
    query = train[users] @ sp.diags(degree ** (-a))
    score = (query @ gram).toarray()
    return np.asarray(score * degree[None, :] ** (-c), dtype=np.float32)


def pair_support_votes(train, users, pair_weight=None):
    """sum_v binom(|H_u intersect H_v|, 2) R_vi, optionally plus m votes.

pair_weight=None gives the pair-only variant; otherwise score=s1+weight*s2,
where s1=sum_v m_uv R_vi. Repeated self history is later masked in evaluation.
"""
    overlap = (train[users] @ train.T).tocsr()
    paired = overlap.copy()
    paired.data = paired.data * (paired.data - 1.) * .5
    paired.eliminate_zeros()
    s2 = (paired @ train).toarray()
    if pair_weight is None:
        return s2.astype(np.float32)
    s1 = (overlap @ train).toarray()
    return np.asarray(s1 + pair_weight * s2, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', required=True)
    p.add_argument('--dataset', default='baby')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        p.error('Output exists; choose a fresh path')
    path = Path(args.data_path) / args.dataset / f'{args.dataset}.inter'
    train, valid, _ = load_splits(path)
    report = dict(source='TRAIN_ONLY', selection='VALID_RECALL20_ONLY',
                  test_evaluated=False, numeric_order_used=False,
                  note='Fixed 8 untruncated three-hop variants plus 4 joint support variants; all candidates retained.',
                  dataset=args.dataset, shape=list(train.shape), train_edges=train.nnz,
                  input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), candidates=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    def record(config, scorer):
        started = time.monotonic()
        row = dict(config=config, valid=evaluate_scores(train, valid, scorer, batch=128),
                   seconds=time.monotonic() - started)
        report['candidates'].append(row)
        report['selected'] = max(report['candidates'], key=lambda x:x['valid']['recall@20'])
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    for b in (0., 1.):
        gram = item_gram(train, b)
        print(json.dumps(dict(cached_item_gram_user_power=b, nnz=gram.nnz)), flush=True)
        for a in (0., 1.):
            for c in (0., .5):
                config = dict(family='untruncated_three_hop', a=a, b=b, c=c,
                              formula='R Di^-a R.T Du^-b R Di^-c')
                record(config, lambda users, a=a, b=b, c=c, g=gram:
                       untruncated_paths(train, users, a, b, c, gram=g))
    for weight in (None, .1, 1., 10.):
        config = dict(family='common_item_pair_support', pair_weight=weight,
                      formula='s2' if weight is None else 's1+lambda*s2')
        record(config, lambda users, w=weight: pair_support_votes(train, users, w))
    print('VALIDATION WINNER ' + json.dumps(report['selected']), flush=True)

if __name__ == '__main__':
    main()
