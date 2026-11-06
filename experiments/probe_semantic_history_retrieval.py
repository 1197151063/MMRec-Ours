#!/usr/bin/env python3
"""Bounded TRAIN-only, item-conditioned semantic history retrieval screen.

Unlike a mean user profile, each candidate receives the maximum (or sum) of
individual historical-item contributions. Item cosine neighbors may be gated
by TRAIN co-consumer evidence. Numeric ID order, timestamps, order teachers,
and held-out labels never enter construction. Validation is evaluated only
after a score matrix has been constructed; no test evaluation is implemented.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from probe_ease_retrieval import load_splits
from probe_order_history_votes import evaluate_votes


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def semantic_vectors(visual, text, centered=False):
    """Equal-size modality contributions are weighted .1 visual / .9 text.

    Centering each feature dimension removes a catalog-wide common component;
    neither TRAIN user aggregation nor any held-out targets are needed.
    """
    visual, text = unit(visual), unit(text)
    if centered:
        visual = unit(visual - visual.mean(axis=0, keepdims=True))
        text = unit(text - text.mean(axis=0, keepdims=True))
    return np.concatenate((np.sqrt(.1) * visual, np.sqrt(.9) * text), axis=1)


def semantic_neighbors(vectors, k=80, block=128, ties=None):
    """Keep positive cosine top-k, excluding self; retain O(items*k) memory."""
    count = len(vectors)
    if not 0 < k < count or block < 1:
        raise ValueError('Require 0 < k < item count and a positive block')
    ties = np.random.default_rng(2026).random(count) if ties is None else np.asarray(ties)
    rows, cols, vals = [], [], []
    for start in range(0, count, block):
        similarities = vectors[start:start + block] @ vectors.T
        for local, scores in enumerate(similarities):
            item = start + local
            scores[item] = -np.inf
            candidates = np.flatnonzero(scores > 0)
            if len(candidates) > k:
                threshold = np.partition(scores[candidates], -k)[-k]
                candidates = candidates[scores[candidates] >= threshold]
            selected = candidates[np.lexsort((ties[candidates], -scores[candidates]))[:k]]
            rows.extend([item] * len(selected)); cols.extend(selected.tolist())
            vals.extend(scores[selected].tolist())
    return sp.csr_matrix((vals, (rows, cols)), shape=(count, count), dtype=np.float32)


def collaborative_edges(train):
    """RA-style item co-consumption; a high-degree user contributes less."""
    degree_u = np.diff(train.indptr).astype(np.float32)
    evidence = (train.T @ sp.diags(1 / np.maximum(degree_u, 1)) @ train).tocsr()
    evidence.setdiag(0); evidence.eliminate_zeros()
    degree_i = np.asarray(train.sum(0)).ravel().astype(np.float32)
    return evidence, degree_i


def edge_weights(neighbors, evidence, degree_i, gated=False):
    """Cubic semantic concentration, with optional bounded CF reliability.

    A positive floor preserves semantically plausible edges with no observed
    co-consumer. This is a fixed hypothesis, not tuned on evaluation labels.
    """
    edges = neighbors.tocoo()
    weights = edges.data ** 3
    if gated:
        observed = np.asarray(evidence[edges.row, edges.col]).ravel()
        weights *= .05 + .95 * observed / (observed + .25)
    weights /= np.maximum(degree_i[edges.col], 1) ** .25
    return sp.csr_matrix((weights, (edges.row, edges.col)), shape=neighbors.shape)


def history_scores(train, edges, aggregation='sum'):
    if aggregation == 'sum':
        return (train @ edges).tocsr()
    if aggregation != 'max':
        raise ValueError('Unknown aggregation')
    row_parts, col_parts, value_parts = [], [], []
    for user in range(train.shape[0]):
        history = train.indices[train.indptr[user]:train.indptr[user + 1]]
        contributions = edges[history].tocoo()
        if not contributions.nnz:
            continue
        candidates, inverse = np.unique(contributions.col, return_inverse=True)
        maxima = np.zeros(len(candidates), dtype=np.float32)
        np.maximum.at(maxima, inverse, contributions.data)
        row_parts.extend([user] * len(candidates)); col_parts.extend(candidates.tolist())
        value_parts.extend(maxima.tolist())
    return sp.csr_matrix((value_parts, (row_parts, col_parts)), shape=train.shape, dtype=np.float32)


def row_max_normalize(scores):
    scores = scores.tocsr()
    maxima = scores.max(axis=1).toarray().ravel()
    return (sp.diags(1 / np.maximum(maxima, 1e-12)) @ scores).tocsr()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--dataset', default='baby')
    parser.add_argument('--output', required=True)
    parser.add_argument('--neighbors', type=int, default=80)
    parser.add_argument('--block', type=int, default=128)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Output already exists; choose a new filename')
    folder = Path(args.data_path) / args.dataset
    paths = [folder / f'{args.dataset}.inter', folder / 'image_feat.npy', folder / 'text_feat.npy']
    train, valid, _ = load_splits(paths[0])
    visual = np.load(paths[1], allow_pickle=False).astype(np.float32)
    text = np.load(paths[2], allow_pickle=False).astype(np.float32)
    if visual.shape[0] != train.shape[1] or text.shape[0] != train.shape[1]:
        raise ValueError('Feature/catalog shape mismatch')
    if not np.isfinite(visual).all() or not np.isfinite(text).all():
        raise ValueError('Nonfinite item features')
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(method='Item-conditioned multi-interest semantic history retrieval',
                  source='TRAIN_INTERACTIONS_AND_RAW_PRETRAINED_ITEM_FEATURES_ONLY',
                  protocol='Full catalog; mask TRAIN only; validation only; no test evaluation; no numeric ID order or teacher',
                  arguments=vars(args), shape=list(train.shape), train_edges=train.nnz,
                  input_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                  predeclared_candidates=12, candidates=[], selected=None, test=None,
                  fixed_parameters=dict(visual_weight=.1, text_weight=.9, semantic_power=3,
                                        candidate_degree_power=.25, reliability_floor=.05,
                                        reliability_shrinkage=.25, hybrid_semantic_weights=[.1, .3]))
    evidence, degree_i = collaborative_edges(train)
    cf_edges = sp.diags(1 / np.maximum(degree_i, 1) ** .5) @ evidence @ sp.diags(1 / np.maximum(degree_i, 1) ** .5)
    cf_scores = row_max_normalize(history_scores(train, cf_edges))
    report['cf_reference'] = dict(name='TRAIN_RA_cosine_full_support',
                                  valid=evaluate_votes(train, valid, cf_scores))
    print('Fixed collaborative reference: ' + json.dumps(report['cf_reference']), flush=True)
    centered_scores = {}
    started = time.monotonic()
    def record(name, scores, parameters):
        row = dict(name=name, parameters=parameters, vote_nnz=scores.nnz,
                   valid=evaluate_votes(train, valid, scores), elapsed_seconds=time.monotonic() - started)
        report['candidates'].append(row)
        if report['selected'] is None or row['valid']['recall@20'] > report['selected']['valid']['recall@20']:
            report['selected'] = row
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    for centered in (False, True):
        geometry = 'centered' if centered else 'raw'
        print('Building ' + geometry + ' item-semantic neighbors', flush=True)
        vectors = semantic_vectors(visual, text, centered)
        neighbors = semantic_neighbors(vectors, args.neighbors, args.block)
        del vectors
        for gated in (False, True):
            edges = edge_weights(neighbors, evidence, degree_i, gated)
            for aggregation in ('sum', 'max'):
                scores = history_scores(train, edges, aggregation)
                record(f'{geometry}_{aggregation}_' + ('cf_gate' if gated else 'semantic'), scores,
                       dict(centered=centered, aggregation=aggregation, cf_reliability_gate=gated))
                if centered and not gated:
                    centered_scores[aggregation] = row_max_normalize(scores)
    for aggregation in ('sum', 'max'):
        for semantic_weight in (.1, .3):
            scores = (cf_scores + semantic_weight * centered_scores[aggregation]).tocsr()
            record(f'cf_plus_centered_{aggregation}_w{semantic_weight:g}', scores,
                   dict(aggregation=aggregation, semantic_weight=semantic_weight,
                        cf='TRAIN item co-consumption / sqrt(candidate and source degrees)', row_scale='max'))
    report['total_seconds'] = time.monotonic() - started
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('Finished: ' + json.dumps(report['selected']), flush=True)


if __name__ == '__main__':
    main()
