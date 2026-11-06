#!/usr/bin/env python3
"""CPU denoising history MLP: TRAIN reconstruction, VALID-only selection.

No user lookup, numeric identifier distances, order kernels, teacher targets,
timestamps, modality features or TEST metrics. Item identifiers are categorical
vocabulary entries: consistently permuting encoder rows and decoder rows leaves
the model equivariant. The sparse EmbeddingBag is exactly a sparse Linear(I,H).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from utils.metrics import recall_, ndcg_


class HistoryMLP(nn.Module):
    def __init__(self, n_items, hidden=64, dropout=.3):
        super().__init__()
        if min(n_items, hidden) < 1 or not 0 <= dropout < 1:
            raise ValueError('Positive vocabulary/hidden size and dropout in [0,1) required')
        self.n_items, self.hidden, self.dropout = n_items, hidden, float(dropout)
        self.encoder = nn.EmbeddingBag(n_items, hidden, mode='sum', include_last_offset=True)
        self.encoder_bias = nn.Parameter(torch.zeros(hidden))
        self.decoder = nn.Linear(hidden, n_items)
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, items, offsets, values):
        # Values are the full-history L2 normalization, not the retained degree.
        values = F.dropout(values, p=self.dropout, training=self.training)
        hidden = self.encoder(items, offsets, per_sample_weights=values)
        return self.decoder(F.relu(hidden + self.encoder_bias))


def sparse_batch(history, users):
    selected = history[np.asarray(users)].tocsr()
    degree = np.diff(selected.indptr)
    normalizer = np.repeat(1. / np.sqrt(np.maximum(degree, 1)), degree).astype(np.float32)
    return (torch.from_numpy(selected.indices.astype(np.int64)),
            torch.from_numpy(selected.indptr.astype(np.int64)),
            torch.from_numpy(normalizer),
            torch.from_numpy(np.repeat(np.arange(len(users)), degree)),
            torch.from_numpy(degree.astype(np.int64)))


def multinomial_loss(logits, items, row_ids):
    log_prob = F.log_softmax(logits, dim=-1)
    return -log_prob[row_ids, items].sum() / logits.shape[0]


def load_train_valid(path):
    raw = pd.read_csv(path, sep='\t', usecols=['userID', 'itemID', 'x_label'])
    # Only vocabulary sizes use all rows. No TEST interaction enters either matrix.
    shape = (int(raw.userID.max()) + 1, int(raw.itemID.max()) + 1)
    matrices = []
    for label in (0, 1):
        part = raw.loc[raw.x_label == label]
        matrix = sp.csr_matrix((np.ones(len(part), np.float32),
                                (part.userID, part.itemID)), shape=shape)
        matrix.sum_duplicates(); matrix.eliminate_zeros(); matrix.data[:] = 1.
        matrices.append(matrix)
    if matrices[0].multiply(matrices[1]).nnz:
        raise ValueError('TRAIN and VALID interactions overlap')
    return matrices


@torch.no_grad()
def evaluate(model, train, valid, batch=512, topk=(10, 20)):
    model.eval()
    users = np.flatnonzero((np.diff(train.indptr) > 0) & (np.diff(valid.indptr) > 0))
    if not len(users) or max(topk) > train.shape[1]:
        raise ValueError('No eligible users or insufficient item vocabulary')
    hits, lengths = [], []
    for start in range(0, len(users), batch):
        selected = users[start:start + batch]
        items, offsets, values, rows, _ = sparse_batch(train, selected)
        scores = model(items, offsets, values)
        scores[rows, items] = -1.e10
        ranking = torch.topk(scores, max(topk), dim=-1).indices.numpy()
        target = valid[selected[:, None], ranking]
        hits.append((target.toarray() if sp.issparse(target) else np.asarray(target)) > 0)
        lengths.append(np.diff(valid.indptr)[selected])
    hit_matrix, counts = np.concatenate(hits), np.concatenate(lengths)
    result = dict(users=int(len(users)))
    for metric, func in [('recall', recall_), ('ndcg', ndcg_)]:
        curve = func(hit_matrix, counts)
        result.update({f'{metric}@{k}': float(curve[k - 1]) for k in topk})
    return result


def train_configuration(train, valid, dropout, hidden=64, epochs=20, batch=512,
                        seed=999, validate_every=5, patience=0, max_seconds=None,
                        progress=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = HistoryMLP(train.shape[1], hidden=hidden, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1.e-4)
    users = np.flatnonzero(np.diff(train.indptr) > 0)
    started = time.monotonic()
    result = dict(dropout=dropout, hidden=hidden, batch_size=batch, seed=seed,
                  optimizer='Adam', learning_rate=.001, weight_decay=.0001,
                  maximum_epochs=epochs, validation_interval=validate_every,
                  early_stopping_patience=patience,
                  trainable_parameters=sum(p.numel() for p in model.parameters()),
                  train_curve=[], valid_curve=[], selected=None, stopping_reason=None)
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = rng.permutation(users)
        loss_sum = 0.
        epoch_started = time.monotonic()
        for start in range(0, len(permutation), batch):
            selected = permutation[start:start + batch]
            items, offsets, values, rows, _ = sparse_batch(train, selected)
            opt.zero_grad(set_to_none=True)
            loss = multinomial_loss(model(items, offsets, values), items, rows)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite TRAIN multinomial loss')
            loss.backward(); opt.step()
            loss_sum += float(loss.detach()) * len(selected)
        row = dict(epoch=epoch, train_loss=loss_sum / len(users),
                   train_seconds=time.monotonic() - epoch_started)
        result['train_curve'].append(row)
        print(f'dropout={dropout:g} epoch={epoch} loss={row["train_loss"]:.6f} '
              f'train_seconds={row["train_seconds"]:.3f}', flush=True)
        if epoch % validate_every == 0 or epoch == epochs:
            eval_started = time.monotonic()
            metrics = evaluate(model, train, valid, batch=batch)
            validation = dict(epoch=epoch, valid=metrics,
                              evaluation_seconds=time.monotonic() - eval_started,
                              elapsed_seconds=time.monotonic() - started)
            result['valid_curve'].append(validation)
            if (result['selected'] is None or metrics['recall@20'] >
                    result['selected']['valid']['recall@20']):
                result['selected'] = validation
                stale = 0
            else:
                stale += 1
            if patience and stale >= patience:
                result['stopping_reason'] = 'VALID_Recall20_patience'
            print(json.dumps(dict(dropout=dropout, **validation)), flush=True)
        result['seconds'] = time.monotonic() - started
        if max_seconds is not None and result['seconds'] >= max_seconds:
            result['stopping_reason'] = 'wall_time_budget'
        if progress:
            progress(result)
        if result['stopping_reason'] is not None:
            break
    if result['stopping_reason'] is None:
        result['stopping_reason'] = 'maximum_epochs'
    result['completed_epochs'] = result['train_curve'][-1]['epoch']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--dataset', default='baby')
    parser.add_argument('--output', required=True)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--dropout', nargs='+', type=float, default=[.3, .6])
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--patience', type=int, default=0,
                        help='Consecutive VALID evaluations without Recall@20 improvement; 0 disables')
    parser.add_argument('--max-seconds', type=float, default=None,
                        help='Optional global wall-time budget after data loading')
    args = parser.parse_args()
    if min(args.hidden, args.epochs, args.batch_size, args.threads) < 1:
        parser.error('Dimensions and counts must be positive')
    output = Path(args.output)
    if output.exists():
        parser.error('Output exists; choose a new filename')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    path = Path(args.data_path) / args.dataset / f'{args.dataset}.inter'
    train, valid = load_train_valid(path)
    report = dict(method='Denoising shared-item categorical history MLP', source='TRAIN_ONLY',
                  vocabulary='Item category lookup only; no numeric order, distance or user parameters',
                  dataset=args.dataset, shape=list(train.shape), train_edges=int(train.nnz),
                  valid_edges=int(valid.nnz),
                  input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  arguments=vars(args),
                  protocol='L2 normalized TRAIN history + inverted input dropout; reconstruct all TRAIN '
                           'positives by multinomial negative log likelihood; full catalog VALID; mask '
                           'TRAIN only; macro Recall/NDCG; select by VALID Recall@20; no TEST evaluation',
                  configurations=[], selected=None)
    run_started = time.monotonic()
    def save():
        temp = output.with_suffix(output.suffix + '.tmp')
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temp.replace(output)
    for dropout in args.dropout:
        remaining = None if args.max_seconds is None else args.max_seconds - (time.monotonic() - run_started)
        if remaining is not None and remaining <= 0:
            report['budget_exhausted'] = True
            break
        current = len(report['configurations'])
        report['configurations'].append({})
        def progress(row):
            report['configurations'][current] = row
            save()
        result = train_configuration(train, valid, dropout=dropout, hidden=args.hidden,
                                     epochs=args.epochs, batch=args.batch_size, seed=args.seed,
                                     patience=args.patience, max_seconds=remaining,
                                     progress=progress)
        report['configurations'][current] = result
        if result['selected'] is not None and (report['selected'] is None or
                result['selected']['valid']['recall@20'] > report['selected']['valid']['recall@20']):
            report['selected'] = dict(dropout=dropout, hidden=args.hidden,
                                      **result['selected'])
        save()
    report['completed'] = len(report['configurations']) == len(args.dropout)
    report['has_validated_winner'] = report['selected'] is not None
    report['seconds'] = time.monotonic() - run_started
    save()
    print('Finished: ' + str(output), flush=True)


if __name__ == '__main__':
    main()
