#!/usr/bin/env python3
"""Run the pinned official REARM, with its own model, losses and training loop."""
import argparse
import csv
import hashlib
import math
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'third_party/rearm'
REVISION = '3038f1ac3b7a957f140550bfecd34b09688a689b'
PRESETS = {
    'baby': dict(num_layer=4, reg_weight=.0005, rank=3, s_drop=.4, m_drop=.6,
                 u_mm_image_weight=.2, i_mm_image_weight=0., uu_co_weight=.4, ii_co_weight=.2,
                 user_knn_k=40, item_knn_k=10, n_ii_layers=1, n_uu_layers=1,
                 cl_tmp=.6, cl_loss_weight=5e-6, diff_loss_weight=5e-5),
    'sports': dict(num_layer=5, reg_weight=.05, rank=7, s_drop=1., m_drop=.2,
                   u_mm_image_weight=0., i_mm_image_weight=.2, uu_co_weight=.9, ii_co_weight=.2,
                   user_knn_k=25, item_knn_k=5, n_ii_layers=2, n_uu_layers=2,
                   cl_tmp=1.5, cl_loss_weight=1e-3, diff_loss_weight=5e-4),
    'clothing': dict(num_layer=4, reg_weight=.00001, rank=3, s_drop=.4, m_drop=.1,
                     u_mm_image_weight=.1, i_mm_image_weight=.1, uu_co_weight=.7, ii_co_weight=.1,
                     user_knn_k=45, item_knn_k=10, n_ii_layers=1, n_uu_layers=1,
                     cl_tmp=.03, cl_loss_weight=1e-6, diff_loss_weight=1e-5),
}


def write_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare_data(source, target, dataset):
    import numpy as np
    target.mkdir(parents=True)
    inter = source / (dataset + '.inter')
    inputs = []
    if inter.exists():
        splits = [[], [], []]
        with inter.open() as stream:
            for row in csv.DictReader(stream, delimiter='\t'):
                label = int(row['x_label'])
                if label not in (0, 1, 2):
                    raise ValueError('Expected x_label in {0,1,2}')
                splits[label].append((int(row['userID']), int(row['itemID'])))
        arrays = [np.asarray(rows, dtype=np.int64).reshape(-1, 2).T for rows in splits]
        inputs.append(inter)
        # Match this repository's evaluation cold-user filtering; never reindex IDs.
        train_users = np.unique(arrays[0][0])
        dropped = []
        for j in (1, 2):
            keep = np.isin(arrays[j][0], train_users)
            dropped.append(int((~keep).sum()))
            arrays[j] = arrays[j][:, keep]
    else:
        inputs += [source / (name + '.npy') for name in ('train', 'valid', 'test')]
        arrays = [np.load(p, allow_pickle=False) for p in inputs]
        dropped = [0, 0]
    for a in arrays:
        if a.ndim != 2 or a.shape[0] != 2 or not a.shape[1] or a.dtype.kind not in 'iu' or (a < 0).any():
            raise ValueError('Each split must be a nonempty integer [2, interactions] array')
    all_edges = np.concatenate(arrays, axis=1)
    users, items = [np.unique(row) for row in all_edges]
    for ids in (users, items):
        if not np.array_equal(ids, np.arange(len(ids))):
            raise ValueError('Official REARM requires contiguous IDs; refusing implicit reindexing')
    if len(np.unique(arrays[0][0])) != len(users):
        raise ValueError('Official user interest averaging requires training history for every user')
    sets = [set(map(tuple, a.T.tolist())) for a in arrays]
    if any(len(s) != a.shape[1] for s, a in zip(sets, arrays)):
        raise ValueError('Duplicate interactions found; refusing silent preprocessing changes')
    if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Train/validation/test overlap detected')
    for name, a in zip(('train', 'valid', 'test'), arrays):
        np.save(target / (name + '.npy'), a)
    for name in ('image_feat.npy', 'text_feat.npy'):
        p = source / name
        x = np.load(p, mmap_mode='r', allow_pickle=False)
        if x.ndim != 2 or x.shape[0] != len(items) or not np.isfinite(x).all():
            raise ValueError('Feature row count/values invalid: ' + str(p))
        (target / name).symlink_to(p.resolve())
        inputs.append(p)
    return dict(users=len(users), items=len(items), interactions=[a.shape[1] for a in arrays],
                dropped_cold_eval=dropped, inputs={str(p): sha(p) for p in inputs})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', required=True, help='Root containing baby/sports/clothing')
    parser.add_argument('--dataset', choices=PRESETS, default='baby')
    parser.add_argument('--output', default='night_runs/baby-rearm-official-1')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--graph-block', type=int, default=256)
    parser.add_argument('--negative-scope', choices=['official', 'train'], default='official')
    parser.add_argument('--item-mode', choices=['original', 'none', 'alignment'], default='original')
    parser.add_argument('--item-alignment-weight', type=float, default=0.1)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.graph_block < 1 or args.num_workers < 0:
        parser.error('Invalid epochs, graph-block or num-workers')
    if not math.isfinite(args.item_alignment_weight) or args.item_alignment_weight < 0:
        parser.error('item-alignment-weight must be finite and nonnegative')
    variant = dict(item_mode=args.item_mode, item_alignment_weight=args.item_alignment_weight if args.item_mode == 'alignment' else 0.0)
    config = dict(PRESETS[args.dataset], dataset=args.dataset, seed=args.seed,
                  num_epoch=args.epochs, num_workers=args.num_workers, gpu_id=args.gpu_id,
                  use_gpu=not args.cpu)
    if args.dry_run:
        print(json.dumps(dict(preset=config, negative_scope=args.negative_scope, variant=variant, upstream=REVISION), indent=2))
        return
    output = Path(args.output).resolve()
    source = Path(args.data_path).resolve() / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / '.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another REARM run owns this output directory')
    if (output / 'status.json').exists():
        parser.error('Output already used; choose a new output directory')
    started = time.time()
    write_json(output / 'status.json', dict(state='preparing', started=started, pid=os.getpid()))
    try:
        data_info = prepare_data(source, output / 'data' / args.dataset, args.dataset)
        if data_info['users'] < config['user_knn_k'] or data_info['items'] < max(20, config['item_knn_k']):
            raise ValueError('Dataset too small for the official top-k configuration')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        sys.path.insert(0, str(UPSTREAM))
        from utils import helper
        from rearm_runtime import install, patch_loader
        install(helper, args.graph_block)
        from utils import data_loader
        patch_loader(data_loader, args.negative_scope)
        from utils.parser import parse_args
        original_argv = sys.argv
        sys.argv = [sys.argv[0]]
        official_args = parse_args()
        sys.argv = original_argv
        for key, value in config.items():
            setattr(official_args, key, value)
        import main as official
        import torch
        if args.item_mode != 'original':
            from rearm_item_loss import REARMItemLoss
            official.REARM = lambda config, dataset: REARMItemLoss(config, dataset, weight=variant['item_alignment_weight'])
        manifest = dict(upstream_revision=REVISION, arguments=vars(official_args),
                        negative_scope=args.negative_scope, variant=variant, data=data_info,
                        torch_version=torch.__version__, python=sys.version,
                        source_sha256={str(p.relative_to(ROOT)): sha(p) for folder in (UPSTREAM, ROOT / 'experiments')
                                       for p in folder.rglob('*.py')})
        write_json(output / 'manifest.json', manifest)
        # Official relative paths are isolated per run. No graph/checkpoint saves.
        os.chdir(output)
        print('REARM variant=' + str(variant) + '; negative_scope=' + args.negative_scope, flush=True)
        print('Official scope uses held-out positives to exclude negatives and for fallback user popularity.', flush=True)
        original_update = official.update_result
        current = {'epoch': None, 'valid': None}
        original_evaluate = official.evaluate_model
        def evaluate(net, epoch, data, t_or_v):
            result = original_evaluate(net, epoch, data, t_or_v)
            if t_or_v == 'valid':
                current.update(epoch=epoch, valid=result[1])
            return result
        def update(net, result):
            original_update(net, result)
            record = dict(model='REARM', dataset=args.dataset, best_epoch=current['epoch'],
                          valid=current['valid'], test=result, negative_scope=args.negative_scope, **variant)
            write_json(output / 'result.json', record)
            row = dict(best_epoch=record['best_epoch'], negative_scope=args.negative_scope, **variant,
                       **{'valid_' + k: v for k, v in record['valid'].items()},
                       **{'test_' + k: v for k, v in result.items()})
            with (output / 'summary.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
        official.evaluate_model = evaluate
        official.update_result = update
        original_train = official.train
        def train(net, epoch):
            if hasattr(net.model, 'reset_alignment_stats'):
                net.model.reset_alignment_stats()
            losses = original_train(net, epoch)
            if any(not torch.isfinite(torch.as_tensor(loss)).all() for loss in losses):
                raise FloatingPointError('Non-finite training loss')
            if getattr(net.model, 'alignment_batches', 0):
                raw = net.model.alignment_sum / net.model.alignment_batches
                net.logger.info('II alignment: raw=%.6f weighted=%.6f', raw, raw * net.model.item_alignment_weight)
            return losses
        official.train = train
        net = official.Net(official_args)
        if not args.cpu and net.device.type != 'cuda':
            raise RuntimeError('CUDA unavailable; use --cpu only for a deliberate CPU run')
        write_json(output / 'status.json', dict(state='running', pid=os.getpid(), started=started))
        net.run()
        if not (output / 'result.json').exists():
            raise RuntimeError('No valid best-epoch result was produced')
        write_json(output / 'status.json', dict(state='complete', started=started, seconds=time.time()-started))
        print('FINAL RESULT: ' + (output / 'result.json').read_text(), flush=True)
    except BaseException as error:
        write_json(output / 'status.json', dict(state='failed', error=repr(error), seconds=time.time()-started))
        raise


if __name__ == '__main__':
    main()
