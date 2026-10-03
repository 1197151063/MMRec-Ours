#!/usr/bin/env python3
"""Sequential train-only REARM relation ablations; no model checkpoints."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_rearm import ROOT, PRESETS, write_json, sha

MODES = ('original', 'merged_single', 'merged_matched', 'typed_fixed', 'typed_global', 'typed_user')


def collect(output, jobs):
    rows = []
    for job in jobs:
        folder = output / job['name']
        status = json.loads((folder / 'status.json').read_text()) if (folder / 'status.json').exists() else {}
        row = dict(name=job['name'], relation_mode=job['mode'], seed=job['seed'],
                   state=status.get('state', 'pending'), seconds=status.get('seconds'),
                   best_epoch=None, parameters=None, error=status.get('error', ''))
        if (folder / 'result.json').exists():
            result = json.loads((folder / 'result.json').read_text())
            row.update(best_epoch=result['best_epoch'], parameters=result.get('parameters'))
            for split in ('valid', 'test'):
                row.update({split + '_' + key: value for key, value in result[split].items()})
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / 'summary.csv.tmp').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    (output / 'summary.csv.tmp').replace(output / 'summary.csv')
    # Summarize only completed jobs. Do not choose configurations on test metrics.
    aggregate = {}
    for mode in MODES:
        done = [r for r in rows if r['relation_mode'] == mode and r['state'] == 'complete']
        if not done:
            continue
        metrics = {}
        for key in done[0]:
            if key.startswith(('valid_', 'test_')):
                values = [r[key] for r in done]
                mean = sum(values) / len(values)
                std = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** .5 if len(values) > 1 else None
                metrics[key] = dict(mean=mean, sample_std=std)
        aggregate[mode] = dict(completed_seeds=[r['seed'] for r in done], metrics=metrics)
    write_json(output / 'summary.json', dict(runs=rows, aggregate=aggregate,
                                           selection_metric='valid_Recall@20'))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--dataset', choices=PRESETS, default='baby')
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2025])
    parser.add_argument('--variants', choices=MODES, nargs='+', default=list(MODES))
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--graph-block', type=int, default=256)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Skip complete runs; refuse failed/interrupted run directories')
    args = parser.parse_args()
    if args.epochs < 1 or args.num_workers < 0 or args.graph_block < 1:
        parser.error('Invalid epochs, workers or graph block')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        parser.error('Duplicate seeds or variants')
    output = Path(args.output).resolve()
    source = Path(args.data_path).resolve()
    jobs = []
    for seed in args.seeds:
        for mode in args.variants:
            name = f'{mode}_s{seed}'
            command = [sys.executable, '-u', str(ROOT / 'experiments/run_rearm.py'),
                       '--data-path', str(source), '--dataset', args.dataset,
                       '--output', str(output / name), '--gpu-id', str(args.gpu_id),
                       '--seed', str(seed), '--epochs', str(args.epochs),
                       '--num-workers', str(args.num_workers), '--graph-block', str(args.graph_block),
                       '--negative-scope', 'train', '--relation-mode', mode]
            if args.cpu:
                command.append('--cpu')
            jobs.append(dict(name=name, seed=seed, mode=mode, command=command))
    manifest = dict(jobs=jobs, protocol='train-only; best validation Recall@20; no checkpoints',
                    dataset=args.dataset, data_path=str(source),
                    source_sha256={str(p.relative_to(ROOT)): sha(p)
                                   for p in list((ROOT / 'third_party/rearm').rglob('*.py')) +
                                   [ROOT / 'experiments' / name for name in
                                    ('run_rearm.py', 'rearm_runtime.py', 'rearm_relations.py', 'run_rearm_relations.py')]})
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.queue.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another relation queue owns this output directory')
    if (output / 'manifest.json').exists():
        if not args.resume:
            parser.error('Queue output already used; use a new directory or --resume')
        if json.loads((output / 'manifest.json').read_text()) != manifest:
            parser.error('Resume plan/source changed; use a new directory')
    else:
        if any((output / job['name']).exists() for job in jobs):
            parser.error('Existing child output without queue manifest; choose a new directory')
        write_json(output / 'manifest.json', manifest)
    started = time.time()
    child = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        # Fail before expensive graph construction if dependencies/device are missing.
        import torch
        import tensorboardX  # noqa: F401
        if not args.cpu and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use --cpu only for a deliberate CPU run')
        for idx, job in enumerate(jobs):
            folder = output / job['name']
            if (folder / 'status.json').exists():
                state = json.loads((folder / 'status.json').read_text())['state']
                if state == 'complete' and (folder / 'result.json').exists():
                    print('Skipping completed ' + job['name'], flush=True)
                    continue
                raise RuntimeError(f'{job["name"]} is {state}; preserve it and use a new output root for retries')
            folder.mkdir(exist_ok=True)
            write_json(output / 'status.json', dict(state='running', job=job['name'], pid=os.getpid(),
                                                    completed=idx, total=len(jobs), started=started))
            print(f'[{idx + 1}/{len(jobs)}] {job["name"]} -> {folder / "console.log"}', flush=True)
            with (folder / 'console.log').open('w') as log:
                child = subprocess.Popen(job['command'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         cwd=ROOT, start_new_session=True, text=True, bufsize=1)
                # The outer nohup log contains real epoch progress as well as queue status.
                for line in child.stdout:
                    log.write(line); log.flush()
                    print(line, end='', flush=True)
                returncode = child.wait()
                child.stdout.close()
                child = None
            if returncode:
                if not (folder / 'status.json').exists():
                    write_json(folder / 'status.json', dict(state='failed', error=f'Process exit {returncode}'))
                raise RuntimeError(f'{job["name"]} exited {returncode}; see {folder / "console.log"}')
            rows = collect(output, jobs)
            if rows[idx]['state'] != 'complete':
                raise RuntimeError('Child exited without complete status: ' + job['name'])
            print(f'Completed {job["name"]}; summary: {output / "summary.csv"}', flush=True)
        collect(output, jobs)
        write_json(output / 'status.json', dict(state='complete', runs=len(jobs), seconds=time.time()-started))
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
            if child.stdout is not None:
                child.stdout.close()
            status_path = folder / 'status.json'
            prior = json.loads(status_path.read_text()) if status_path.exists() else {}
            if prior.get('state') not in ('complete', 'failed'):
                write_json(status_path, dict(state='interrupted', error=repr(error),
                                             seconds=time.time()-prior.get('started', started)))
        collect(output, jobs)
        write_json(output / 'status.json', dict(state='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                               error=repr(error), seconds=time.time()-started))
        raise


if __name__ == '__main__':
    main()
