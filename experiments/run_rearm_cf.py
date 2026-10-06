#!/usr/bin/env python3
"""Run a fixed, diverse REARM+CF suite sequentially. No model checkpoints."""
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

from run_rearm import ROOT, PRESETS, sha, write_json
from rearm_cf_plan import experiments, QUICK


def historical_references(dataset, seeds, data_hashes, epochs, workers, cpu):
    records = []
    for seed in seeds:
        path = ROOT / 'experiments/references' / f'rearm_{dataset}_s{seed}.json'
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        command = record['job']['command']
        reasons = []
        if data_hashes != record['data_sha256']:
            # Root paths can change when moving the same dataset to another server.
            old = {Path(k).name: v for k,v in record['data_sha256'].items()}
            new = {Path(k).name: v for k,v in data_hashes.items()}
            if old != new:
                reasons.append('Data fingerprints differ')
        if epochs != int(command[command.index('--epochs')+1]):
            reasons.append('Epoch limit differs')
        if workers != int(command[command.index('--num-workers')+1]):
            reasons.append('Data-loader worker count differs')
        if cpu != ('--cpu' in command):
            reasons.append('CPU/CUDA execution mode differs')
        records.append(dict(source=str(path.relative_to(ROOT)), sha256=sha(path), run=record['run'],
                            input_and_run_settings_match=not reasons, mismatches=reasons,
                            note='Historical observation, not a newly executed job; compare only under a compatible training protocol'))
    return records


def collect(output, jobs):
    rows = []
    for job in jobs:
        folder = output / job['name']
        status = json.loads((folder / 'status.json').read_text()) if (folder / 'status.json').exists() else {}
        row = dict(name=job['name'], variant=job['variant'], family=job['family'], seed=job['seed'],
                   state=status.get('state', 'pending'), seconds=status.get('seconds'), error=status.get('error', ''))
        if (folder / 'result.json').exists():
            result = json.loads((folder / 'result.json').read_text())
            row.update(best_epoch=result['best_epoch'], parameters=result['parameters'])
            for group, counts in result.get('parameter_breakdown', {}).items():
                row['params_' + group] = counts['total']
                row['trainable_params_' + group] = counts['trainable']
            if result.get('parameter_breakdown'):
                row['trainable_parameters'] = sum(v['trainable'] for v in result['parameter_breakdown'].values())
            profile = result.get('cf_diagnostics', {}).get('training_profile', {})
            row.update(train_seconds_at_best=profile.get('train_seconds'),
                       peak_cuda_allocated_mb_at_best=profile.get('peak_cuda_allocated_mb'))
            for split in ('valid', 'test'):
                row.update({split + '_' + k: v for k, v in result[split].items()})
        rows.append(row)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (output / 'summary.csv.tmp').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (output / 'summary.csv.tmp').replace(output / 'summary.csv')
    done = [r for r in rows if r['state'] == 'complete']
    aggregates = []
    for variant in dict.fromkeys(r['variant'] for r in done):
        group = [r for r in done if r['variant'] == variant]
        values = [r['valid_Recall@20'] for r in group]
        mean = sum(values) / len(values)
        aggregates.append(dict(variant=variant, seeds=[r['seed'] for r in group], valid_recall20_mean=mean,
                               sample_std=(sum((v-mean)**2 for v in values)/(len(values)-1))**.5 if len(values)>1 else None))
    aggregates.sort(key=lambda r: r['valid_recall20_mean'], reverse=True)
    manifest_path = output / 'manifest.json'
    history = json.loads(manifest_path.read_text()).get('historical_references', []) if manifest_path.exists() else []
    write_json(output / 'summary.json', dict(runs=rows, validation_ranking=aggregates, historical_references=history,
               selection_metric='valid_Recall@20', note='Single-seed screening is not significance evidence; no test-based ranking'))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--dataset', choices=PRESETS, default='baby')
    parser.add_argument('--output', required=True)
    parser.add_argument('--suite', choices=['all', 'quick', 'capacity', 'capacity_extended', 'preferences', 'preferences_extended'], default='all')
    parser.add_argument('--variants', nargs='+', help='Exact variant names; overrides --suite')
    parser.add_argument('--include-reference', action='store_true',
                        help='Explicitly rerun original REARM; default suites skip the reference')
    parser.add_argument('--seeds', nargs='+', type=int, default=[2025])
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--graph-block', type=int, default=256)
    parser.add_argument('--max-consecutive-failures', type=int, default=3)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-failed', action='store_true', help='With --resume: archive old failed/interrupted runs before retrying')
    args = parser.parse_args()
    if min(args.epochs, args.graph_block, args.max_consecutive_failures) < 1 or args.num_workers < 0:
        parser.error('Invalid count or worker setting')
    if len(set(args.seeds)) != len(args.seeds) or (args.retry_failed and not args.resume):
        parser.error('Seeds must be unique; --retry-failed requires --resume')
    if args.suite.startswith('preferences'):
        from rearm_preferences_plan import experiments as preference_experiments
        plan = preference_experiments(extended=args.suite == 'preferences_extended')
    elif args.suite.startswith('capacity'):
        from rearm_capacity_plan import experiments as capacity_experiments
        plan = capacity_experiments(extended=args.suite == 'capacity_extended')
    else:
        plan = experiments()
    selected = args.variants if args.variants else QUICK if args.suite == 'quick' else [j['name'] for j in plan]
    if not args.variants and not args.include_reference:
        selected = [name for name in selected if name not in ('reference', 'cap_original')]
    if len(set(selected)) != len(selected) or set(selected) - {j['name'] for j in plan}:
        parser.error('Unknown or duplicate variants: ' + str(set(selected) - {j['name'] for j in plan}))
    plan = [j for j in plan if j['name'] in selected]
    output = Path(args.output).resolve()
    source = Path(args.data_path).resolve()
    jobs = []
    for seed in args.seeds:
        for spec in plan:
            name = spec['name'] + '_s' + str(seed)
            command = [sys.executable, '-u', str(ROOT / 'experiments/run_rearm.py'), '--data-path', str(source),
                       '--dataset', args.dataset, '--output', str(output/name), '--gpu-id', str(args.gpu_id),
                       '--seed', str(seed), '--epochs', str(args.epochs), '--num-workers', str(args.num_workers),
                       '--graph-block', str(args.graph_block), '--negative-scope', 'train',
                       '--cf-config', str(output/'configs'/(name+'.json'))]
            if args.cpu:
                command.append('--cpu')
            jobs.append(dict(name=name, variant=spec['name'], family=spec['family'], question=spec['question'],
                             seed=seed, config=spec['config'], command=command))
    paths = list((ROOT/'third_party/rearm').rglob('*.py')) + list((ROOT/'experiments').glob('rearm*.py')) + [ROOT/'experiments/run_rearm.py', Path(__file__)]
    manifest = dict(jobs=jobs, dataset=args.dataset, protocol='TRAIN-only; validation Recall@20 selection; no checkpoints',
                    reference_policy='Original REARM is opt-in; historical results are separate from new runs',
                    source_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths})
    if args.dry_run:
        print(json.dumps(manifest, indent=2)); return
    data_dir = source / args.dataset
    inputs = [data_dir/'image_feat.npy', data_dir/'text_feat.npy']
    inter = data_dir/(args.dataset+'.inter')
    inputs += [inter] if inter.exists() else [data_dir/(name+'.npy') for name in ('train','valid','test')]
    manifest['data_sha256'] = {str(p): sha(p) for p in inputs}
    manifest['historical_references'] = historical_references(args.dataset, args.seeds, manifest['data_sha256'],
                                                             args.epochs, args.num_workers, args.cpu)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'.queue.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another CF queue is running in this output directory')
    if (output/'manifest.json').exists():
        if not args.resume or json.loads((output/'manifest.json').read_text()) != manifest:
            parser.error('Output already used or plan/source changed; use a new directory, or --resume with identical plan/source')
    else:
        if any((output/j['name']).exists() for j in jobs):
            parser.error('Child output exists without matching manifest')
        write_json(output/'manifest.json', manifest)
    (output/'configs').mkdir(exist_ok=True)
    for job in jobs:
        write_json(output/'configs'/(job['name']+'.json'), job['config'])
    started, child, failures = time.time(), None, 0
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        import torch
        import tensorboardX  # noqa: F401
        if not args.cpu and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; refusing silent CPU fallback')
        collect(output, jobs)
        for idx, job in enumerate(jobs):
            folder = output/job['name']
            status = json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else None
            if status:
                if status['state'] == 'complete' and (folder/'result.json').exists():
                    print('Skipping completed ' + job['name'], flush=True); continue
                if not args.retry_failed:
                    print('Preserving unsuccessful run ' + job['name'] + '; use --resume --retry-failed to retry', flush=True)
                    continue
                # A still-running orphan may retain the single-run lock. Never move it.
                with (folder/'.lock').open('a') as run_lock:
                    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    archive = output/'attempts'; archive.mkdir(exist_ok=True)
                    folder.rename(archive/(job['name']+'_'+str(time.time_ns())))
            folder.mkdir(exist_ok=True)
            write_json(output/'status.json', dict(state='running', job=job['name'], index=idx+1, total=len(jobs), pid=os.getpid()))
            print(f'[{idx+1}/{len(jobs)}] {job["name"]} -> {folder/"console.log"}', flush=True)
            with (folder/'console.log').open('w') as log:
                child = subprocess.Popen(job['command'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         cwd=ROOT, start_new_session=True, text=True, bufsize=1)
                for line in child.stdout:
                    log.write(line); log.flush(); print(line, end='', flush=True)
                code = child.wait(); child.stdout.close(); child = None
            status = json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
            if code or status.get('state') != 'complete':
                write_json(folder/'status.json', dict(state='failed', error=status.get('error', f'Process exit {code}'),
                                                      seconds=status.get('seconds')))
                failures += 1
            else:
                failures = 0
            collect(output, jobs)
            if failures >= args.max_consecutive_failures:
                raise RuntimeError(f'{failures} consecutive failures; stopping to avoid wasting the remaining queue')
        rows = collect(output, jobs)
        write_json(output/'status.json', dict(state='complete' if all(r['state']=='complete' for r in rows) else 'complete_with_failures',
                                              seconds=time.time()-started, completed=sum(r['state']=='complete' for r in rows), total=len(jobs)))
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
            if child.stdout:
                child.stdout.close()
            write_json(folder/'status.json', dict(state='interrupted', error=repr(error)))
        collect(output, jobs)
        write_json(output/'status.json', dict(state='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
