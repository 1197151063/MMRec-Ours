#!/usr/bin/env python3
"""Sequential, resumable, time-bounded experiment queue. Uses current Python environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import csv
from collections import deque

from night_plan import build_plan

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def summarize(output, jobs):
    rows = []
    for job in jobs:
        folder = output / job['name']
        status = json.loads((folder / 'status.json').read_text()) if (folder / 'status.json').exists() else {'state': 'pending'}
        row = dict(name=job['name'], hypothesis=job['hypothesis'], model=job['model'], **status)
        if status['state'] == 'complete':
            result = json.loads((folder / 'result.json').read_text())[0]
            row['best_epoch'] = result['best_epoch']
            for split in ('valid', 'test'):
                row.update({split + '_' + key: value for key, value in result[split].items()})
        rows.append(row)
    write_json(output / 'summary.json', rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--dataset', default='baby')
    parser.add_argument('--suite', choices=['order', 'init', 'profile', 'corr'], default='order')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--hours', type=float, default=8)
    parser.add_argument('--job-minutes', type=float, default=45)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[999])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', default='night_runs/baby-order-audit')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-consecutive-failures', type=int, default=3)
    args = parser.parse_args()
    if args.hours <= 0 or args.job_minutes <= 0 or args.epochs <= 0 or args.max_consecutive_failures < 1 or (args.limit is not None and args.limit < 1):
        parser.error('Budgets, epochs and limit must be positive')
    if args.suite == 'corr':
        from corr_plan import build_plan as selected_plan
    elif args.suite == 'profile':
        from profile_plan import build_plan as selected_plan
    elif args.suite == 'init':
        from init_plan import build_plan as selected_plan
    else:
        selected_plan = build_plan
    jobs = selected_plan(args.seeds)
    if args.limit:
        jobs = jobs[:args.limit]
    output = Path(args.output).resolve()
    data = Path(args.data_path).resolve()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    # Include source content so dirty edits cannot silently reuse previous results.
    digest = hashlib.sha256()
    for folder in ('src', 'experiments'):
        for path in sorted((ROOT / folder).rglob('*')):
            if path.suffix in ('.py', '.yaml'):
                digest.update(str(path.relative_to(ROOT)).encode())
                digest.update(path.read_bytes())
    manifest = dict(revision=revision, source_digest=digest.hexdigest(), data_path=str(data),
                    dataset=args.dataset, epochs=args.epochs, cpu=args.cpu, jobs=jobs)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return
    if not data.is_dir():
        parser.error('data-path must be an existing dataset root')
    output.mkdir(parents=True, exist_ok=True)
    # Single queue owner per output directory; OS releases lock even after interruption.
    import fcntl
    lock = (output / '.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another queue is using this output directory')
    previous = output / 'manifest.json'
    if previous.exists() and json.loads(previous.read_text()) != manifest:
        parser.error('Manifest changed; choose a new output directory to avoid mixing experiments')
    write_json(previous, manifest)
    deadline = time.monotonic() + args.hours * 3600
    import yaml
    summarize(output, jobs)
    print(f'Python: {sys.executable} ({sys.version.split()[0]}); data: {data}', flush=True)
    consecutive_failures = 0
    for index, job in enumerate(jobs, 1):
        if time.monotonic() >= deadline:
            break
        folder = output / job['name']
        folder.mkdir(exist_ok=True)
        status = folder / 'status.json'
        if status.exists() and json.loads(status.read_text())['state'] == 'complete':
            continue
        shutil.copytree(ROOT / 'src/configs', folder / 'configs', dirs_exist_ok=True)
        config = dict(job['overrides'], epochs=args.epochs, use_gpu=not args.cpu,
                      evaluate_train=False, save_recommended_topk=False,
                      result_file=str(folder / 'result.json'))
        # Scalars are supported by the experiment grid; each subprocess runs one config.
        (folder / 'override.yaml').write_text(yaml.safe_dump(config))
        if (folder / 'result.json').exists():
            (folder / 'result.json').unlink()
        command = [sys.executable, str(ROOT / 'src/main.py'), '--model', job['model'],
                   '--dataset', args.dataset, '--data-path', str(data), '--gpu-id', str(args.gpu_id),
                   '--config', str(folder / 'override.yaml')]
        write_json(status, dict(state='running'))
        print(f'[{index}/{len(jobs)}] {job["name"]}', flush=True)
        start = time.monotonic()
        state = 'failed'
        returncode = None
        try:
            with (folder / 'console.log').open('w') as stream:
                result = subprocess.run(command, cwd=folder, stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=min(args.job_minutes * 60, max(0.1, deadline-start)))
            returncode = result.returncode
            if returncode == 0 and (folder / 'result.json').exists():
                state = 'complete'
        except subprocess.TimeoutExpired:
            state = 'timeout'
        except KeyboardInterrupt:
            write_json(status, dict(state='interrupted', seconds=time.monotonic()-start))
            summarize(output, jobs)
            raise
        error_tail = ''
        if state != 'complete':
            with (folder / 'console.log').open(errors='replace') as stream:
                error_tail = ''.join(deque(stream, maxlen=35))
            print(f'  Error log: {folder / "console.log"}\n{error_tail}', flush=True)
            consecutive_failures += 1
        else:
            consecutive_failures = 0
        write_json(status, dict(state=state, returncode=returncode,
                                seconds=round(time.monotonic()-start, 2), error_tail=error_tail))
        summarize(output, jobs)
        print(f'  {state}; summary: {output / "summary.csv"}', flush=True)
        if consecutive_failures >= args.max_consecutive_failures:
            print('Stopping after consecutive failures. Fix the shared error before restarting.', flush=True)
            raise SystemExit(1)
    summarize(output, jobs)


if __name__ == '__main__':
    main()
