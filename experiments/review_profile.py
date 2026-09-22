"""Summarize only complete seed groups, sorted by validation Recall@20."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def review(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    summaries = {r['name']: r for r in json.loads((folder / 'summary.json').read_text())}
    groups = defaultdict(list)
    for job in manifest['jobs']:
        parameters = {k: v for k, v in job['overrides'].items() if k != 'seed'}
        groups[json.dumps(parameters, sort_keys=True)].append((job, summaries.get(job['name'], {})))
    eligible, diagnostic, incomplete = [], [], 0
    for encoded, members in groups.items():
        if not all(row.get('state') == 'complete' for _, row in members):
            incomplete += 1
            continue
        params = json.loads(encoded)
        valid = [r['valid_recall@20'] for _, r in members]
        test = [r['test_recall@20'] for _, r in members]
        record = dict(name=members[0][0]['name'].rsplit('_s', 1)[0], seeds=len(members),
                      valid_mean=statistics.mean(valid), valid_sd=statistics.stdev(valid) if len(valid)>1 else None,
                      test_mean=statistics.mean(test), test_sd=statistics.stdev(test) if len(test)>1 else None)
        (eligible if params['leave_one_out'] else diagnostic).append(record)
    eligible.sort(key=lambda r: (-r['valid_mean'], r['name']))
    return dict(selection='Validation Recall@20 mean only; sample SD; no significance claims',
                incomplete_groups=incomplete, eligible=eligible, target_included_diagnostic=diagnostic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_directory')
    args = parser.parse_args()
    print(json.dumps(review(args.output_directory), indent=2))
