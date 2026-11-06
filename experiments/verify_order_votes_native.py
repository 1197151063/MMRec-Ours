#!/usr/bin/env python3
"""Recheck a FROZEN validation-selected vote configuration with native MMRec loaders/evaluator."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common.order_history_votes import build_order_history_votes
from utils.dataset import RecDataset
from utils.dataloader import EvalDataLoader
from utils.topk_evaluator import TopKEvaluator


def verify(data_path, dataset_name, report):
    config = yaml.safe_load((ROOT / 'src/configs/overall.yaml').read_text())
    config.update(yaml.safe_load((ROOT / 'src/configs/dataset' / f'{dataset_name}.yaml').read_text()))
    config.update(data_path=str(Path(data_path).resolve()) + '/', dataset=dataset_name,
                  device=torch.device('cpu'), save_recommended_topk=False)
    path = Path(config['data_path']) / dataset_name / config['inter_file_name']
    if hashlib.sha256(path.read_bytes()).hexdigest() != report['input_sha256']:
        raise ValueError('Input hash differs from selection report')
    full = RecDataset(config)
    train, valid, test = full.split()
    r = sp.csr_matrix((np.ones(len(train.df), dtype=np.float32),
                      (train.df[train.uid_field], train.df[train.iid_field])),
                     shape=(full.user_num, full.item_num))
    selected = report['selected']
    votes = build_order_history_votes(r, selected['bandwidth'], selected['radius'], selected['user_degree_power'])
    popularity = np.asarray(r.sum(0)).ravel().astype(np.float32)
    fallback = 1e-8 * popularity / max(float(popularity.max()), 1.)
    output = dict(source='FROZEN_CONFIG_NATIVE_RECDATASET_EVALDATALOADER_TOPKEVALUATOR',
                  selected={key: selected[key] for key in ('bandwidth', 'radius', 'user_degree_power')},
                  input_sha256=report['input_sha256'])
    for split, ds in [('valid', valid), ('test', test)]:
        if split == 'test' and report.get('test') is None:
            continue
        # Native quick_start logs each split before constructing its loader;
        # RecDataset.__str__ initializes inter_num as a side effect there.
        str(ds)
        loader = EvalDataLoader(config, ds, additional_dataset=train, batch_size=1024)
        ranked = []
        for users, masked in loader:
            scores = torch.from_numpy(votes[users.numpy()].toarray() + fallback)
            scores[masked[0], masked[1]] = -1e10
            ranked.append(scores.topk(max(config['topk']), dim=-1).indices)
        metrics = TopKEvaluator(config).evaluate(ranked, loader)
        reference = selected['valid'] if split == 'valid' else report['test']
        for key, value in metrics.items():
            if abs(value - reference[key]) > 0.000051:
                raise AssertionError(f'{split} {key}: native={value}, probe={reference[key]}')
        output[split] = metrics
    output['matches_probe'] = True
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--dataset', default='baby')
    parser.add_argument('--report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Output already exists; choose a fresh filename')
    result = verify(args.data_path, args.dataset, json.loads(Path(args.report).read_text()))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
