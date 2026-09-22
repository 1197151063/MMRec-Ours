"""Training-only user adjacency statistics; no fitting or held-out ranking."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def diagnose(data_root, dataset, pairs=10000, seed=2026):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / 'src/configs/overall.yaml').read_text())
    config.update(yaml.safe_load((root / 'src/configs/dataset' / (dataset + '.yaml')).read_text()))
    folder = Path(data_root) / dataset
    uid, iid, label = config['USER_ID_FIELD'], config['ITEM_ID_FIELD'], config['inter_splitting_label']
    full = pd.read_csv(folder / config['inter_file_name'], sep=config['field_separator'], usecols=[uid, iid, label])
    train = full.loc[full[label] == 0, [uid, iid]].drop_duplicates()
    if train.empty:
        raise ValueError('No training interactions')
    n_users, n_items = int(train[uid].max())+1, int(train[iid].max())+1
    matrix = sp.csr_matrix((np.ones(len(train), dtype=np.float32), (train[uid], train[iid])), shape=(n_users, n_items))
    degree = np.diff(matrix.indptr)
    users = np.flatnonzero(degree)
    adjacent = users[:-1][np.diff(users) == 1]
    if len(adjacent) == 0 or len(users) < 2:
        raise ValueError('Need adjacent numeric users with training histories')
    rng = np.random.default_rng(seed)
    anchors = rng.choice(adjacent, pairs, replace=True)
    neighbors = anchors + 1
    random = rng.choice(users, pairs, replace=True)
    while np.any(random == anchors):
        bad = random == anchors
        random[bad] = rng.choice(users, bad.sum(), replace=True)
    buckets = np.floor(np.log2(np.maximum(degree, 1))).astype(int)
    groups = {b: users[buckets[users] == b] for b in np.unique(buckets[users])}
    matched = np.array([rng.choice(groups[buckets[v]][groups[buckets[v]] != u]) for u,v in zip(anchors, neighbors)])
    partners = {'adjacent': neighbors, 'random': random, 'degree_matched': matched}
    popularity = np.asarray(matrix.sum(axis=0)).ravel()
    weights = 1 / np.log2(2 + popularity)
    records = {}
    for name, others in partners.items():
        jac, overlap, weighted = [], [], []
        for u, v in zip(anchors, others):
            a, b = matrix.indices[matrix.indptr[u]:matrix.indptr[u+1]], matrix.indices[matrix.indptr[v]:matrix.indptr[v+1]]
            common, union = np.intersect1d(a,b), np.union1d(a,b)
            jac.append(len(common)/len(union))
            overlap.append(bool(len(common)))
            weighted.append(float(weights[common].sum()/weights[union].sum()))
        records[name] = dict(mean_jaccard=float(np.mean(jac)), fraction_any_overlap=float(np.mean(overlap)),
                             mean_popularity_weighted_jaccard=float(np.mean(weighted)),
                             mean_anchor_degree=float(degree[anchors].mean()), mean_partner_degree=float(degree[others].mean()))
    mean_matrix = sp.diags(1 / np.maximum(degree,1)) @ matrix
    for modality, filename in [('visual', config['vision_feature_file']), ('text', config['text_feature_file'])]:
        features = np.load(folder / filename).astype(np.float32)
        if len(features) < n_items:
            raise ValueError('Feature rows do not cover training item IDs')
        history = normalize(np.asarray(mean_matrix @ normalize(features[:n_items]), dtype=np.float32))
        for name, others in partners.items():
            cosine = []
            for start in range(0, pairs, 256):
                cosine.extend(np.sum(history[anchors[start:start+256]] * history[others[start:start+256]], axis=1))
            records[name][modality + '_history_cosine'] = float(np.mean(cosine))
    return dict(dataset=dataset, seed=seed, sampled_pairs=pairs, unique_anchors=int(len(np.unique(anchors))),
                training_pairs=len(train), results=records,
                note='Only training interactions construct statistics. Adjacent means numeric user difference 1. '
                     'Random controls share anchors; matched partners share the adjacent partner log2-degree bin. '
                     'Pairs sampled with replacement; these are descriptive means, not independent significance tests.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--dataset', default='baby')
    parser.add_argument('--pairs', type=int, default=10000)
    parser.add_argument('--output', default='user-neighbors.json')
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error('pairs must be positive')
    result = diagnose(args.data_path, args.dataset, args.pairs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
