"""Explicit order-dependence audit of the original LightMRec architecture.

Only the PE coordinate assignment/side changes. Entity lookups and feature rows
stay fixed so initialization, dropout, and negative sampling remain comparable.
"""
import hashlib
import logging
import math

import numpy as np
import torch

from models.lightmrecnope import LightMRecNoPE


def first_seen_positions(ids, size, rng):
    """Training-first order; append unseen catalog IDs in a fixed random order."""
    seen = list(dict.fromkeys(int(value) for value in ids))
    present = np.zeros(size, dtype=bool)
    present[seen] = True
    unseen = np.flatnonzero(~present)
    order = np.concatenate((np.asarray(seen, dtype=np.int64), rng.permutation(unseen)))
    positions = np.empty(size, dtype=np.int64)
    positions[order] = np.arange(size)
    return positions, len(unseen)


def make_positions(train, uid, iid, n_users, n_items, source, seed):
    # Local RNG: generating coordinates must not advance model/sampler RNG streams.
    rng = np.random.default_rng(seed)
    if source == 'original':
        return np.arange(n_users), np.arange(n_items)
    if source in ('random', 'random_user', 'random_item'):
        users = rng.permutation(n_users) if source != 'random_item' else np.arange(n_users)
        items = rng.permutation(n_items) if source != 'random_user' else np.arange(n_items)
        return users, items
    if source == 'train_degree':
        def degree_order(values, size):
            count = np.bincount(np.asarray(values, dtype=np.int64), minlength=size)
            tie = rng.permutation(size)
            order = tie[np.argsort(-count[tie], kind='stable')]
            positions = np.empty(size, dtype=np.int64)
            positions[order] = np.arange(size)
            return positions
        return degree_order(train[uid], n_users), degree_order(train[iid], n_items)
    if source not in ('train_file', 'train_shuffled'):
        raise ValueError('Unknown order_source: ' + source)
    if source == 'train_shuffled':
        train = train.iloc[rng.permutation(len(train))]
    users, _ = first_seen_positions(train[uid], n_users, rng)
    items, _ = first_seen_positions(train[iid], n_items, rng)
    return users, items


def sinusoidal(positions, dim):
    if dim % 2:
        raise ValueError('Order audit requires an even embedding_size')
    position = torch.as_tensor(positions, dtype=torch.float32)[:, None]
    frequency = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    encoding = torch.empty(len(positions), dim)
    encoding[:, 0::2] = torch.sin(position * frequency)
    encoding[:, 1::2] = torch.cos(position * frequency)
    return encoding


class LightMRecOrder(LightMRecNoPE):
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self.pe_side = config['pe_side']
        if self.pe_side not in ('none', 'user', 'item', 'both'):
            raise ValueError('pe_side must be none, user, item, or both')
        # dataset_bk is the pre-shuffle training snapshot, never validation/test rows.
        train = dataloader.dataset_bk.df
        uid, iid = self.USER_ID, self.ITEM_ID
        users, items = make_positions(train, uid, iid, self.n_users, self.n_items,
                                      config['order_source'], int(config['order_seed']))
        def option(key, default):
            return config[key] if key in config and config[key] is not None else default
        self.pe_scale = float(option('pe_scale', 1.0))
        self.pe_application = option('pe_application', 'forward')
        if self.pe_application not in ('forward', 'init'):
            raise ValueError('pe_application must be forward or init')
        if self.pe_application == 'init' and self.pe_side != 'user':
            raise ValueError('Initialization equivalence is restricted to user-only PE')
        self.eval_cosine = bool(option('eval_cosine', False))
        if option('freeze_features', False):
            self.image_embedding.weight.requires_grad_(False)
            self.text_embedding.weight.requires_grad_(False)
        dim = config['embedding_size']
        kind = option('pe_kind', 'sinusoidal')
        rng = np.random.default_rng(int(config['order_seed']))
        def encode(positions, offset):
            if kind == 'sinusoidal':
                return sinusoidal(np.asarray(positions) + offset, dim)
            if kind == 'gaussian':
                values = torch.from_numpy(rng.standard_normal((len(positions), dim)).astype(np.float32))
                return torch.nn.functional.normalize(values, dim=-1) * math.sqrt(dim / 2)
            if kind == 'constant':
                return sinusoidal(np.zeros(len(positions)), dim)
            raise ValueError('Unknown pe_kind: ' + kind)
        self.register_buffer('user_pe', encode(users, 0))
        self.register_buffer('item_pe', encode(items, int(option('item_pe_offset', 0))))
        if self.pe_application == 'init':
            with torch.no_grad():
                self.user_embedding.weight.add_(self.pe_scale * self.user_pe.to(self.user_embedding.weight.device))
        digest = lambda x: hashlib.sha256(np.asarray(x, dtype='<i8').tobytes()).hexdigest()[:16]
        logging.getLogger().info(
            'ORDER AUDIT source=%s side=%s order_seed=%s user_map=%s item_map=%s '
            'train_rows=%s unseen_users=%s unseen_items=%s. '
            'train_file means current .inter row order, NOT reconstructed raw-source order.',
            config['order_source'], self.pe_side, config['order_seed'], digest(users), digest(items),
            len(train), self.n_users - train[uid].nunique(), self.n_items - train[iid].nunique())

    def forward(self):
        user, item = super().forward()
        if self.pe_side in ('user', 'both') and self.pe_application == 'forward':
            user = user + self.pe_scale * self.user_pe
        if self.pe_side in ('item', 'both'):
            item = item + self.pe_scale * self.item_pe
        return user, item

    def full_sort_predict(self, interaction):
        users = interaction[0] if isinstance(interaction, (list, tuple)) else interaction
        user, item = self.forward()
        if self.eval_cosine:
            user = torch.nn.functional.normalize(user, dim=-1)
            item = torch.nn.functional.normalize(item, dim=-1)
        return user[users] @ item.T
