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
    if source == 'random':
        return rng.permutation(n_users), rng.permutation(n_items)
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
        self.register_buffer('user_pe', sinusoidal(users, config['embedding_size']))
        self.register_buffer('item_pe', sinusoidal(items, config['embedding_size']))
        digest = lambda x: hashlib.sha256(np.asarray(x, dtype='<i8').tobytes()).hexdigest()[:16]
        logging.getLogger().info(
            'ORDER AUDIT source=%s side=%s order_seed=%s user_map=%s item_map=%s '
            'train_rows=%s unseen_users=%s unseen_items=%s. '
            'train_file means current .inter row order, NOT reconstructed raw-source order.',
            config['order_source'], self.pe_side, config['order_seed'], digest(users), digest(items),
            len(train), self.n_users - train[uid].nunique(), self.n_items - train[iid].nunique())

    def forward(self):
        user, item = super().forward()
        if self.pe_side in ('user', 'both'):
            user = user + self.user_pe
        if self.pe_side in ('item', 'both'):
            item = item + self.item_pe
        return user, item
