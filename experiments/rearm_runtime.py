"""Compatibility/resource adapters for the pinned REARM release; no model edits."""
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


def install(helper, block_size=256):
    def norm_adj(model, interaction):
        r = interaction.tocsr().astype(np.float32)
        r.sum_duplicates(); r.data[:] = 1
        a = sp.bmat([[None, r], [r.T, None]], format='coo')
        a = helper.sparse_mx_to_torch_sparse_tensor(a).to(model.device)
        return helper.torch_sparse_tensor_norm_adj(a, a, a.shape, model.device)

    def cooccurrence(kind, edges, start, size):
        # Same set-intersection counts and zero diagonal as upstream's nested loops.
        u, i = edges.T
        u = u - (start if kind == 'user' else 0)
        i = i - (start if kind == 'item' else int(i.min()))
        r = sp.csr_matrix((np.ones(len(u), dtype=np.float32), (u, i)),
                          shape=(size if kind == 'user' else int(u.max()) + 1,
                                 size if kind == 'item' else int(i.max()) + 1))
        r.sum_duplicates(); r.data[:] = 1
        return r if kind == 'user' else r.T.tocsr()

    def dict_graph(incidence, size):
        result = {}
        for begin in range(0, size, block_size):
            counts = (incidence[begin:begin + block_size] @ incidence.T).toarray()
            for offset, values in enumerate(counts):
                row = begin + offset
                values[row] = 0
                x = torch.from_numpy(values)
                v, ids = torch.topk(x, min(200, int(torch.count_nonzero(x))))
                result[row] = [ids.tolist(), v.tolist()]
        return result

    def knn(features, k, device):
        x = F.normalize(features, dim=1)
        indices, values = [], []
        for begin in range(0, len(x), block_size):
            sims = (x[begin:begin + block_size] @ x.T).cpu()
            val, col = torch.topk(sims, k, dim=-1)
            row = torch.arange(begin, begin + len(val))[:, None].expand_as(col)
            indices.append(torch.stack((row.flatten(), col.flatten())))
            values.append(val.flatten())
        ids, val = torch.cat(indices, 1), torch.cat(values)
        shape = (len(x), len(x))
        adj = torch.sparse_coo_tensor(ids, val, shape).to(device)
        degree = torch.sparse_coo_tensor(ids, torch.ones_like(val), shape)
        return helper.torch_sparse_tensor_norm_adj(adj, degree, shape, device)

    def no_stale_cache(logger, kind, description, dataset, name, function, *args):
        logger.info('%s%s: building from this run inputs (no external graph cache)', kind, description)
        return function(*args)

    helper.get_norm_adj_mat = norm_adj
    helper.creat_co_occur_matrix = cooccurrence
    helper.creat_dict_graph = dict_graph
    helper.get_knn_adj_mat = knn
    helper.load_or_create_matrix = no_stale_cache


def patch_loader(module, negative_scope='official'):
    original_dict = module.BaseDataset.dict_user_items

    def dictionaries(self):
        original_dict(self)
        if negative_scope == 'train':
            # Upstream mutates dict_train_u_i with held-out positives; rebuild it.
            from collections import defaultdict
            self.dict_train_u_i = module.update_dict('user', self.train_data, defaultdict(set))
            self.user_items_dict = self.dict_train_u_i
            ordered = sorted(self.dict_train_u_i.items(), key=lambda pair: len(pair[1]), reverse=True)
            self.topK_users = [u for u, _ in ordered]
            self.topK_users_counts = [len(items) for _, items in ordered]

    def sample(self, index):
        user, positive = self.train_data[index]
        # Tuple preserves iteration order of the upstream set; Python 3.11 no longer samples sets.
        candidates = tuple(self.all_set - self.user_items_dict[user])
        if not candidates:
            raise ValueError('User has no eligible negative items')
        negative = random.sample(candidates, 1)[0]
        return torch.LongTensor([user, user]), torch.LongTensor([positive, negative])

    original_eval = module.Load_eval_dataset.__init__
    def eval_init(self, *args, **kwargs):
        original_eval(self, *args, **kwargs)
        # Upstream iterates users but uses interaction count as its stopping bound.
        self.n_inters = len(self.eval_u)

    module.BaseDataset.dict_user_items = dictionaries
    module.Load_dataset.__getitem__ = sample
    module.Load_eval_dataset.__init__ = eval_init
