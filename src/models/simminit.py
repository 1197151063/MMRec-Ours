"""One-time user initialization; inherits the unchanged content model/loss/inference."""
import logging
import time
import torch
from torch.nn import functional as F
from models.simmrec import SIMMRec


def history_directions(keys, n_users, n_items, item_vectors):
    """Binary training histories only. Empty histories produce zero vectors."""
    keys = keys.unique().cpu()
    indices = torch.stack((keys // n_items, keys % n_items))
    counts = torch.bincount(indices[0], minlength=n_users).clamp_min(1)
    values = 1.0 / counts[indices[0]].float()
    matrix = torch.sparse_coo_tensor(indices, values, (n_users, n_items)).coalesce()
    return F.normalize(torch.sparse.mm(matrix, item_vectors.cpu()), dim=-1)


class SIMMInit(SIMMRec):
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        if self.representation != 'content' or self.gamma != 0:
            raise ValueError('Initialization experiments require content-only SIMMRec')
        kind = config['user_init']
        strength = float(config['init_strength'])
        if strength < 0 or kind not in ('none', 'random', 'interaction', 'content', 'mismatched'):
            raise ValueError('Invalid user initialization configuration')
        start = time.perf_counter()
        # Projectors and loaded feature tensors must be on the same device on CUDA servers.
        self.to(self.device)
        if kind != 'none':
            generator = torch.Generator(device='cpu').manual_seed(int(config['init_seed']))
            dim = config['embedding_size']
            with torch.no_grad():
                if kind == 'random':
                    direction = F.normalize(torch.randn(self.n_users, dim, generator=generator), dim=-1)
                elif kind == 'content':
                    pieces = []
                    for start_item in range(0, self.n_items, 2048):
                        ids = torch.arange(start_item, min(start_item + 2048, self.n_items), device=self.device)
                        pieces.append(self.item_representations(ids).cpu())
                    direction = history_directions(self.train_positive_keys, self.n_users, self.n_items, torch.cat(pieces))
                else:
                    # Fixed entity-bound random codes, not numeric-ID/position functions.
                    item_vectors = F.normalize(torch.randn(self.n_items, dim, generator=generator), dim=-1)
                    direction = history_directions(self.train_positive_keys, self.n_users, self.n_items, item_vectors)
                    if kind == 'mismatched':
                        direction = direction[torch.randperm(self.n_users, generator=generator)]
                self.user_embedding.weight.add_(strength * direction.to(self.device))
        logging.getLogger().info('USER INIT kind=%s strength=%s seed=%s seconds=%.4f; training history only; no persistent aggregation',
                                 kind, strength, config['init_seed'], time.perf_counter()-start)
