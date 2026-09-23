"""Replace only REARM's item homogeneous propagation with ID neighbor attraction."""
import torch
import torch.nn.functional as F
from model import REARM


class REARMItemLoss(REARM):
    def __init__(self, config, dataset, weight=0.1):
        super().__init__(config, dataset)
        self.n_ii_layers = 0  # upstream last_layer=True returns its input at zero layers
        self.item_alignment_weight = weight
        graph = self.stre_ii_graph.detach().coalesce()
        row, col = graph.indices()
        val = graph.values()
        if not torch.isfinite(val).all():
            raise ValueError('Non-finite II graph weights')
        keep = (row != col) & (val > 0)
        row, col, val = row[keep], col[keep], val[keep]
        counts = torch.bincount(row, minlength=self.n_items)
        width = max(1, int(counts.max()))
        ids = torch.zeros((self.n_items, width), dtype=torch.long, device=row.device)
        weights = torch.zeros((self.n_items, width), dtype=val.dtype, device=val.device)
        starts = counts.cumsum(0) - counts
        slots = torch.arange(len(row), device=row.device) - starts[row]
        ids[row, slots] = col
        weights[row, slots] = val
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
        self.register_buffer('alignment_ids', ids)
        self.register_buffer('alignment_weights', weights)
        self.reset_alignment_stats()

    def reset_alignment_stats(self):
        self.alignment_sum = 0.0
        self.alignment_batches = 0

    @staticmethod
    def weighted_alignment(embedding, items, ids, weights):
        anchors = embedding[items].unsqueeze(1)
        neighbors = embedding[ids[items]]
        scores = (anchors * neighbors).sum(-1)
        # -log(sigmoid(s)) == softplus(-s); average anchors, normalized edge weights.
        return (weights[items] * F.softplus(-scores)).sum(-1).mean()

    def loss(self, user_tensor, item_tensor):
        losses = list(super().loss(user_tensor, item_tensor))
        if self.item_alignment_weight:
            # Official batch columns are positive/negative items with the user offset.
            items = torch.unique(item_tensor[:, 0].to(self.device) - self.n_users)
            alignment = self.weighted_alignment(self.item_id_embedding.weight, items,
                                                self.alignment_ids, self.alignment_weights)
            losses[0] = losses[0] + self.item_alignment_weight * alignment
            self.alignment_sum += alignment.detach().item()
            self.alignment_batches += 1
        return tuple(losses)
