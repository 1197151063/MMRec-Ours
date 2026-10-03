"""Relation-separated REARM. Upstream parameters, attention and propagation are shared.

Each branch executes the unchanged upstream forward with a different II graph.
This is a controlled accuracy experiment, not a computational simplification.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from model import REARM
from utils.helper import ssl_loss, cal_diff_loss

RELATIONS = ('cooccurrence', 'agreement', 'visual_only', 'text_only')
MODES = ('original', 'merged_single', 'merged_matched', 'typed_fixed', 'typed_global', 'typed_user')


@torch.no_grad()
def semantic_knn(features, k, block=256):
    """Positive cosine top-k, excluding self; O(block * items) scratch space."""
    x = F.normalize(features.detach(), dim=1)
    n = len(x)
    k = min(k, n - 1)
    indices, values = [], []
    for start in range(0, n, block):
        sim = x[start:start + block] @ x.T
        local = torch.arange(len(sim), device=x.device)
        sim[local, start + local] = -torch.inf
        val, col = sim.topk(k, dim=1)
        row = (start + local)[:, None].expand_as(col)
        keep = val > 0
        indices.append(torch.stack((row[keep], col[keep])))
        values.append(val[keep])
    return torch.sparse_coo_tensor(torch.cat(indices, 1), torch.cat(values), (n, n)).coalesce()


def row_stochastic(graph):
    """Positive row-normalization; an empty row is identity, never a random neighbor."""
    graph = graph.coalesce()
    row, col = graph.indices()
    val = graph.values()
    keep = (row != col) & (val > 0)
    row, col, val = row[keep], col[keep], val[keep]
    sums = torch.zeros(graph.size(0), device=val.device, dtype=val.dtype)
    sums.index_add_(0, row, val)
    val = val / sums[row].clamp_min(1e-12)
    empty = torch.where(sums == 0)[0]
    return torch.sparse_coo_tensor(
        torch.stack((torch.cat((row, empty)), torch.cat((col, empty)))),
        torch.cat((val, torch.ones(len(empty), device=val.device))), graph.shape).coalesce()


def partition_semantics(visual, text):
    """Partition directed top-k supports. No claim that exclusive means negative."""
    visual, text = visual.coalesce(), text.coalesce()
    n = visual.size(0)
    vk = visual.indices()[0] * n + visual.indices()[1]
    tk = text.indices()[0] * n + text.indices()[1]
    # Coalesced COO indices are lexicographically sorted.
    locations = torch.searchsorted(tk, vk)
    present = locations < len(tk)
    if len(tk):
        present = present & (tk[locations.clamp_max(len(tk) - 1)] == vk)
    common_keys = vk[present]
    common_vals = (visual.values()[present] + text.values()[locations[present]]) / 2
    t_common = torch.isin(tk, common_keys)
    pairs = ((common_keys, common_vals), (vk[~present], visual.values()[~present]),
             (tk[~t_common], text.values()[~t_common]))
    return [torch.sparse_coo_tensor(torch.stack((key // n, key % n)), val,
                                    (n, n)).coalesce() for key, val in pairs]


class REARMRelations(REARM):
    def __init__(self, config, dataset, mode='typed_fixed', graph_block=256):
        if mode not in MODES[1:]:
            raise ValueError('Unsupported relation mode: ' + mode)
        super().__init__(config, dataset)
        self.relation_mode = mode
        visual = semantic_knn(self.i_v_feat, config.item_knn_k, graph_block)
        text = semantic_knn(self.i_t_feat, config.item_knn_k, graph_block)
        raw_graphs = [self.item_co_graph] + partition_semantics(visual, text)
        self.graph_diagnostics = {}
        for name, raw in zip(RELATIONS, raw_graphs):
            graph = row_stochastic(raw)
            self.register_buffer('relation_' + name, graph)
            r, c = graph.indices()
            self.graph_diagnostics[name] = dict(
                nonself_edges=int((r != c).sum()), identity_rows=int((r == c).sum()),
                active_row_fraction=float(torch.unique(r[r != c]).numel() / self.n_items))
        graphs = self.graphs()
        merged = (graphs[0] + graphs[1] + graphs[2] + graphs[3]) * .25
        self.register_buffer('relation_merged', merged.coalesce())
        # Same gate parameter count for global/user/matched controls. Profiles are
        # fixed TRAIN-history means; no numeric-ID feature or held-out interaction.
        self.image_u_trs.to(self.device)
        self.text_u_trs.to(self.device)
        with torch.no_grad():
            profile = torch.cat((F.normalize(self.image_u_trs(dataset.u_v_interest), dim=1),
                                 F.normalize(self.text_u_trs(dataset.u_t_interest), dim=1)), dim=1)
        # Fixed initial projection keeps the routing context small (users x 128).
        self.register_buffer('gate_profile', profile.detach())
        if mode in ('typed_global', 'typed_user', 'merged_matched'):
            # Keep RNG state unchanged so downstream sampling is paired across arms.
            with torch.random.fork_rng(devices=[]):
                self.relation_gate = nn.Linear(2 * self.embedding_dim, 4)
            nn.init.zeros_(self.relation_gate.weight)
            nn.init.zeros_(self.relation_gate.bias)
        else:
            self.relation_gate = None
        self._eval_representation = None
        self._branch_features = []
        self.last_pair_score_std = []

    def graphs(self):
        return [getattr(self, 'relation_' + name) for name in RELATIONS]

    def train(self, mode=True):
        self._eval_representation = None
        return super().train(mode)

    def gate_weights(self):
        if self.relation_gate is None:
            return self.user_id_embedding.weight.new_full((self.n_users, 4), .25)
        logits = self.relation_gate(self.gate_profile)
        if self.relation_mode == 'typed_global':
            logits = logits.mean(0, keepdim=True).expand(self.n_users, -1)
        return logits.softmax(dim=1)

    def forward(self):
        original_graph = self.stre_ii_graph
        graphs = self.graphs()
        if self.relation_mode.startswith('merged'):
            graphs = [self.relation_merged] * (1 if self.relation_mode == 'merged_single' else 4)
        reps, features = [], []
        try:
            for graph in graphs:
                self.stre_ii_graph = graph
                reps.append(super().forward())
                features.append(self.fin_feat_prefer)
        finally:
            self.stre_ii_graph = original_graph
        self._branch_features = features
        if len(reps) == 1:
            self._last_gates = None
            return reps[0]
        weights = self.gate_weights()
        self._last_gates = weights.detach()
        # Concatenation implements sum_r g_ur <u_ur,i_ir> with no cross-terms.
        # Applying g only to users preserves the upstream dot-product score scale.
        weighted = [torch.cat((rep[:self.n_users] * weights[:, r:r+1], rep[self.n_users:]))
                    for r, rep in enumerate(reps)]
        return torch.cat(weighted, dim=1)

    def loss(self, user_tensor, item_tensor):
        out = self.forward()
        u, i = user_tensor.flatten().to(out.device), item_tensor.flatten().to(out.device)
        pair = (out[u] * out[i]).view(len(u), -1, 3 * self.embedding_dim).sum(-1)
        scores = pair.sum(-1).view(-1, 2)
        bpr = F.softplus(scores[:, 1] - scores[:, 0]).mean()
        cl, diff = out.new_zeros(()), out.new_zeros(())
        for feat in self._branch_features:
            v, t = feat[:, :self.embedding_dim], feat[:, self.embedding_dim:]
            cl = cl + ssl_loss(v, t, i, self.cl_tmp) + ssl_loss(v, t, u, self.cl_tmp)
            diff = diff + cal_diff_loss(feat, user_tensor, self.embedding_dim)
            diff = diff + cal_diff_loss(feat, item_tensor, self.embedding_dim)
        cl = cl * self.cl_loss_weight / len(self._branch_features)
        diff = diff * self.diff_loss_weight / len(self._branch_features)
        self.last_pair_score_std = pair.detach().std(0, unbiased=False).cpu().tolist()
        # Do not retain an extra reference to four training graphs after backward.
        self._branch_features = []
        return bpr + cl + diff, bpr, 0, cl, diff

    def full_sort_predict(self, interaction):
        if self.training:
            raise RuntimeError('Evaluation requires model.eval()')
        if self._eval_representation is None:
            self._eval_representation = self.forward().detach()
            self._branch_features = []
        out = self._eval_representation
        return out[interaction[0]] @ out[self.n_users:].T

    @torch.no_grad()
    def relation_diagnostics(self):
        weights = self.gate_weights()
        return dict(mode=self.relation_mode, graphs=self.graph_diagnostics,
                    relation_order=list(RELATIONS),
                    gate_mean=weights.mean(0).cpu().tolist(),
                    gate_user_std=weights.std(0, unbiased=False).cpu().tolist(),
                    gate_entropy=float(-(weights * weights.clamp_min(1e-12).log()).sum(1).mean()),
                    gate_max_possible_entropy=math.log(4),
                    last_train_weighted_branch_score_std=self.last_pair_score_std)
