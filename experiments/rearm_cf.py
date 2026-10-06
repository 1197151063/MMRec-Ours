"""Full REARM with separately switchable CF objectives and train-only priors.

All changes are adapters; the pinned upstream implementation remains untouched.
Losses use per-example means. The user's SSM is negative-only log-sum-exp.
"""
import numpy as np
import torch
import torch.nn.functional as F

from model import REARM
from utils.helper import propgt_info
from rearm_cf_config import validate
from rearm_cf_graphs import ultra_neighbors, cagcn, tensor


def sampled_loss(anchor, positive, negative, tau, include_positive=False):
    a, p, n = [F.normalize(x, dim=-1) for x in (anchor, positive, negative)]
    pos = (a * p).sum(-1) / tau
    neg = (a[:, None] * n).sum(-1) / tau
    denominator = torch.cat([pos[:, None], neg], 1) if include_positive else neg
    return torch.logsumexp(denominator, 1) - pos


class REARMCF(REARM):
    def __init__(self, config, dataset, options, graph_block=256):
        super().__init__(config, dataset)
        self.cf = validate({'options': options})['options']
        self.r = dataset.sparse_inter_matrix(form='csr').tocsr().astype(np.float32)
        self.r.sum_duplicates(); self.r.data[:] = 1
        from rearm_capacity import configure
        configure(self, self.cf)
        if self.cf['shared_users'] != 'none':
            from rearm_shared_users import configure_shared_users
            configure_shared_users(self)
        self.graph_block = graph_block
        self.epoch = 0
        self._eval_out = None
        self._projected = {}
        for name in ('image_i_trs', 'text_i_trs', 'image_u_trs', 'text_u_trs'):
            getattr(self, name).register_forward_hook(self._capture(name))
        du, di = np.asarray(self.r.sum(1)).ravel(), np.asarray(self.r.sum(0)).ravel()
        self.register_buffer('degree_u', torch.tensor(du, device=self.device))
        self.register_buffer('degree_i', torch.tensor(di, device=self.device))
        row, col = self.r.nonzero()
        keys = np.sort(row.astype(np.int64) * self.n_items + col)
        self.register_buffer('train_keys', torch.tensor(keys, device=self.device))
        beta = np.sqrt(du[row]+1) / np.maximum(du[row],1) / np.sqrt(di[col]+1)
        self.register_buffer('mean_beta', torch.tensor(float(beta.mean()), device=self.device))
        self.register_buffer('edge_ema', torch.zeros(len(keys), device=self.device))
        self.register_buffer('edge_seen', torch.zeros(len(keys), dtype=torch.bool, device=self.device))
        # Inverse directions support item-to-user NT loss without a dense U x I mask.
        self.graph_diagnostics = dict(train_edges=self.r.nnz)
        if self.cf['ultra_ii']:
            ids, weights = ultra_neighbors(self.r, self.cf['ultra_k'], graph_block)
            self.register_buffer('ultra_ids', torch.tensor(ids, device=self.device))
            self.register_buffer('ultra_weights', torch.tensor(weights, device=self.device))
            self.register_buffer('mean_omega_sum', torch.tensor(float(weights.sum(1).mean()), device=self.device))
            self.graph_diagnostics['ultra_nonzero_edges'] = int((weights > 0).sum())
        if self.cf['cir'] != 'none':
            adj = cagcn(self.r, self.cf['cir'], self.cf['cir_mix'], graph_block)
            self.norm_adj = tensor(adj, self.device)
            self.graph_diagnostics['cir'] = dict(similarity=self.cf['cir'], mix=self.cf['cir_mix'], edges=adj.nnz)
        if self.cf['freeze_features']:
            for name in ('image_embedding', 'text_embedding'):
                getattr(self, name).weight.requires_grad_(False)
            for name in ('user_v_prefer', 'user_t_prefer'):
                getattr(self, name).requires_grad_(False)
        self.before_epoch(0)

    def _capture(self, name):
        def hook(module, inputs, output):
            self._projected[name] = output
        return hook

    def meta_extra_share(self, id_embed, prefer_or_feat):
        self.cf_id = id_embed
        if self.cf['meta'] == 'none':
            # Keep the 192-d output and the third branch's original ID residual.
            return torch.zeros_like(id_embed)
        return super().meta_extra_share(id_embed, prefer_or_feat)

    def parameter_breakdown(self):
        from rearm_capacity import parameter_breakdown
        return parameter_breakdown(self)

    def forward(self):
        out = super().forward()
        self.cf_out = out
        if self.cf['interest_heads']:
            self.interest_context = self.interest_bank.context(self.fin_feat_prefer, self.n_users, self.embedding_dim)
        return out

    def pair_score(self, out, users, items):
        score = (out[users] * out[items + self.n_users]).sum(-1)
        if self.cf['interest_heads']:
            score = score + self.cf['interest_weight'] * self.interest_bank.paired(self.interest_context, users, items)
        return score

    def train(self, mode=True):
        self._eval_out = None
        return super().train(mode)

    def full_sort_predict(self, interaction):
        if self.training or torch.is_grad_enabled():
            out = self.forward()
        else:
            if self._eval_out is None:
                self._eval_out = self.forward()
            out = self._eval_out
        users = interaction[0]
        scores = out[users] @ out[self.n_users:].T
        if self.cf['interest_heads']:
            for begin in range(0, self.n_items, self.cf['interest_chunk']):
                end = min(begin + self.cf['interest_chunk'], self.n_items)
                scores[:, begin:end] = scores[:, begin:end] + self.cf['interest_weight'] * self.interest_bank.block(self.interest_context, users, begin, end)
        return scores

    def before_epoch(self, epoch):
        self.epoch = epoch
        self.stats = {}
        self.batches = 0

    def diagnostics(self):
        result = dict(epoch=self.epoch, batches=self.batches,
                    training_profile=getattr(self, 'training_profile', {}),
                    losses={k: v / max(1, self.batches) for k, v in self.stats.items()})
        if self.cf['interest_heads'] and hasattr(self, 'interest_context'):
            with torch.no_grad():
                k = self.cf['interest_heads']
                result['interests'] = dict(heads=k, weight=self.cf['interest_weight'])
                for name, (q, _) in zip(('visual', 'text'), self.interest_context):
                    # Average pairwise cosine between unit-norm interest heads.
                    result['interests'][name + '_mean_head_cosine'] = float(
                        ((q.sum(1).square().sum(-1) - k)/(k*(k-1))).mean()) if k > 1 else None
        if self.cf['shared_users'] != 'none' and hasattr(self, 'cf_out'):
            from rearm_shared_users import geometry
            result['user_geometry'] = geometry(self)
        return result

    def contains(self, users, items):
        keys = users * self.n_items + items
        positions = torch.searchsorted(self.train_keys, keys)
        return (positions < len(self.train_keys)) & (self.train_keys[positions.clamp_max(len(self.train_keys)-1)] == keys)

    @torch.no_grad()
    def sample(self, anchors, count, reverse=False, exclude=True, popularity=0.):
        size = self.n_users if reverse else self.n_items
        degrees = self.degree_u if reverse else self.degree_i
        forbidden = self.degree_i[anchors] if reverse else self.degree_u[anchors]
        if exclude and torch.any(forbidden >= size):
            raise ValueError('No eligible training-negative candidate for an anchor')
        def draw(n):
            if popularity:
                return torch.multinomial((degrees + 1).pow(popularity), n, replacement=True)
            return torch.randint(size, (n,), device=self.train_keys.device)
        candidates = draw(anchors.numel() * count).reshape(-1, count)
        if exclude:
            for _ in range(100):
                invalid = self.contains(candidates, anchors[:, None]) if reverse else self.contains(anchors[:, None], candidates)
                if not invalid.any():
                    return candidates
                candidates[invalid] = draw(int(invalid.sum()))
            # Finite exact-complement fallback for unusually dense histories.
            for b in invalid.any(1).nonzero().flatten().tolist():
                all_ids = torch.arange(size, device=candidates.device)
                bad = self.contains(all_ids, anchors[b]) if reverse else self.contains(anchors[b], all_ids)
                eligible = all_ids[~bad]
                candidates[b, invalid[b]] = eligible[torch.randint(len(eligible), (int(invalid[b].sum()),), device=eligible.device)]
        return candidates

    def id_parts(self):
        """Exact split of the linear UI ID branch by starting endpoint type.

        Homogeneous processing/normalization is retained; this is NOT a linear
        decomposition of REARM's final nonlinear attention/meta representation.
        """
        d = self.embedding_dim
        items = torch.cat([self.item_id_embedding.weight, self._projected['image_i_trs'], self._projected['text_i_trs']], 1)
        users = torch.cat([self.user_id_embedding.weight, self._projected['image_u_trs'], self._projected['text_u_trs']], 1)
        u0 = F.normalize(propgt_info(users, self.n_uu_layers, self.stre_uu_graph, last_layer=True), dim=1)[:, :d]
        i0 = F.normalize(propgt_info(items, self.n_ii_layers, self.stre_ii_graph, last_layer=True), dim=1)[:, :d]
        from_users = propgt_info(torch.cat([u0, torch.zeros_like(i0)], 0), self.n_layers, self.norm_adj)
        return from_users, self.cf_id - from_users

    def nt_loss(self, users, pos, negatives):
        a, b = self.id_parts()
        total = self.cf_id
        norms = total.norm(dim=1, keepdim=True).clamp_min(1e-12)
        full = total / norms
        # Shared full norms are essential: alpha=(1,1) exactly recovers SSM.
        weighted = (self.cf['nt_user'] * a + self.cf['nt_item'] * b) / norms
        def direction(anchor, positive, negative):
            positive_score = (full[anchor] * full[positive]).sum(-1) / self.cf['tau']
            negative_score = (full[anchor, None] * weighted[negative]).sum(-1) / self.cf['tau']
            return torch.logsumexp(torch.cat([positive_score[:, None], negative_score], 1), 1) - positive_score
        value = direction(users, pos + self.n_users, negatives + self.n_users)
        if self.cf['nt_bidirectional']:
            reverse = self.sample(pos, self.cf['negatives'], reverse=True)
            value = .5 * (value + direction(pos + self.n_users, users, reverse))
        return value.mean()

    @torch.no_grad()
    def denoise_weights(self, users, positives, per_example):
        c = self.cf
        progress = min(1., max(0., (self.epoch - c['warmup'] + 1) / c['ramp']))
        weights = torch.ones_like(per_example)
        if c['denoise'] == 'ema':
            idx = torch.searchsorted(self.train_keys, users * self.n_items + positives)
            unique, inverse = idx.unique(return_inverse=True)
            means = torch.zeros(len(unique), device=idx.device).scatter_add_(0, inverse, per_example)
            means /= torch.bincount(inverse, minlength=len(unique))
            old = self.edge_ema[unique]
            self.edge_ema[unique] = torch.where(self.edge_seen[unique], c['ema_decay'] * old + (1-c['ema_decay']) * means, means)
            self.edge_seen[unique] = True
            difficulty = self.edge_ema[idx]
            # Smooth rank-based downweighting; no claim these are known noisy labels.
            rank = difficulty.argsort().argsort().float() / max(1, len(difficulty)-1)
            weights = 1 - progress * c['drop_rate'] * rank
        elif c['denoise'] == 'trim':
            drop = min(len(weights)-1, int(len(weights) * c['drop_rate'] * progress))
            if drop:
                weights[per_example.topk(drop).indices] = 0
        return weights / weights.mean().clamp_min(1e-12)

    def loss(self, user_tensor, item_tensor):
        # This computes the original forward and its CL/diff losses unchanged.
        original = super().loss(user_tensor, item_tensor)
        c = self.cf
        users = user_tensor[:, 0].to(self.train_keys.device)
        positive = item_tensor[:, 0].to(users.device) - self.n_users
        negative = item_tensor[:, 1].to(users.device) - self.n_users
        out = self.cf_out
        if c['negative_pool'] > 1 or c['negative_popularity']:
            candidates = self.sample(users, c['negative_pool'], popularity=c['negative_popularity'])
            with torch.no_grad():
                scores = (out[users, None] * out[candidates + self.n_users]).sum(-1)
                rank = scores.argsort(dim=1, descending=True)[:, c['negative_rank']]
                negative = candidates.gather(1, rank[:, None]).squeeze(1)
        ps = self.pair_score(out, users, positive)
        ns = self.pair_score(out, users, negative)
        raw_main = F.softplus(ns - ps)
        need_samples = c['aux'] in ('ssm', 'both') or c['main'] == 'ssm' or c['nt_weight']
        sampled = self.sample(users, c['negatives'], exclude=c['exclude_train']) if need_samples else None
        if c['main'] == 'ssm':
            raw_main = sampled_loss(out[users], out[positive+self.n_users], out[sampled+self.n_users], c['tau'], c['include_positive'])
        elif c['main'] == 'bce':
            raw_main = F.softplus(-ps) + F.softplus(ns)
        weights = self.denoise_weights(users, positive, raw_main.detach())
        if c['degree_power']:
            beta = (self.degree_u[users] + 1).sqrt() / self.degree_u[users].clamp_min(1) / (self.degree_i[positive] + 1).sqrt()
            beta = beta.pow(c['degree_power'])
            weights = weights * beta / beta.mean().clamp_min(1e-12)
        main = (weights * raw_main).mean()
        aux = original[3] if c['aux'] in ('official', 'both') else out.new_zeros(())
        ssm = out.new_zeros(())
        if c['aux'] in ('ssm', 'both'):
            if c['target'] == 'projected':
                v, t = self._projected['image_i_trs'], self._projected['text_i_trs']
            else:
                v, t = self.fin_feat_prefer[self.n_users:].split(self.embedding_dim, dim=1)
            lv = sampled_loss(self.cf_id[users], v[positive], v[sampled], c['tau'], c['include_positive'])
            lt = sampled_loss(self.cf_id[users], t[positive], t[sampled], c['tau'], c['include_positive'])
            ssm = c['ssm_weight'] * (weights * (c['image_weight']*lv + (1-c['image_weight'])*lt)).mean()
            aux = aux + ssm
        ui, ii, nt = [out.new_zeros(()) for _ in range(3)]
        if c['ultra_ui']:
            bu = (self.degree_u[users]+1).sqrt() / self.degree_u[users].clamp_min(1)
            bp, bn = bu / (self.degree_i[positive]+1).sqrt(), bu / (self.degree_i[negative]+1).sqrt()
            # Raw UltraGCN degree coefficients, globally mean-scaled for stable lambda.
            ui = c['ultra_ui'] * (bp*F.softplus(-ps) + bn*F.softplus(ns)).mean() / self.mean_beta.clamp_min(1e-12)
        if c['ultra_ii']:
            neighbor = self.ultra_ids[positive]
            omega = self.ultra_weights[positive]
            scores = (out[users, None] * out[neighbor+self.n_users]).sum(-1)
            # Mean of per-positive neighbor sums, scaled by full-graph mean omega sum.
            norm = self.mean_omega_sum.clamp_min(1e-12)
            ii = c['ultra_ii'] * (omega * F.softplus(-scores)).sum(1).mean() / norm
        if c['nt_weight']:
            nt = c['nt_weight'] * self.nt_loss(users, positive, sampled)
        regularizer = ui + ii + nt
        total = main + aux + original[4] + regularizer
        if not torch.isfinite(total):
            raise FloatingPointError('Non-finite REARM CF loss')
        self.batches += 1
        for k, v in dict(main=main, ssm=ssm, aux=aux, diff=original[4], ultra_ui=ui, ultra_ii=ii,
                         nt=nt, total=total, weight_min=weights.min(), weight_max=weights.max()).items():
            self.stats[k] = self.stats.get(k, 0.) + float(torch.as_tensor(v).detach())
        return total, main, regularizer, aux, original[4]
