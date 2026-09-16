"""Content recommendation with an optional item identity residual; no positional encoding."""
import torch
from torch import nn
from torch.nn import functional as F

from common.abstract_recommender import GeneralRecommender


class SIMMRec(GeneralRecommender):
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        dim = config['embedding_size']
        self.alpha = float(config['alpha'])
        self.gamma = float(config['item_id_weight'])
        self.temperature = float(config['temperature'])
        self.num_negatives = int(config['num_negatives'])
        self.reg_weight = float(config['reg_weight'])
        if not 0 <= self.alpha <= 1 or self.gamma < 0 or self.reg_weight < 0:
            raise ValueError('alpha must be in [0, 1]; residual and regularization weights must be nonnegative')
        if self.temperature <= 0 or self.num_negatives < 1:
            raise ValueError('temperature and num_negatives must be positive')
        self.representation = config['representation'] if 'representation' in config else 'content'
        if self.representation not in ('content', 'id'):
            raise ValueError('representation must be content or id')
        if self.representation == 'content':
            if self.v_feat is None or self.t_feat is None:
                raise ValueError('SIMMRec requires image_feat.npy and text_feat.npy')
            for features in (self.v_feat, self.t_feat):
                if features.ndim != 2 or features.shape[0] != self.n_items:
                    raise ValueError('Feature rows must match the complete item ID mapping')
                if not torch.isfinite(features).all():
                    raise ValueError('Modality features must be finite')
            self.image_embedding = nn.Embedding.from_pretrained(
                F.normalize(self.v_feat, dim=-1), freeze=True)
            self.text_embedding = nn.Embedding.from_pretrained(
                F.normalize(self.t_feat, dim=-1), freeze=True)
            self.image_trs = nn.Linear(self.v_feat.shape[1], dim)
            self.text_trs = nn.Linear(self.t_feat.shape[1], dim)
        self.user_embedding = nn.Embedding(self.n_users, dim)
        nn.init.normal_(self.user_embedding.weight, std=dim ** -0.5)
        self.item_embedding = nn.Embedding(self.n_items, dim) if self.gamma or self.representation == 'id' else None
        if self.item_embedding is not None:
            nn.init.normal_(self.item_embedding.weight, std=dim ** -0.5)

        # Sparse sorted keys allow rejection of training positives without a dense U x I mask.
        matrix = dataloader.inter_matrix(form='coo')
        keys = torch.as_tensor(matrix.row, dtype=torch.long) * self.n_items
        keys += torch.as_tensor(matrix.col, dtype=torch.long)
        keys = keys.unique(sorted=True)
        if keys.numel() == 0:
            raise ValueError('SIMMRec requires at least one training interaction')
        self.register_buffer('train_positive_keys', keys, persistent=False)
        counts = torch.bincount(keys // self.n_items, minlength=self.n_users)
        if (counts >= self.n_items).any():
            raise ValueError('Cannot sample negatives for a user who interacted with every item')
        self._eval_embeddings = None

    def item_representations(self, items):
        if self.representation == 'id':
            return F.normalize(self.item_embedding(items), dim=-1)
        visual = F.normalize(self.image_trs(self.image_embedding(items)), dim=-1)
        textual = F.normalize(self.text_trs(self.text_embedding(items)), dim=-1)
        content = self.alpha * textual + (1 - self.alpha) * visual
        if self.item_embedding is not None:
            content = content + self.gamma * self.item_embedding(items)
        return F.normalize(content, dim=-1)

    def forward(self):
        items = torch.arange(self.n_items, device=self.user_embedding.weight.device)
        return F.normalize(self.user_embedding.weight, dim=-1), self.item_representations(items)

    def train(self, mode=True):
        self._eval_embeddings = None
        return super().train(mode)

    def sample_negatives(self, users):
        negatives = torch.randint(self.n_items, (users.numel(), self.num_negatives), device=users.device)
        for _ in range(100):
            keys = users[:, None] * self.n_items + negatives
            positions = torch.searchsorted(self.train_positive_keys, keys)
            safe_positions = positions.clamp(max=self.train_positive_keys.numel() - 1)
            invalid = (positions < self.train_positive_keys.numel()) & (self.train_positive_keys[safe_positions] == keys)
            if not invalid.any():
                return negatives
            negatives[invalid] = torch.randint(self.n_items, (int(invalid.sum()),), device=users.device)
        # Exact complement fallback for unusually dense users; no unbounded rejection loop.
        catalog = torch.arange(self.n_items, device=users.device)
        for row in invalid.any(dim=1).nonzero().flatten().tolist():
            known = self.train_positive_keys[self.train_positive_keys // self.n_items == users[row]] % self.n_items
            allowed = catalog[~torch.isin(catalog, known)]
            count = int(invalid[row].sum())
            negatives[row, invalid[row]] = allowed[torch.randint(allowed.numel(), (count,), device=users.device)]
        return negatives

    def calculate_loss(self, interaction):
        self._eval_embeddings = None
        users, positives = interaction[0], interaction[1]
        negatives = self.sample_negatives(users)
        candidates = torch.cat((positives[:, None], negatives), dim=1)
        # Project each sampled item only once, rather than projecting the full catalog per batch.
        unique, inverse = candidates.unique(return_inverse=True)
        items = self.item_representations(unique)[inverse]
        user = F.normalize(self.user_embedding(users), dim=-1)
        logits = (items * user[:, None]).sum(dim=-1) / self.temperature
        ranking = F.cross_entropy(logits, torch.zeros_like(users))
        regularization = self.user_embedding(users).square().sum(dim=-1).mean()
        if self.item_embedding is not None:
            regularization = regularization + self.item_embedding(candidates).square().sum(dim=-1).mean()
        return ranking + self.reg_weight * regularization

    @torch.no_grad()
    def full_sort_predict(self, interaction):
        users = interaction[0] if isinstance(interaction, (list, tuple)) else interaction
        if self.training:
            user, items = self.forward()
        else:
            if self._eval_embeddings is None:
                self._eval_embeddings = self.forward()
            user, items = self._eval_embeddings
        return user[users] @ items.T
