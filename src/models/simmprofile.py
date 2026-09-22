"""Training-history means with shared modality projectors; no iterative propagation."""
import logging
import torch
from torch.nn import functional as F
from models.simmrec import SIMMRec


class SIMMProfile(SIMMRec):
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        if self.gamma or self.representation != 'content':
            raise ValueError('SIMMProfile requires frozen content without item IDs')
        self.profile_weight = float(config['profile_weight'])
        self.profile_mode = config['profile_mode']
        self.history_power = float(config['history_power'])
        self.leave_one_out = bool(config['leave_one_out'])
        self.use_user_id = bool(config['use_user_id'])
        self.detach_profile = bool(config['detach_profile'])
        if self.profile_weight < 0 or self.history_power < 0 or self.profile_mode not in ('own', 'shuffled'):
            raise ValueError('Invalid profile configuration')
        if not self.use_user_id and self.profile_weight == 0:
            raise ValueError('User representation would be identically zero')
        self.user_embedding.weight.requires_grad_(self.use_user_id)
        # Build binary weighted history sums once, on CPU; features stay frozen.
        keys = self.train_positive_keys.cpu()
        u, i = keys // self.n_items, keys % self.n_items
        popularity = torch.bincount(i, minlength=self.n_items).float().clamp_min(1)
        weights = popularity.pow(-self.history_power)
        matrix = torch.sparse_coo_tensor(torch.stack((u, i)), weights[i],
                                         (self.n_users, self.n_items)).coalesce()
        mass = torch.zeros(self.n_users).scatter_add_(0, u, weights[i])
        source = torch.arange(self.n_users)
        if self.profile_mode == 'shuffled':
            generator = torch.Generator().manual_seed(int(config['profile_seed']))
            source = torch.randperm(self.n_users, generator=generator)
        self.register_buffer('profile_source', source)
        self.register_buffer('history_mass', mass)
        self.register_buffer('history_item_weights', weights)
        for name, embedding in [('visual_sum', self.image_embedding), ('text_sum', self.text_embedding)]:
            self.register_buffer(name, torch.sparse.mm(matrix, embedding.weight.detach().cpu()))
        logging.getLogger().info('PROFILE mode=%s weight=%s LOO=%s user_id=%s power=%s; binary TRAIN histories only',
                                self.profile_mode, self.profile_weight, self.leave_one_out,
                                self.use_user_id, self.history_power)

    def history_means(self, users, targets=None):
        sources = self.profile_source[users]
        mass = self.history_mass[sources]
        visual, text = self.visual_sum[sources], self.text_sum[sources]
        if targets is not None:
            # Also correct for shuffled controls: subtract only if the target belongs
            # to the source user's history. Never subtract an absent item.
            keys = sources * self.n_items + targets
            positions = torch.searchsorted(self.train_positive_keys, keys)
            present = (positions < self.train_positive_keys.numel()) & (
                self.train_positive_keys[positions.clamp_max(self.train_positive_keys.numel()-1)] == keys)
            removed = self.history_item_weights[targets] * present
            visual = visual - removed[:, None] * self.image_embedding(targets)
            text = text - removed[:, None] * self.text_embedding(targets)
            mass = (mass - removed).clamp_min(0)
        nonempty = mass > 1e-7
        scale = mass.clamp_min(1e-7)[:, None]
        return visual / scale, text / scale, nonempty

    def user_representations(self, users, targets=None):
        residual = self.user_embedding(users)
        if not self.use_user_id:
            residual = torch.zeros_like(residual)
        if self.profile_weight == 0:
            return F.normalize(residual, dim=-1)
        visual, text, nonempty = self.history_means(users, targets)
        visual = F.normalize(self.image_trs(visual), dim=-1)
        text = F.normalize(self.text_trs(text), dim=-1)
        profile = F.normalize(self.alpha * text + (1-self.alpha) * visual, dim=-1)
        # Empty histories must not turn projection biases into a synthetic history.
        profile = profile * nonempty[:, None]
        if self.detach_profile:
            profile = profile.detach()
        return F.normalize(residual + self.profile_weight * profile, dim=-1)

    def forward(self):
        device = self.user_embedding.weight.device
        users = torch.arange(self.n_users, device=device)
        # Bound temporary memory for high-dimensional history projections.
        user = torch.cat([self.user_representations(chunk) for chunk in users.split(2048)])
        return user, self.item_representations(torch.arange(self.n_items, device=device))

    def calculate_loss(self, interaction):
        self._eval_embeddings = None
        users, positives = interaction[0], interaction[1]
        candidates = torch.cat((positives[:, None], self.sample_negatives(users)), dim=1)
        unique, inverse = candidates.unique(return_inverse=True)
        items = self.item_representations(unique)[inverse]
        user = self.user_representations(users, positives if self.leave_one_out else None)
        logits = (items * user[:, None]).sum(-1) / self.temperature
        ranking = F.cross_entropy(logits, torch.zeros_like(users))
        regularization = self.user_embedding(users).square().sum(-1).mean() if self.use_user_id else 0.
        return ranking + self.reg_weight * regularization
