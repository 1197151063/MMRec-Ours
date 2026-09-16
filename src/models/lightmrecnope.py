"""Diagnostic reconstruction of the supplied LightMRec, removing ONLY both PEs.

Retains trainable raw features, the full-catalog MLP/BatchNorm/dropout, unfiltered
uniform negatives, decoupled loss, and raw-dot evaluation. Not a corrected baseline.
"""
import torch
from torch import nn
from torch.nn import functional as F
from common.abstract_recommender import GeneralRecommender


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, out_dim))

    def forward(self, x):
        return self.mlp(x)


class LightMRecNoPE(GeneralRecommender):
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self.alpha = float(config['alpha'])
        self.temperature = float(config['temperature'])
        self.num_negatives = int(config['num_negatives'])
        if self.v_feat is None or self.t_feat is None:
            raise ValueError('Both modality feature files are required')
        self.user_embedding = nn.Embedding(self.n_users, config['embedding_size'])
        nn.init.xavier_uniform_(self.user_embedding.weight)
        self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        self.image_trs = MLP(self.v_feat.shape[1], config['embedding_size'])
        self.text_trs = MLP(self.t_feat.shape[1], config['embedding_size'])

    def forward(self):
        visual = self.image_trs(self.image_embedding.weight)
        textual = self.text_trs(self.text_embedding.weight)
        return self.user_embedding.weight, self.alpha * textual + (1 - self.alpha) * visual

    def calculate_loss(self, interaction):
        users, positives = interaction[0], interaction[1]
        user_table, item_table = self.forward()
        negatives = torch.randint(self.n_items, (users.numel(), self.num_negatives), device=users.device)
        user = F.normalize(user_table[users], dim=-1)
        candidates = torch.cat((positives[:, None], negatives), dim=1)
        items = F.normalize(item_table[candidates], dim=-1)
        logits = (items * user[:, None]).sum(dim=-1) / self.temperature
        # Algebraically identical to -log(exp(pos)/sum(exp(neg))), evaluated stably.
        return (torch.logsumexp(logits[:, 1:], dim=-1) - logits[:, 0]).mean()

    def full_sort_predict(self, interaction):
        users = interaction[0] if isinstance(interaction, (list, tuple)) else interaction
        user_table, item_table = self.forward()
        return user_table[users] @ item_table.T
