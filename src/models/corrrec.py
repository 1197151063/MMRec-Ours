"""Order-correlated initialization and isolated recommendation components."""
import math
import logging
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from models.lightmrecnope import LightMRecNoPE
from models.lightmrecorder import make_positions


def correlated_normal(n, dim, rho, eta, std, seed, shared=False):
    if not 0 <= rho <= 1 or not 0 <= eta < 1 or std <= 0:
        raise ValueError('Require rho in [0,1], eta in [0,1), std > 0')
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, dim))
    innovation = noise if shared else rng.standard_normal((n, dim))
    z = np.empty_like(noise)
    z[0] = innovation[0]  # Stationary N(0,I) initial state.
    for u in range(1, n):
        z[u] = eta*z[u-1] + math.sqrt(1-eta*eta)*innovation[u]
    return torch.from_numpy((std*(math.sqrt(1-rho)*noise + math.sqrt(rho)*z)).astype(np.float32))


def as_torch(matrix):
    coo = matrix.tocoo()
    return torch.sparse_coo_tensor(torch.from_numpy(np.stack((coo.row, coo.col))).long(),
                                   torch.from_numpy(coo.data.astype(np.float32)), coo.shape).coalesce()


def topk_rows(values, k, offset=0):
    # Exclude diagonal and nonpositive similarities; avoid a catalog-square tensor.
    values = values.copy()
    values[np.arange(len(values)), np.arange(len(values))+offset] = 0
    k = min(k, values.shape[1]-1)
    cols = np.argpartition(values, -k, axis=1)[:, -k:]
    rows = np.repeat(np.arange(len(values)), k)
    vals = values[rows, cols.ravel()]
    keep = vals > 0
    return sp.csr_matrix((vals[keep], (rows[keep], cols.ravel()[keep])), shape=values.shape)


def content_knn(features, k, block=256):
    features = F.normalize(features.detach(), dim=-1)
    pieces = []
    for start in range(0, len(features), block):
        sim = (features[start:start+block] @ features.T).cpu().numpy()
        # Reference uses binary modality KNN adjacency, not similarity weights.
        piece = topk_rows(sim, k, start)
        piece.data[:] = 1
        pieces.append(piece)
    return sp.vstack(pieces, format='csr')


def ppr_items(binary, k, alpha=.15, iterations=20, block=128):
    """Exact finite polynomial from the supplied code, evaluated in row blocks.

    alpha is propagation probability (restart=1-alpha). No intermediate top-k;
    zero transition diagonal is NOT renormalized, matching the supplied code.
    Final diagonal is excluded to make k count actual neighbors.
    """
    du = np.asarray(binary.sum(1)).ravel().clip(1)
    di = np.asarray(binary.sum(0)).ravel().clip(1)
    transition = sp.diags(1/di) @ binary.T @ sp.diags(1/du) @ binary
    transition = transition.tocsr(); transition.setdiag(0); transition.eliminate_zeros()
    n = binary.shape[1]
    pieces = []
    for start in range(0, n, block):
        size = min(block, n-start)
        eye = np.zeros((size, n), dtype=np.float32)
        eye[np.arange(size), np.arange(size)+start] = 1
        x = (1-alpha)*eye
        for _ in range(iterations):
            x = alpha*(transition.T @ x.T).T + (1-alpha)*eye
        pieces.append(topk_rows(x, k, start))
    return sp.vstack(pieces, format='csr')


class CorrRec(LightMRecNoPE):
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self.layers = int(config['ui_layers'])
        self.item_weight = float(config['item_id_weight'])
        self.mm_weight = float(config['mm_weight'])
        self.aux_weight = float(config['aux_weight'])
        self.visual_weight = float(config['visual_weight'])
        self.dropout_rate = float(config['user_dropout'])
        self.loss_kind = config['loss_kind']
        self.reg = float(config['corr_reg'])
        self.eval_cosine = bool(config['eval_cosine'])
        if self.layers < 0 or self.loss_kind not in ('ssm', 'bpr'):
            raise ValueError('Invalid layer/loss setting')
        if min(self.item_weight,self.mm_weight,self.aux_weight,self.reg,self.visual_weight) < 0 or not 0 <= self.dropout_rate < 1:
            raise ValueError('Invalid component weight/dropout')
        dim = config['embedding_size']
        if config['corr_init']:
            positions,_ = make_positions(dataloader.dataset_bk.df, self.USER_ID, self.ITEM_ID,
                                         self.n_users, self.n_items, config['order_source'], int(config['order_seed']))
            weights = correlated_normal(self.n_users, dim, float(config['rho']), float(config['eta']),
                                        float(config['init_std']), int(config['init_seed']), bool(config['shared_noise']))
            with torch.no_grad():
                self.user_embedding.weight.copy_(weights[positions])
        # Independent RNG stream keeps optional item lookup from changing training RNG.
        self.item_embedding = None
        if self.item_weight:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(config['init_seed'])+1)
                self.item_embedding = nn.Embedding(self.n_items, dim)
                nn.init.xavier_uniform_(self.item_embedding.weight)
        if bool(config['freeze_features']):
            self.image_embedding.weight.requires_grad_(False)
            self.text_embedding.weight.requires_grad_(False)
        matrix = dataloader.inter_matrix(form='coo').tocsr().astype(np.float32)
        matrix.sum_duplicates(); matrix.data[:] = 1
        if self.layers:
            adj = sp.bmat([[None,matrix],[matrix.T,None]], format='csr')
            degree = np.asarray(adj.sum(1)).ravel().clip(1)
            d = sp.diags(degree**-.5)
            self.register_buffer('ui_adj', as_torch(d@adj@d))
        if self.mm_weight:
            k = int(config['knn_k'])
            mix = float(config['ppr_mix'])
            if k < 1 or self.n_items < 2 or not 0 <= mix <= 1:
                raise ValueError('Invalid item graph configuration')
            visual = content_knn(self.image_embedding.weight, k)
            textual = content_knn(self.text_embedding.weight, k)
            adj = .1*visual + .9*textual
            if mix:
                adj = (1-mix)*adj + mix*ppr_items(matrix, k)
            # Explicit symmetric normalization, no implicit directed-edge conventions.
            adj = (adj+adj.T)*.5
            degree = np.asarray(adj.sum(1)).ravel().clip(1e-12)
            d = sp.diags(degree**-.5)
            self.register_buffer('mm_adj', as_torch(d@adj@d))
        logging.getLogger().info('CORR rho=%s eta=%s std=%s order=%s shared_noise=%s ui=%s mm=%s aux=%s',
                                config['rho'],config['eta'],config['init_std'],config['order_source'],
                                config['shared_noise'],self.layers,self.mm_weight,self.aux_weight)

    def encode(self):
        visual = self.image_trs(self.image_embedding.weight)
        text = self.text_trs(self.text_embedding.weight)
        item = self.alpha*text + (1-self.alpha)*visual
        if self.item_embedding is not None:
            item = item + self.item_weight*self.item_embedding.weight
        user = self.user_embedding.weight
        if self.layers:
            x = torch.cat((user,item)); outputs = [x]
            for _ in range(self.layers):
                x = torch.sparse.mm(self.ui_adj,x); outputs.append(x)
            user,item = torch.stack(outputs).mean(0).split((self.n_users,self.n_items))
        if self.mm_weight:
            item = item + self.mm_weight*torch.sparse.mm(self.mm_adj,item)
        return user,item,visual,text

    def forward(self):
        user,item,_,_ = self.encode()
        return user,item

    def contrastive(self, user, items, candidates):
        user = F.normalize(F.dropout(user, p=self.dropout_rate, training=self.training),dim=-1)
        logits = (F.normalize(items[candidates],dim=-1)*user[:,None]).sum(-1)/self.temperature
        return (torch.logsumexp(logits[:,1:],dim=-1)-logits[:,0]).mean()

    def calculate_loss(self, interaction):
        users,positives = interaction[:2]
        user,item,visual,text = self.encode()  # one forward, reuse across all objectives
        neg = torch.randint(self.n_items,(len(users),self.num_negatives),device=users.device)
        candidates = torch.cat((positives[:,None],neg),dim=1)
        if self.loss_kind == 'bpr':
            pos = (user[users]*item[positives]).sum(-1)
            negatives = (user[users,None]*item[neg]).sum(-1)
            loss = F.softplus(negatives-pos[:,None]).mean()
        else:
            loss = self.contrastive(user[users],item,candidates)
        if self.aux_weight:
            loss = loss + self.aux_weight*(self.contrastive(user[users],text,candidates)+
                                          self.visual_weight*self.contrastive(user[users],visual,candidates))
        if self.reg:
            penalty = self.user_embedding(users).square().sum(-1).mean()
            if self.item_embedding is not None:
                penalty = penalty+self.item_embedding(candidates).square().sum(-1).mean()
            loss = loss+self.reg*penalty
        return loss

    def full_sort_predict(self, interaction):
        users = interaction[0] if isinstance(interaction,(list,tuple)) else interaction
        user,item = self.forward()
        if self.eval_cosine:
            user,item = F.normalize(user,dim=-1),F.normalize(item,dim=-1)
        return user[users]@item.T
