"""Finite-loss training and in-memory teacher/student orchestration."""
import random
import time
import numpy as np
import torch

from rearm_cf_config import DEFAULTS
from rearm_cf_graphs import graphda, tensor


def train_cf(net, epoch):
    net.model.before_epoch(epoch)
    net.model.train()
    sums, batches = np.zeros(5), 0
    for users, items in net.train_data:
        net.optimizer.zero_grad(set_to_none=True)
        losses = net.model.loss(users, items)
        if not all(torch.isfinite(torch.as_tensor(x)).all() for x in losses):
            raise FloatingPointError('Non-finite loss before backward')
        losses[0].backward()
        net.optimizer.step()
        sums += [float(torch.as_tensor(x).detach()) for x in losses]
        batches += 1
    if not batches:
        raise ValueError('Empty training loader')
    return [torch.tensor(x / batches) for x in sums]


def prepare_teacher(net):
    """Fixed-epoch official-CL REARM teacher, then a freshly initialized student.

    No validation/test calls and no disk checkpoints. Initial parameter values
    and RNG states are held in CPU RAM, restored before student optimization.
    Teacher's optimizer moments are discarded. Main-stage epoch numbering and
    best-validation tracking have not started yet.
    """
    model = net.model
    options = model.cf
    if not options['graphda_epochs']:
        return None
    started = time.time()
    initial = {k: p.detach().cpu().clone() for k, p in model.named_parameters()}
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    model.cf = dict(DEFAULTS, aux='official')
    # Teacher uses the same architecture/lr as its student, and only BPR+CL+diff.
    for epoch in range(options['graphda_epochs']):
        losses = train_cf(net, epoch)
        net.lr_scheduler.step()
        net.logger.info('GraphDA teacher %d/%d loss=%.6f (training only)', epoch+1,
                        options['graphda_epochs'], losses[0])
    model.eval()
    with torch.no_grad():
        representations = model.forward().detach()
        adj, stats = graphda(representations, model.n_users, model.r,
                            k=options['graphda_k'], homogeneous_k=options['graphda_homogeneous_k'],
                            mix=options['graphda_mix'], block=model.graph_block)
        model.norm_adj = tensor(adj, model.device)
        for k, p in model.named_parameters():
            p.copy_(initial[k].to(p.device))
        model.edge_ema.zero_(); model.edge_seen.zero_()
    del initial, representations
    model.cf = options
    random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
    if rng[3] is not None:
        torch.cuda.set_rng_state_all(rng[3])
    net.optimizer = torch.optim.AdamW(model.parameters(), lr=net.learning_rate, weight_decay=net.reg_weight)
    schedule = net.config.learning_rate_scheduler
    net.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(net.optimizer, lambda e: schedule[0] ** (e / schedule[1]))
    model.before_epoch(0); model.train()
    stats.update(teacher_epochs=options['graphda_epochs'], seconds=time.time()-started,
                 student_reinitialized=True, teacher_objective='official BPR+CL+diff',
                 checkpoints_saved=False)
    model.graph_diagnostics['graphda'] = stats
    net.logger.info('GraphDA student starts from original initialization: %s', stats)
    return stats
