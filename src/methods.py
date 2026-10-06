"""Meta-learning methods sharing one MLP encoder per task.

Conventions: sx/qx torch (n,d) float32, sy/qy (n,) int64 with local ids 0..N-1.
Each class exposes `train_step(...) -> float | None` (one meta-update) and
`predict(...) -> logits` (adaptation included, no grad). Timers live in run.py.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import Encoder


class ProtoNet(nn.Module):
    def __init__(self, in_dim, n_way):
        super().__init__()
        self.enc = Encoder(in_dim)
        self.n_way = n_way

    def _logits(self, sx, sy, qx):
        se, qe = self.enc(sx), self.enc(qx)
        protos = torch.stack([se[sy == k].mean(0) for k in range(self.n_way)])
        return -torch.cdist(qe, protos)

    def train_step(self, sx, sy, qx, qy, opt):
        opt.zero_grad()
        loss = F.cross_entropy(self._logits(sx, sy, qx), qy)
        loss.backward(); opt.step()
        return loss.item()

    @torch.no_grad()
    def predict(self, sx, sy, qx):
        return self._logits(sx, sy, qx)


class MatchingNet(nn.Module):
    def __init__(self, in_dim, n_way, tau=0.5, scale=10.0):
        super().__init__()
        self.enc = Encoder(in_dim)
        self.n_way = n_way; self.tau = tau; self.scale = scale

    def _logits(self, sx, sy, qx):
        se = F.normalize(self.enc(sx), dim=-1)
        qe = F.normalize(self.enc(qx), dim=-1)
        sim = (qe @ se.T) / self.tau
        attn = F.softmax(sim, dim=-1)
        onehot = F.one_hot(sy, self.n_way).float()
        return (attn @ onehot) * self.scale

    def train_step(self, sx, sy, qx, qy, opt):
        opt.zero_grad()
        loss = F.cross_entropy(self._logits(sx, sy, qx), qy)
        loss.backward(); opt.step()
        return loss.item()

    @torch.no_grad()
    def predict(self, sx, sy, qx):
        return self._logits(sx, sy, qx)


class _HeadModel(nn.Module):
    def __init__(self, in_dim, n_way):
        super().__init__()
        self.enc = Encoder(in_dim)
        self.head = nn.Linear(64, n_way)


def _sgd_inner(model, sx, sy, steps, lr, head_only=False):
    params = [p for n, p in model.named_parameters()
              if (not head_only or n.startswith("head."))]
    for _ in range(steps):
        loss = F.cross_entropy(model.enc(sx) @ model.head.weight.T + model.head.bias
                               if False else model.head(model.enc(sx)), sy)
        g = torch.autograd.grad(loss, params, create_graph=False)
        with torch.no_grad():
            for p, gi in zip(params, g):
                p.sub_(lr * gi)


class FOMAML(nn.Module):
    """First-order MAML: inner SGD on a deepcopy, grads copied back (no 2nd order)."""
    def __init__(self, in_dim, n_way, inner_steps=3, inner_lr=0.1):
        super().__init__()
        self.model = _HeadModel(in_dim, n_way)
        self.inner_steps = inner_steps; self.inner_lr = inner_lr

    def train_step(self, sx, sy, qx, qy, opt):
        fast = copy.deepcopy(self.model)
        fast.train()
        _sgd_inner(fast, sx, sy, self.inner_steps, self.inner_lr)
        opt.zero_grad()
        loss = F.cross_entropy(fast.head(fast.enc(qx)), qy)
        g = torch.autograd.grad(loss, fast.parameters())
        for p, gi in zip(self.model.parameters(), g):
            p.grad = gi.detach().clone()
        opt.step()
        return loss.item()

    def predict(self, sx, sy, qx):
        fast = copy.deepcopy(self.model)
        fast.train()
        with torch.enable_grad():
            _sgd_inner(fast, sx, sy, self.inner_steps, self.inner_lr)
        fast.eval()
        with torch.no_grad():
            return fast.head(fast.enc(qx))


class Reptile(nn.Module):
    def __init__(self, in_dim, n_way, inner_steps=3, inner_lr=0.1, outer_eps=0.1):
        super().__init__()
        self.model = _HeadModel(in_dim, n_way)
        self.inner_steps = inner_steps; self.inner_lr = inner_lr
        self.outer_eps = outer_eps

    def train_step(self, sx, sy, qx, qy, opt=None):
        fast = copy.deepcopy(self.model)
        fast.train()
        with torch.enable_grad():
            _sgd_inner(fast, sx, sy, self.inner_steps, self.inner_lr)
        with torch.no_grad():
            for p, pf in zip(self.model.parameters(), fast.parameters()):
                p.add_(self.outer_eps * (pf - p))
        with torch.no_grad():
            loss = F.cross_entropy(fast.head(fast.enc(qx)), qy).item()
        return loss

    def predict(self, sx, sy, qx):
        fast = copy.deepcopy(self.model)
        fast.train()
        with torch.enable_grad():
            _sgd_inner(fast, sx, sy, self.inner_steps, self.inner_lr)
        fast.eval()
        with torch.no_grad():
            return fast.head(fast.enc(qx))


class ANIL(nn.Module):
    """Head-only inner loop (Raghu et al. 2020)."""
    def __init__(self, in_dim, n_way, inner_steps=3, inner_lr=0.1):
        super().__init__()
        self.model = _HeadModel(in_dim, n_way)
        self.inner_steps = inner_steps; self.inner_lr = inner_lr

    def train_step(self, sx, sy, qx, qy, opt):
        fast = copy.deepcopy(self.model)
        fast.train()
        _sgd_inner(fast, sx, sy, self.inner_steps, self.inner_lr, head_only=True)
        opt.zero_grad()
        loss = F.cross_entropy(fast.head(fast.enc(qx)), qy)
        g = torch.autograd.grad(loss, fast.parameters())
        for p, gi in zip(self.model.parameters(), g):
            p.grad = gi.detach().clone()
        opt.step()
        return loss.item()

    def predict(self, sx, sy, qx):
        fast = copy.deepcopy(self.model)
        fast.train()
        with torch.enable_grad():
            _sgd_inner(fast, sx, sy, self.inner_steps, self.inner_lr, head_only=True)
        fast.eval()
        with torch.no_grad():
            return fast.head(fast.enc(qx))


class FineTune(nn.Module):
    """Transfer baseline: supervised pretrain on train pools, then SGD fine-tune.

    Pretraining uses global train-pool labels; adaptation fits a fresh N-way
    head (+ encoder) on the support set only.
    """

    def __init__(self, in_dim, n_way, ft_steps=20, ft_lr=0.05):
        super().__init__()
        self.enc = Encoder(in_dim)
        self.n_way = n_way
        self.ft_steps = ft_steps; self.ft_lr = ft_lr
        self.pre_head = None

    def pretrain(self, task, steps=400, batch=64, lr=1e-3, seed=0):
        import numpy as np
        from .tasks import sample_episode  # noqa (keeps import graph acyclic at module load)
        from . import tasks as T
        rng = np.random.RandomState(seed)
        # build a flat supervised pool from the train split
        if task == "vision":
            pools = T._digits_pseudo_classes(0)
            keys = sorted(pools)
            tkeys = [k for k in keys][:12]
            Xs = np.concatenate([pools[k] for k in tkeys])
            ys = np.concatenate([np.full(len(pools[k]), i) for i, k in enumerate(tkeys)])
        elif task == "tabular":
            pools = T._tabular_pools(0)
            Xs = np.concatenate([pools[("wine", c)] for c in range(3)] +
                                [pools[("iris", c)] for c in range(3)])
            ys = np.array([c for c in range(3) for _ in pools[("wine", c)]] +
                          [3 + c for c in range(3) for _ in pools[("iris", c)]])
            ys = np.array(ys)
        elif task == "chess":
            X, y = T._chess_dataset(0)
            n = len(X); cut = int(0.8 * n)
            # stratified-ish: recompute per class below; simple global cut is fine for pretrain
            Xs, ys = X[:cut], y[:cut]
        elif task == "coding":
            pools = T._coding_pairs(0)
            tids = [0, 1, 3, 4, 5, 6]
            Xs = np.concatenate([pools[t] for t in tids])
            ys = np.concatenate([np.full(len(pools[t]), i) for i, t in enumerate(tids)])
        else:
            raise ValueError(task)
        import numpy as _np
        Xs = _np.asarray(Xs, dtype=_np.float32)
        ncls = int(ys.max()) + 1
        self.pre_head = nn.Linear(64, ncls)
        opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.pre_head.parameters()), lr=lr)
        Xs_t = torch.from_numpy(Xs); ys_t = torch.from_numpy(np.asarray(ys, dtype=np.int64))
        for _ in range(steps):
            idx = rng.choice(len(Xs_t), batch)
            opt.zero_grad()
            loss = F.cross_entropy(self.pre_head(self.enc(Xs_t[idx])), ys_t[idx])
            loss.backward(); opt.step()

    def train_step(self, sx, sy, qx, qy, opt=None):
        return None  # no episodic meta-training; pretrain() covers learning

    def predict(self, sx, sy, qx):
        head = nn.Linear(64, self.n_way)
        opt = torch.optim.SGD(list(self.enc.parameters()) + list(head.parameters()), lr=self.ft_lr)
        self.enc.train(); head.train()
        with torch.enable_grad():
            for _ in range(self.ft_steps):
                opt.zero_grad()
                loss = F.cross_entropy(head(self.enc(sx)), sy)
                loss.backward(); opt.step()
        self.enc.eval(); head.eval()
        with torch.no_grad():
            return head(self.enc(qx))


class PAHO(nn.Module):
    """Prototype-Anchored Hypernetwork Optimizer (novel to this lab).

    Support embeddings -> prototypes c_k + DeepSets task vector z ->
    hypernetwork emits per-class delta weights + bias, anchored on c_k,
    plus a learned gate g(z) scaling one optional support-gradient step.
    Adaptation = 1 forward pass (+<=1 GD step): no unrolled inner loop.
    """

    def __init__(self, in_dim, n_way, emb=64, tdim=32, refine_lr=0.5):
        super().__init__()
        self.enc = Encoder(in_dim, out=emb)
        self.n_way = n_way; self.emb = emb
        self.task_mlp = nn.Sequential(nn.Linear(emb, 64), nn.ReLU(), nn.Linear(64, tdim))
        self.hyper = nn.Sequential(nn.Linear(emb + tdim, 64), nn.ReLU(),
                                   nn.Linear(64, emb + 1))
        self.gate = nn.Sequential(nn.Linear(tdim, 1), nn.Sigmoid())
        self.refine_lr = nn.Parameter(torch.tensor(float(refine_lr)))

    def _synthesize(self, sx, sy):
        se = self.enc(sx)
        protos = torch.stack([se[sy == k].mean(0) for k in range(self.n_way)])
        z = self.task_mlp(se.mean(0))
        deltas = torch.stack([self.hyper(torch.cat([protos[k], z])) for k in range(self.n_way)])
        dW, db = deltas[:, :self.emb], deltas[:, self.emb]
        W_proto = F.normalize(protos, dim=-1).T  # (emb, N)
        W = W_proto + dW.T
        b = db
        g = self.gate(z).squeeze(-1)
        return se, protos, z, W, b, g

    def _forward_logits(self, qx, W, b):
        return self.enc(qx) @ W + b

    def train_step(self, sx, sy, qx, qy, opt):
        opt.zero_grad()
        se, protos, z, W, b, g = self._synthesize(sx, sy)
        logits = self._forward_logits(qx, W, b)
        # one gated refinement step on support loss (differentiable)
        s_logits = se @ W + b
        s_loss = F.cross_entropy(s_logits, sy)
        gw, gb = torch.autograd.grad(s_loss, (W, b), create_graph=True)
        Wr = W - torch.sigmoid(g) * torch.clamp(self.refine_lr, 0, 1.0) * gw
        br = b - torch.sigmoid(g) * torch.clamp(self.refine_lr, 0, 1.0) * gb
        logits_r = self._forward_logits(qx, Wr, br)
        loss = 0.5 * (F.cross_entropy(logits, qy) + F.cross_entropy(logits_r, qy))
        loss.backward(); opt.step()
        return loss.item()

    @torch.no_grad()
    def _predict_nograd(self, sx, sy, qx):
        se, protos, z, W, b, g = self._synthesize(sx, sy)
        return self._forward_logits(qx, W, b), (W, b, g, se)

    def predict(self, sx, sy, qx):
        with torch.no_grad():
            logits0, (W, b, g, se) = self._predict_nograd(sx, sy, qx)
        # single gated GD correction (at most 1 step -> minimal learned optimizer)
        with torch.enable_grad():
            W_ = W.detach().requires_grad_(True); b_ = b.detach().requires_grad_(True)
            s_logits = se.detach() @ W_ + b_
            s_loss = F.cross_entropy(s_logits, sy.detach() if isinstance(sy, torch.Tensor) else sy)
            gw, gb = torch.autograd.grad(s_loss, (W_, b_))
            step = torch.sigmoid(g).item() * float(torch.clamp(self.refine_lr.detach(), 0, 1.0).item())
        with torch.no_grad():
            Wr, br = W - step * gw, b - step * gb
            qe = self.enc(qx)
            logits_r = qe @ Wr + br
        # pick the refined head (it trains to dominate); fallback keeps zero-step path valid
        return logits_r


def build(name, in_dim, n_way):
    name = name.lower()
    if name == "protonet":
        return ProtoNet(in_dim, n_way)
    if name == "matching":
        return MatchingNet(in_dim, n_way)
    if name == "fomaml":
        return FOMAML(in_dim, n_way)
    if name == "reptile":
        return Reptile(in_dim, n_way)
    if name == "anil":
        return ANIL(in_dim, n_way)
    if name == "finetune":
        return FineTune(in_dim, n_way)
    if name == "paho":
        return PAHO(in_dim, n_way)
    raise ValueError(name)
