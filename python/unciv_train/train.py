"""REINFORCE-with-baseline over the recorded learner steps.

advantage = return − running-mean baseline. Per-head masked log-prob (illegal logits → −1e9, then
log-softmax), gathered at the chosen action; a head whose action is −1 (no decision that turn)
contributes zero — and critically is NEVER negative-indexed (the gather index is clamped and then
masked out). loss = −advantage · Σ_head logp. CPU is fine.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .contract import Dims
from .dataset import TrainStep
from .model import PolicyNet


def _masked_logp(logits: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """log π(a|s) over the legal-masked softmax; 0 where the head did not act (action < 0)."""
    neg = torch.where(mask > 0, logits, torch.full_like(logits, -1e9))
    logp = F.log_softmax(neg, dim=1)
    acted = actions >= 0
    idx = actions.clamp(min=0).unsqueeze(1)               # clamp so -1 never negative-indexes
    chosen = logp.gather(1, idx).squeeze(1)
    return torch.where(acted, chosen, torch.zeros_like(chosen))


def train(
    steps: list[TrainStep],
    dims: Dims,
    *,
    epochs: int = 8,
    lr: float = 1e-3,
    seed: int = 0,
    entropy_coef: float = 0.0,
) -> tuple[PolicyNet, dict]:
    torch.manual_seed(seed)
    net = PolicyNet(dims)
    if not steps:
        return net, {"loss": 0.0, "baseline": 0.0, "n": 0, "note": "no steps"}

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    obs = torch.tensor(np.stack([s.obs for s in steps]))
    rets = torch.tensor([s.ret for s in steps], dtype=torch.float32)
    baseline = rets.mean()
    adv = rets - baseline                                  # undiscounted single terminal reward (D11)
    a_tech = torch.tensor([s.a_tech for s in steps])
    a_policy = torch.tensor([s.a_policy for s in steps])
    m_tech = torch.tensor(np.stack([s.mask_tech for s in steps]))
    m_policy = torch.tensor(np.stack([s.mask_policy for s in steps]))

    last = 0.0
    for _ in range(epochs):
        opt.zero_grad()
        tl, pl = net(obs)
        logp = _masked_logp(tl, a_tech, m_tech) + _masked_logp(pl, a_policy, m_policy)
        loss = -(adv * logp).mean()
        if entropy_coef > 0:
            ent = _entropy(tl, m_tech) + _entropy(pl, m_policy)
            loss = loss - entropy_coef * ent.mean()
        loss.backward()
        opt.step()
        last = loss.item()
    return net, {"loss": last, "baseline": float(baseline), "n": len(steps),
                 "ret_mean": float(rets.mean()), "ret_pos": int((rets > 0).sum())}


def _entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg = torch.where(mask > 0, logits, torch.full_like(logits, -1e9))
    p = F.softmax(neg, dim=1)
    logp = F.log_softmax(neg, dim=1)
    return -(p * logp).sum(dim=1)
