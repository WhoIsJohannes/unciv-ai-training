"""Trainers over recorded learner steps.

Two algorithms, dispatched by `--variant` in run_loop:

* `train_reinforce` — v1: REINFORCE with a running-mean baseline (advantage = return − mean).
  Preserved unchanged for the attributable `v1-reinforce` baseline curve.
* `train_actor_critic_blind` / `train_actor_critic_rich` — v2: a learned value critic with
  Generalized Advantage Estimation. Reward is TERMINAL-ONLY ±1 (0 elsewhere); `compute_gae`
  derives per-state advantages and value targets, V(terminal)=0. Advantages/returns are
  recomputed each epoch from the CURRENT critic (no frozen importance ratio → no stale-ratio
  instability → no PPO clip needed; an optional clip knob exists but is off by default). The
  per-head masked-logp machinery + the −1/no-action handling are shared with v1 VERBATIM.

The reward is the ONLY external signal; the critic is the only new credit mechanism (learned from
the terminal outcome — not a shaped intermediate reward).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .contract import Dims
from .dataset import TrainStep, TrainTrajectory
from .model import PolicyNet


def _masked_logp(logits: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """log π(a|s) over the legal-masked softmax; 0 where the head did not act (action < 0)."""
    neg = torch.where(mask > 0, logits, torch.full_like(logits, -1e9))
    logp = F.log_softmax(neg, dim=1)
    acted = actions >= 0
    idx = actions.clamp(min=0).unsqueeze(1)               # clamp so -1 never negative-indexes
    chosen = logp.gather(1, idx).squeeze(1)
    return torch.where(acted, chosen, torch.zeros_like(chosen))


def _entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg = torch.where(mask > 0, logits, torch.full_like(logits, -1e9))
    p = F.softmax(neg, dim=1)
    logp = F.log_softmax(neg, dim=1)
    return -(p * logp).sum(dim=1)


def compute_gae(rewards, values, gamma: float = 0.99, lam: float = 0.95):
    """Episodic GAE over ONE trajectory. reward terminal-only (0 except the last step);
    V(terminal)=0 bootstrap. Returns (advantages, returns) as float32 arrays of the same length.

      delta_t = r_t + gamma*V_{t+1} - V_t      (V_T := 0)
      A_t     = delta_t + gamma*lam*A_{t+1}     (A_T := 0)
      R_t     = A_t + V_t
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in range(n - 1, -1, -1):
        next_v = values[t + 1] if t + 1 < n else 0.0   # V(terminal) = 0
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    ret = adv + values
    return adv, ret


# --------------------------------------------------------------------------------------------------
# v1: REINFORCE with running-mean baseline (preserved for the attributable baseline)
# --------------------------------------------------------------------------------------------------
def train_reinforce(
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
    adv = rets - baseline                                  # undiscounted single terminal reward
    a_tech = torch.tensor([s.a_tech for s in steps])
    a_policy = torch.tensor([s.a_policy for s in steps])
    m_tech = torch.tensor(np.stack([s.mask_tech for s in steps]))
    m_policy = torch.tensor(np.stack([s.mask_policy for s in steps]))

    last = 0.0
    for _ in range(epochs):
        opt.zero_grad()
        tl, pl, _ = net(obs)                               # value head ignored by REINFORCE
        logp = _masked_logp(tl, a_tech, m_tech) + _masked_logp(pl, a_policy, m_policy)
        loss = -(adv * logp).mean()
        if entropy_coef > 0:
            ent = _entropy(tl, m_tech) + _entropy(pl, m_policy)
            loss = loss - entropy_coef * ent.mean()
        loss.backward()
        opt.step()
        last = loss.item()
    return net, {"loss": last, "baseline": float(baseline), "n": len(steps),
                 "value_loss": 0.0, "entropy": 0.0, "mean_adv": float(adv.mean()),
                 "mean_value": 0.0, "grad_norm": 0.0,
                 "ret_mean": float(rets.mean()), "ret_pos": int((rets > 0).sum())}


# --------------------------------------------------------------------------------------------------
# v2: actor-critic + GAE (shared optimizer core; blind & rich wrappers build the batch)
# --------------------------------------------------------------------------------------------------
def _optimize_actor_critic(
    net,
    forward_fn,
    *,
    a_tech: torch.Tensor,
    a_policy: torch.Tensor,
    m_tech: torch.Tensor,
    m_policy: torch.Tensor,
    rewards_np: np.ndarray,
    traj_lens: list[int],
    n_pos: int,
    epochs: int,
    lr: float,
    gamma: float,
    lam: float,
    value_coef: float,
    entropy_coef: float,
    clip_eps: float | None,
    norm_adv: bool,
) -> tuple[object, dict]:
    import copy
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = int(a_tech.shape[0])
    stats = {"n": n, "n_traj": len(traj_lens), "ret_pos": n_pos}
    safe_state = copy.deepcopy(net.state_dict())  # restore on divergence (council 🔴: no NaN export)

    # --- Compute advantages + value targets ONCE per round from a V-snapshot (standard PPO/A2C).
    # Recomputing each epoch makes the target chase V (degenerate value_loss); a fixed target ≈ the
    # bounded discounted-terminal return at λ≈1, so the critic regresses toward a stable signal.
    with torch.no_grad():
        tl0, pl0, val0 = forward_fn()
        val0 = val0.reshape(-1)
        old_logp = (_masked_logp(tl0, a_tech, m_tech) + _masked_logp(pl0, a_policy, m_policy)).detach()
    v_np = val0.cpu().numpy()
    adv_np = np.zeros(n, dtype=np.float32)
    ret_np = np.zeros(n, dtype=np.float32)
    off = 0
    for L in traj_lens:
        a, r = compute_gae(rewards_np[off:off + L], v_np[off:off + L], gamma, lam)
        adv_np[off:off + L] = a
        ret_np[off:off + L] = r
        off += L
    adv = torch.tensor(adv_np)
    ret = torch.tensor(ret_np)                              # fixed value target for the round
    if norm_adv and adv.numel() > 1:                        # BATCH-level normalization (not per-traj)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    adv = adv.detach()

    for ep in range(epochs):
        opt.zero_grad()
        tl, pl, val = forward_fn()
        val = val.reshape(-1)
        logp = _masked_logp(tl, a_tech, m_tech) + _masked_logp(pl, a_policy, m_policy)
        if clip_eps:                                        # PPO clip (default ON; 0/None ⇒ plain A2C)
            logratio = (logp - old_logp).clamp(-20.0, 20.0)  # guard exp() overflow on big policy shifts
            ratio = torch.exp(logratio)
            surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
            policy_loss = -surr.mean()
        else:                                               # plain A2C (single-epoch use)
            policy_loss = -(adv * logp).mean()
        value_loss = F.mse_loss(val, ret)
        ent = (_entropy(tl, m_tech) + _entropy(pl, m_policy)).mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * ent

        if not torch.isfinite(loss):                        # divergence guard (R8 + council 🔴)
            net.load_state_dict(safe_state)                 # restore last finite weights — never export NaN
            stats["note"] = f"non-finite loss at epoch {ep} — restored last-good weights"
            stats["diverged"] = True
            return net, stats

        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
        opt.step()
        safe_state = copy.deepcopy(net.state_dict())        # checkpoint last-good (finite) weights
        stats.update(loss=float(loss.item()), policy_loss=float(policy_loss.item()),
                     value_loss=float(value_loss.item()), entropy=float(ent.item()),
                     mean_adv=float(adv.mean().item()), mean_value=float(val.mean().item()),
                     grad_norm=float(gnorm))
    stats.setdefault("diverged", False)
    return net, stats


def _stack_traj(trajectories: list[TrainTrajectory]):
    a_tech = torch.tensor(np.concatenate([t.a_tech for t in trajectories]))
    a_policy = torch.tensor(np.concatenate([t.a_policy for t in trajectories]))
    m_tech = torch.tensor(np.concatenate([t.mask_tech for t in trajectories]))
    m_policy = torch.tensor(np.concatenate([t.mask_policy for t in trajectories]))
    rewards_np = np.concatenate([t.rewards for t in trajectories]).astype(np.float32)
    traj_lens = [int(len(t.rewards)) for t in trajectories]
    n_pos = int(sum(1 for t in trajectories if t.rewards[-1] > 0))
    return a_tech, a_policy, m_tech, m_policy, rewards_np, traj_lens, n_pos


def train_actor_critic_blind(
    trajectories: list[TrainTrajectory],
    dims: Dims,
    *,
    epochs: int = 8,
    lr: float = 1e-3,
    seed: int = 0,
    gamma: float = 0.99,
    lam: float = 0.95,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    clip_eps: float | None = None,
    norm_adv: bool = True,
) -> tuple[PolicyNet, dict]:
    torch.manual_seed(seed)
    net = PolicyNet(dims)
    if not trajectories:
        return net, {"loss": 0.0, "n": 0, "note": "no steps", "ret_pos": 0}
    a_tech, a_policy, m_tech, m_policy, rewards_np, traj_lens, n_pos = _stack_traj(trajectories)
    obs = torch.tensor(np.concatenate([t.obs for t in trajectories]))

    def forward_fn():
        return net(obs)

    return _optimize_actor_critic(
        net, forward_fn, a_tech=a_tech, a_policy=a_policy, m_tech=m_tech, m_policy=m_policy,
        rewards_np=rewards_np, traj_lens=traj_lens, n_pos=n_pos, epochs=epochs, lr=lr,
        gamma=gamma, lam=lam, value_coef=value_coef, entropy_coef=entropy_coef,
        clip_eps=clip_eps, norm_adv=norm_adv,
    )


def train_actor_critic_rich(
    trajectories: list[TrainTrajectory],
    dims: Dims,
    token_specs: dict[str, int],
    *,
    epochs: int = 8,
    lr: float = 1e-3,
    seed: int = 0,
    gamma: float = 0.99,
    lam: float = 0.95,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    clip_eps: float | None = None,
    norm_adv: bool = True,
) -> tuple[object, dict]:
    """Rich variant. Trajectories carry `rich` dicts of padded per-step token tensors + masks
    (assembled by features.build_rich_batch). The whole round is one padded batch."""
    from .features import build_rich_batch
    from .model import RichPolicyValueNet

    torch.manual_seed(seed)
    net = RichPolicyValueNet(dims, token_specs)
    if not trajectories:
        return net, {"loss": 0.0, "n": 0, "note": "no steps", "ret_pos": 0}
    a_tech, a_policy, m_tech, m_policy, rewards_np, traj_lens, n_pos = _stack_traj(trajectories)
    inputs = build_rich_batch(trajectories, dims, token_specs)  # dict[name -> tensor], padded

    def forward_fn():
        return net(inputs)

    return _optimize_actor_critic(
        net, forward_fn, a_tech=a_tech, a_policy=a_policy, m_tech=m_tech, m_policy=m_policy,
        rewards_np=rewards_np, traj_lens=traj_lens, n_pos=n_pos, epochs=epochs, lr=lr,
        gamma=gamma, lam=lam, value_coef=value_coef, entropy_coef=entropy_coef,
        clip_eps=clip_eps, norm_adv=norm_adv,
    )
