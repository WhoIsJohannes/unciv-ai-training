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
    optimizer,                       # v5: INJECTED (was constructed here) — carries Adam moments across rounds
    forward_chunk_fn=None,           # v5: slice-aware forward(lo, hi) for micro-batched traversal
    micro_batch_steps: int | None = None,  # v5: chunk size; falsy / >= n ⇒ whole-batch (no-op)
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
    opt = optimizer                  # v5: persistent optimizer (built once in the trainer, carried by run_loop)
    n = int(a_tech.shape[0])
    use_micro = (micro_batch_steps is not None and micro_batch_steps > 0
                 and forward_chunk_fn is not None and micro_batch_steps < n)  # <=0 ⇒ whole-batch no-op
    stats = {"n": n, "n_traj": len(traj_lens), "ret_pos": n_pos}
    safe_state = copy.deepcopy(net.state_dict())  # restore on divergence (council 🔴: no NaN export)
    safe_opt = copy.deepcopy(opt.state_dict())    # v5: roll back optimizer moments too (else a diverged round poisons them)

    # --- Compute advantages + value targets ONCE per round from a V-snapshot (standard PPO/A2C).
    # Recomputing each epoch makes the target chase V (degenerate value_loss); a fixed target ≈ the
    # bounded discounted-terminal return at λ≈1, so the critic regresses toward a stable signal.
    with torch.no_grad():
        if use_micro:                                       # v5: chunk the snapshot; cat → math-identical
            v_parts, lp_parts = [], []
            for lo in range(0, n, micro_batch_steps):
                hi = min(lo + micro_batch_steps, n)
                tl0, pl0, v0 = forward_chunk_fn(lo, hi)
                v_parts.append(v0.reshape(-1))
                lp_parts.append(_masked_logp(tl0, a_tech[lo:hi], m_tech[lo:hi])
                                + _masked_logp(pl0, a_policy[lo:hi], m_policy[lo:hi]))
            val0 = torch.cat(v_parts)
            old_logp = torch.cat(lp_parts).detach()
        else:
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
        if use_micro:
            # v5 MICRO-BATCH path (the permitted TRAVERSAL change): same per-update arithmetic as the
            # whole-batch else-branch below, applied per K-step chunk and size-weighted (chunk_n/N) so
            # the summed .mean()s == the whole-batch .mean()s; one backward-accumulate, one opt.step().
            loss_total = pl_sum = vl_sum = ent_sum = mv_sum = 0.0
            for lo in range(0, n, micro_batch_steps):
                hi = min(lo + micro_batch_steps, n)
                w = (hi - lo) / n
                tl, pl, val = forward_chunk_fn(lo, hi)
                val = val.reshape(-1)
                logp = _masked_logp(tl, a_tech[lo:hi], m_tech[lo:hi]) + _masked_logp(pl, a_policy[lo:hi], m_policy[lo:hi])
                if clip_eps:
                    logratio = (logp - old_logp[lo:hi]).clamp(-20.0, 20.0)
                    ratio = torch.exp(logratio)
                    surr = torch.min(ratio * adv[lo:hi], torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv[lo:hi])
                    policy_loss = -surr.mean()
                else:
                    policy_loss = -(adv[lo:hi] * logp).mean()
                value_loss = F.mse_loss(val, ret[lo:hi])
                ent = (_entropy(tl, m_tech[lo:hi]) + _entropy(pl, m_policy[lo:hi])).mean()
                loss_c = (policy_loss + value_coef * value_loss - entropy_coef * ent) * w
                loss_c.backward()                           # accumulate grad; free the chunk graph
                loss_total += float(loss_c.detach())
                pl_sum += float(policy_loss.detach()) * w
                vl_sum += float(value_loss.detach()) * w
                ent_sum += float(ent.detach()) * w
                mv_sum += float(val.detach().sum())
            if not np.isfinite(loss_total):                 # divergence guard — same semantics
                net.load_state_dict(safe_state)
                opt.load_state_dict(safe_opt)               # v5: restore optimizer moments too
                stats["note"] = f"non-finite loss at epoch {ep} — restored last-good weights"
                stats["diverged"] = True
                return net, stats
            gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
            opt.step()
            safe_state = copy.deepcopy(net.state_dict())
            safe_opt = copy.deepcopy(opt.state_dict())
            stats.update(loss=loss_total, policy_loss=pl_sum, value_loss=vl_sum, entropy=ent_sum,
                         mean_adv=float(adv.mean().item()), mean_value=mv_sum / n, grad_norm=float(gnorm))
        else:
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
                opt.load_state_dict(safe_opt)                   # v5: restore optimizer moments too
                stats["note"] = f"non-finite loss at epoch {ep} — restored last-good weights"
                stats["diverged"] = True
                return net, stats

            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
            opt.step()
            safe_state = copy.deepcopy(net.state_dict())        # checkpoint last-good (finite) weights
            safe_opt = copy.deepcopy(opt.state_dict())          # v5: checkpoint last-good optimizer moments
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
    net=None,                        # v5: warm net (continual); None ⇒ fresh
    optimizer=None,                  # v5: warm optimizer; None ⇒ built once here
    micro_batch_steps: int | None = None,
) -> tuple[PolicyNet, dict, object]:
    if net is None:
        torch.manual_seed(seed)
        net = PolicyNet(dims)
    if optimizer is None:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    if not trajectories:
        return net, {"loss": 0.0, "n": 0, "note": "no steps", "ret_pos": 0}, optimizer
    a_tech, a_policy, m_tech, m_policy, rewards_np, traj_lens, n_pos = _stack_traj(trajectories)
    obs = torch.tensor(np.concatenate([t.obs for t in trajectories]))

    def forward_fn():
        return net(obs)

    def forward_chunk_fn(lo, hi):
        return net(obs[lo:hi])

    net, stats = _optimize_actor_critic(
        net, forward_fn, optimizer=optimizer, forward_chunk_fn=forward_chunk_fn,
        micro_batch_steps=micro_batch_steps,
        a_tech=a_tech, a_policy=a_policy, m_tech=m_tech, m_policy=m_policy,
        rewards_np=rewards_np, traj_lens=traj_lens, n_pos=n_pos, epochs=epochs, lr=lr,
        gamma=gamma, lam=lam, value_coef=value_coef, entropy_coef=entropy_coef,
        clip_eps=clip_eps, norm_adv=norm_adv,
    )
    return net, stats, optimizer


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
    net=None,                        # v5: warm net (continual); None ⇒ fresh
    optimizer=None,                  # v5: warm optimizer; None ⇒ built once here
    micro_batch_steps: int | None = None,
) -> tuple[object, dict, object]:
    """Rich variant. Trajectories carry `rich` dicts of padded per-step token tensors + masks
    (assembled by features.build_rich_batch). The whole round is one padded batch.
    v5: warm-start + micro-batched traversal; returns the optimizer for cross-round carry."""
    from .features import build_rich_batch
    from .model import RichPolicyValueNet

    if net is None:
        torch.manual_seed(seed)
        net = RichPolicyValueNet(dims, token_specs)
    if optimizer is None:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    if not trajectories:
        return net, {"loss": 0.0, "n": 0, "note": "no steps", "ret_pos": 0}, optimizer
    a_tech, a_policy, m_tech, m_policy, rewards_np, traj_lens, n_pos = _stack_traj(trajectories)
    inputs = build_rich_batch(trajectories, dims, token_specs)  # dict[name -> tensor], padded

    def forward_fn():
        return net(inputs)

    def forward_chunk_fn(lo, hi):
        return net({k: v[lo:hi] for k, v in inputs.items()})

    net, stats = _optimize_actor_critic(
        net, forward_fn, optimizer=optimizer, forward_chunk_fn=forward_chunk_fn,
        micro_batch_steps=micro_batch_steps,
        a_tech=a_tech, a_policy=a_policy, m_tech=m_tech, m_policy=m_policy,
        rewards_np=rewards_np, traj_lens=traj_lens, n_pos=n_pos, epochs=epochs, lr=lr,
        gamma=gamma, lam=lam, value_coef=value_coef, entropy_coef=entropy_coef,
        clip_eps=clip_eps, norm_adv=norm_adv,
    )
    return net, stats, optimizer


def train_actor_critic_structured(
    trajectories: list[TrainTrajectory],
    dims: Dims,
    token_specs: dict[str, int],
    vocab_counts: dict,
    rung: dict,
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
    net=None,                        # v5: warm net (continual); None ⇒ fresh (round 0 / from-scratch)
    optimizer=None,                  # v5: warm optimizer; None ⇒ built once here and carried by run_loop
    micro_batch_steps: int | None = None,  # v5: chunk the dense traversal (medium rung on Medium)
) -> tuple[object, dict, object]:
    """v4 STRUCTURED variant: same encoder-AGNOSTIC core (_optimize_actor_critic) + same rich
    batch/stacking as train_actor_critic_rich — ONLY the nn.Module is swapped for
    StructuredPolicyValueNet (embeddings + hex-GNN + attention, sized by `rung`). The trainer core +
    train_actor_critic_rich stay UNTOUCHED (the FROZEN seam).
    v5: warm-start (`net`/`optimizer` reused across rounds; manual_seed only on the fresh branch) +
    micro-batched traversal. Returns the optimizer so the caller can carry it to the next round."""
    from .features import build_rich_batch
    from .model import StructuredPolicyValueNet

    if net is None:                  # fresh round: deterministic init
        torch.manual_seed(seed)
        net = StructuredPolicyValueNet(dims, token_specs, vocab_counts, **rung)
    if optimizer is None:            # build the persistent optimizer ONCE (round 0 / from-scratch round)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    if not trajectories:
        return net, {"loss": 0.0, "n": 0, "note": "no steps", "ret_pos": 0}, optimizer
    a_tech, a_policy, m_tech, m_policy, rewards_np, traj_lens, n_pos = _stack_traj(trajectories)
    inputs = build_rich_batch(trajectories, dims, token_specs)

    def forward_fn():
        return net(inputs)

    def forward_chunk_fn(lo, hi):    # slice every [B,...] tensor on axis 0 (per-row independent forward)
        return net({k: v[lo:hi] for k, v in inputs.items()})

    net, stats = _optimize_actor_critic(
        net, forward_fn, optimizer=optimizer, forward_chunk_fn=forward_chunk_fn,
        micro_batch_steps=micro_batch_steps,
        a_tech=a_tech, a_policy=a_policy, m_tech=m_tech, m_policy=m_policy,
        rewards_np=rewards_np, traj_lens=traj_lens, n_pos=n_pos, epochs=epochs, lr=lr,
        gamma=gamma, lam=lam, value_coef=value_coef, entropy_coef=entropy_coef,
        clip_eps=clip_eps, norm_adv=norm_adv,
    )
    return net, stats, optimizer
