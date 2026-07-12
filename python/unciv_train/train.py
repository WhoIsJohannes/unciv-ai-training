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


def _construction_logp_sum(logits: torch.Tensor, actions: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
    """v7 per-step construction logp: Σ_cities log π(a_city | s) over the legal-masked softmax, 0 where
    the city did not act (action < 0). `logits`/`mask`: [n, M, W]; `actions`: [n, M] → returns [n]. The
    −1 gating makes an OFF step (all cities −1) contribute EXACTLY 0 (bit-identical no-op vs v6)."""
    neg = torch.where(mask > 0, logits, torch.full_like(logits, -1e9))
    logp = F.log_softmax(neg, dim=2)                       # over the W construction-action axis
    acted = actions >= 0                                   # [n, M]
    idx = actions.clamp(min=0).unsqueeze(2)                # [n, M, 1]
    chosen = logp.gather(2, idx).squeeze(2)               # [n, M]
    chosen = torch.where(acted, chosen, torch.zeros_like(chosen))
    return chosen.sum(dim=1)                               # [n]


def _construction_entropy_sum(logits: torch.Tensor, actions: torch.Tensor,
                              mask: torch.Tensor) -> torch.Tensor:
    """v7 per-step construction entropy, summed over ACTING cities only (gated by action ≥ 0) so an
    OFF step contributes 0 — preserving the bit-identical no-op. [n, M, W] → [n]."""
    neg = torch.where(mask > 0, logits, torch.full_like(logits, -1e9))
    p = F.softmax(neg, dim=2)
    logp = F.log_softmax(neg, dim=2)
    ent = -(p * logp).sum(dim=2)                           # [n, M]
    ent = torch.where(actions >= 0, ent, torch.zeros_like(ent))
    return ent.sum(dim=1)                                  # [n]


# v8: per-unit INTENT logp/entropy sums are the SAME masked-softmax-over-the-W-axis math as construction
# ([n, M-or-U, W] → [n], entity-agnostic); aliased for readability at the unit-intent joint-PG call sites.
# Units have no per-unit economy signal, so they ALWAYS take the joint shared-advantage path (no per-unit
# credit term); this is the direct analog of the construction shared-adv (credit-coef 0) behaviour.
_unit_intent_logp_sum = _construction_logp_sum
_unit_intent_entropy_sum = _construction_entropy_sum


def _construction_logp_percity(logits: torch.Tensor, actions: torch.Tensor,
                               mask: torch.Tensor) -> torch.Tensor:
    """v7.3: PER-CITY construction logp (NOT summed over cities) → [n, M], 0 where the city did not act.
    Same masked-softmax gather as `_construction_logp_sum` but keeps the city axis so each city's logp can
    be weighted by ITS OWN advantage in the separate per-city construction PG term."""
    neg = torch.where(mask > 0, logits, torch.full_like(logits, -1e9))
    logp = F.log_softmax(neg, dim=2)
    acted = actions >= 0                                   # [n, M]
    idx = actions.clamp(min=0).unsqueeze(2)
    chosen = logp.gather(2, idx).squeeze(2)               # [n, M]
    return torch.where(acted, chosen, torch.zeros_like(chosen))


def _construction_entropy_percity(logits: torch.Tensor, actions: torch.Tensor,
                                  mask: torch.Tensor) -> torch.Tensor:
    """v7.3: per-city construction entropy [n, M], 0 for non-acting cities (mirrors the per-city logp)."""
    neg = torch.where(mask > 0, logits, torch.full_like(logits, -1e9))
    p = F.softmax(neg, dim=2)
    logp = F.log_softmax(neg, dim=2)
    ent = -(p * logp).sum(dim=2)                           # [n, M]
    return torch.where(actions >= 0, ent, torch.zeros_like(ent))


def _per_city_gae(econ, city_val, present, traj_lens, gamma: float, lam: float):
    """v7.3 PER-CITY GAE. `econ`/`city_val`/`present`: [n, M] (padded to the batch max city count M).
    Each city row is credited by ITS OWN economy return: the per-city reward is the raw log-economy
    `econ[t, i]` (a LEVEL, not ΔΦ — ΔΦ telescopes to a state function and gives no action credit), the
    per-city value `city_val[t, i]` is the baseline, and GAE runs down each trajectory independently per
    city. Padded/absent cities (present=0) get advantage/return 0. Returns (A_city, R_city) as [n, M]
    float32 arrays. This is the ATTRIBUTION fix: construction in city i is credited by city i's outcome,
    not the single civ-wide scalar shared across all heads+cities."""
    econ = np.asarray(econ, dtype=np.float32)
    city_val = np.asarray(city_val, dtype=np.float32)
    present = np.asarray(present, dtype=np.float32)
    A = np.zeros_like(econ)
    off = 0
    for L in traj_lens:
        gae = np.zeros(econ.shape[1], dtype=np.float32)     # [M], per-city running GAE
        for t in range(L - 1, -1, -1):
            row = off + t
            next_v = city_val[row + 1] if t + 1 < L else np.zeros(econ.shape[1], np.float32)
            delta = econ[row] + gamma * next_v - city_val[row]
            gae = delta + gamma * lam * gae
            A[row] = gae
        off += L
    R = A + city_val
    A = A * present                                         # zero the padded/absent city rows
    R = R * present
    return A, R


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
    stored_old_logp: torch.Tensor | None = None,  # v6: per-step SUM of head behavior logps for REPLAYED
                                                  # (off-policy) data. None ⇒ recompute = on-policy = v5 path.
    a_construction: torch.Tensor | None = None,   # v7: per-step per-city construction action [n, M] (−1 = no decision)
    m_construction: torch.Tensor | None = None,   # v7: per-city construction legal mask [n, M, W]
    phi_np: np.ndarray | None = None,             # v7.2: per-step economy potential Φ(s), flat-batch [n]
    reward_shaping_coef: float = 0.0,             # v7.2: PBRS coefficient; 0 ⇒ no shaping (terminal-only)
    econ_city: torch.Tensor | None = None,        # v7.3: per-city raw log-economy [n, M] (padded); the per-city reward
    city_present: torch.Tensor | None = None,     # v7.3: per-city presence mask [n, M] (1 real city, 0 padding)
    construction_credit_coef: float = 0.0,        # v7.3: weight of the per-city economy advantage (0 ⇒ shared-adv only)
    old_logp_construction_pc: torch.Tensor | None = None,  # v7.3: per-city construction behavior logp [n, M] for the
                                                  # per-city PPO importance ratio (valid under replay). None ⇒ A2C (no ratio).
    ref_construction_logp: torch.Tensor | None = None,  # v7.4: frozen BC-clone construction log-probs [n, M, W] (masked)
    construction_kl_coef: float = 0.0,            # v7.4: KL-to-clone leash weight; >0 anchors construction near the clone
    a_unit_intent: torch.Tensor | None = None,    # v8: per-step per-unit intent action [n, U] (−1 = no decision)
    m_unit_intent: torch.Tensor | None = None,    # v8: per-unit intent legal mask [n, U, W]
    ref_unit_intent_logp: torch.Tensor | None = None,  # v8: frozen BC-clone unit-intent log-probs [n, U, W] (masked)
    unit_kl_coef: float = 0.0,                    # v8: KL-to-clone leash weight for the unit-intent head
) -> tuple[object, dict]:
    import copy
    opt = optimizer                  # v5: persistent optimizer (built once in the trainer, carried by run_loop)
    n = int(a_tech.shape[0])
    use_construction = a_construction is not None   # v7: forward_fn returns a 5-tuple (+construction logits, +city value)
    # v7.3 PER-CITY CREDIT: when econ_city is supplied, construction is pulled OUT of the joint PPO ratio
    # and trained by a SEPARATE per-city policy-gradient term whose advantage is the shared civ advantage
    # PLUS a per-city economy advantage (so each city's construction is credited by its OWN outcome). When
    # econ_city is None (OFF arm, or the legacy v7.2 shared-adv mode) construction stays in the joint logp.
    use_percity_credit = use_construction and econ_city is not None
    constr_in_joint = use_construction and not use_percity_credit
    use_kl_leash = use_construction and ref_construction_logp is not None and construction_kl_coef > 0
    # v8: per-unit intent. No per-unit economy signal ⇒ ALWAYS the joint shared-advantage path (unit logp
    # summed into the joint PPO ratio, credited by the same civ GAE advantage). The forward returns the
    # EXTENDED tuple whenever EITHER head is wired (so unit-intent works even if construction is off).
    use_unit_intent = a_unit_intent is not None
    unit_in_joint = use_unit_intent
    use_unit_kl_leash = use_unit_intent and ref_unit_intent_logp is not None and unit_kl_coef > 0
    extended = use_construction or use_unit_intent

    def _unpack(out):
        """forward_fn / forward_chunk_fn return (tl, pl, val) normally; the 6-tuple
        (tl, pl, cl, cval, ul, val) when the construction/unit-intent heads are wired. Normalize to
        (tl, pl, cl|None, cval|None, ul|None, val)."""
        if extended:
            tl, pl, cl, cval, ul, val = out
            return tl, pl, cl, cval, ul, val
        tl, pl, val = out
        return tl, pl, None, None, None, val

    def _policy_logp(tl, pl, cl, ul, lo, hi):
        """Per-step joint logp = tech + policy (+ Σ_cities construction in the legacy shared-adv mode)
        (+ Σ_units unit-intent). In v7.3 per-city-credit mode construction is a SEPARATE term (see the epoch
        loop) and is excluded here; unit-intent is always joint. Slices to [lo:hi] (None ⇒ whole batch)."""
        s = _masked_logp(tl, a_tech[lo:hi], m_tech[lo:hi]) + _masked_logp(pl, a_policy[lo:hi], m_policy[lo:hi])
        if constr_in_joint and cl is not None:
            ac, mc = a_construction[lo:hi], m_construction[lo:hi]
            if cl.shape[1] != ac.shape[1]:
                raise ValueError(f"construction city-axis mismatch: net {cl.shape[1]} != actions {ac.shape[1]} "
                                 "(own_cities padding must equal construction padding — same orderedOwnCities count)")
            s = s + _construction_logp_sum(cl, ac, mc)
        if unit_in_joint and ul is not None:
            au, mu = a_unit_intent[lo:hi], m_unit_intent[lo:hi]
            if ul.shape[1] != au.shape[1]:
                raise ValueError(f"unit-intent unit-axis mismatch: net {ul.shape[1]} != actions {au.shape[1]} "
                                 "(own_units padding must equal unit_intent padding — same orderedOwnUnits count)")
            s = s + _unit_intent_logp_sum(ul, au, mu)
        return s

    def _policy_entropy(tl, pl, cl, ul, lo, hi):
        e = _entropy(tl, m_tech[lo:hi]) + _entropy(pl, m_policy[lo:hi])
        if constr_in_joint and cl is not None:
            e = e + _construction_entropy_sum(cl, a_construction[lo:hi], m_construction[lo:hi])
        if unit_in_joint and ul is not None:
            e = e + _unit_intent_entropy_sum(ul, a_unit_intent[lo:hi], m_unit_intent[lo:hi])
        return e

    def _construction_credit_loss(cl, cval, lo, hi):
        """v7.3 SEPARATE per-city construction objective for chunk [lo:hi]. Returns (pg_loss, value_loss,
        entropy) as scalar tensors. pg_loss = −Σ_cities logπ(a_city)·A_constr_city (on-policy A2C, no PPO
        ratio — replay-window 1); A_constr_city = shared_adv + coef·A_city (per-city economy advantage,
        detached). value_loss = masked MSE(V_city, R_city). All means/sums are over the STEP axis so they
        compose with the size-weighted micro-batch accumulation exactly like the civ terms."""
        ac, mc = a_construction[lo:hi], m_construction[lo:hi]
        if cl.shape[1] != ac.shape[1]:
            raise ValueError(f"construction city-axis mismatch: net {cl.shape[1]} != actions {ac.shape[1]}")
        pres = city_present[lo:hi]                           # [chunk, M]
        # MEAN over present cities per step (NOT sum) — city-count-invariant so the construction gradient
        # stays comparable to a single civ head. Summing over ~6 cities was the v7 objective-domination bug:
        # grad clipping would then shrink the civ (tech/policy) gradient too. Per-step denom = #present cities.
        denom = pres.sum(dim=1).clamp(min=1.0)              # [chunk]
        acted = (ac >= 0).float()                            # [chunk, M] — only deciding cities contribute
        logp_pc = _construction_logp_percity(cl, ac, mc)    # [chunk, M], 0 for non-acting cities
        constr_adv = (adv[lo:hi].unsqueeze(1) + construction_credit_coef * A_city[lo:hi]).detach()  # [chunk, M]
        if clip_eps and old_logp_construction_pc is not None:
            # v7.3 per-city PPO clip: importance ratio vs the recorded per-city behavior policy makes the
            # per-city term correct under REPLAY (off-policy). On-policy the behavior logp ≈ round-start logp
            # so the ratio starts ≈1 and drifts across epochs — standard PPO. Mask to acting cities: a
            # non-acting city has 0 logp-diff ⇒ ratio 1, which would otherwise leak constr_adv into the sum.
            logratio = (logp_pc - old_logp_construction_pc[lo:hi]).clamp(-20.0, 20.0)
            ratio = torch.exp(logratio)
            surr = torch.min(ratio * constr_adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * constr_adv)
            pg = -((surr * acted).sum(dim=1) / denom).mean()
        else:
            pg = -((logp_pc * constr_adv).sum(dim=1) / denom).mean()   # A2C fallback (no ratio)
        ent = (_construction_entropy_percity(cl, ac, mc).sum(dim=1) / denom).mean()
        cval = cval.reshape(pres.shape)
        se = (cval - R_city[lo:hi]).pow(2) * pres           # mask padded/absent cities
        vloss = se.sum() / pres.sum().clamp(min=1.0)
        return pg, vloss, ent

    def _construction_kl_loss(cl, lo, hi):
        """v7.4 KL-to-clone leash: KL(current construction || frozen BC clone) averaged over deciding cities,
        then over steps. Anchors the head near the clone so RL can't drift it back into the collapse basin
        (the residual variance). Works in BOTH the joint (coef 0) and per-city credit modes."""
        mc = m_construction[lo:hi]
        neg = torch.where(mc > 0, cl, torch.full_like(cl, -1e9))
        cur = F.log_softmax(neg, dim=2)                     # [chunk, M, W]
        ref = ref_construction_logp[lo:hi]                  # [chunk, M, W] frozen clone log-probs (masked)
        kl = (cur.exp() * (cur - ref)).sum(dim=2)           # [chunk, M] KL(cur||ref) per city
        has_legal = (mc.sum(dim=2) > 0).float()             # only cities with legal constructions (deciders)
        denom = has_legal.sum(dim=1).clamp(min=1.0)
        return ((kl * has_legal).sum(dim=1) / denom).mean()

    def _unit_intent_kl_loss(ul, lo, hi):
        """v8 KL-to-clone leash for the unit-intent head: KL(current unit-intent || frozen BC clone) averaged
        over deciding units, then over steps — the direct analog of `_construction_kl_loss`, anchoring the
        intent head near the clone so RL can't drift it back into the collapse basin."""
        mu = m_unit_intent[lo:hi]
        neg = torch.where(mu > 0, ul, torch.full_like(ul, -1e9))
        cur = F.log_softmax(neg, dim=2)                     # [chunk, U, W]
        ref = ref_unit_intent_logp[lo:hi]                   # [chunk, U, W] frozen clone log-probs (masked)
        kl = (cur.exp() * (cur - ref)).sum(dim=2)           # [chunk, U] KL(cur||ref) per unit
        has_legal = (mu.sum(dim=2) > 0).float()             # only units with legal intents (deciders)
        denom = has_legal.sum(dim=1).clamp(min=1.0)
        return ((kl * has_legal).sum(dim=1) / denom).mean()
    use_micro = (micro_batch_steps is not None and micro_batch_steps > 0
                 and forward_chunk_fn is not None and micro_batch_steps < n)  # <=0 ⇒ whole-batch no-op
    # v6 GUARD: with stored behavior logp (off-policy replay), clip_eps MUST be truthy — the
    # clip_eps-falsy policy-loss branch `-(adv*logp).mean()` never references old_logp, so it would
    # silently ignore the stored behavior logp and apply replayed advantages as on-policy (biased).
    if stored_old_logp is not None and not clip_eps:
        raise ValueError(
            "off-policy replay (stored_old_logp set) requires a truthy clip_eps — the clip_eps-falsy "
            "policy loss ignores old_logp and would apply replayed advantages on-policy (biased "
            "gradient). Pass clip_eps>0 (default 0.2) or disable replay (--replay-window 1).")
    stats = {"n": n, "n_traj": len(traj_lens), "ret_pos": n_pos}
    safe_state = copy.deepcopy(net.state_dict())  # restore on divergence (council 🔴: no NaN export)
    safe_opt = copy.deepcopy(opt.state_dict())    # v5: roll back optimizer moments too (else a diverged round poisons them)

    # --- Compute advantages + value targets ONCE per round from a V-snapshot (standard PPO/A2C).
    # Recomputing each epoch makes the target chase V (degenerate value_loss); a fixed target ≈ the
    # bounded discounted-terminal return at λ≈1, so the critic regresses toward a stable signal.
    # v6: val0 (current-net V) is ALWAYS recomputed here — GAE must use the CURRENT critic, never a
    # stored value. Only the policy-logp SOURCE switches: stored behavior logp for replayed steps,
    # else the recomputed current-net logp (the literal v5 on-policy path).
    cval0_parts = []                                        # v7.3: per-city value snapshot [·, M] chunks
    with torch.no_grad():
        if use_micro:                                       # v5: chunk the snapshot; cat → math-identical
            v_parts, lp_parts = [], []
            for lo in range(0, n, micro_batch_steps):
                hi = min(lo + micro_batch_steps, n)
                tl0, pl0, cl0, cv0, ul0, v0 = _unpack(forward_chunk_fn(lo, hi))
                v_parts.append(v0.reshape(-1))
                if use_percity_credit:
                    cval0_parts.append(cv0)
                if stored_old_logp is None:                 # only recompute when on-policy
                    lp_parts.append(_policy_logp(tl0, pl0, cl0, ul0, lo, hi))
            val0 = torch.cat(v_parts)
            old_logp = stored_old_logp.detach() if stored_old_logp is not None else torch.cat(lp_parts).detach()
        else:
            tl0, pl0, cl0, cv0, ul0, val0 = _unpack(forward_fn())
            val0 = val0.reshape(-1)
            if use_percity_credit:
                cval0_parts.append(cv0)
            old_logp = (stored_old_logp.detach() if stored_old_logp is not None
                        else _policy_logp(tl0, pl0, cl0, ul0, None, None).detach())
    v_np = val0.cpu().numpy()
    # v7.3: per-city GAE off the value snapshot — A_city[t,i] credits city i's construction by ITS OWN
    # discounted economy return (baseline V_city). Computed ONCE per round (like the civ adv/ret).
    A_city = R_city = None
    if use_percity_credit:
        cval0_np = torch.cat(cval0_parts).cpu().numpy()     # [n, M]
        econ_np = econ_city.cpu().numpy()
        pres_np = city_present.cpu().numpy()
        # v7.3: bound the per-city value target to O(1) so it can't dwarf the civ policy/value losses and
        # swamp the shared trunk. Two steps, both advantage-preserving up to the norm_adv re-scale below:
        #  (1) per-round STANDARDIZE econ over present cities → the signal is each city's economy RELATIVE
        #      to the round mean (COMA-flavored); a constant offset shifts R_city and V_city equally.
        #  (2) AVERAGE-REWARD scale ×(1−γ): a city's economy is autocorrelated (≈constant over its own
        #      steps), so a raw discounted sum multiplies the level by ~1/(1−γ)≈100. Scaling by (1−γ) makes
        #      R_city a γ-weighted AVERAGE of the standardized economy ⇒ O(1) value target, O(1) value loss.
        pm = pres_np > 0
        if pm.sum() > 1:
            mu, sd = float(econ_np[pm].mean()), float(econ_np[pm].std()) + 1e-6
            econ_np = np.where(pm, (1.0 - gamma) * (econ_np - mu) / sd, 0.0).astype(np.float32)
        a_city_np, r_city_np = _per_city_gae(econ_np, cval0_np, pres_np, traj_lens, gamma, lam)
        if norm_adv and a_city_np.size > 1:                 # normalize over PRESENT cities only
            m = pres_np > 0
            if m.sum() > 1:
                vals = a_city_np[m]
                a_city_np = np.where(m, (a_city_np - vals.mean()) / (vals.std() + 1e-8), 0.0).astype(np.float32)
        A_city = torch.tensor(a_city_np)
        R_city = torch.tensor(r_city_np)
    # v7.2 POTENTIAL-BASED REWARD SHAPING (Ng-Harada, policy-invariant): add F_t = coef·(γ·Φ_{t+1}−Φ_t)
    # to each non-terminal step WITHIN a trajectory; the LAST step keeps only the terminal ±1 (F=0 there,
    # so the terminal reward stands alone). The F terms telescope to a constant, leaving the optimal
    # policy unchanged — they only shorten the credit horizon (a Granary's economic payoff registers in
    # a few steps via Φ instead of ~200 turns later via the terminal critic). coef=0 ⇒ exact terminal-only.
    shaped_rewards = rewards_np
    if reward_shaping_coef and phi_np is not None:
        shaped_rewards = rewards_np.copy()
        off = 0
        for L in traj_lens:
            if L >= 2:
                ph = phi_np[off:off + L]
                shaped_rewards[off:off + L - 1] += reward_shaping_coef * (gamma * ph[1:] - ph[:-1])
            off += L
    adv_np = np.zeros(n, dtype=np.float32)
    ret_np = np.zeros(n, dtype=np.float32)
    off = 0
    for L in traj_lens:
        a, r = compute_gae(shaped_rewards[off:off + L], v_np[off:off + L], gamma, lam)
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
            cpg_sum = cvl_sum = 0.0                           # v7.3 per-city construction diagnostics
            ratio_sum = clip_count = 0.0                     # v6 read-only diagnostics (off-policy health)
            for lo in range(0, n, micro_batch_steps):
                hi = min(lo + micro_batch_steps, n)
                w = (hi - lo) / n
                tl, pl, cl, cval, ul, val = _unpack(forward_chunk_fn(lo, hi))
                val = val.reshape(-1)
                logp = _policy_logp(tl, pl, cl, ul, lo, hi)
                if clip_eps:
                    logratio = (logp - old_logp[lo:hi]).clamp(-20.0, 20.0)
                    ratio = torch.exp(logratio)
                    surr = torch.min(ratio * adv[lo:hi], torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv[lo:hi])
                    policy_loss = -surr.mean()
                    with torch.no_grad():
                        ratio_sum += float(ratio.sum())
                        clip_count += float(((ratio - 1.0).abs() > clip_eps).float().sum())
                else:
                    policy_loss = -(adv[lo:hi] * logp).mean()
                value_loss = F.mse_loss(val, ret[lo:hi])
                ent = _policy_entropy(tl, pl, cl, ul, lo, hi).mean()
                loss_step = policy_loss + value_coef * value_loss - entropy_coef * ent
                if use_percity_credit:                       # v7.3 separate per-city construction objective
                    c_pg, c_vloss, c_ent = _construction_credit_loss(cl, cval, lo, hi)
                    loss_step = loss_step + c_pg + value_coef * c_vloss - entropy_coef * c_ent
                    cpg_sum += float(c_pg.detach()) * w
                    cvl_sum += float(c_vloss.detach()) * w
                if use_kl_leash and cl is not None:          # v7.4 KL-to-clone leash (anti-drift)
                    loss_step = loss_step + construction_kl_coef * _construction_kl_loss(cl, lo, hi)
                if use_unit_kl_leash and ul is not None:     # v8 KL-to-clone leash for the unit-intent head
                    loss_step = loss_step + unit_kl_coef * _unit_intent_kl_loss(ul, lo, hi)
                loss_c = loss_step * w
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
            if use_percity_credit:
                stats.update(construction_pg=cpg_sum, construction_value_loss=cvl_sum)
            if clip_eps:
                stats.update(mean_ratio=ratio_sum / n, clip_frac=clip_count / n)
        else:
            tl, pl, cl, cval, ul, val = _unpack(forward_fn())
            val = val.reshape(-1)
            logp = _policy_logp(tl, pl, cl, ul, None, None)
            if clip_eps:                                        # PPO clip (default ON; 0/None ⇒ plain A2C)
                logratio = (logp - old_logp).clamp(-20.0, 20.0)  # guard exp() overflow on big policy shifts
                ratio = torch.exp(logratio)
                surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
                policy_loss = -surr.mean()
                with torch.no_grad():                            # v6 read-only diagnostics (off-policy health)
                    ep_mean_ratio = float(ratio.mean())
                    ep_clip_frac = float(((ratio - 1.0).abs() > clip_eps).float().mean())
            else:                                               # plain A2C (single-epoch use)
                policy_loss = -(adv * logp).mean()
            value_loss = F.mse_loss(val, ret)
            ent = _policy_entropy(tl, pl, cl, ul, None, None).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * ent
            if use_percity_credit:                              # v7.3 separate per-city construction objective
                c_pg, c_vloss, c_ent = _construction_credit_loss(cl, cval, None, None)
                loss = loss + c_pg + value_coef * c_vloss - entropy_coef * c_ent
                stats["construction_pg"] = float(c_pg.detach())
                stats["construction_value_loss"] = float(c_vloss.detach())
            if use_kl_leash and cl is not None:                 # v7.4 KL-to-clone leash (anti-drift)
                kl_term = _construction_kl_loss(cl, None, None)
                loss = loss + construction_kl_coef * kl_term
                stats["construction_kl"] = float(kl_term.detach())
            if use_unit_kl_leash and ul is not None:            # v8 KL-to-clone leash for the unit-intent head
                ukl_term = _unit_intent_kl_loss(ul, None, None)
                loss = loss + unit_kl_coef * ukl_term
                stats["unit_intent_kl"] = float(ukl_term.detach())

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
            if clip_eps:
                stats.update(mean_ratio=ep_mean_ratio, clip_frac=ep_clip_frac)
    stats.setdefault("diverged", False)
    stats.setdefault("mean_ratio", 1.0)   # v6: on-policy / plain-A2C path ⇒ ratio≡1, no clipping
    stats.setdefault("clip_frac", 0.0)
    return net, stats


def _stack_traj(trajectories: list[TrainTrajectory]):
    a_tech = torch.tensor(np.concatenate([t.a_tech for t in trajectories]))
    a_policy = torch.tensor(np.concatenate([t.a_policy for t in trajectories]))
    m_tech = torch.tensor(np.concatenate([t.mask_tech for t in trajectories]))
    m_policy = torch.tensor(np.concatenate([t.mask_policy for t in trajectories]))
    rewards_np = np.concatenate([t.rewards for t in trajectories]).astype(np.float32)
    traj_lens = [int(len(t.rewards)) for t in trajectories]
    n_pos = int(sum(1 for t in trajectories if t.rewards[-1] > 0))
    # v6: per-step behavior logp per head, in the SAME flat-batch order as a_tech/a_policy. A synthetic
    # trajectory with no recorded logp (b_logp_*=None) ⇒ zeros (it never exercises the off-policy path —
    # the trainer only consumes `stored` when behavior_logp is enabled, i.e. real replayed shards).
    def _blogp(attr, t):
        v = getattr(t, attr)
        return v if v is not None else np.zeros(len(t.rewards), dtype=np.float32)
    b_logp_tech = torch.tensor(np.concatenate([_blogp("b_logp_tech", t) for t in trajectories]))
    b_logp_policy = torch.tensor(np.concatenate([_blogp("b_logp_policy", t) for t in trajectories]))
    # v7.2: per-step economy potential Φ(s), flat-batch order; None ⇒ zeros (no shaping for synthetic trajs).
    # nan_to_num guards against any non-finite Φ leaking into the shaping reward (→ NaN loss → divergence).
    phi_np = np.nan_to_num(np.concatenate([_blogp("phi", t) for t in trajectories]).astype(np.float32),
                           nan=0.0, posinf=0.0, neginf=0.0)
    return a_tech, a_policy, m_tech, m_policy, rewards_np, traj_lens, n_pos, b_logp_tech, b_logp_policy, phi_np


def _stack_construction(trajectories: list[TrainTrajectory], constr_w: int):
    """v7: per-step ragged construction (action / logp / mask, in `rich` dicts) → dense padded tensors
    aligned to the SAME max city count `build_rich_batch` pads own_cities to (per-step own_cities count
    == construction count == orderedOwnCities size). Pad action=−1 / logp=0 / mask=0 (inert rows → 0
    logp). Returns (a_construction[n,M] int, b_logp_construction[n] f32, m_construction[n,M,W] f32)."""
    acts, logps, masks, econs = [], [], [], []
    for t in trajectories:
        rich = t.rich
        for i in range(len(t.rewards)):
            step = rich[i] if rich is not None else None
            a = np.asarray(step["construction_action"], np.int64).reshape(-1) if step is not None and "construction_action" in step else np.zeros(0, np.int64)
            lp = np.asarray(step["construction_logp"], np.float32).reshape(-1) if step is not None and "construction_logp" in step else np.zeros(0, np.float32)
            mk = np.asarray(step["mask_construction"], np.float32) if step is not None and "mask_construction" in step else np.zeros((0, constr_w), np.float32)
            if mk.ndim == 1:
                mk = mk.reshape(0, constr_w)
            ec = np.asarray(step["econ_city"], np.float32).reshape(-1) if step is not None and "econ_city" in step else np.zeros(0, np.float32)
            acts.append(a); logps.append(lp); masks.append(mk); econs.append(ec)
    n = len(acts)
    big_m = max((a.shape[0] for a in acts), default=0)
    m_dim = max(1, big_m)                                   # ≥1 so the tensors aren't degenerate
    a_c = np.full((n, m_dim), -1, np.int64)
    lp_c = np.zeros((n, m_dim), np.float32)
    mk_c = np.zeros((n, m_dim, constr_w), np.float32)
    ec_c = np.zeros((n, m_dim), np.float32)                 # v7.3: per-city economy (0 in padded rows)
    pres_c = np.zeros((n, m_dim), np.float32)               # v7.3: per-city presence (1 real, 0 padding)
    has_econ = False
    for i, (a, lp, mk, ec) in enumerate(zip(acts, logps, masks, econs)):
        k = a.shape[0]
        if k:
            a_c[i, :k] = a
            lp_c[i, :k] = lp
            if mk.shape[0] == k and mk.shape[1] == constr_w:
                mk_c[i, :k] = mk
            elif mk.size and mk.shape[1] != constr_w:       # fail loud (was: silently zero → dead construction grad)
                raise ValueError(f"construction mask width {mk.shape[1]} != net.constr_w {constr_w} — shard "
                                 "ruleset/vocab differs from the training net (warm-start arch mismatch)")
            pres_c[i, :k] = 1.0                             # every own city is a present row (acted or not)
            if ec.shape[0] == k:
                ec_c[i, :k] = ec
                has_econ = True
    econ_t = torch.tensor(ec_c) if has_econ else None       # None ⇒ v6-era shard w/o econ_city (no per-city credit)
    present_t = torch.tensor(pres_c) if has_econ else None
    # Return BOTH the per-step SUM (for the legacy joint-ratio stored old_logp) and the PER-CITY behavior
    # logp [n,M] (for the v7.3 per-city PPO importance ratio — makes the per-city term valid under replay).
    return torch.tensor(a_c), torch.tensor(lp_c.sum(axis=1)), torch.tensor(mk_c), econ_t, present_t, torch.tensor(lp_c)


def _stack_unit_intent(trajectories: list[TrainTrajectory], intent_w: int):
    """v8: per-step ragged unit-intent (action / logp / mask, in `rich` dicts) → dense padded tensors aligned
    to the SAME max unit count `build_rich_batch` pads own_units to (per-step own_units == unit_intent count
    == orderedOwnUnits size). Pad action=−1 / logp=0 / mask=0 (inert rows → 0 logp). Units use the SHARED civ
    advantage (no per-unit econ/credit), so this returns just (a_unit_intent[n,U] int, b_logp[n] f32 sum,
    m_unit_intent[n,U,W] f32) — the direct analog of the construction joint path."""
    acts, logps, masks = [], [], []
    for t in trajectories:
        rich = t.rich
        for i in range(len(t.rewards)):
            step = rich[i] if rich is not None else None
            a = np.asarray(step["unit_intent_action"], np.int64).reshape(-1) if step is not None and "unit_intent_action" in step else np.zeros(0, np.int64)
            lp = np.asarray(step["unit_intent_logp"], np.float32).reshape(-1) if step is not None and "unit_intent_logp" in step else np.zeros(0, np.float32)
            mk = np.asarray(step["mask_unit_intent"], np.float32) if step is not None and "mask_unit_intent" in step else np.zeros((0, intent_w), np.float32)
            if mk.ndim == 1:
                mk = mk.reshape(0, intent_w)
            acts.append(a); logps.append(lp); masks.append(mk)
    n = len(acts)
    big_u = max((a.shape[0] for a in acts), default=0)
    u_dim = max(1, big_u)
    a_u = np.full((n, u_dim), -1, np.int64)
    lp_u = np.zeros((n, u_dim), np.float32)
    mk_u = np.zeros((n, u_dim, intent_w), np.float32)
    for i, (a, lp, mk) in enumerate(zip(acts, logps, masks)):
        k = a.shape[0]
        if k:
            a_u[i, :k] = a
            lp_u[i, :k] = lp
            if mk.shape[0] == k and mk.shape[1] == intent_w:
                mk_u[i, :k] = mk
            elif mk.size and mk.shape[1] != intent_w:            # fail loud (never silently zero the grad)
                raise ValueError(f"unit-intent mask width {mk.shape[1]} != net.intent_w {intent_w} — shard "
                                 "intent enum differs from the training net (warm-start arch mismatch)")
    return torch.tensor(a_u), torch.tensor(lp_u.sum(axis=1)), torch.tensor(mk_u)


def _stack_bc_targets(trajectories: list[TrainTrajectory], constr_w: int):
    """v7.4: per-step per-city BC target = the heuristic's currently-building construction (`construction_current`,
    0-indexed mask idx; −1 = idle/no-target), padded to the batch max city count M. Returns (targets[n,M] int,
    mask[n,M,W] f32). Rows with target −1 (or an out-of-range idx) are excluded from the BC loss."""
    tgts, masks = [], []
    for t in trajectories:
        rich = t.rich
        for i in range(len(t.rewards)):
            step = rich[i] if rich is not None else None
            cc = (np.asarray(step["construction_current"], np.int64).reshape(-1)
                  if step is not None and "construction_current" in step else np.zeros(0, np.int64))
            mk = np.asarray(step["mask_construction"], np.float32) if step is not None and "mask_construction" in step else np.zeros((0, constr_w), np.float32)
            if mk.ndim == 1:
                mk = mk.reshape(0, constr_w)
            tgts.append(cc); masks.append(mk)
    n = len(tgts)
    big_m = max((a.shape[0] for a in tgts), default=0)
    m_dim = max(1, big_m)
    tg = np.full((n, m_dim), -1, np.int64)
    mk_c = np.zeros((n, m_dim, constr_w), np.float32)
    for i, (cc, mk) in enumerate(zip(tgts, masks)):
        k = cc.shape[0]
        if k:
            tg[i, :k] = cc
            if mk.shape[0] == k and mk.shape[1] == constr_w:
                mk_c[i, :k] = mk
            elif mk.size and mk.shape[1] != constr_w:                # fail loud on a genuine width mismatch
                raise ValueError(f"BC mask width {mk.shape[1]} != net.constr_w {constr_w} — BC dataset "
                                 "ruleset/vocab differs from the training net (cross-ruleset BC is invalid)")
    tg[(tg < 0) | (tg >= constr_w)] = -1                        # drop out-of-range / idle targets
    return torch.tensor(tg), torch.tensor(mk_c)


def _stack_bc_targets_unit(trajectories: list[TrainTrajectory], intent_w: int):
    """v8: per-step per-unit BC target = the heuristic's first-firing ladder rung (`unit_intent_current`,
    0-indexed intent idx; −1 = none/non-land-military), padded to the batch max unit count U. Returns
    (targets[n,U] int, mask[n,U,W] f32). Rows with target −1 (or out-of-range) are excluded from the BC loss.
    The direct analog of [_stack_bc_targets] for the unit-intent head."""
    tgts, masks = [], []
    for t in trajectories:
        rich = t.rich
        for i in range(len(t.rewards)):
            step = rich[i] if rich is not None else None
            uc = (np.asarray(step["unit_intent_current"], np.int64).reshape(-1)
                  if step is not None and "unit_intent_current" in step else np.zeros(0, np.int64))
            mk = np.asarray(step["mask_unit_intent"], np.float32) if step is not None and "mask_unit_intent" in step else np.zeros((0, intent_w), np.float32)
            if mk.ndim == 1:
                mk = mk.reshape(0, intent_w)
            tgts.append(uc); masks.append(mk)
    n = len(tgts)
    big_u = max((a.shape[0] for a in tgts), default=0)
    u_dim = max(1, big_u)
    tg = np.full((n, u_dim), -1, np.int64)
    mk_u = np.zeros((n, u_dim, intent_w), np.float32)
    for i, (uc, mk) in enumerate(zip(tgts, masks)):
        k = uc.shape[0]
        if k:
            tg[i, :k] = uc
            if mk.shape[0] == k and mk.shape[1] == intent_w:
                mk_u[i, :k] = mk
            elif mk.size and mk.shape[1] != intent_w:
                raise ValueError(f"BC unit-intent mask width {mk.shape[1]} != net.intent_w {intent_w}")
    tg[(tg < 0) | (tg >= intent_w)] = -1
    return torch.tensor(tg), torch.tensor(mk_u)


def _bc_masked_ce(logits: torch.Tensor, mk: torch.Tensor, tg: torch.Tensor):
    """Masked-softmax cross-entropy of a per-entity head chunk vs the heuristic target idx, + (correct, valid)
    counts. Targets must be LEGAL in the mask — a mid-build/stale target's −1e9 masked logit would otherwise
    blow up the loss (a handful poison the whole gradient). Returns (loss, correct, valid)."""
    neg = torch.where(mk > 0, logits, torch.full_like(logits, -1e9))
    logp = F.log_softmax(neg, dim=2)
    idx = tg.clamp(min=0).unsqueeze(2)
    legal_at_tgt = (mk.gather(2, idx).squeeze(2) > 0)
    valid = (tg >= 0) & legal_at_tgt                           # entities with a LEGAL heuristic target
    chosen = logp.gather(2, idx).squeeze(2)
    vsum = valid.float().sum().clamp(min=1.0)
    loss = -(chosen * valid.float()).sum() / vsum
    with torch.no_grad():
        pred = neg.argmax(dim=2)                                # masked argmax
        corr = float(((pred == tg) & valid).float().sum())
        val = float(valid.float().sum())
    return loss, corr, val


def bc_pretrain_construction(net, trajectories, dims, token_specs, *, epochs=8, lr=1e-3,
                             seed=0, micro_batch_steps=512, clone_unit_intent=True):
    """v7.4/v8 BEHAVIOR CLONING: supervised-pretrain the per-entity head(s) to mimic the heuristic's picks
    (`construction_current` / `unit_intent_current`, gen'd with control OFF). Cross-entropy over the
    legal-masked head logits vs the target idx, for entities with a valid legal target. Only the head-loss
    path (head + shared trunk/encoder) gets gradient — tech/policy/value heads are untouched (not in the loss).
    v8: when the net carries a unit-intent head AND [clone_unit_intent], the unit-intent head is cloned in the
    SAME optimizer pass (one warm-start serves both heads + both KL leashes). Returns (net, stats)."""
    from .features import build_rich_batch
    torch.manual_seed(seed)
    if not trajectories:
        return net, {"bc_loss": 0.0, "bc_acc": 0.0, "n": 0}
    tgt, m_c = _stack_bc_targets(trajectories, net.constr_w)
    clone_u = clone_unit_intent and hasattr(net, "unit_intent_head")
    tgt_u = m_u = None
    if clone_u:
        tgt_u, m_u = _stack_bc_targets_unit(trajectories, net.intent_w)
    have_c = (tgt >= 0).sum() > 0
    have_u = clone_u and (tgt_u >= 0).sum() > 0
    if not have_c and not have_u:
        return net, {"bc_loss": 0.0, "bc_acc": 0.0, "n": 0, "note": "no BC targets"}
    inputs = build_rich_batch(trajectories, dims, token_specs)
    n = tgt.shape[0]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    mb = micro_batch_steps if (micro_batch_steps and 0 < micro_batch_steps < n) else n
    last = {"bc_loss": 0.0, "bc_acc": 0.0, "n": n}
    for ep in range(epochs):
        opt.zero_grad()
        tot_loss = tot_correct = tot_valid = 0.0
        tot_u_correct = tot_u_valid = 0.0
        for lo in range(0, n, mb):
            hi = min(lo + mb, n)
            w = (hi - lo) / n
            out = net({k: v[lo:hi] for k, v in inputs.items()}, with_construction=True)
            c_loss, c_corr, c_val = _bc_masked_ce(out[2], m_c[lo:hi], tgt[lo:hi])   # construction logits @ out[2]
            loss = c_loss
            if clone_u:
                u_loss, u_corr, u_val = _bc_masked_ce(out[4], m_u[lo:hi], tgt_u[lo:hi])  # unit-intent logits @ out[4]
                loss = loss + u_loss
                tot_u_correct += u_corr; tot_u_valid += u_val
            (loss * w).backward()
            tot_correct += c_corr; tot_valid += c_val
            tot_loss += float(loss.detach()) * w
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
        opt.step()
        last = {"bc_loss": tot_loss, "bc_acc": (tot_correct / tot_valid if tot_valid else 0.0), "n": n}
        if clone_u:
            last["bc_unit_acc"] = (tot_u_correct / tot_u_valid if tot_u_valid else 0.0)
    return net, last


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
    behavior_logp: bool = False,     # v6: use stored behavior logp as old_logp (off-policy replay); False ⇒ v5 recompute
    reward_shaping_coef: float = 0.0,  # v7.2: PBRS coefficient (F = coef·(γ·Φ'−Φ)); 0 ⇒ terminal-only
) -> tuple[PolicyNet, dict, object]:
    if net is None:
        torch.manual_seed(seed)
        net = PolicyNet(dims)
    if optimizer is None:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    if not trajectories:
        return net, {"loss": 0.0, "n": 0, "note": "no steps", "ret_pos": 0}, optimizer
    a_tech, a_policy, m_tech, m_policy, rewards_np, traj_lens, n_pos, b_logp_tech, b_logp_policy, phi_np = _stack_traj(trajectories)
    # v6: stored behavior logp (sum of per-head logps) for off-policy replayed data; None ⇒ on-policy (v5).
    stored = (b_logp_tech + b_logp_policy) if behavior_logp else None
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
        clip_eps=clip_eps, norm_adv=norm_adv, stored_old_logp=stored,
        phi_np=phi_np, reward_shaping_coef=reward_shaping_coef,
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
    behavior_logp: bool = False,     # v6: use stored behavior logp as old_logp (off-policy replay); False ⇒ v5 recompute
    reward_shaping_coef: float = 0.0,  # v7.2: PBRS coefficient (F = coef·(γ·Φ'−Φ)); 0 ⇒ terminal-only
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
    a_tech, a_policy, m_tech, m_policy, rewards_np, traj_lens, n_pos, b_logp_tech, b_logp_policy, phi_np = _stack_traj(trajectories)
    # v6: stored behavior logp (sum of per-head logps) for off-policy replayed data; None ⇒ on-policy (v5).
    stored = (b_logp_tech + b_logp_policy) if behavior_logp else None
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
        clip_eps=clip_eps, norm_adv=norm_adv, stored_old_logp=stored,
        phi_np=phi_np, reward_shaping_coef=reward_shaping_coef,
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
    behavior_logp: bool = False,     # v6: use stored behavior logp as old_logp (off-policy replay); False ⇒ v5 recompute
    reward_shaping_coef: float = 0.0,  # v7.2: PBRS coefficient (F = coef·(γ·Φ'−Φ)); 0 ⇒ terminal-only
    construction: bool = True,       # v7: train the per-city construction head; False ⇒ pure v6 path (no-op oracle)
    construction_credit_coef: float = 0.0,  # v7.3: per-city economy-advantage weight; >0 ⇒ per-city credit path
    bc_ref=None,                     # v7.4: frozen BC-clone net (KL-leash reference); None ⇒ no leash
    construction_kl_coef: float = 0.0,  # v7.4: KL-to-clone leash weight
    unit_intent: bool = False,       # v8: train the per-unit intent head; False ⇒ head inert (no-op oracle)
    unit_kl_coef: float = 0.0,       # v8: KL-to-clone leash weight for the unit-intent head
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
    a_tech, a_policy, m_tech, m_policy, rewards_np, traj_lens, n_pos, b_logp_tech, b_logp_policy, phi_np = _stack_traj(trajectories)
    # v6: stored behavior logp (sum of per-head logps) for off-policy replayed data; None ⇒ on-policy (v5).
    stored = (b_logp_tech + b_logp_policy) if behavior_logp else None
    inputs = build_rich_batch(trajectories, dims, token_specs)

    # v7: per-city construction tensors (action / behavior-logp-sum / mask), padded to the SAME city
    # axis build_rich_batch pads own_cities to. The construction summand is part of the per-step joint
    # logp AND (for replay) the stored old_logp — the importance ratio covers all heads jointly.
    a_construction = m_construction = None
    econ_city_t = city_present_t = old_logp_construction_pc = None
    if construction:
        a_construction, b_logp_construction, m_construction, econ_all, present_all, b_logp_construction_pc = \
            _stack_construction(trajectories, net.constr_w)
        # v7.3: per-city credit is active ONLY when coef>0 AND the shards carry econ_city. Then construction
        # trains as a SEPARATE per-city term (excluded from the joint ratio ⇒ NOT added to stored old_logp),
        # with its OWN per-city PPO importance ratio (b_logp_construction_pc) so it is valid under replay.
        # coef==0 reproduces the legacy v7.2 shared-adv path (construction in the joint PPO ratio + stored).
        use_pcc = construction_credit_coef > 0 and econ_all is not None
        if use_pcc:
            econ_city_t, city_present_t = econ_all, present_all
            old_logp_construction_pc = b_logp_construction_pc
        if stored is not None and not use_pcc:
            stored = stored + b_logp_construction

    # v8: per-unit intent tensors (action / behavior-logp-sum / mask), padded to the SAME unit axis
    # build_rich_batch pads own_units to. Units always ride in the joint PPO ratio (shared civ advantage,
    # no per-unit credit) ⇒ their behavior-logp sum is added to the stored old_logp under replay.
    a_unit_intent = m_unit_intent = None
    if unit_intent:
        a_unit_intent, b_logp_unit_intent, m_unit_intent = _stack_unit_intent(trajectories, net.intent_w)
        if stored is not None:
            stored = stored + b_logp_unit_intent

    # v8: the forward returns the EXTENDED 6-tuple whenever EITHER head is trained, so unit-intent works even
    # if construction is off (and vice versa).
    extended_fwd = construction or unit_intent

    def forward_fn():
        return net(inputs, with_construction=extended_fwd)

    def forward_chunk_fn(lo, hi):    # slice every [B,...] tensor on axis 0 (per-row independent forward)
        return net({k: v[lo:hi] for k, v in inputs.items()}, with_construction=extended_fwd)

    # v7.4 KL-to-clone leash: forward the FROZEN BC clone ONCE (no_grad, chunked) → reference construction
    # log-probs [n,M,W] (masked). The trainer penalizes KL(current || ref) each epoch so RL can't drift the
    # head back into the collapse basin. Chunk to bound memory; concat is math-identical.
    # v8: compute BOTH reference log-prob tensors in ONE bc_ref forward loop (the 6-tuple carries both heads).
    ref_construction_logp = None
    ref_unit_intent_logp = None
    need_cref = construction and bc_ref is not None and construction_kl_coef > 0 and m_construction is not None
    need_uref = unit_intent and bc_ref is not None and unit_kl_coef > 0 and m_unit_intent is not None
    if need_cref or need_uref:
        bc_ref.eval()
        with torch.no_grad():
            n_all = a_tech.shape[0]
            step = micro_batch_steps if (micro_batch_steps and 0 < micro_batch_steps < n_all) else n_all
            cparts, uparts = [], []
            for lo in range(0, n_all, step):
                hi = min(lo + step, n_all)
                out = bc_ref({k: v[lo:hi] for k, v in inputs.items()}, with_construction=True)
                if need_cref:
                    rcl, mc = out[2], m_construction[lo:hi]                                 # [chunk,M,W]
                    cparts.append(F.log_softmax(torch.where(mc > 0, rcl, torch.full_like(rcl, -1e9)), dim=2))
                if need_uref:
                    rul, mu = out[4], m_unit_intent[lo:hi]                                  # [chunk,U,W]
                    uparts.append(F.log_softmax(torch.where(mu > 0, rul, torch.full_like(rul, -1e9)), dim=2))
            if need_cref:
                ref_construction_logp = torch.cat(cparts).detach()
            if need_uref:
                ref_unit_intent_logp = torch.cat(uparts).detach()

    net, stats = _optimize_actor_critic(
        net, forward_fn, optimizer=optimizer, forward_chunk_fn=forward_chunk_fn,
        micro_batch_steps=micro_batch_steps,
        a_tech=a_tech, a_policy=a_policy, m_tech=m_tech, m_policy=m_policy,
        rewards_np=rewards_np, traj_lens=traj_lens, n_pos=n_pos, epochs=epochs, lr=lr,
        gamma=gamma, lam=lam, value_coef=value_coef, entropy_coef=entropy_coef,
        clip_eps=clip_eps, norm_adv=norm_adv, stored_old_logp=stored,
        phi_np=phi_np, reward_shaping_coef=reward_shaping_coef,
        a_construction=a_construction, m_construction=m_construction,
        econ_city=econ_city_t, city_present=city_present_t,
        construction_credit_coef=construction_credit_coef,
        old_logp_construction_pc=old_logp_construction_pc,
        ref_construction_logp=ref_construction_logp,
        construction_kl_coef=construction_kl_coef,
        a_unit_intent=a_unit_intent, m_unit_intent=m_unit_intent,
        ref_unit_intent_logp=ref_unit_intent_logp, unit_kl_coef=unit_kl_coef,
    )
    return net, stats, optimizer
