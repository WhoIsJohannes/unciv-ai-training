"""v6 AC3 — off-policy importance ratio uses the STORED behavior logp, and at K=1 the stored logp
equals the round-start recompute ⇒ ratio ≡ 1 (within fp). Two layers:

  (1) FORMULA: the Kotlin MaskedChoice.chooseWithLogp value ln(exps[pos]/sum) equals the Python
      trainer's _masked_logp = log_softmax(masked)[chosen] within 1e-6 (so the JVM-recorded logp and
      the Python recompute are the SAME quantity).
  (2) DIAGNOSTIC: when the stored behavior logp == the round-start recompute, the new per-round
      `mean_ratio` diagnostic ≈ 1.0 and `clip_frac` == 0 (no trust-region binding) — proving the
      stored value flows into the ratio AND that K=1 is on-policy.

RED until v6 adds `behavior_logp=` + the `mean_ratio`/`clip_frac` diagnostics.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from unciv_train.contract import Dims  # noqa: E402
from unciv_train.dataset import TrainTrajectory  # noqa: E402
from unciv_train.features import build_rich_batch  # noqa: E402
from unciv_train.model import RUNGS, StructuredPolicyValueNet, _SPATIAL_FIELD_PLAN  # noqa: E402
from unciv_train.train import _masked_logp, train_actor_critic_structured  # noqa: E402

_TS = {"spatial": len(_SPATIAL_FIELD_PLAN), "own_units": 9, "opp_units": 9,
       "own_cities": 17, "opp_cities": 17, "civ_tokens": 84}
_VOCAB = {"terrain": 6, "resource": 5, "improvement": 4, "religion": 3, "era": 4,
          "building": 7, "unit": 8, "nation": 2, "promotion": 3}


def test_kotlin_logp_formula_matches_masked_logp():
    """ln(exps[pos]/sum) over the legal support == log_softmax(masked logits)[chosen]."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(2, 30))
        logits = rng.standard_normal(n).astype(np.float32) * 5
        mask = (rng.random(n) > 0.4)
        if not mask.any():
            continue
        legal = np.where(mask)[0]
        chosen = int(rng.choice(legal))
        # Kotlin chooseWithLogp formula:
        m = logits[legal]
        exps = np.exp((m - m.max()).astype(np.float64))
        kotlin_logp = float(np.log(exps[list(legal).index(chosen)] / exps.sum()))
        # Python _masked_logp:
        py = _masked_logp(torch.tensor(logits).unsqueeze(0),
                          torch.tensor([chosen]), torch.tensor(mask.astype(np.float32)).unsqueeze(0)).item()
        assert abs(kotlin_logp - py) < 1e-5, f"formula mismatch: kotlin={kotlin_logp} py={py}"
        # ratio = exp(logp - old_logp) ≈ 1 when both are the same quantity
        assert abs(np.exp(kotlin_logp - py) - 1.0) < 1e-4


def _traj(dims, n_steps, b_logp_tech=None, b_logp_policy=None):
    rng = np.random.default_rng(1)
    steps = []
    for i in range(n_steps):
        nt = 3 + (i % 3)
        steps.append({
            "global": np.zeros(dims.global_w, np.float32), "acting_civ": np.zeros(dims.acting_w, np.float32),
            "spatial": rng.integers(0, 4, size=(nt, _TS["spatial"])).astype(np.float32),
            "spatial_coords": np.stack([np.arange(nt), np.arange(nt) % 2], 1).astype(np.float32),
            "own_units": rng.standard_normal((2, 9)).astype(np.float32), "opp_units": np.zeros((0, 9), np.float32),
            "own_cities": rng.standard_normal((1, 17)).astype(np.float32), "opp_cities": np.zeros((0, 17), np.float32),
            "civ_tokens": rng.standard_normal((2, 84)).astype(np.float32),
        })
    rewards = np.zeros(n_steps, np.float32); rewards[-1] = 1.0
    return TrainTrajectory(
        np.zeros((n_steps, dims.input_w), np.float32), np.zeros(n_steps, np.int64), np.zeros(n_steps, np.int64),
        np.ones((n_steps, dims.tech_w), np.float32), np.ones((n_steps, dims.policy_w), np.float32),
        rewards, steps, b_logp_tech=b_logp_tech, b_logp_policy=b_logp_policy)


def test_stored_equals_recompute_gives_ratio_one():
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    tj = _traj(dims, 4)
    torch.manual_seed(0)
    net0 = StructuredPolicyValueNet(dims, _TS, _VOCAB, **RUNGS["small"])
    inputs = build_rich_batch([tj], dims, _TS)
    with torch.no_grad():
        tl, pl, _ = net0(inputs)
        recompute = (_masked_logp(tl, torch.tensor(tj.a_tech), torch.tensor(tj.mask_tech))
                     + _masked_logp(pl, torch.tensor(tj.a_policy), torch.tensor(tj.mask_policy))).numpy()
    tj_stored = _traj(dims, 4, b_logp_tech=recompute.astype(np.float32),
                      b_logp_policy=np.zeros_like(recompute, np.float32))
    _, stats, _ = train_actor_critic_structured(
        [tj_stored], dims, _TS, _VOCAB, RUNGS["small"], epochs=1, lr=1e-3, seed=0,
        clip_eps=0.2, net=copy.deepcopy(net0), behavior_logp=True)
    # mean_ratio + clip_frac are TRAINER-level diagnostics (frac_replayed is a run_loop metric).
    assert abs(stats["mean_ratio"] - 1.0) < 1e-4, f"mean_ratio={stats['mean_ratio']} (expected ≈1 at K=1)"
    assert stats["clip_frac"] == pytest.approx(0.0, abs=1e-6), f"clip_frac={stats['clip_frac']} (expected 0)"
