"""v6 AC1 — the K=1 / window-gated REPLAY NO-OP.

When the replay window is 1 the run loop passes `behavior_logp=None`, so `_optimize_actor_critic`
takes the ORIGINAL recompute branch (the literal v5 path). This test proves the equivalence at the
trainer level: when the STORED behavior logp exactly equals the round-start recompute (which is the
case at K=1 because the gen net IS that round's warm net), passing it as `behavior_logp=stored` is
bit-identical to passing `behavior_logp=None`. ⇒ the importance ratio ≡ 1 ⇒ zero numerical drift.

RED until v6 adds the `behavior_logp=` kwarg to the structured trainer + `stored_old_logp` to
_optimize_actor_critic + the `b_logp_*` fields on TrainTrajectory → TypeError until then.
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


def _traj(dims, n_steps=4, *, b_logp_tech=None, b_logp_policy=None):
    rng = np.random.default_rng(0)
    steps = []
    for i in range(n_steps):
        n_tiles = 3 + (i % 3)
        coords = np.stack([np.arange(n_tiles), (np.arange(n_tiles) % 2)], axis=1).astype(np.float32)
        steps.append({
            "global": np.zeros(dims.global_w, np.float32), "acting_civ": np.zeros(dims.acting_w, np.float32),
            "spatial": rng.integers(0, 4, size=(n_tiles, _TS["spatial"])).astype(np.float32),
            "spatial_coords": coords,
            "own_units": rng.standard_normal((1 + i % 3, 9)).astype(np.float32),
            "opp_units": np.zeros((0, 9), np.float32),
            "own_cities": rng.standard_normal((1, 17)).astype(np.float32),
            "opp_cities": np.zeros((0, 17), np.float32),
            "civ_tokens": rng.standard_normal((2, 84)).astype(np.float32),
        })
    rewards = np.zeros(n_steps, np.float32)
    rewards[-1] = 1.0
    a_tech = np.zeros(n_steps, np.int64)          # head acts (index 0); all-legal mask
    a_policy = np.zeros(n_steps, np.int64)
    return TrainTrajectory(
        np.zeros((n_steps, dims.input_w), np.float32), a_tech, a_policy,
        np.ones((n_steps, dims.tech_w), np.float32), np.ones((n_steps, dims.policy_w), np.float32),
        rewards, steps, b_logp_tech=b_logp_tech, b_logp_policy=b_logp_policy)


def _weights(net):
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()])


def test_k1_stored_equals_none_bit_identical():
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    tj = _traj(dims)
    # round-start recompute under a fixed net == the behavior logp at K=1 (gen net IS the warm net).
    torch.manual_seed(0)
    net0 = StructuredPolicyValueNet(dims, _TS, _VOCAB, **RUNGS["small"])
    inputs = build_rich_batch([tj], dims, _TS)
    with torch.no_grad():
        tl, pl, _ = net0(inputs)
        recompute = (_masked_logp(tl, torch.tensor(tj.a_tech), torch.tensor(tj.mask_tech))
                     + _masked_logp(pl, torch.tensor(tj.a_policy), torch.tensor(tj.mask_policy)))
    stored_tech = recompute.numpy().astype(np.float32)          # put the whole sum in the tech head
    stored_policy = np.zeros_like(stored_tech)
    tj_stored = _traj(dims, b_logp_tech=stored_tech, b_logp_policy=stored_policy)

    net_none, _, _ = train_actor_critic_structured(
        [tj], dims, _TS, _VOCAB, RUNGS["small"], epochs=1, lr=1e-3, seed=0,
        clip_eps=0.2, net=copy.deepcopy(net0), behavior_logp=None)
    net_stored, _, _ = train_actor_critic_structured(
        [tj_stored], dims, _TS, _VOCAB, RUNGS["small"], epochs=1, lr=1e-3, seed=0,
        clip_eps=0.2, net=copy.deepcopy(net0), behavior_logp=True)

    max_dw = (_weights(net_none) - _weights(net_stored)).abs().max().item()
    assert max_dw < 1e-6, f"K=1 stored-vs-None weights diverge: max|Δw|={max_dw} (no-op broken)"
