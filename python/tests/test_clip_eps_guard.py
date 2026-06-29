"""v6 AC3 — the clip_eps-truthy guard. With replay active (stored behavior logp in use), clip_eps
MUST be truthy: the clip_eps-falsy policy-loss branch is `-(adv*logp).mean()`, which NEVER references
old_logp — so it would silently ignore the stored behavior logp and apply replayed advantages as if
on-policy (a biased gradient). The trainer must fail LOUD instead.

RED until v6 adds the `behavior_logp=` plumb-through + the guard.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from unciv_train.contract import Dims  # noqa: E402
from unciv_train.dataset import TrainTrajectory  # noqa: E402
from unciv_train.model import RUNGS, _SPATIAL_FIELD_PLAN  # noqa: E402
from unciv_train.train import train_actor_critic_structured  # noqa: E402

_TS = {"spatial": len(_SPATIAL_FIELD_PLAN), "own_units": 9, "opp_units": 9,
       "own_cities": 17, "opp_cities": 17, "civ_tokens": 84}
_VOCAB = {"terrain": 6, "resource": 5, "improvement": 4, "religion": 3, "era": 4,
          "building": 7, "unit": 8, "nation": 2, "promotion": 3}


def _traj(dims, n=3):
    rng = np.random.default_rng(0)
    steps = []
    for i in range(n):
        nt = 3 + (i % 3)
        steps.append({
            "global": np.zeros(dims.global_w, np.float32), "acting_civ": np.zeros(dims.acting_w, np.float32),
            "spatial": rng.integers(0, 4, (nt, _TS["spatial"])).astype(np.float32),
            "spatial_coords": np.stack([np.arange(nt), np.arange(nt) % 2], 1).astype(np.float32),
            "own_units": rng.standard_normal((1, 9)).astype(np.float32), "opp_units": np.zeros((0, 9), np.float32),
            "own_cities": rng.standard_normal((1, 17)).astype(np.float32), "opp_cities": np.zeros((0, 17), np.float32),
            "civ_tokens": rng.standard_normal((2, 84)).astype(np.float32),
        })
    rewards = np.zeros(n, np.float32); rewards[-1] = 1.0
    blp = np.zeros(n, np.float32)
    return TrainTrajectory(
        np.zeros((n, dims.input_w), np.float32), np.zeros(n, np.int64), np.zeros(n, np.int64),
        np.ones((n, dims.tech_w), np.float32), np.ones((n, dims.policy_w), np.float32),
        rewards, steps, b_logp_tech=blp, b_logp_policy=blp)


def test_clip_eps_zero_with_replay_raises():
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    with pytest.raises((ValueError, AssertionError)):
        train_actor_critic_structured([_traj(dims)], dims, _TS, _VOCAB, RUNGS["small"],
                                      epochs=1, seed=0, clip_eps=0.0, behavior_logp=True)


def test_clip_eps_zero_without_replay_is_allowed():
    # No replay (behavior_logp falsy) ⇒ plain A2C single-epoch path is still valid with clip_eps=0.
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    net, stats, _ = train_actor_critic_structured([_traj(dims)], dims, _TS, _VOCAB, RUNGS["small"],
                                                  epochs=1, seed=0, clip_eps=0.0, behavior_logp=None)
    assert np.isfinite(stats["loss"])
