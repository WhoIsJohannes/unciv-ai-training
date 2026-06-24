"""D7 integration smoke: train_actor_critic_structured runs end-to-end (model + build_rich_batch
with derived neighbor tensors + the FROZEN _optimize_actor_critic core) for both a GNN-only rung
and an attention rung — no gradle, no NaN/divergence."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from unciv_train.contract import Dims  # noqa: E402
from unciv_train.dataset import TrainTrajectory  # noqa: E402
from unciv_train.model import RUNGS, _SPATIAL_FIELD_PLAN  # noqa: E402
from unciv_train.train import train_actor_critic_structured  # noqa: E402


def _tiny_trajectory(dims, token_specs):
    rng = np.random.default_rng(0)
    coords = np.array([(0, 0), (1, 1), (0, 1), (-1, 0)], np.float32)
    n = coords.shape[0]
    g = np.zeros(dims.global_w, np.float32)
    g[5], g[6], g[7] = 0.0, 0.0, 1.0   # effWrapRadius, worldWrap, shape (map dims)
    step = {
        "global": g, "acting_civ": np.zeros(dims.acting_w, np.float32),
        "spatial": rng.integers(0, 4, size=(n, token_specs["spatial"])).astype(np.float32),
        "spatial_coords": coords,
        "own_units": rng.standard_normal((2, 9)).astype(np.float32),
        "opp_units": np.zeros((0, 9), np.float32),       # empty → NaN-guard path
        "own_cities": rng.standard_normal((1, 17)).astype(np.float32),
        "opp_cities": np.zeros((0, 17), np.float32),
        "civ_tokens": rng.standard_normal((2, 84)).astype(np.float32),
    }
    return TrainTrajectory(
        np.zeros((1, dims.input_w), np.float32), np.array([0]), np.array([0]),
        np.ones((1, dims.tech_w), np.float32), np.ones((1, dims.policy_w), np.float32),
        np.array([1.0], np.float32), [step],
    )


@pytest.mark.parametrize("rung", ["small", "medium"])  # GNN-only + attention
def test_structured_trainer_runs(rung):
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    token_specs = {"spatial": len(_SPATIAL_FIELD_PLAN), "own_units": 9, "opp_units": 9,
                   "own_cities": 17, "opp_cities": 17, "civ_tokens": 84}
    vocab_counts = {"terrain": 6, "resource": 5, "improvement": 4, "religion": 3, "era": 4,
                    "building": 7, "unit": 8, "nation": 2, "promotion": 3}
    tj = _tiny_trajectory(dims, token_specs)
    net, stats = train_actor_critic_structured(
        [tj], dims, token_specs, vocab_counts, RUNGS[rung], epochs=2, lr=1e-3, seed=0)
    assert net is not None
    assert np.isfinite(float(stats.get("loss", 0.0))), f"{rung} loss not finite"
    assert not stats.get("diverged", False), f"{rung} diverged (NaN guard tripped)"
    assert stats.get("n", 0) >= 1
