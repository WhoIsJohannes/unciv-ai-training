"""v5 AC6: micro-batched forward/backward traversal is numerically equivalent to the whole-batch
path. The frozen per-update objective is unchanged — only the TRAVERSAL is chunked, with each chunk's
.mean()-reduced loss size-weighted by n_c/N so the accumulated gradient equals the whole-batch
gradient (within fp tolerance). Oracle = the whole-batch path itself.

RED until v5 adds `micro_batch_steps=` to train_actor_critic_structured (+ the chunked path in
_optimize_actor_critic). Today the kwarg does not exist → TypeError → this fails for the right reason.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from unciv_train.contract import Dims  # noqa: E402
from unciv_train.dataset import TrainTrajectory  # noqa: E402
from unciv_train.model import RUNGS, _SPATIAL_FIELD_PLAN  # noqa: E402
from unciv_train.train import train_actor_critic_structured  # noqa: E402

_TS = {"spatial": len(_SPATIAL_FIELD_PLAN), "own_units": 9, "opp_units": 9,
       "own_cities": 17, "opp_cities": 17, "civ_tokens": 84}
_VOCAB = {"terrain": 6, "resource": 5, "improvement": 4, "religion": 3, "era": 4,
          "building": 7, "unit": 8, "nation": 2, "promotion": 3}


def _multi_step_trajectory(dims, n_steps=6):
    """One trajectory with n_steps decision steps, VARYING spatial-tile and entity counts per step so
    build_rich_batch pads to the batch-wide max — exercising the 'slice the already-padded full batch'
    path that chunking relies on."""
    rng = np.random.default_rng(0)
    steps = []
    for i in range(n_steps):
        n_tiles = 3 + (i % 3)                          # 3..5 tiles → varying maxn padding
        coords = np.stack([np.arange(n_tiles), (np.arange(n_tiles) % 2)], axis=1).astype(np.float32)
        g = np.zeros(dims.global_w, np.float32)
        g[5], g[6], g[7] = 0.0, 0.0, 1.0
        n_units = 1 + (i % 3)
        steps.append({
            "global": g, "acting_civ": np.zeros(dims.acting_w, np.float32),
            "spatial": rng.integers(0, 4, size=(n_tiles, _TS["spatial"])).astype(np.float32),
            "spatial_coords": coords,
            "own_units": rng.standard_normal((n_units, 9)).astype(np.float32),
            "opp_units": np.zeros((0, 9), np.float32),
            "own_cities": rng.standard_normal((1, 17)).astype(np.float32),
            "opp_cities": np.zeros((0, 17), np.float32),
            "civ_tokens": rng.standard_normal((2, 84)).astype(np.float32),
        })
    rewards = np.zeros(n_steps, np.float32)
    rewards[-1] = 1.0                                  # terminal-only ±1 reward (unchanged)
    return TrainTrajectory(
        np.zeros((n_steps, dims.input_w), np.float32), np.zeros(n_steps, np.int64),
        np.zeros(n_steps, np.int64), np.ones((n_steps, dims.tech_w), np.float32),
        np.ones((n_steps, dims.policy_w), np.float32), rewards, steps,
    )


def _run(micro_batch_steps):
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    tj = _multi_step_trajectory(dims, n_steps=6)
    # seed=0 → identical init + identical fresh optimizer; clip_eps ON (the real objective).
    net, stats, _opt = train_actor_critic_structured(
        [tj], dims, _TS, _VOCAB, RUNGS["small"], epochs=1, lr=1e-3, seed=0,
        clip_eps=0.2, micro_batch_steps=micro_batch_steps)
    flat = torch.cat([p.detach().reshape(-1) for p in net.parameters()])
    return stats, flat


def test_microbatch_matches_whole_batch():
    whole_stats, whole_w = _run(micro_batch_steps=None)       # oracle: whole-batch
    micro_stats, micro_w = _run(micro_batch_steps=2)          # 6 steps → 3 chunks
    assert np.isfinite(whole_stats["loss"]) and np.isfinite(micro_stats["loss"])
    assert abs(whole_stats["loss"] - micro_stats["loss"]) < 1e-5, \
        f"loss mismatch: whole={whole_stats['loss']} micro={micro_stats['loss']}"
    assert abs(whole_stats["grad_norm"] - micro_stats["grad_norm"]) < 1e-4, \
        f"grad_norm mismatch: whole={whole_stats['grad_norm']} micro={micro_stats['grad_norm']}"
    max_dw = (whole_w - micro_w).abs().max().item()
    assert max_dw < 1e-5, f"post-step weights diverge: max|Δw|={max_dw}"


def test_microbatch_noop_when_K_ge_N():
    """micro_batch_steps >= N must be byte-equivalent to the whole-batch path (the small-rung primary
    arm uses a no-op so it stays apples-to-apples with v4)."""
    _, whole_w = _run(micro_batch_steps=None)
    _, big_w = _run(micro_batch_steps=9999)                   # >= N → single chunk → no-op
    assert (whole_w - big_w).abs().max().item() < 1e-6
