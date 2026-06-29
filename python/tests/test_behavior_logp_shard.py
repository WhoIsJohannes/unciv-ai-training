"""v6 AC4 — the per-step `behavior_logp` FIXED f32 block round-trips through the layout-driven reader
into TrainTrajectory.b_logp_tech / b_logp_policy, and a v3 shard (no block, old version) refuses to load.

RED until v6 adds SCHEMA_VERSION=4, the TrainTrajectory.b_logp_* fields, and the load_trajectories
read of `behavior_logp`.
"""
from __future__ import annotations

import json
import struct
import zlib

import numpy as np
import pytest

from unciv_dataplane.schema import SCHEMA_VERSION  # noqa: E402
from unciv_train import dataset  # noqa: E402

MAGIC = b"UNCVSMP1"


def _build_shard(*, version=SCHEMA_VERSION, fingerprint="deadbeef", lp_tech=-2.31, lp_policy=-1.05,
                 global_w=4, acting_w=3, tech_w=5, policy_w=4, terminal_reward=1.0) -> bytes:
    """One non-terminal learner step (civ_slot=0) + terminal record, with an `actions` block AND a
    `behavior_logp` FIXED f32 block (width 4, slots [tech, policy, 0, 0])."""
    layout = [
        {"name": "global", "dtype": "<f4", "kind": "fixed", "perItem": 0, "len": global_w},
        {"name": "acting_civ", "dtype": "<f4", "kind": "fixed", "perItem": 0, "len": acting_w},
        {"name": "mask_tech", "dtype": "<u1", "kind": "fixed", "perItem": 0, "len": tech_w},
        {"name": "mask_policy", "dtype": "<u1", "kind": "fixed", "perItem": 0, "len": policy_w},
        {"name": "actions", "dtype": "<f4", "kind": "fixed", "perItem": 0, "len": 4},
        {"name": "behavior_logp", "dtype": "<f4", "kind": "fixed", "perItem": 0, "len": 4},
    ]
    header = json.dumps({
        "schemaVersion": version, "rulesetFingerprint": fingerprint, "gameId": "ep-0", "seed": 1,
        "majorCivSlots": [{"slot": 0, "civId": "SimulationCiv1"}], "nTiles": 10, "layout": layout,
    }).encode("utf-8")

    def step(turn, is_terminal, reward, a_tech, a_policy, blp_t, blp_p):
        body = struct.pack("<ii4Bf", turn, 0, 0, 0, is_terminal, 0, reward)
        body += np.zeros(global_w, "<f4").tobytes()
        body += np.zeros(acting_w, "<f4").tobytes()
        body += np.ones(tech_w, "<u1").tobytes()
        body += np.ones(policy_w, "<u1").tobytes()
        body += np.array([a_tech, a_policy, -1, -1], "<f4").tobytes()
        body += np.array([blp_t, blp_p, 0, 0], "<f4").tobytes()
        return body

    records = b""
    for body in (step(5, 0, 0.0, 2.0, 1.0, lp_tech, lp_policy),
                 step(6, 1, terminal_reward, -1.0, -1.0, 0.0, 0.0)):
        records += struct.pack("<I", len(body)) + body
    out = MAGIC + struct.pack("<H", version) + struct.pack("<I", len(header)) + header + records
    out += struct.pack("<II", 2, zlib.crc32(records) & 0xFFFFFFFF)
    return out


def test_behavior_logp_round_trips(tmp_path):
    p = tmp_path / "v4.bin"
    p.write_bytes(_build_shard(lp_tech=-2.31, lp_policy=-1.05))
    trajs = dataset.load_trajectories([p], expected_version=SCHEMA_VERSION,
                                      expected_fingerprint="deadbeef", rich=False)
    assert len(trajs) == 1
    t = trajs[0]
    assert t.b_logp_tech.shape == (1,) and t.b_logp_policy.shape == (1,)
    assert float(t.b_logp_tech[0]) == pytest.approx(-2.31, abs=1e-5)
    assert float(t.b_logp_policy[0]) == pytest.approx(-1.05, abs=1e-5)


def test_v3_shard_refuses_under_v4_reader(tmp_path):
    p = tmp_path / "v3.bin"
    p.write_bytes(_build_shard(version=3))                 # old layout version → reader refuses
    with pytest.raises(Exception):
        dataset.load_trajectories([p], expected_version=SCHEMA_VERSION,
                                  expected_fingerprint="deadbeef", rich=False)
