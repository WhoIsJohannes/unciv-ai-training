"""AC6 PROVENANCE — unciv_train.dataset refuses mismatched shards and extracts (obs, action, return).

RED-first: this imports `unciv_train.dataset`, which the build creates. It fails at import until then.
Once `unciv_train` exists, these assertions pin the provenance gate + the (obs, action, return) extraction.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

# The module under test (created in Phase 3). A hard import → collection error until built = the RED signal.
from unciv_train import dataset
from unciv_dataplane.schema import SCHEMA_VERSION

MAGIC = b"UNCVSMP1"


def _build_v2_shard(
    *,
    version: int = SCHEMA_VERSION,
    fingerprint: str = "deadbeef",
    global_w: int = 4,
    acting_w: int = 3,
    tech_w: int = 5,
    policy_w: int = 4,
    terminal_reward: float = 1.0,
) -> bytes:
    """Minimal current-schema (v4) shard: one non-terminal learner step (civ_slot=0) + one terminal
    record. v6: carries a `behavior_logp` FIXED f32 block right after `actions` (same width)."""
    layout = [
        {"name": "global", "dtype": "<f4", "kind": "fixed", "perItem": 0, "len": global_w},
        {"name": "acting_civ", "dtype": "<f4", "kind": "fixed", "perItem": 0, "len": acting_w},
        {"name": "mask_tech", "dtype": "<u1", "kind": "fixed", "perItem": 0, "len": tech_w},
        {"name": "mask_policy", "dtype": "<u1", "kind": "fixed", "perItem": 0, "len": policy_w},
        {"name": "actions", "dtype": "<f4", "kind": "fixed", "perItem": 0, "len": 4},
        {"name": "behavior_logp", "dtype": "<f4", "kind": "fixed", "perItem": 0, "len": 4},
    ]
    import json

    header = json.dumps({
        "schemaVersion": version,
        "rulesetFingerprint": fingerprint,
        "gameId": "ep-test-0",
        "seed": 1,
        "majorCivSlots": [{"slot": 0, "civId": "SimulationCiv1"}],
        "nTiles": 10,
        "layout": layout,
    }).encode("utf-8")

    def step(turn, civ_slot, is_terminal, reward, a_tech, a_policy):
        # step header: i32 turn | i32 civSlot | u8 isFirst | u8 isLast | u8 isTerminal | u8 overflow | f32 reward
        body = struct.pack("<ii4Bf", turn, civ_slot, 0, 0, is_terminal, 0, reward)
        body += np.zeros(global_w, "<f4").tobytes()
        body += np.zeros(acting_w, "<f4").tobytes()
        body += np.ones(tech_w, "<u1").tobytes()      # all legal (test)
        body += np.ones(policy_w, "<u1").tobytes()
        body += np.array([a_tech, a_policy, -1, -1], "<f4").tobytes()
        body += np.array([-1.5, -0.7, 0, 0], "<f4").tobytes()   # behavior_logp per head
        return body

    records = b""
    for body in (
        step(5, 0, 0, 0.0, 2.0, 1.0),                 # learner non-terminal step
        step(6, 0, 1, terminal_reward, -1.0, -1.0),   # terminal record carries the reward
    ):
        records += struct.pack("<I", len(body)) + body

    out = MAGIC + struct.pack("<H", version) + struct.pack("<I", len(header)) + header + records
    out += struct.pack("<II", 2, zlib.crc32(records) & 0xFFFFFFFF)
    return out


def test_refuses_schema_version_mismatch(tmp_path):
    p = tmp_path / "v1.bin"
    p.write_bytes(_build_v2_shard(version=1))   # reader.load refuses VERSION 1 against the live SCHEMA_VERSION
    with pytest.raises(Exception):
        dataset.load_training_steps([p], expected_version=SCHEMA_VERSION, expected_fingerprint="deadbeef")


def test_refuses_fingerprint_mismatch(tmp_path):
    p = tmp_path / "wrongfp.bin"
    p.write_bytes(_build_v2_shard(fingerprint="00000000"))
    with pytest.raises(dataset.ProvenanceError):
        dataset.load_training_steps([p], expected_version=SCHEMA_VERSION, expected_fingerprint="deadbeef")


def test_extracts_obs_action_return(tmp_path):
    p = tmp_path / "ok.bin"
    p.write_bytes(_build_v2_shard(terminal_reward=1.0))
    steps = dataset.load_training_steps([p], expected_version=SCHEMA_VERSION, expected_fingerprint="deadbeef")
    assert len(steps) == 1                          # one non-terminal learner step
    s = steps[0]
    assert s.obs.shape[0] == 4 + 3                  # concat(global, acting_civ)
    assert s.a_tech == 2 and s.a_policy == 1        # chosen indices from the actions block
    assert s.ret == pytest.approx(1.0)              # terminal reward broadcast as return-to-go
