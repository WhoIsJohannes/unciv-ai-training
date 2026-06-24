"""v2 unit tests closing the traceability gaps: trajectory loader (R1 — keep no-action steps,
terminal-only reward), masked-pool NaN guard (R3), rich batch assembly, rich export value-drop."""
from __future__ import annotations

import json
import struct
import zlib

import numpy as np
import pytest

from unciv_train import dataset

torch = pytest.importorskip("torch")

MAGIC = b"UNCVSMP1"


def _shard_with_steps(steps_spec, *, global_w=4, acting_w=3, tech_w=5, policy_w=4,
                      terminal_reward=1.0, fingerprint="deadbeef"):
    """steps_spec = list of (a_tech, a_policy) for non-terminal learner steps, then a terminal."""
    n_tiles = 10
    layout = [
        {"name": "global", "dtype": "<f4", "kind": "fixed", "perItem": 0, "len": global_w},
        {"name": "acting_civ", "dtype": "<f4", "kind": "fixed", "perItem": 0, "len": acting_w},
        {"name": "mask_tech", "dtype": "<u1", "kind": "fixed", "perItem": 0, "len": tech_w},
        {"name": "mask_policy", "dtype": "<u1", "kind": "fixed", "perItem": 0, "len": policy_w},
        {"name": "actions", "dtype": "<f4", "kind": "fixed", "perItem": 0, "len": 4},
        {"name": "spatial", "dtype": "<u1", "kind": "fixed", "perItem": 0, "len": n_tiles * 13},
        {"name": "own_units", "dtype": "<f4", "kind": "var", "perItem": 8, "len": 0},
        {"name": "opp_units", "dtype": "<f4", "kind": "var", "perItem": 8, "len": 0},
        {"name": "own_cities", "dtype": "<f4", "kind": "var", "perItem": 16, "len": 0},
        {"name": "opp_cities", "dtype": "<f4", "kind": "var", "perItem": 16, "len": 0},
        {"name": "civ_tokens", "dtype": "<f4", "kind": "var", "perItem": 84, "len": 0},
    ]
    header = json.dumps({
        "schemaVersion": 2, "rulesetFingerprint": fingerprint, "gameId": "ep-0", "seed": 1,
        "majorCivSlots": [{"slot": 0, "civId": "SimulationCiv1"}], "nTiles": n_tiles, "layout": layout,
    }).encode("utf-8")

    def step(turn, is_terminal, reward, a_tech, a_policy):
        body = struct.pack("<ii4Bf", turn, 0, 0, 0, is_terminal, 0, reward)
        body += np.zeros(global_w, "<f4").tobytes()
        body += np.zeros(acting_w, "<f4").tobytes()
        body += np.ones(tech_w, "<u1").tobytes()
        body += np.ones(policy_w, "<u1").tobytes()
        body += np.array([a_tech, a_policy, -1, -1], "<f4").tobytes()
        body += np.zeros(n_tiles * 13, "<u1").tobytes()           # spatial (FIXED u8)
        for _ in range(5):                                         # 5 VARIABLE entity blocks, count 0
            body += struct.pack("<H", 0)
        return body

    records = b""
    t = 1
    for (at, ap) in steps_spec:
        records += (lambda b: struct.pack("<I", len(b)) + b)(step(t, 0, 0.0, at, ap)); t += 1
    term = step(t, 1, terminal_reward, -1.0, -1.0)
    records += struct.pack("<I", len(term)) + term
    out = MAGIC + struct.pack("<H", 2) + struct.pack("<I", len(header)) + header + records
    out += struct.pack("<II", len(steps_spec) + 1, zlib.crc32(records) & 0xFFFFFFFF)
    return out


def test_load_trajectories_keeps_noaction_and_terminal_only_reward(tmp_path):
    # one no-action step (both heads -1) FOLLOWED by an acting step → R1: BOTH kept, in order.
    p = tmp_path / "traj.bin"
    p.write_bytes(_shard_with_steps([(-1, -1), (2, 1)], terminal_reward=1.0))
    trajs = dataset.load_trajectories([p], expected_version=2, expected_fingerprint="deadbeef")
    assert len(trajs) == 1
    tj = trajs[0]
    assert len(tj.rewards) == 2, "no-action step must NOT be dropped (GAE needs the full sequence)"
    assert tj.a_tech.tolist() == [-1, 2] and tj.a_policy.tolist() == [-1, 1]
    np.testing.assert_array_equal(tj.rewards, [0.0, 1.0])  # terminal-only: 0 except last


def test_load_trajectories_rich_blocks_present(tmp_path):
    p = tmp_path / "traj.bin"
    p.write_bytes(_shard_with_steps([(2, 1)]))
    trajs = dataset.load_trajectories([p], expected_version=2, expected_fingerprint="deadbeef",
                                      rich=True, expected_spatial_channels=13)
    assert trajs[0].rich is not None and "spatial" in trajs[0].rich[0]


def test_masked_pool_empty_set_is_zero_no_nan():
    from unciv_train.model import masked_pool
    tokens = torch.zeros(1, 1, 2)
    mask = torch.zeros(1, 1)               # no present tokens
    out = masked_pool(tokens, mask)
    assert out.shape == (1, 4)             # mean(2) ‖ max(2)
    assert torch.isfinite(out).all() and float(out.abs().sum()) == 0.0


def test_masked_pool_mean_max_correct():
    from unciv_train.model import masked_pool
    tokens = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [9.0, 9.0]]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])  # third token is padding → excluded
    out = masked_pool(tokens, mask)[0]
    assert out[:2].tolist() == [2.0, 3.0]   # mean of [1,3],[2,4]
    assert out[2:].tolist() == [3.0, 4.0]   # max excludes the padded [9,9]


def test_build_rich_batch_pads_and_masks():
    from unciv_train.features import build_rich_batch
    from unciv_train.contract import Dims
    from unciv_train.dataset import TrainTrajectory
    specs = {"spatial": 3, "own_units": 2}
    rich = [{"global": np.zeros(4, np.float32), "acting_civ": np.zeros(4, np.float32),
             "spatial": np.ones((5, 3), np.float32), "own_units": np.zeros((0, 2), np.float32)}]
    tj = TrainTrajectory(np.zeros((1, 8), np.float32), np.array([0]), np.array([0]),
                         np.ones((1, 5), np.float32), np.ones((1, 4), np.float32),
                         np.array([1.0], np.float32), rich)
    b = build_rich_batch([tj], Dims(4, 4, 5, 4), specs)
    assert b["spatial"].shape == (1, 5, 3) and b["spatial_mask"].sum().item() == 5
    assert b["own_units"].shape == (1, 1, 2) and b["own_units_mask"].sum().item() == 0  # empty → pad1 mask0


def test_export_rich_drops_value(tmp_path):
    import onnx
    from unciv_train.contract import Dims
    from unciv_train.model import RichPolicyValueNet
    from unciv_train import export_onnx
    dims = Dims(4, 4, 5, 4)
    specs = {"spatial": 3, "own_units": 2}
    net = RichPolicyValueNet(dims, specs, token_dim=8, hidden=16)
    out = tmp_path / "rich.onnx"
    export_onnx.export_rich(net, dims, specs, out, schema_version=2, ruleset_fingerprint="x")
    names = {o.name for o in onnx.load(str(out)).graph.output}
    assert names == {"tech_logits", "policy_logits"}, f"value head leaked: {names}"
