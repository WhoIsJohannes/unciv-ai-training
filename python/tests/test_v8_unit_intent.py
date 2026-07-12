"""v8 — per-unit INTENT control head: schema contract + trainer no-op/BC/parity (mirrors test_v7_construction).

Covers, on the Python side:
  * SCHEMA_VERSION lockstep bump (8 -> 9) + the three new VARIABLE f32 blocks (unit_intent_action/_logp/
    _current, perItem=1, one row per own unit) + mask_unit_intent round-trip with NO reader change (AC#4/#5).
  * AC#2 NO-OP zero-summand oracle: with every unit action −1, wiring the unit-intent summand must NOT
    change the SHARED weights vs the head-off path (max|Δw|<1e-6).
  * Sanity: real unit actions DO move the head (the summand isn't inert when it shouldn't be).
  * AC (BC): behavior-cloning the unit-intent head reaches high accuracy on the heuristic target and leaves
    the civ (tech/policy) heads untouched.
  * AC#3 parity: exported unit_intent_logits (ORT) == the torch net's unit-intent head, atol 1e-4. The JVM
    OnnxPolicy reads the SAME ORT output, so ORT==torch is the cross-boundary guarantee (as for construction).
"""

import json
import os
import struct
import tempfile
import zlib
from pathlib import Path

import pytest

from unciv_dataplane import SCHEMA_VERSION, load

MAGIC = b"UNCVSMP1"
_STEP = struct.Struct("<ii4Bf")  # turn, civSlot, isFirst,isLast,isTerminal,overflow, reward


def _build_shard_v9(*, version, n_steps=3, n_units=3, intent_w=6) -> bytes:
    """A shard whose layout includes the three new v8 VARIABLE f32 blocks + the u8 mask, aligned to own_units."""
    layout = [
        {"name": "global", "dtype": "<f4", "kind": "fixed", "perItem": 0, "len": 4},
        {"name": "mask_tech", "dtype": "<u1", "kind": "fixed", "perItem": 0, "len": 8},
        {"name": "mask_unit_intent", "dtype": "<u1", "kind": "var", "perItem": intent_w, "len": 0},
        {"name": "actions", "dtype": "<f4", "kind": "fixed", "perItem": 0, "len": 4},
        {"name": "behavior_logp", "dtype": "<f4", "kind": "fixed", "perItem": 0, "len": 4},
        {"name": "unit_intent_action", "dtype": "<f4", "kind": "var", "perItem": 1, "len": 0},
        {"name": "unit_intent_logp", "dtype": "<f4", "kind": "var", "perItem": 1, "len": 0},
        {"name": "unit_intent_current", "dtype": "<f4", "kind": "var", "perItem": 1, "len": 0},  # BC target
        {"name": "phi", "dtype": "<f4", "kind": "fixed", "perItem": 0, "len": 1},
    ]
    header = {
        "schemaVersion": version, "uncivVersionText": "4.20.15", "uncivVersionNumber": 1229,
        "compatibilityNumber": 4, "rulesetFingerprint": "abc123", "gitSha": None,
        "gameId": "fixed-game-id", "seed": 42, "nTiles": 91, "caps": {"maxMajorCivs": 16},
        "spatialChannels": ["visibility_state"], "layout": layout,
    }
    hdr = json.dumps(header).encode("utf-8")
    records = bytearray()
    for i in range(n_steps):
        payload = bytearray()
        payload += _STEP.pack(i, 0, 1 if i == 0 else 0, 1 if i == n_steps - 1 else 0,
                              1 if i == n_steps - 1 else 0, 0, 0.0)
        for blk in layout:
            if blk.get("kind") == "var":
                rows = n_units
                payload += struct.pack("<H", rows)
                n = rows * blk["perItem"]
                payload += (struct.pack(f"<{n}f", *([0.0] * n)) if blk["dtype"] == "<f4" else bytes(n))
            else:
                n = blk["len"]
                payload += (struct.pack(f"<{n}f", *([0.0] * n)) if blk["dtype"] == "<f4" else bytes(n))
        records += struct.pack("<I", len(payload)) + payload
    out = bytearray()
    out += MAGIC + struct.pack("<H", version) + struct.pack("<I", len(hdr)) + hdr + records
    out += struct.pack("<II", n_steps, zlib.crc32(bytes(records)) & 0xFFFFFFFF)
    return bytes(out)


def test_schema_version_is_9():
    """Lockstep bump landed on the Python side (mirrors Kotlin SampleSchema.VERSION). v9 = v8 unit-intent."""
    assert SCHEMA_VERSION == 9, f"v8 unit-intent requires SCHEMA_VERSION 9 (got {SCHEMA_VERSION})"


def test_unit_intent_blocks_round_trip(tmp_path):
    """AC#4/#5: the new VARIABLE f32 blocks + the u8 mask round-trip with NO reader change."""
    n_units, intent_w = 3, 6
    p = tmp_path / "v9.bin"
    p.write_bytes(_build_shard_v9(version=SCHEMA_VERSION, n_units=n_units, intent_w=intent_w))
    shard = load(p)
    s0 = shard.steps[0]
    assert s0.blocks["unit_intent_action"].shape == (n_units, 1)
    assert s0.blocks["unit_intent_logp"].shape == (n_units, 1)
    assert s0.blocks["unit_intent_current"].shape == (n_units, 1)
    assert s0.blocks["mask_unit_intent"].shape == (n_units, intent_w)


def test_old_v8_shard_refused(tmp_path):
    """A v8 shard must refuse to load against the v9 reader (perishable old data)."""
    p = tmp_path / "old.bin"
    p.write_bytes(_build_shard_v9(version=8))
    from unciv_dataplane import ShardError
    with pytest.raises(ShardError, match="VERSION"):
        load(p)


# ---- v8 trainer: no-op zero-summand oracle + BC + ORT parity ---------------------------------------
import copy as _copy

import numpy as np
import pytest as _pytest

_pytest.importorskip("torch")
import torch  # noqa: E402

from unciv_train import contract as C  # noqa: E402
from unciv_train.contract import Dims  # noqa: E402
from unciv_train.dataset import TrainTrajectory  # noqa: E402
from unciv_train.export_onnx import export_rich  # noqa: E402
from unciv_train.model import RUNGS, StructuredPolicyValueNet, _SPATIAL_FIELD_PLAN  # noqa: E402
from unciv_train.train import bc_pretrain_construction, train_actor_critic_structured  # noqa: E402

_TS = {"spatial": len(_SPATIAL_FIELD_PLAN), "own_units": 9, "opp_units": 9,
       "own_cities": 17, "opp_cities": 17, "civ_tokens": 84}
_VC = {"terrain": 6, "resource": 5, "improvement": 4, "religion": 3, "era": 4,
       "building": 7, "unit": 8, "nation": 2, "promotion": 3, "unitIntent": 6}
_INTENT_W = _VC["unitIntent"]


def _traj_with_unit_intent(dims, n_steps=4, *, n_units=2, acted=False):
    """A structured trajectory whose per-step rich dicts carry unit-intent blocks. `acted=False` → every
    unit records action −1 (the OFF / no-op case); `acted=True` → real legal actions."""
    rng = np.random.default_rng(0)
    steps = []
    for i in range(n_steps):
        n_tiles = 3 + (i % 3)
        coords = np.stack([np.arange(n_tiles), (np.arange(n_tiles) % 2)], axis=1).astype(np.float32)
        umask = np.zeros((n_units, _INTENT_W), np.float32)
        umask[:, :5] = 1.0                                   # first 5 intents legal
        if acted:
            ua = np.array([1 + (j % 4) for j in range(n_units)], np.float32)  # legal idx in [0,5)
            ulp = np.full(n_units, -1.4, np.float32)
        else:
            ua = np.full(n_units, -1.0, np.float32)
            ulp = np.zeros(n_units, np.float32)
        steps.append({
            "global": np.zeros(dims.global_w, np.float32), "acting_civ": np.zeros(dims.acting_w, np.float32),
            "spatial": rng.integers(0, 4, size=(n_tiles, _TS["spatial"])).astype(np.float32),
            "spatial_coords": coords,
            "own_units": rng.standard_normal((n_units, 9)).astype(np.float32), "opp_units": np.zeros((0, 9), np.float32),
            "own_cities": rng.standard_normal((1, 17)).astype(np.float32),
            "opp_cities": np.zeros((0, 17), np.float32), "civ_tokens": rng.standard_normal((2, 84)).astype(np.float32),
            "mask_unit_intent": umask, "unit_intent_action": ua, "unit_intent_logp": ulp,
            # BC target: a legal per-unit intent (idx in [0,5) legal per the mask) to clone.
            "unit_intent_current": np.array([1 + (j % 4) for j in range(n_units)], np.float32),
        })
    rewards = np.zeros(n_steps, np.float32); rewards[-1] = 1.0
    return TrainTrajectory(np.zeros((n_steps, dims.input_w), np.float32),
                           np.zeros(n_steps, np.int64), np.zeros(n_steps, np.int64),
                           np.ones((n_steps, dims.tech_w), np.float32), np.ones((n_steps, dims.policy_w), np.float32),
                           rewards, steps)


def _weights(net):
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()])


def test_unit_intent_offarm_is_bit_identical_noop():
    """AC#2 deterministic oracle: with every unit action −1 (OFF), wiring the unit-intent summand must NOT
    change the SHARED weights vs the head-off path (unit_intent=False) — max|Δw|<1e-6."""
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    tj = _traj_with_unit_intent(dims, acted=False)
    torch.manual_seed(0); net0 = StructuredPolicyValueNet(dims, _TS, _VC, **RUNGS["small"])
    net_off, _, _ = train_actor_critic_structured(
        [tj], dims, _TS, _VC, RUNGS["small"], epochs=2, lr=1e-3, seed=0, clip_eps=0.2,
        net=_copy.deepcopy(net0), construction=False, unit_intent=False)
    net_on, _, _ = train_actor_critic_structured(
        [tj], dims, _TS, _VC, RUNGS["small"], epochs=2, lr=1e-3, seed=0, clip_eps=0.2,
        net=_copy.deepcopy(net0), construction=False, unit_intent=True)
    # compare only the SHARED params (the unit-intent head exists in both deepcopies, untouched at init).
    shared = [k for k in net0.state_dict() if not k.startswith("unit_intent_head.")]
    max_dw = max((net_off.state_dict()[k] - net_on.state_dict()[k]).abs().max().item() for k in shared)
    assert max_dw < 1e-6, f"OFF unit-intent summand is not a no-op: max|Δw|={max_dw}"


def test_unit_intent_trains_when_units_act():
    """Sanity: when units DO record real intent actions, the summand changes the gradient (unit_intent=True
    diverges from the unit_intent=False path) — the head actually learns."""
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    tj = _traj_with_unit_intent(dims, acted=True)
    torch.manual_seed(1); net0 = StructuredPolicyValueNet(dims, _TS, _VC, **RUNGS["small"])
    net_off, _, _ = train_actor_critic_structured(
        [tj], dims, _TS, _VC, RUNGS["small"], epochs=2, lr=1e-3, seed=0, clip_eps=0.2,
        net=_copy.deepcopy(net0), construction=False, unit_intent=False)
    net_on, _, _ = train_actor_critic_structured(
        [tj], dims, _TS, _VC, RUNGS["small"], epochs=2, lr=1e-3, seed=0, clip_eps=0.2,
        net=_copy.deepcopy(net0), construction=False, unit_intent=True)
    assert (_weights(net_off) - _weights(net_on)).abs().max().item() > 1e-5, \
        "unit intent actions did not affect training — the summand is inert when it shouldn't be"


def test_unit_intent_noop_holds_under_replay():
    """The no-op must also hold with v6 replay (behavior_logp=True): an OFF step's stored old_logp includes
    a 0 unit-intent term, so the importance ratio and weights stay identical."""
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    tj = _traj_with_unit_intent(dims, acted=False)
    torch.manual_seed(2); net0 = StructuredPolicyValueNet(dims, _TS, _VC, **RUNGS["small"])
    net_off, _, _ = train_actor_critic_structured(
        [tj], dims, _TS, _VC, RUNGS["small"], epochs=1, lr=1e-3, seed=0, clip_eps=0.2,
        net=_copy.deepcopy(net0), behavior_logp=True, construction=False, unit_intent=False)
    net_on, _, _ = train_actor_critic_structured(
        [tj], dims, _TS, _VC, RUNGS["small"], epochs=1, lr=1e-3, seed=0, clip_eps=0.2,
        net=_copy.deepcopy(net0), behavior_logp=True, construction=False, unit_intent=True)
    shared = [k for k in net0.state_dict() if not k.startswith("unit_intent_head.")]
    max_dw = max((net_off.state_dict()[k] - net_on.state_dict()[k]).abs().max().item() for k in shared)
    assert max_dw < 1e-6, f"OFF no-op broken under replay: max|Δw|={max_dw}"


def test_bc_pretrain_learns_unit_intent_picks():
    """The behavior-cloning pass drives the unit-intent head to predict the heuristic's recorded rung
    (`unit_intent_current`) with high accuracy — the v8 analog of the construction BC-acc test."""
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    tj = _traj_with_unit_intent(dims, n_steps=6, n_units=3, acted=True)
    torch.manual_seed(4); net = StructuredPolicyValueNet(dims, _TS, _VC, **RUNGS["small"])
    net, stats = bc_pretrain_construction(net, [tj], dims, _TS, epochs=150, lr=1e-3, micro_batch_steps=0)
    assert stats.get("bc_unit_acc", 0.0) > 0.9, f"BC did not learn the unit-intent picks: {stats}"


def test_bc_pretrain_leaves_civ_heads_untouched():
    """BC's unit-intent loss path (head + shared trunk) must NOT leak gradient into the civ tech head."""
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    tj = _traj_with_unit_intent(dims, n_steps=6, n_units=3, acted=True)
    torch.manual_seed(4); net = StructuredPolicyValueNet(dims, _TS, _VC, **RUNGS["small"])
    before = _copy.deepcopy(net.tech_head.state_dict())
    net, _ = bc_pretrain_construction(net, [tj], dims, _TS, epochs=20, lr=1e-3, micro_batch_steps=0)
    dmax = max((net.tech_head.state_dict()[k] - before[k]).abs().max().item() for k in before)
    assert dmax < 1e-7, f"BC leaked gradient into the tech head: max|Δw|={dmax}"


def test_unit_intent_logits_ort_matches_torch():
    """AC#3: the exported per-unit intent logits (ORT) equal the torch net's unit-intent head (atol 1e-4)."""
    ort = _pytest.importorskip("onnxruntime")
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    torch.manual_seed(3)
    net = StructuredPolicyValueNet(dims, _TS, _VC, **RUNGS["medium"]); net.eval()
    N, U = 5, 3
    inp = {"global": torch.randn(1, 8), "acting_civ": torch.randn(1, 6),
           "spatial": torch.randn(1, N, _TS["spatial"]), "spatial_mask": torch.ones(1, N),
           "neighbor_index": torch.zeros(1, N, 6, dtype=torch.long), "neighbor_mask": torch.ones(1, N, 6)}
    for n in ("own_units", "opp_units", "own_cities", "opp_cities", "civ_tokens"):
        cnt = U if n == "own_units" else 1
        inp[n] = torch.randn(1, cnt, _TS[n]); inp[n + "_mask"] = torch.ones(1, cnt)
    with torch.no_grad():
        _t, _p, _c, _cv, ui_torch, _v = net(inp, with_construction=True)   # v8: forward is a 6-tuple
    tmp = tempfile.mktemp(suffix=".onnx")
    export_rich(net, dims, _TS, tmp, schema_version=SCHEMA_VERSION, ruleset_fingerprint="fp",
                sample_inputs={k: v.numpy() for k, v in inp.items()}, neighbors=True,
                contract_version=C.CONTRACT_VERSION_STRUCTURED)
    sess = ort.InferenceSession(tmp)
    feed = {i.name: (inp[i.name].numpy() if i.name != "neighbor_index" else inp[i.name].numpy().astype("int64"))
            for i in sess.get_inputs()}
    ui_ort = sess.run([C.OUTPUT_UNIT_INTENT], feed)[0]
    os.remove(tmp)
    assert ui_ort.shape == tuple(ui_torch.shape), f"{ui_ort.shape} != {tuple(ui_torch.shape)}"
    assert np.allclose(ui_torch.numpy(), ui_ort, atol=1e-4), \
        f"unit-intent logits ORT != torch: max|Δ|={np.abs(ui_torch.numpy()-ui_ort).max()}"
