"""v7 — per-city construction control head: schema contract (AC#4).

RED until the v7 schema bump lands:
  * Kotlin SampleSchema.VERSION 4 -> 5 AND Python unciv_dataplane.SCHEMA_VERSION 4 -> 5 (lockstep).
  * The descriptor-generic reader must round-trip the two NEW variable f32 blocks
    (`construction_action`, `construction_logp`, perItem=1, one row per own city) with NO reader change.

These tests fail today because SCHEMA_VERSION == 4 (so `assert == 5` fails AND a v5 shard is refused
by `expect_compatible`). They go GREEN once the lockstep bump lands — and the round-trip test proves
AC#4 (new variable blocks decode without touching reader.py). The deeper trainer/no-op/parity/legality
assertions (AC#1,#2,#5) live in the Kotlin suite + the construction-aware trainer tests added in build.
"""

import json
import struct
import zlib
from pathlib import Path

import pytest

from unciv_dataplane import SCHEMA_VERSION, load

MAGIC = b"UNCVSMP1"
_STEP = struct.Struct("<ii4Bf")  # turn, civSlot, isFirst,isLast,isTerminal,overflow, reward


def _build_shard_v7(*, version, n_steps=3, n_cities=3, constr_w=5) -> bytes:
    """A shard whose layout includes the two new v7 VARIABLE f32 blocks aligned to own_cities."""
    layout = [
        {"name": "global", "dtype": "<f4", "kind": "fixed", "perItem": 0, "len": 4},
        {"name": "mask_tech", "dtype": "<u1", "kind": "fixed", "perItem": 0, "len": 8},
        {"name": "mask_construction", "dtype": "<u1", "kind": "var", "perItem": constr_w, "len": 0},
        {"name": "actions", "dtype": "<f4", "kind": "fixed", "perItem": 0, "len": 4},
        {"name": "behavior_logp", "dtype": "<f4", "kind": "fixed", "perItem": 0, "len": 4},
        {"name": "construction_action", "dtype": "<f4", "kind": "var", "perItem": 1, "len": 0},
        {"name": "construction_logp", "dtype": "<f4", "kind": "var", "perItem": 1, "len": 0},
        {"name": "econ_city", "dtype": "<f4", "kind": "var", "perItem": 1, "len": 0},  # v7.3 per-city log-economy
        {"name": "construction_current", "dtype": "<f4", "kind": "var", "perItem": 1, "len": 0},  # v7.4 BC target
        {"name": "phi", "dtype": "<f4", "kind": "fixed", "perItem": 0, "len": 1},  # v7.2 economy potential
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
                rows = n_cities
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


def test_schema_version_is_8():
    """Lockstep bump landed on the Python side (mirrors Kotlin SampleSchema.VERSION). v8 = v7.4 BC target."""
    assert SCHEMA_VERSION == 8, f"v7.4 requires SCHEMA_VERSION 8 (got {SCHEMA_VERSION})"


def test_construction_blocks_round_trip(tmp_path):
    """AC#4: the new VARIABLE f32 blocks (construction + v7.3 econ_city) round-trip with NO reader change."""
    n_cities, constr_w = 3, 5
    p = tmp_path / "v7.bin"
    p.write_bytes(_build_shard_v7(version=SCHEMA_VERSION, n_cities=n_cities, constr_w=constr_w))
    shard = load(p)  # refuses today (SCHEMA_VERSION==4 != 5); GREEN after the bump
    s0 = shard.steps[0]
    assert "construction_action" in s0.blocks and "construction_logp" in s0.blocks
    assert s0.blocks["construction_action"].shape == (n_cities, 1)
    assert s0.blocks["construction_logp"].shape == (n_cities, 1)
    assert s0.blocks["econ_city"].shape == (n_cities, 1)          # v7.3 per-city economy
    assert s0.blocks["construction_current"].shape == (n_cities, 1)  # v7.4 BC target
    assert s0.blocks["mask_construction"].shape == (n_cities, constr_w)


def test_old_v4_shard_refused(tmp_path):
    """A v4 shard must refuse to load against the v5 reader (perishable old data)."""
    p = tmp_path / "old.bin"
    p.write_bytes(_build_shard_v7(version=4))
    from unciv_dataplane import ShardError
    with pytest.raises(ShardError, match="VERSION"):
        load(p)


# ---- v7 trainer: no-op zero-summand oracle (AC#5 / PR1) + construction-trains sanity ----------------
import copy as _copy

import numpy as np
import pytest as _pytest

_pytest.importorskip("torch")
import torch  # noqa: E402

from unciv_train.contract import Dims  # noqa: E402
from unciv_train.dataset import TrainTrajectory  # noqa: E402
from unciv_train.model import RUNGS, StructuredPolicyValueNet, _SPATIAL_FIELD_PLAN  # noqa: E402
from unciv_train.train import train_actor_critic_structured  # noqa: E402

_TS = {"spatial": len(_SPATIAL_FIELD_PLAN), "own_units": 9, "opp_units": 9,
       "own_cities": 17, "opp_cities": 17, "civ_tokens": 84}
_VC = {"terrain": 6, "resource": 5, "improvement": 4, "religion": 3, "era": 4,
       "building": 7, "unit": 8, "nation": 2, "promotion": 3}
_CONSTR_W = _VC["building"] + _VC["unit"]


def _traj_with_construction(dims, n_steps=4, *, n_cities=2, acted=False):
    """A structured trajectory whose per-step rich dicts carry construction blocks. `acted=False` →
    every city records action −1 (the OFF / no-op case); `acted=True` → real legal actions."""
    rng = np.random.default_rng(0)
    steps = []
    for i in range(n_steps):
        n_tiles = 3 + (i % 3)
        coords = np.stack([np.arange(n_tiles), (np.arange(n_tiles) % 2)], axis=1).astype(np.float32)
        mask = np.zeros((n_cities, _CONSTR_W), np.float32)
        mask[:, :4] = 1.0                                   # first 4 constructions legal
        if acted:
            a = np.array([1 + (j % 3) for j in range(n_cities)], np.float32)  # legal idx in [0,4)
            lp = np.full(n_cities, -1.3, np.float32)
        else:
            a = np.full(n_cities, -1.0, np.float32)
            lp = np.zeros(n_cities, np.float32)
        steps.append({
            "global": np.zeros(dims.global_w, np.float32), "acting_civ": np.zeros(dims.acting_w, np.float32),
            "spatial": rng.integers(0, 4, size=(n_tiles, _TS["spatial"])).astype(np.float32),
            "spatial_coords": coords,
            "own_units": rng.standard_normal((1, 9)).astype(np.float32), "opp_units": np.zeros((0, 9), np.float32),
            "own_cities": rng.standard_normal((n_cities, 17)).astype(np.float32),
            "opp_cities": np.zeros((0, 17), np.float32), "civ_tokens": rng.standard_normal((2, 84)).astype(np.float32),
            "mask_construction": mask, "construction_action": a, "construction_logp": lp,
            # v7.3 per-city raw log-economy (distinct per city so the per-city advantage is non-degenerate).
            "econ_city": np.array([2.0 + 0.5 * j + 0.1 * i for j in range(n_cities)], np.float32),
            # v7.4 BC target: a legal per-city construction (idx in [0,4) legal per the mask) to clone.
            "construction_current": np.array([1 + (j % 3) for j in range(n_cities)], np.float32),
        })
    rewards = np.zeros(n_steps, np.float32); rewards[-1] = 1.0
    return TrainTrajectory(np.zeros((n_steps, dims.input_w), np.float32),
                           np.zeros(n_steps, np.int64), np.zeros(n_steps, np.int64),
                           np.ones((n_steps, dims.tech_w), np.float32), np.ones((n_steps, dims.policy_w), np.float32),
                           rewards, steps)


def _weights(net):
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()])


def test_construction_offarm_is_bit_identical_noop():
    """AC#5 / PR1 deterministic oracle: with every city action −1 (OFF), wiring the construction
    summand must NOT change the SHARED weights vs the pure-v6 path (construction=False) — max|Δw|<1e-6."""
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    tj = _traj_with_construction(dims, acted=False)
    torch.manual_seed(0); net0 = StructuredPolicyValueNet(dims, _TS, _VC, **RUNGS["small"])
    net_off, _, _ = train_actor_critic_structured(
        [tj], dims, _TS, _VC, RUNGS["small"], epochs=2, lr=1e-3, seed=0, clip_eps=0.2,
        net=_copy.deepcopy(net0), construction=False)
    net_on, _, _ = train_actor_critic_structured(
        [tj], dims, _TS, _VC, RUNGS["small"], epochs=2, lr=1e-3, seed=0, clip_eps=0.2,
        net=_copy.deepcopy(net0), construction=True)
    # compare only the SHARED params (the construction head exists in both deepcopies, untouched at init).
    shared = [k for k in net0.state_dict() if not k.startswith("construction_head.")]
    max_dw = max((net_off.state_dict()[k] - net_on.state_dict()[k]).abs().max().item() for k in shared)
    assert max_dw < 1e-6, f"OFF construction summand is not a no-op: max|Δw|={max_dw}"


def test_construction_trains_when_cities_act():
    """Sanity: when cities DO record real construction actions, the construction summand changes the
    gradient (construction=True diverges from the construction=False path) — the head actually learns."""
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    tj = _traj_with_construction(dims, acted=True)
    torch.manual_seed(1); net0 = StructuredPolicyValueNet(dims, _TS, _VC, **RUNGS["small"])
    net_off, _, _ = train_actor_critic_structured(
        [tj], dims, _TS, _VC, RUNGS["small"], epochs=2, lr=1e-3, seed=0, clip_eps=0.2,
        net=_copy.deepcopy(net0), construction=False)
    net_on, _, _ = train_actor_critic_structured(
        [tj], dims, _TS, _VC, RUNGS["small"], epochs=2, lr=1e-3, seed=0, clip_eps=0.2,
        net=_copy.deepcopy(net0), construction=True)
    assert (_weights(net_off) - _weights(net_on)).abs().max().item() > 1e-5, \
        "construction actions did not affect training — the summand is inert when it shouldn't be"


def test_construction_noop_holds_under_replay():
    """The no-op must also hold with v6 replay (behavior_logp=True): an OFF step's stored old_logp
    includes a 0 construction term, so the importance ratio and weights stay v6-identical."""
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    tj = _traj_with_construction(dims, acted=False)
    torch.manual_seed(2); net0 = StructuredPolicyValueNet(dims, _TS, _VC, **RUNGS["small"])
    net_off, _, _ = train_actor_critic_structured(
        [tj], dims, _TS, _VC, RUNGS["small"], epochs=1, lr=1e-3, seed=0, clip_eps=0.2,
        net=_copy.deepcopy(net0), behavior_logp=True, construction=False)
    net_on, _, _ = train_actor_critic_structured(
        [tj], dims, _TS, _VC, RUNGS["small"], epochs=1, lr=1e-3, seed=0, clip_eps=0.2,
        net=_copy.deepcopy(net0), behavior_logp=True, construction=True)
    shared = [k for k in net0.state_dict() if not k.startswith("construction_head.")]
    max_dw = max((net_off.state_dict()[k] - net_on.state_dict()[k]).abs().max().item() for k in shared)
    assert max_dw < 1e-6, f"OFF no-op broken under replay: max|Δw|={max_dw}"


def test_construction_logits_ort_matches_torch():
    """AC#2 (parity): the ONNX-runtime per-city construction logits == the torch construction head on
    a fixed observation (atol 1e-4). The JVM OnnxPolicy reads the SAME ORT output, so ORT==torch is the
    cross-boundary numerical guarantee for the new per-city head."""
    ort = _pytest.importorskip("onnxruntime")
    import tempfile, os
    import unciv_train.contract as C
    from unciv_train.export_onnx import export_rich
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    torch.manual_seed(3)
    net = StructuredPolicyValueNet(dims, _TS, _VC, **RUNGS["medium"]); net.eval()
    N, M = 5, 3
    inp = {"global": torch.randn(1, 8), "acting_civ": torch.randn(1, 6),
           "spatial": torch.randn(1, N, _TS["spatial"]), "spatial_mask": torch.ones(1, N),
           "neighbor_index": torch.zeros(1, N, 6, dtype=torch.long), "neighbor_mask": torch.ones(1, N, 6)}
    for n in ("own_units", "opp_units", "own_cities", "opp_cities", "civ_tokens"):
        cnt = M if n == "own_cities" else 1
        inp[n] = torch.randn(1, cnt, _TS[n]); inp[n + "_mask"] = torch.ones(1, cnt)
    with torch.no_grad():
        _t, _p, c_torch, _cv, _v = net(inp, with_construction=True)   # v7.3: forward is a 5-tuple (+city value)
    tmp = tempfile.mktemp(suffix=".onnx")
    export_rich(net, dims, _TS, tmp, schema_version=SCHEMA_VERSION, ruleset_fingerprint="fp",
                sample_inputs={k: v.numpy() for k, v in inp.items()}, neighbors=True,
                contract_version=C.CONTRACT_VERSION_STRUCTURED)
    sess = ort.InferenceSession(tmp)
    feed = {i.name: (inp[i.name].numpy() if i.name != "neighbor_index" else inp[i.name].numpy().astype("int64"))
            for i in sess.get_inputs()}
    c_ort = sess.run([C.OUTPUT_CONSTRUCTION], feed)[0]
    os.remove(tmp)
    assert c_ort.shape == tuple(c_torch.shape), f"{c_ort.shape} != {tuple(c_torch.shape)}"
    assert np.allclose(c_torch.numpy(), c_ort, atol=1e-4), \
        f"construction logits ORT != torch: max|Δ|={np.abs(c_torch.numpy()-c_ort).max()}"


# ---- v7.2 potential-based reward shaping (PBRS) ----------------------------------------------------
def test_pbrs_telescopes_to_constant_offset():
    """Policy-invariance (Ng-Harada): the shaping the trainer applies — shaped[t] += coef*(γ*φ[t+1]-φ[t])
    for t<L-1, last step unshaped — adds to the DISCOUNTED return exactly coef*(γ^(L-1)*φ[L-1] - φ[0]),
    a constant determined only by the (fixed-start) φ[0] and a vanishing γ^(L-1) tail. So the optimal
    policy over the discounted return is unchanged; only the credit timing shifts."""
    rng = np.random.default_rng(0)
    gamma, coef, L = 0.99, 0.1, 40
    phi = rng.standard_normal(L).astype(np.float32) * 3 + 10
    rewards = np.zeros(L, np.float32); rewards[-1] = 1.0
    shaped = rewards.copy()
    shaped[:L-1] += coef * (gamma * phi[1:] - phi[:-1])               # the exact code formula
    disc = gamma ** np.arange(L)
    offset = float((disc * (shaped - rewards)).sum())
    expected = coef * (gamma ** (L - 1) * phi[L-1] - phi[0])          # telescoped form
    assert abs(offset - expected) < 1e-4, f"PBRS does not telescope to a constant: {offset} vs {expected}"


def test_pbrs_coef_zero_and_none_phi_are_noops():
    """coef=0 OR phi=None ⇒ no shaping ⇒ bit-identical training to the terminal-only path."""
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    tj = _traj_with_construction(dims, acted=False)   # phi defaults None on this synthetic trajectory
    torch.manual_seed(0); net0 = StructuredPolicyValueNet(dims, _TS, _VC, **RUNGS["small"])
    base, _, _ = train_actor_critic_structured([tj], dims, _TS, _VC, RUNGS["small"], epochs=2, lr=1e-3,
        seed=0, clip_eps=0.2, net=_copy.deepcopy(net0), construction=False, reward_shaping_coef=0.0)
    # coef>0 but phi is None (synthetic) → the None-guard must make it a no-op.
    coefpos, _, _ = train_actor_critic_structured([tj], dims, _TS, _VC, RUNGS["small"], epochs=2, lr=1e-3,
        seed=0, clip_eps=0.2, net=_copy.deepcopy(net0), construction=False, reward_shaping_coef=0.1)
    max_dw = (_weights(base) - _weights(coefpos)).abs().max().item()
    assert max_dw < 1e-6, f"PBRS not a no-op when phi is None: max|Δw|={max_dw}"


# ---- v7.3 per-city credit assignment ---------------------------------------------------------------
def test_per_city_gae_credits_each_city_by_its_own_return():
    """`_per_city_gae` runs an INDEPENDENT GAE per city. A city with a high, rising economy return must
    get a different advantage than a flat-economy city — the whole point of per-city attribution. Padded
    (present=0) rows are zeroed."""
    from unciv_train.train import _per_city_gae
    # 1 trajectory, L=3, M=2. City 0 economy rises; city 1 flat. Baselines all zero.
    econ = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]], np.float32)
    cval = np.zeros((3, 2), np.float32)
    present = np.array([[1, 1], [1, 1], [1, 0]], np.float32)   # city1 absent at t=2 (padding)
    A, R = _per_city_gae(econ, cval, present, [3], gamma=0.99, lam=0.95)
    assert A.shape == (3, 2) and R.shape == (3, 2)
    assert A[0, 0] != A[0, 1], "distinct per-city economies must yield distinct per-city advantages"
    assert A[2, 1] == 0.0 and R[2, 1] == 0.0, "padded/absent city row must be zeroed"
    # At λ=1 with zero baseline, R == the Monte-Carlo discounted econ return: city0 R[0]=1+.99*2+.99^2*3.
    A1, R1 = _per_city_gae(econ, cval, present, [3], gamma=0.99, lam=1.0)
    assert abs(R1[0, 0] - (1.0 + 0.99 * 2.0 + 0.99 ** 2 * 3.0)) < 1e-3


def test_per_city_credit_changes_training_vs_shared_adv():
    """coef>0 (per-city credit: construction pulled out of the joint ratio, per-city advantage) must
    diverge from coef=0 (legacy shared-adv, construction in the joint ratio) — the per-city economy
    advantage actually flows into the construction head. Uses --replay-window-1-equivalent on-policy path."""
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    tj = _traj_with_construction(dims, acted=True)            # cities act + carry econ_city
    torch.manual_seed(5); net0 = StructuredPolicyValueNet(dims, _TS, _VC, **RUNGS["small"])
    shared, _, _ = train_actor_critic_structured(
        [tj], dims, _TS, _VC, RUNGS["small"], epochs=2, lr=1e-3, seed=0, clip_eps=0.2,
        net=_copy.deepcopy(net0), construction=True, construction_credit_coef=0.0)
    percity, _, _ = train_actor_critic_structured(
        [tj], dims, _TS, _VC, RUNGS["small"], epochs=2, lr=1e-3, seed=0, clip_eps=0.2,
        net=_copy.deepcopy(net0), construction=True, construction_credit_coef=0.5)
    assert (_weights(shared) - _weights(percity)).abs().max().item() > 1e-5, \
        "per-city credit (coef>0) did not change training vs shared-adv (coef=0)"


def test_per_city_credit_trains_city_value_head():
    """The per-city value head must receive gradient under per-city credit (it's the per-city baseline).
    Under coef=0 it is untouched (no per-city term); under coef>0 it moves."""
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    tj = _traj_with_construction(dims, acted=True)
    torch.manual_seed(6); net0 = StructuredPolicyValueNet(dims, _TS, _VC, **RUNGS["small"])
    cv0 = torch.cat([p.detach().reshape(-1) for p in net0.city_value_head.parameters()])
    percity, _, _ = train_actor_critic_structured(
        [tj], dims, _TS, _VC, RUNGS["small"], epochs=2, lr=1e-3, seed=0, clip_eps=0.2,
        net=_copy.deepcopy(net0), construction=True, construction_credit_coef=0.5)
    cv1 = torch.cat([p.detach().reshape(-1) for p in percity.city_value_head.parameters()])
    assert (cv0 - cv1).abs().max().item() > 1e-6, "city value head received no gradient under per-city credit"


def test_bc_pretrain_learns_heuristic_picks():
    """v7.4: behavior-cloning the construction head raises its accuracy at predicting the recorded target
    (construction_current) from ~random toward the target — the non-collapsed ~heuristic start for RL."""
    from unciv_train.train import bc_pretrain_construction
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    tj = _traj_with_construction(dims, n_steps=6, n_cities=2, acted=True)   # carries construction_current
    torch.manual_seed(4); net = StructuredPolicyValueNet(dims, _TS, _VC, **RUNGS["small"])
    _, s0 = bc_pretrain_construction(net, [tj], dims, _TS, epochs=0, lr=1e-2)   # measure init acc (0 epochs)
    net2, s1 = bc_pretrain_construction(net, [tj], dims, _TS, epochs=40, lr=1e-2, micro_batch_steps=0)
    assert s1["n"] > 0, "no BC targets found"
    assert s1["bc_acc"] > 0.9, f"BC did not learn the heuristic picks (acc={s1['bc_acc']:.2f})"


def test_bc_pretrain_leaves_civ_heads_untouched():
    """BC's construction-only loss must NOT move the tech/policy/value heads (they're not in the loss path)."""
    from unciv_train.train import bc_pretrain_construction
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    tj = _traj_with_construction(dims, n_steps=6, n_cities=2, acted=True)
    torch.manual_seed(4); net = StructuredPolicyValueNet(dims, _TS, _VC, **RUNGS["small"])
    tech0 = torch.cat([p.detach().reshape(-1) for p in net.tech_head.parameters()])
    bc_pretrain_construction(net, [tj], dims, _TS, epochs=20, lr=1e-2, micro_batch_steps=0)
    tech1 = torch.cat([p.detach().reshape(-1) for p in net.tech_head.parameters()])
    assert (tech0 - tech1).abs().max().item() < 1e-7, "BC leaked gradient into the tech head"


def test_per_city_credit_trains_under_replay():
    """v7.3 per-city PPO ratio: with behavior_logp=True (off-policy replay), the per-city construction term
    uses its own importance ratio and must train FINITE without divergence — the guarantee that lets the
    experiment run at replay-window>1 (the strong ~40% baseline regime), not just on-policy rw1."""
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    tj = _traj_with_construction(dims, acted=True)
    torch.manual_seed(8); net0 = StructuredPolicyValueNet(dims, _TS, _VC, **RUNGS["small"])
    net, stats, _ = train_actor_critic_structured(
        [tj], dims, _TS, _VC, RUNGS["small"], epochs=3, lr=1e-3, seed=0, clip_eps=0.2,
        net=_copy.deepcopy(net0), construction=True, construction_credit_coef=0.5, behavior_logp=True)
    assert not stats.get("diverged", False), "per-city credit diverged under replay"
    assert torch.isfinite(_weights(net)).all(), "non-finite weights under replay per-city credit"
    assert (_weights(net0) - _weights(net)).abs().max().item() > 1e-6, "no training happened under replay"


def test_per_city_credit_offarm_still_bit_identical_noop():
    """v7.3 must not break the OFF no-op: construction=False with the per-city coef set is still a
    bit-identical no-op on the shared weights (the coef is inert when construction is off)."""
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    tj = _traj_with_construction(dims, acted=False)
    torch.manual_seed(7); net0 = StructuredPolicyValueNet(dims, _TS, _VC, **RUNGS["small"])
    net_off, _, _ = train_actor_critic_structured(
        [tj], dims, _TS, _VC, RUNGS["small"], epochs=2, lr=1e-3, seed=0, clip_eps=0.2,
        net=_copy.deepcopy(net0), construction=False, construction_credit_coef=0.5)
    net_ref, _, _ = train_actor_critic_structured(
        [tj], dims, _TS, _VC, RUNGS["small"], epochs=2, lr=1e-3, seed=0, clip_eps=0.2,
        net=_copy.deepcopy(net0), construction=False, construction_credit_coef=0.0)
    max_dw = (_weights(net_off) - _weights(net_ref)).abs().max().item()
    assert max_dw < 1e-6, f"per-city coef leaked into the OFF path: max|Δw|={max_dw}"
