"""v5 continual-training correctness:
  (1) warm-from-memory == warm-from-disk — carrying (net, opt) in-process is identical to checkpoint →
      reload → continue (the --resume path), so an interrupted run resumes bit-for-bit.
  (2) AC6 parity — the warm net a round starts from IS the net that was exported (and that generated
      the round's data), so warm-start stays on-policy.

RED until v5 adds net=/optimizer= warm params + the 3-tuple (net, stats, optimizer) return to the
trainers, and the opt sidecar save/load. Today these kwargs/return shape don't exist → fails.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from unciv_train.contract import Dims  # noqa: E402
from unciv_train.dataset import TrainTrajectory  # noqa: E402
from unciv_train.model import RUNGS, StructuredPolicyValueNet, _SPATIAL_FIELD_PLAN  # noqa: E402
from unciv_train.train import train_actor_critic_structured  # noqa: E402

_TS = {"spatial": len(_SPATIAL_FIELD_PLAN), "own_units": 9, "opp_units": 9,
       "own_cities": 17, "opp_cities": 17, "civ_tokens": 84}
_VOCAB = {"terrain": 6, "resource": 5, "improvement": 4, "religion": 3, "era": 4,
          "building": 7, "unit": 8, "nation": 2, "promotion": 3}


def _traj(dims, seed):
    rng = np.random.default_rng(seed)
    coords = np.array([(0, 0), (1, 1), (0, 1), (-1, 0)], np.float32)
    n = coords.shape[0]
    g = np.zeros(dims.global_w, np.float32); g[7] = 1.0
    step = {"global": g, "acting_civ": np.zeros(dims.acting_w, np.float32),
            "spatial": rng.integers(0, 4, size=(n, _TS["spatial"])).astype(np.float32),
            "spatial_coords": coords,
            "own_units": rng.standard_normal((2, 9)).astype(np.float32),
            "opp_units": np.zeros((0, 9), np.float32),
            "own_cities": rng.standard_normal((1, 17)).astype(np.float32),
            "opp_cities": np.zeros((0, 17), np.float32),
            "civ_tokens": rng.standard_normal((2, 84)).astype(np.float32)}
    return TrainTrajectory(
        np.zeros((1, dims.input_w), np.float32), np.array([0]), np.array([0]),
        np.ones((1, dims.tech_w), np.float32), np.ones((1, dims.policy_w), np.float32),
        np.array([1.0], np.float32), [step])


def _flat(net):
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()])


def test_warm_from_memory_equals_warm_from_disk(tmp_path):
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    r0_data = [_traj(dims, 0)]
    r1_data = [_traj(dims, 1)]                                # round 1's (different) on-policy batch

    # Round 0 fresh → (net0, opt0). seed=0.
    net0, _s0, opt0 = train_actor_critic_structured(
        r0_data, dims, _TS, _VOCAB, RUNGS["small"], epochs=2, lr=1e-3, seed=0, clip_eps=0.2)

    # Persist round-0 state (this is what run_loop saves as ckpt_round_0.pt + opt_round_0.pt).
    ckpt = tmp_path / "ckpt_round_0.pt"; optf = tmp_path / "opt_round_0.pt"
    torch.save(net0.state_dict(), ckpt); torch.save(opt0.state_dict(), optf)

    # (a) Continue round 1 WARM IN-MEMORY (reuse the live net+opt).
    net_mem = copy.deepcopy(net0)
    opt_mem = torch.optim.Adam(net_mem.parameters(), lr=1e-3); opt_mem.load_state_dict(opt0.state_dict())
    net_a, _sa, _oa = train_actor_critic_structured(
        r1_data, dims, _TS, _VOCAB, RUNGS["small"], epochs=2, lr=1e-3, seed=1, clip_eps=0.2,
        net=net_mem, optimizer=opt_mem)

    # (b) Continue round 1 WARM FROM DISK (the --resume path).
    net_disk = StructuredPolicyValueNet(dims, _TS, _VOCAB, **RUNGS["small"])
    net_disk.load_state_dict(torch.load(ckpt, weights_only=True))
    opt_disk = torch.optim.Adam(net_disk.parameters(), lr=1e-3)
    opt_disk.load_state_dict(torch.load(optf, weights_only=True))
    net_b, _sb, _ob = train_actor_critic_structured(
        r1_data, dims, _TS, _VOCAB, RUNGS["small"], epochs=2, lr=1e-3, seed=1, clip_eps=0.2,
        net=net_disk, optimizer=opt_disk)

    max_dw = (_flat(net_a) - _flat(net_b)).abs().max().item()
    assert max_dw < 1e-6, f"warm-from-memory != warm-from-disk: max|Δw|={max_dw}"


def test_warm_round_does_not_reinit_weights():
    """A warm round must NOT re-seed/re-init: passing net= reuses those exact weights as the start
    point (the optimizer keeps stepping from them), it does not reconstruct from manual_seed(seed)."""
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    net0, _s0, opt0 = train_actor_critic_structured(
        [_traj(dims, 0)], dims, _TS, _VOCAB, RUNGS["small"], epochs=1, lr=1e-3, seed=0, clip_eps=0.2)
    start = _flat(net0).clone()
    # Warm round with a tiny LR-equivalent: pass the SAME net; its starting weights are `start`, not a
    # fresh manual_seed(7) init. After training they move from `start`, so they must stay closer to
    # `start` than a fresh seed=7 net would be.
    net_warm, _s, _o = train_actor_critic_structured(
        [_traj(dims, 1)], dims, _TS, _VOCAB, RUNGS["small"], epochs=1, lr=1e-4, seed=7, clip_eps=0.2,
        net=net0, optimizer=opt0)
    fresh = StructuredPolicyValueNet(dims, _TS, _VOCAB, **RUNGS["small"])
    torch.manual_seed(7); fresh = StructuredPolicyValueNet(dims, _TS, _VOCAB, **RUNGS["small"])
    d_warm = (_flat(net_warm) - start).abs().mean().item()
    d_fresh = (_flat(fresh) - start).abs().mean().item()
    assert d_warm < d_fresh, "warm round appears to have re-initialized weights instead of reusing net"


def test_ac6_warm_net_matches_exported_onnx(tmp_path):
    """AC6: the warm net a round starts from produces logits matching the ONNX exported from it — the
    on-policy guarantee (warm net == gen net). Compares the warm net's torch tech/policy logits to the
    exported ONNX logits on the SAME inputs (atol 1e-4). Skips cleanly if onnxruntime is unavailable."""
    ort = pytest.importorskip("onnxruntime")
    import numpy as np
    from unciv_train.export_onnx import export_rich
    from unciv_train.features import build_rich_batch
    from unciv_train import contract as C

    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    traj = _traj(dims, 0)
    net, _s, _o = train_actor_critic_structured(
        [traj], dims, _TS, _VOCAB, RUNGS["small"], epochs=1, lr=1e-3, seed=0, clip_eps=0.2)
    onnx_path = tmp_path / "policy_round_0.onnx"
    export_rich(net, dims, _TS, onnx_path, schema_version=C.CONTRACT_VERSION_STRUCTURED,
                ruleset_fingerprint="testfp", neighbors=True,
                contract_version=C.CONTRACT_VERSION_STRUCTURED)
    assert onnx_path.is_file()

    # The warm net (torch) vs the exported ONNX must agree on the policy-relevant heads (tech+policy);
    # the value head is train-only and dropped at export, so it is NOT part of the parity surface.
    sess = ort.InferenceSession(str(onnx_path))
    out_names = [o.name for o in sess.get_outputs()]
    # v7/v8: the STRUCTURED export emits tech + policy + the per-city construction head + the per-unit intent
    # head; both value heads are train-only and dropped.
    assert set(out_names) == {C.OUTPUT_TECH, C.OUTPUT_POLICY, C.OUTPUT_CONSTRUCTION, C.OUTPUT_UNIT_INTENT}, \
        f"structured export must expose tech+policy+construction+unit_intent (values dropped); got {out_names}"

    inputs = build_rich_batch([traj], dims, _TS)            # the SAME inputs for both forwards
    net.eval()
    with torch.no_grad():
        t_tech, t_policy, _t_val = net(inputs)
    feed = {i.name: inputs[i.name].numpy() for i in sess.get_inputs()}
    o_tech, o_policy = sess.run([C.OUTPUT_TECH, C.OUTPUT_POLICY], feed)
    assert np.allclose(t_tech.numpy(), o_tech, atol=1e-4), "AC6: warm net tech logits != exported ONNX"
    assert np.allclose(t_policy.numpy(), o_policy, atol=1e-4), "AC6: warm net policy logits != exported ONNX"
