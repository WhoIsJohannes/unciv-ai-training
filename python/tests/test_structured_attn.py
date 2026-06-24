"""Phase-B attention gate for the v4 StructuredPolicyValueNet (D4 self-attn + entity↔node join +
D5 single-query cross-attn), exercised on the MEDIUM rung (attn_layers>0).

Covers:
- medium-rung forward: shapes + no NaN (attention path is live).
- fully-masked entity set + all-isolated nodes (and the all-entities-masked extreme) → no NaN
  (the masked-softmax NaN guard, FND-0025).
- export medium rung to ONNX opset 17, assert op types contain NO Scatter*/Attention/
  MultiHeadAttention (hand-rolled attention only), reload in onnxruntime, ORT≈torch atol 1e-4 with
  N differing between trace and run (proves the n_spatial / entity dynamic axes survive attention).

Phase A (small rung, GNN-only) is gated by test_structured_smoke.py and must stay green — this file
only adds the attention coverage and never touches the GNN-only path.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

import onnx  # noqa: E402

from unciv_train import contract  # noqa: E402
from unciv_train.contract import Dims  # noqa: E402
from unciv_train.export_onnx import export_rich  # noqa: E402
from unciv_train.model import (  # noqa: E402
    RUNGS,
    StructuredPolicyValueNet,
    _ENTITY_TILE_FIELD,
    _SPATIAL_FIELD_PLAN,
)

ATOL = 1e-4
SPATIAL_W = len(_SPATIAL_FIELD_PLAN)  # 13
RUNG = "medium"  # attn_layers>0 → Phase-B path


def _tiny_setup():
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    token_specs = {"spatial": SPATIAL_W, "own_units": 9, "opp_units": 9,
                   "own_cities": 17, "opp_cities": 17, "civ_tokens": 84}
    vocab_counts = {"terrain": 6, "resource": 5, "improvement": 4,
                    "religion": 3, "era": 4, "building": 7, "unit": 8,
                    "nation": 2, "promotion": 3}
    return dims, token_specs, vocab_counts


def _random_inputs(dims, token_specs, *, n_spatial=4, batch=1, seed=0):
    """Like the smoke test, but writes VALID in-range tile indices into the entity join columns
    (own/opp_units field 8, own/opp_cities field 12) so the entity↔node gather is exercised."""
    rng = np.random.default_rng(seed)
    inputs: dict[str, torch.Tensor] = {
        "global": torch.tensor(rng.standard_normal((batch, dims.global_w)).astype(np.float32)),
        "acting_civ": torch.tensor(rng.standard_normal((batch, dims.acting_w)).astype(np.float32)),
    }
    sp = rng.integers(0, 4, size=(batch, n_spatial, token_specs["spatial"])).astype(np.float32)
    sp[..., 7] = 255.0   # owner_slot self sentinel
    sp[..., 10] = 255.0  # unit_owner_slot self sentinel
    inputs["spatial"] = torch.tensor(sp)
    inputs["spatial_mask"] = torch.ones(batch, n_spatial)
    for name in ("own_units", "opp_units", "own_cities", "opp_cities", "civ_tokens"):
        m = int(rng.integers(1, 4))
        tok = rng.standard_normal((batch, m, token_specs[name])).astype(np.float32)
        fld = _ENTITY_TILE_FIELD.get(name)
        if fld is not None:
            # valid tile indices into [0, n_spatial) for the join gather
            tok[..., fld] = rng.integers(0, n_spatial, size=(batch, m))
        inputs[name] = torch.tensor(tok)
        inputs[name + "_mask"] = torch.ones(batch, m)
    nbr = rng.integers(0, n_spatial + 1, size=(batch, n_spatial, 6)).astype(np.int64)
    inputs["neighbor_index"] = torch.tensor(nbr)
    inputs["neighbor_mask"] = torch.tensor(
        rng.integers(0, 2, size=(batch, n_spatial, 6)).astype(np.float32))
    return inputs


def _net(seed=7):
    dims, token_specs, vocab_counts = _tiny_setup()
    torch.manual_seed(seed)
    net = StructuredPolicyValueNet(dims, token_specs, vocab_counts, **RUNGS[RUNG]).eval()
    return dims, token_specs, vocab_counts, net


def test_medium_rung_uses_attention():
    _, _, _, net = _net()
    assert net.attn_layers > 0, "medium rung must turn attention ON"
    assert net.use_attn is True


def test_medium_forward_shapes_and_no_nan():
    dims, token_specs, _, net = _net(seed=11)
    inputs = _random_inputs(dims, token_specs, n_spatial=5, seed=1)
    with torch.no_grad():
        tech, policy, value = net(inputs)
    assert tech.shape == (1, dims.tech_w)
    assert policy.shape == (1, dims.policy_w)
    assert value.shape == (1, 1)
    for t in (tech, policy, value):
        assert torch.isfinite(t).all(), "attention forward produced NaN/Inf"
    assert (value >= -1).all() and (value <= 1).all(), "tanh value must be in [-1,1]"


def test_fully_masked_entity_set_and_isolated_nodes_nan_safe():
    """A fully-masked entity SELF-attention set can be all −inf after masked_fill → NaN after
    softmax. The guard must force ZEROS. Also all-isolated GNN nodes (deg 0) must not NaN."""
    dims, token_specs, _, net = _net(seed=3)
    inputs = _random_inputs(dims, token_specs, n_spatial=4, seed=2)
    inputs["opp_units_mask"] = torch.zeros_like(inputs["opp_units_mask"])  # fully-masked self-attn set
    inputs["neighbor_mask"] = torch.zeros_like(inputs["neighbor_mask"])    # all-isolated nodes
    with torch.no_grad():
        out = net(inputs)
    for t in out:
        assert torch.isfinite(t).all(), "NaN guard failed (masked self-attn set / isolated nodes)"


def test_all_entity_sets_masked_nan_safe():
    """Every entity set fully masked: only the GNN nodes remain as cross-attn keys (always ≥1), and
    every per-entity self-attention set is fully empty → all must produce ZEROS, never NaN."""
    dims, token_specs, _, net = _net(seed=4)
    inputs = _random_inputs(dims, token_specs, n_spatial=4, seed=5)
    for name in ("own_units", "opp_units", "own_cities", "opp_cities", "civ_tokens"):
        inputs[name + "_mask"] = torch.zeros_like(inputs[name + "_mask"])
    with torch.no_grad():
        out = net(inputs)
    for t in out:
        assert torch.isfinite(t).all(), "NaN guard failed when ALL entity sets are masked"


def test_export_onnx_opset17_no_scatter_or_attention_and_ort_matches(tmp_path):
    dims, token_specs, _, net = _net(seed=7)

    # Trace with n=6, run with n=8 (DIFFERENT N) → proves the n_spatial / entity dynamic axes work
    # through the attention path.
    inputs = _random_inputs(dims, token_specs, n_spatial=6, seed=5)

    model = tmp_path / "structured_medium.onnx"
    sample = {k: (v.numpy() if hasattr(v, "numpy") else np.asarray(v)) for k, v in inputs.items()}
    export_rich(net, dims, token_specs, model, schema_version=3, ruleset_fingerprint="attntest",
                sample_inputs=sample, opset=17, neighbors=True,
                contract_version=contract.CONTRACT_VERSION_STRUCTURED)

    # Op-type audit: hand-rolled attention only — NO Scatter*/Attention/MultiHeadAttention.
    m = onnx.load(str(model))
    op_types = sorted({n.op_type for n in m.graph.node})
    forbidden = [o for o in op_types
                 if o.startswith("Scatter") or "Attention" in o or "MultiHeadAttention" in o]
    assert not forbidden, f"forbidden ops in exported graph: {forbidden} (all ops: {op_types})"
    # Softmax + LayerNormalization must be present (proves the attention is actually realized).
    assert "Softmax" in op_types, op_types
    assert "LayerNormalization" in op_types, op_types

    # ORT ≈ torch at a DIFFERENT N than the trace.
    run_inputs = _random_inputs(dims, token_specs, n_spatial=8, seed=99)
    with torch.no_grad():
        t_tech, t_policy, _ = net(run_inputs)
    sess = ort.InferenceSession(str(model))
    feed = {}
    for i in sess.get_inputs():
        v = run_inputs[i.name]
        arr = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
        feed[i.name] = arr.astype(np.int64) if "int64" in i.type else arr.astype(np.float32)
    ort_tech, ort_policy = sess.run([contract.OUTPUT_TECH, contract.OUTPUT_POLICY], feed)

    assert np.allclose(t_tech.numpy(), ort_tech, atol=ATOL), \
        f"tech logits drift > {ATOL}:\n{t_tech.numpy()}\nvs\n{ort_tech}"
    assert np.allclose(t_policy.numpy(), ort_policy, atol=ATOL), \
        f"policy logits drift > {ATOL}:\n{t_policy.numpy()}\nvs\n{ort_policy}"
