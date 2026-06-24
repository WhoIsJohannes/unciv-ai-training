"""Phase-A smoke test for the v4 StructuredPolicyValueNet (embeddings + hex-GNN, attn off).

Gates the load-bearing export risk EARLY (FND-0008/0023): build a tiny structured net on the SMALL
rung, run a forward pass (shape + NaN asserts), export to ONNX opset 17 with the two NEW int64/f32
neighbor graph tensors, reload in onnxruntime, and assert ORT logits ≈ torch logits (atol 1e-4).

All ops must be opset-17 core (Gather/Mul/ReduceSum/MatMul/LayerNorm/Softmax/tanh) — no scatter, no
nn.MultiheadAttention/SDPA. The export round-trip proves the gather-GNN graph is export-safe.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from unciv_train import contract  # noqa: E402
from unciv_train.contract import Dims  # noqa: E402
from unciv_train.export_onnx import export_rich  # noqa: E402
from unciv_train.model import RUNGS, StructuredPolicyValueNet, _SPATIAL_FIELD_PLAN  # noqa: E402

ATOL = 1e-4
SPATIAL_W = len(_SPATIAL_FIELD_PLAN)  # 13


def _tiny_setup():
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    token_specs = {"spatial": SPATIAL_W, "own_units": 9, "opp_units": 9,
                   "own_cities": 17, "opp_cities": 17, "civ_tokens": 84}
    # Small synthetic vocab cardinalities (counts; +1 sentinel applied inside the model).
    vocab_counts = {"terrain": 6, "resource": 5, "improvement": 4,
                    "religion": 3, "era": 4, "building": 7, "unit": 8,
                    "nation": 2, "promotion": 3}
    return dims, token_specs, vocab_counts


def _random_inputs(dims, token_specs, *, n_spatial=4, batch=1, seed=0):
    rng = np.random.default_rng(seed)
    inputs: dict[str, torch.Tensor] = {
        "global": torch.tensor(rng.standard_normal((batch, dims.global_w)).astype(np.float32)),
        "acting_civ": torch.tensor(rng.standard_normal((batch, dims.acting_w)).astype(np.float32)),
    }
    # Spatial: numeric channels random-ish; categorical channels small non-negative ints incl. the
    # 255 self-slot sentinel to exercise the reindex path.
    sp = rng.integers(0, 4, size=(batch, n_spatial, token_specs["spatial"])).astype(np.float32)
    sp[..., 7] = 255.0   # owner_slot self sentinel
    sp[..., 10] = 255.0  # unit_owner_slot self sentinel
    inputs["spatial"] = torch.tensor(sp)
    inputs["spatial_mask"] = torch.ones(batch, n_spatial)
    for name in ("own_units", "opp_units", "own_cities", "opp_cities", "civ_tokens"):
        m = int(rng.integers(1, 4))
        inputs[name] = torch.tensor(rng.standard_normal((batch, m, token_specs[name])).astype(np.float32))
        inputs[name + "_mask"] = torch.ones(batch, m)
    # Degree-6 neighbor graph: indices in [0, n_spatial] (n_spatial == the model's zero pad row).
    nbr = rng.integers(0, n_spatial + 1, size=(batch, n_spatial, 6)).astype(np.int64)
    inputs["neighbor_index"] = torch.tensor(nbr)
    inputs["neighbor_mask"] = torch.tensor(
        rng.integers(0, 2, size=(batch, n_spatial, 6)).astype(np.float32))
    return inputs


def test_forward_shapes_and_no_nan():
    dims, token_specs, vocab_counts = _tiny_setup()
    torch.manual_seed(11)
    net = StructuredPolicyValueNet(dims, token_specs, vocab_counts, **RUNGS["small"])
    assert net.attn_layers == 0, "Phase A must be GNN-only (attn_layers=0)"
    inputs = _random_inputs(dims, token_specs, n_spatial=5)
    net.eval()
    with torch.no_grad():
        tech, policy, value = net(inputs)
    assert tech.shape == (1, dims.tech_w)
    assert policy.shape == (1, dims.policy_w)
    assert value.shape == (1, 1)
    for t in (tech, policy, value):
        assert torch.isfinite(t).all(), "structured forward produced NaN/Inf"
    assert (value >= -1).all() and (value <= 1).all(), "tanh value must be in [-1,1]"


def test_all_padding_entity_set_is_nan_safe():
    """A fully-masked entity set must yield zeros (mirror masked_pool), never NaN."""
    dims, token_specs, vocab_counts = _tiny_setup()
    torch.manual_seed(3)
    net = StructuredPolicyValueNet(dims, token_specs, vocab_counts, **RUNGS["small"]).eval()
    inputs = _random_inputs(dims, token_specs, n_spatial=4)
    inputs["opp_units_mask"] = torch.zeros_like(inputs["opp_units_mask"])  # all padding
    inputs["neighbor_mask"] = torch.zeros_like(inputs["neighbor_mask"])    # all isolated nodes
    with torch.no_grad():
        tech, policy, value = net(inputs)
    for t in (tech, policy, value):
        assert torch.isfinite(t).all(), "NaN guard failed on fully-masked set / isolated nodes"


def test_export_onnx_opset17_and_ort_matches_torch(tmp_path):
    dims, token_specs, vocab_counts = _tiny_setup()
    torch.manual_seed(7)
    net = StructuredPolicyValueNet(dims, token_specs, vocab_counts, **RUNGS["small"]).eval()

    # Trace/export dummy uses n0=2 spatial rows; the run-time feed below uses a DIFFERENT N to prove
    # the n_spatial dynamic axis works.
    inputs = _random_inputs(dims, token_specs, n_spatial=6, seed=5)
    with torch.no_grad():
        t_tech, t_policy, _ = net(inputs)

    model = tmp_path / "structured_small.onnx"
    sample = {k: (v.numpy() if hasattr(v, "numpy") else np.asarray(v)) for k, v in inputs.items()}
    export_rich(net, dims, token_specs, model, schema_version=3, ruleset_fingerprint="smoketest",
                sample_inputs=sample, opset=17, neighbors=True,
                contract_version=contract.CONTRACT_VERSION_STRUCTURED)

    sess = ort.InferenceSession(str(model))
    in_names = [i.name for i in sess.get_inputs()]
    # The two new graph tensors are present, neighbor_index is int64.
    assert "neighbor_index" in in_names and "neighbor_mask" in in_names
    ni_type = next(i.type for i in sess.get_inputs() if i.name == "neighbor_index")
    assert "int64" in ni_type, f"neighbor_index must export as int64, got {ni_type}"

    feed = {}
    for i in sess.get_inputs():
        v = inputs[i.name]
        arr = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
        feed[i.name] = arr.astype(np.int64) if "int64" in i.type else arr.astype(np.float32)
    ort_tech, ort_policy = sess.run([contract.OUTPUT_TECH, contract.OUTPUT_POLICY], feed)

    assert np.allclose(t_tech.numpy(), ort_tech, atol=ATOL), \
        f"tech logits drift > {ATOL}:\n{t_tech.numpy()}\nvs\n{ort_tech}"
    assert np.allclose(t_policy.numpy(), ort_policy, atol=ATOL), \
        f"policy logits drift > {ATOL}:\n{t_policy.numpy()}\nvs\n{ort_policy}"


def test_export_input_names_order(tmp_path):
    """Document/freeze the ONNX input order: global, acting_civ, spatial(+mask), neighbor_index,
    neighbor_mask, then the remaining entity sets (+masks)."""
    dims, token_specs, vocab_counts = _tiny_setup()
    torch.manual_seed(1)
    net = StructuredPolicyValueNet(dims, token_specs, vocab_counts, **RUNGS["small"]).eval()
    model = tmp_path / "structured_names.onnx"
    export_rich(net, dims, token_specs, model, schema_version=3, ruleset_fingerprint="smoketest",
                neighbors=True, contract_version=contract.CONTRACT_VERSION_STRUCTURED)
    sess = ort.InferenceSession(str(model))
    in_names = [i.name for i in sess.get_inputs()]
    head = ["global", "acting_civ", "spatial", "spatial_mask", "neighbor_index", "neighbor_mask"]
    assert in_names[:len(head)] == head, in_names
