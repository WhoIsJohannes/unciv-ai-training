"""Export the trained net to `policy.onnx` with the FIXED contract names + provenance metadata.

The metadata (schema version, ruleset fingerprint, contract version, widths) is read back on the
JVM via `session.getMetadata().getCustomMetadata()` so EVAL refuses a contract-mismatched model.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import onnx
import torch
import torch.nn as nn

from . import contract
from .contract import Dims
from .model import PolicyNet


def _atomic_save(model, out_path: str) -> None:
    """Atomic ONNX write: serialize to a sibling .tmp then os.replace (mirrors ShardFormat's atomic
    finalize — a crash mid-write never leaves a half-written model the JVM might load)."""
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".onnx.tmp")
    os.close(fd)
    try:
        onnx.save(model, tmp)
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


class _PolicyOnly(nn.Module):
    """Export wrapper for the BLIND net (single obs tensor): drops the value head so the play
    ONNX has exactly two outputs (tech_logits, policy_logits)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, obs):
        tech, policy, _value = self.net(obs)
        return tech, policy


class _RichPolicyOnly(nn.Module):
    """Export wrapper for the RICH net: takes POSITIONAL tensors (in `names` order — ONNX can't
    trace a dict input), reassembles the dict the net expects, drops the value head. v7: when
    [with_construction] (structured nets), ALSO emits the per-city `construction_logits` output."""

    def __init__(self, net: nn.Module, names: list[str], with_construction: bool = False,
                 with_unit_intent: bool = False):
        super().__init__()
        self.net = net
        self.names = names
        self.with_construction = with_construction
        self.with_unit_intent = with_unit_intent

    def forward(self, *tensors):
        inputs = {name: t for name, t in zip(self.names, tensors)}
        if self.with_construction:
            # v8: with_construction=True now returns the 6-tuple (+unit_intent before value). Drop both
            # value heads; emit unit_intent as a 4th output when the net carries the v8 head.
            tech, policy, construction, _city_value, unit_intent, _value = self.net(inputs, with_construction=True)
            if self.with_unit_intent:
                return tech, policy, construction, unit_intent
            return tech, policy, construction   # city_value + value are train-only, dropped from the export
        tech, policy, _value = self.net(inputs)
        return tech, policy


def export(
    net: PolicyNet,
    dims: Dims,
    out_path: str | Path,
    *,
    schema_version: int,
    ruleset_fingerprint: str,
    opset: int = 17,
) -> None:
    """Blind/contract-v1 export: single `obs` input, policy-only outputs (value dropped). The JVM
    bridge contract is UNCHANGED — v1-reinforce and blind-critic models load on the v1 path."""
    out_path = str(out_path)
    net.eval()
    wrapped = _PolicyOnly(net)
    dummy = torch.zeros(1, dims.input_w)
    dyn = {0: "batch"}
    torch.onnx.export(
        wrapped, dummy, out_path,
        input_names=[contract.INPUT_NAME],
        output_names=[contract.OUTPUT_TECH, contract.OUTPUT_POLICY],
        dynamic_axes={contract.INPUT_NAME: dyn, contract.OUTPUT_TECH: dyn, contract.OUTPUT_POLICY: dyn},
        opset_version=opset,
    )
    model = onnx.load(out_path)
    onnx.helper.set_model_props(model, {
        contract.META_SCHEMA_VERSION: str(schema_version),
        contract.META_RULESET_FINGERPRINT: ruleset_fingerprint,
        contract.META_CONTRACT_VERSION: str(contract.CONTRACT_VERSION),
        contract.META_INPUT_WIDTH: str(dims.input_w),
        contract.META_TECH_WIDTH: str(dims.tech_w),
        contract.META_POLICY_WIDTH: str(dims.policy_w),
    })
    onnx.save(model, out_path)


def export_rich(
    net,
    dims: Dims,
    token_specs: dict[str, int],
    out_path: str | Path,
    *,
    schema_version: int,
    ruleset_fingerprint: str,
    sample_inputs: dict | None = None,
    opset: int = 17,
    neighbors: bool = False,
    contract_version: int | None = None,
) -> None:
    """Rich/contract-v2(+v3) export: MULTI-TENSOR named input (global, acting_civ, each token set +
    its presence mask) with dynamic axes; policy-only outputs (value dropped).

    Stamps CONTRACT_VERSION_RICH (=2) by default so the JVM bridge selects the multi-tensor path.
    When `neighbors=True` (v4 structured encoder), also appends the degree-6 graph tensors
    `neighbor_index` (int64) + `neighbor_mask` (f32) — built OUTSIDE the float32 sample coercion so
    the int64 index dtype survives — both sharing spatial's `n_spatial` dynamic axis (degree axis
    static 6), and the caller should pass `contract_version=CONTRACT_VERSION_STRUCTURED`. Atomic write.
    """
    import numpy as np

    out_path = str(out_path)
    net.eval()
    if contract_version is None:
        contract_version = contract.CONTRACT_VERSION_RICH

    # Ordered tensor names: global, acting_civ, then each token set + its presence mask. The two
    # neighbor tensors (when present) are appended immediately after the spatial pair so the
    # positional export order is unambiguous and documents the coupling to spatial's node axis.
    names = [contract.INPUT_GLOBAL, contract.INPUT_ACTING]
    dummy: dict[str, torch.Tensor] = {
        contract.INPUT_GLOBAL: torch.zeros(1, dims.global_w),
        contract.INPUT_ACTING: torch.zeros(1, dims.acting_w),
    }
    dyn = {contract.INPUT_GLOBAL: {0: "batch"}, contract.INPUT_ACTING: {0: "batch"},
           contract.OUTPUT_TECH: {0: "batch"}, contract.OUTPUT_POLICY: {0: "batch"}}
    n0 = 2
    for name, width in token_specs.items():
        dummy[name] = torch.zeros(1, n0, width)
        dummy[name + "_mask"] = torch.ones(1, n0)
        names += [name, name + "_mask"]
        dyn[name] = {0: "batch", 1: "n_" + name}
        dyn[name + "_mask"] = {0: "batch", 1: "n_" + name}
        if neighbors and name == "spatial":
            # Reuse spatial's dummy row count (n0) AND its dynamic label so the neighbor node axis is
            # byte-identically bound to spatial's "n_spatial" — values 0..n0-1 are valid Gather rows.
            n_sp = int(dummy["spatial"].shape[1])
            ni, nm = contract.INPUT_NEIGHBOR_INDEX, contract.INPUT_NEIGHBOR_MASK
            dummy[ni] = torch.zeros(1, n_sp, contract.NEIGHBOR_DEGREE, dtype=torch.int64)
            dummy[nm] = torch.ones(1, n_sp, contract.NEIGHBOR_DEGREE)
            names += [ni, nm]
            dyn[ni] = {0: "batch", 1: "n_spatial"}
            dyn[nm] = {0: "batch", 1: "n_spatial"}
    if sample_inputs is not None:
        # Coerce float32 for every override EXCEPT neighbor_index, which must stay int64 (ORT Gather
        # index dtype) — handle it OUTSIDE the float32 dict so it is never down-cast to float.
        for k, v in sample_inputs.items():
            if k == contract.INPUT_NEIGHBOR_INDEX:
                dummy[k] = torch.as_tensor(np.asarray(v)).long()
            else:
                dummy[k] = torch.as_tensor(np.asarray(v, dtype=np.float32))

    # v7: the structured net carries a per-city construction head → emit `construction_logits`
    # [batch, n_own_cities, constr_w] (the n_own_cities axis is BOUND to the own_cities input's
    # dynamic label, so the output city axis tracks the input city count). Rich-v2 nets lack the head.
    with_construction = neighbors and hasattr(net, "construction_head")
    # v8: the structured net also carries a per-unit intent head → emit `unit_intent_logits`
    # [batch, n_own_units, intent_w] (the n_own_units axis is BOUND to the own_units input's dynamic label).
    with_unit_intent = neighbors and hasattr(net, "unit_intent_head")
    output_names = [contract.OUTPUT_TECH, contract.OUTPUT_POLICY]
    if with_construction:
        output_names.append(contract.OUTPUT_CONSTRUCTION)
        dyn[contract.OUTPUT_CONSTRUCTION] = {0: "batch", 1: "n_own_cities"}
    if with_unit_intent:
        output_names.append(contract.OUTPUT_UNIT_INTENT)
        dyn[contract.OUTPUT_UNIT_INTENT] = {0: "batch", 1: "n_own_units"}

    wrapped = _RichPolicyOnly(net, names, with_construction, with_unit_intent)
    args = tuple(dummy[n] for n in names)  # POSITIONAL tensors in name order (ONNX-traceable)
    torch.onnx.export(
        wrapped, args, out_path,
        input_names=names,
        output_names=output_names,
        dynamic_axes=dyn,
        opset_version=opset,
    )
    props = {
        contract.META_SCHEMA_VERSION: str(schema_version),
        contract.META_RULESET_FINGERPRINT: ruleset_fingerprint,
        contract.META_CONTRACT_VERSION: str(contract_version),
        contract.META_INPUT_WIDTH: str(dims.input_w),
        contract.META_TECH_WIDTH: str(dims.tech_w),
        contract.META_POLICY_WIDTH: str(dims.policy_w),
        contract.META_INPUT_NAMES: ",".join(names),
    }
    if with_construction:
        props[contract.META_CONSTRUCTION_WIDTH] = str(int(net.constr_w))
    if with_unit_intent:
        props[contract.META_UNIT_INTENT_WIDTH] = str(int(net.intent_w))
    model = onnx.load(out_path)
    onnx.helper.set_model_props(model, props)
    _atomic_save(model, out_path)
