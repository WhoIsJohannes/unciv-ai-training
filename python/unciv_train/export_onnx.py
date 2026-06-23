"""Export the trained net to `policy.onnx` with the FIXED contract names + provenance metadata.

The metadata (schema version, ruleset fingerprint, contract version, widths) is read back on the
JVM via `session.getMetadata().getCustomMetadata()` so EVAL refuses a contract-mismatched model.
"""
from __future__ import annotations

from pathlib import Path

import onnx
import torch
import torch.nn as nn

from . import contract
from .contract import Dims
from .model import PolicyNet


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
    trace a dict input), reassembles the dict the net expects, drops the value head."""

    def __init__(self, net: nn.Module, names: list[str]):
        super().__init__()
        self.net = net
        self.names = names

    def forward(self, *tensors):
        inputs = {name: t for name, t in zip(self.names, tensors)}
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
) -> None:
    """Rich/contract-v2 export: MULTI-TENSOR named input (global, acting_civ, each token set +
    its presence mask) with dynamic axes; policy-only outputs (value dropped). Stamps
    CONTRACT_VERSION_RICH so the JVM bridge selects the multi-tensor path."""
    import numpy as np

    out_path = str(out_path)
    net.eval()

    # Ordered tensor names: global, acting_civ, then each token set + its presence mask.
    names = [contract.INPUT_GLOBAL, contract.INPUT_ACTING]
    dummy: dict[str, torch.Tensor] = {
        contract.INPUT_GLOBAL: torch.zeros(1, dims.global_w),
        contract.INPUT_ACTING: torch.zeros(1, dims.acting_w),
    }
    dyn = {contract.INPUT_GLOBAL: {0: "batch"}, contract.INPUT_ACTING: {0: "batch"},
           contract.OUTPUT_TECH: {0: "batch"}, contract.OUTPUT_POLICY: {0: "batch"}}
    for name, width in token_specs.items():
        n0 = 2
        dummy[name] = torch.zeros(1, n0, width)
        dummy[name + "_mask"] = torch.ones(1, n0)
        names += [name, name + "_mask"]
        dyn[name] = {0: "batch", 1: "n_" + name}
        dyn[name + "_mask"] = {0: "batch", 1: "n_" + name}
    if sample_inputs is not None:
        dummy.update({k: torch.as_tensor(np.asarray(v, dtype=np.float32)) for k, v in sample_inputs.items()})

    wrapped = _RichPolicyOnly(net, names)
    args = tuple(dummy[n] for n in names)  # POSITIONAL tensors in name order (ONNX-traceable)
    torch.onnx.export(
        wrapped, args, out_path,
        input_names=names,
        output_names=[contract.OUTPUT_TECH, contract.OUTPUT_POLICY],
        dynamic_axes=dyn,
        opset_version=opset,
    )
    model = onnx.load(out_path)
    onnx.helper.set_model_props(model, {
        contract.META_SCHEMA_VERSION: str(schema_version),
        contract.META_RULESET_FINGERPRINT: ruleset_fingerprint,
        contract.META_CONTRACT_VERSION: str(contract.CONTRACT_VERSION_RICH),
        contract.META_INPUT_WIDTH: str(dims.input_w),
        contract.META_TECH_WIDTH: str(dims.tech_w),
        contract.META_POLICY_WIDTH: str(dims.policy_w),
        contract.META_INPUT_NAMES: ",".join(names),
    })
    onnx.save(model, out_path)
