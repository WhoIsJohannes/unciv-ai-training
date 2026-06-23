"""Export the trained net to `policy.onnx` with the FIXED contract names + provenance metadata.

The metadata (schema version, ruleset fingerprint, contract version, widths) is read back on the
JVM via `session.getMetadata().getCustomMetadata()` so EVAL refuses a contract-mismatched model.
"""
from __future__ import annotations

from pathlib import Path

import onnx
import torch

from . import contract
from .contract import Dims
from .model import PolicyNet


def export(
    net: PolicyNet,
    dims: Dims,
    out_path: str | Path,
    *,
    schema_version: int,
    ruleset_fingerprint: str,
    opset: int = 17,
) -> None:
    out_path = str(out_path)
    net.eval()
    dummy = torch.zeros(1, dims.input_w)
    dyn = {0: "batch"}
    torch.onnx.export(
        net, dummy, out_path,
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
