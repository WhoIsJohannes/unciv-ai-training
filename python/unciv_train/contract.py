"""ONNX I/O contract (Python side) — the lockstep mirror of Kotlin `SampleSchema.OnnxContract`.

Tensor NAMES are fixed here; tensor WIDTHS are runtime-derived from the generated `schema.json`
(never hardcoded — the GnK tech/policy counts come from the loaded ruleset). The cross-boundary
PARITY test guards that these names + the produced logits match the JVM session byte-for-byte
(within fp tolerance).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CONTRACT_VERSION = 1

INPUT_NAME = "obs"
OUTPUT_TECH = "tech_logits"
OUTPUT_POLICY = "policy_logits"
MODELED_HEADS = ["tech", "policy"]

# ONNX metadata_props keys (provenance — read back on the JVM via session.getMetadata()).
META_SCHEMA_VERSION = "schema_version"
META_RULESET_FINGERPRINT = "ruleset_fingerprint"
META_CONTRACT_VERSION = "contract_version"
META_INPUT_WIDTH = "input_width"
META_TECH_WIDTH = "tech_width"
META_POLICY_WIDTH = "policy_width"

# Learner identity (the pinned nation Constants.simulationCiv1). The trainer filters shards to this
# civ's steps via the header's per-game slot↔civId mapping.
LEARNER_CIV_ID = "SimulationCiv1"


@dataclass(frozen=True)
class Dims:
    """Net I/O widths, read from a generated schema.json layout."""
    global_w: int
    acting_w: int
    tech_w: int
    policy_w: int

    @property
    def input_w(self) -> int:
        return self.global_w + self.acting_w


def _schema(schema_path: str | Path) -> dict:
    return json.loads(Path(schema_path).read_text("utf-8"))


def dims_from_schema(schema_path: str | Path) -> Dims:
    layout = {b["name"]: b for b in _schema(schema_path)["layout"]}
    return Dims(
        global_w=int(layout["global"]["len"]),
        acting_w=int(layout["acting_civ"]["len"]),
        tech_w=int(layout["mask_tech"]["len"]),
        policy_w=int(layout["mask_policy"]["len"]),
    )


def fingerprint_from_schema(schema_path: str | Path) -> str:
    return str(_schema(schema_path)["rulesetFingerprint"])


def schema_version_from_schema(schema_path: str | Path) -> int:
    return int(_schema(schema_path)["schemaVersion"])
