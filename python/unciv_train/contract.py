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

# Contract v1 = blind single-tensor input ("obs"); used by v1-reinforce + blind-critic (JVM bridge
# unchanged). Contract v2 = rich MULTI-TENSOR input; used by rich-critic. The JVM OnnxPolicy reads
# META_CONTRACT_VERSION and selects the single- vs multi-tensor build path, so both coexist.
CONTRACT_VERSION = 1
CONTRACT_VERSION_RICH = 2

INPUT_NAME = "obs"                 # contract v1 single input
OUTPUT_TECH = "tech_logits"
OUTPUT_POLICY = "policy_logits"
MODELED_HEADS = ["tech", "policy"]

# Contract v2 named multi-tensor inputs. Token sets each pair with a "<name>_mask" presence mask.
INPUT_GLOBAL = "global"
INPUT_ACTING = "acting_civ"
# Token sets (name -> perItem width is runtime-derived from schema via token_specs_from_schema).
RICH_TOKEN_NAMES = ["spatial", "own_units", "opp_units", "own_cities", "opp_cities", "civ_tokens"]

# ONNX metadata_props keys (provenance — read back on the JVM via session.getMetadata()).
META_SCHEMA_VERSION = "schema_version"
META_RULESET_FINGERPRINT = "ruleset_fingerprint"
META_CONTRACT_VERSION = "contract_version"
META_INPUT_WIDTH = "input_width"
META_TECH_WIDTH = "tech_width"
META_POLICY_WIDTH = "policy_width"
META_INPUT_NAMES = "input_names"   # comma-joined ordered tensor names (contract v2)

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


# Hardcoded fallbacks if a schema layout omits perItem (matches the v1 Featurizer widths).
_TOKEN_WIDTH_FALLBACK = {"spatial": 13, "own_units": 8, "opp_units": 8,
                         "own_cities": 16, "opp_cities": 16, "civ_tokens": 84}


def token_specs_from_schema(schema_path: str | Path) -> dict:
    """Per-token feature widths for the rich variant, derived from the generated schema.

    spatial → number of spatial channels (per-tile width); entity blocks → their layout perItem.
    Falls back to the known v1 Featurizer widths if the schema omits a field.
    """
    sch = _schema(schema_path)
    layout = {b["name"]: b for b in sch.get("layout", [])}
    specs: dict[str, int] = {}
    chans = sch.get("spatial_channels") or sch.get("spatialChannels")
    specs["spatial"] = len(chans) if chans else _TOKEN_WIDTH_FALLBACK["spatial"]
    for name in ("own_units", "opp_units", "own_cities", "opp_cities", "civ_tokens"):
        entry = layout.get(name)
        if entry and entry.get("perItem"):
            specs[name] = int(entry["perItem"])
        else:
            specs[name] = _TOKEN_WIDTH_FALLBACK[name]
    return specs
