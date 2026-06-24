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
# Contract v3 = structured encoder (embeddings + hex-GNN). Same multi-tensor base as v2 PLUS two
# NEW graph tensors (neighbor_index/neighbor_mask) that drive the GNN gather. Bumped in lockstep
# with Kotlin SampleSchema.OnnxContract.CONTRACT_VERSION_RICH / SampleSchema.VERSION (2→3).
CONTRACT_VERSION_STRUCTURED = 3

INPUT_NAME = "obs"                 # contract v1 single input
OUTPUT_TECH = "tech_logits"
OUTPUT_POLICY = "policy_logits"
MODELED_HEADS = ["tech", "policy"]

# Contract v2 named multi-tensor inputs. Token sets each pair with a "<name>_mask" presence mask.
INPUT_GLOBAL = "global"
INPUT_ACTING = "acting_civ"
# Token sets (name -> perItem width is runtime-derived from schema via token_specs_from_schema).
RICH_TOKEN_NAMES = ["spatial", "own_units", "opp_units", "own_cities", "opp_cities", "civ_tokens"]

# Contract v3 (structured) NEW graph tensors. They are NOT token sets (no "<name>_mask" loop entry)
# and are NOT folded into spatial — they index the spatial node axis (a different shape [B,N,6]) and
# are derived from coords+map-dims, not a per-channel value. They share spatial's "n_spatial"
# dynamic axis; the degree axis is the static hex degree. neighbor_index is int64 (ORT Gather index
# dtype); neighbor_mask is float32. Mirrored Kotlin↔Python (SampleSchema.OnnxContract).
INPUT_NEIGHBOR_INDEX = "neighbor_index"   # [batch, n_spatial, 6] int64
INPUT_NEIGHBOR_MASK = "neighbor_mask"     # [batch, n_spatial, 6] float32
NEIGHBOR_INPUT_NAMES = [INPUT_NEIGHBOR_INDEX, INPUT_NEIGHBOR_MASK]
NEIGHBOR_DEGREE = 6                        # hex degree-6 (== hexgraph.HEX_DEGREE / OFFSETS length)

# Global head map-dim slots: buildGlobal appends the 3 pre-resolved map-dim scalars right after the
# 5 fixed head scalars (turns, era, tileCount, knownMajors, aliveMajors) and BEFORE the demographics
# agg block, at head slots 5,6,7 — read positionally by the Python adjacency builder. (Kotlin C1
# also surfaces them as named schema fields; this positional offset is the lockstep with buildGlobal.)
GLOBAL_MAPDIM_OFFSET = 5
MAPDIM_SLOTS = ("eff_wrap_radius", "world_wrap", "shape")

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


def spatial_channels_from_schema(schema_path: str | Path) -> int:
    """Number of spatial (per-tile) channels — the v4 god-constant. FAIL-LOUD (FND-0007/0011): the
    SSOT is Kotlin `SampleSchema.SPATIAL_CHANNELS`; if the schema omits it we RAISE rather than
    silently reverting to a stale hardcoded width (which would mis-shape every shard)."""
    sch = _schema(schema_path)
    chans = sch.get("spatial_channels") or sch.get("spatialChannels")
    if not chans:
        raise ValueError(
            f"{schema_path}: schema omits 'spatial_channels'/'spatialChannels' — refusing silent "
            "fallback (spatial channel count is the v4 god-constant; SSOT is Kotlin SampleSchema)"
        )
    return len(chans)


def token_specs_from_schema(schema_path: str | Path) -> dict:
    """Per-token feature widths for the rich/structured variant, derived from the generated schema.

    spatial → number of spatial channels (per-tile width); entity blocks → their layout perItem.
    FAIL-LOUD (FND-0007/0011): if the schema omits spatial_channels or an entity perItem we RAISE
    instead of silently using a stale fallback width (which `features._pad_token_set`'s min(width)
    would then silently truncate — exactly the drift the council flagged).
    """
    sch = _schema(schema_path)
    layout = {b["name"]: b for b in sch.get("layout", [])}
    specs: dict[str, int] = {"spatial": spatial_channels_from_schema(schema_path)}
    for name in ("own_units", "opp_units", "own_cities", "opp_cities", "civ_tokens"):
        entry = layout.get(name)
        if not (entry and entry.get("perItem")):
            raise ValueError(
                f"{schema_path}: layout[{name!r}] missing 'perItem' — refusing silent fallback "
                "(entity token width is schema-driven; SSOT is Kotlin SampleSchema)"
            )
        specs[name] = int(entry["perItem"])
    return specs


def vocab_counts_from_schema(schema_path: str | Path) -> dict:
    """Categorical vocabulary cardinalities (terrain/resource/improvement/religion/era/building/
    unit/nation/promotion), emitted in schema.json's `vocabCounts` object by the Kotlin header.

    Embedding `num_embeddings` is derived from these (count + 1 sentinel row) — NEVER hardcoded.
    FAIL-LOUD (FND-0007/0011): raise if the schema omits the block (regenerate shards on contract v3).
    """
    sch = _schema(schema_path)
    vc = sch.get("vocabCounts") or sch.get("vocab_counts")
    if not vc:
        raise ValueError(
            f"{schema_path}: schema missing 'vocabCounts' — regenerate shards on contract v3 "
            "(embedding cardinalities must come from the schema, never hardcoded)"
        )
    return {str(k): int(v) for k, v in vc.items()}
