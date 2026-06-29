"""``schema.json`` loader + the SampleSchema VERSION mirror.

``SCHEMA_VERSION`` MUST equal Kotlin ``SampleSchema.VERSION``. Any feature/mask LAYOUT change
bumps BOTH (the reader refuses a shard whose embedded version mismatches). Layout-affecting
*ruleset content* changes are caught separately by the per-shard ``rulesetFingerprint``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Mirror of com.unciv.logic.simulation.dataplane.SampleSchema.VERSION — keep in lockstep.
# v5 (v7 per-city construction head): adds two per-step VARIABLE f32 blocks aligned to own_cities —
# `construction_action` (chosen 0-indexed construction-mask idx; −1 = no decision) and
# `construction_logp` (its behavior log-prob; 0 where no decision). A v4 shard lacks the blocks ⇒ not
# layout-compatible ⇒ reader refuses (v4/v5 replay never mix). The descriptor-generic reader decodes
# the new VARIABLE blocks with no code change.
# v4 (v6 off-policy replay): adds the per-step `behavior_logp` block (per-head behavior-policy log π_b
# recorded at sampling time). A v3 shard lacks the block ⇒ not layout-compatible ⇒ reader refuses.
# v3 (v4 structured encoder): adds the per-tile spatial_coords (f32 x,y) block, map dims in global,
# per-entity tile-index, and the construction-namespace fix. v2/v1 shards are not layout-compatible.
SCHEMA_VERSION = 5


class SchemaError(Exception):
    pass


@dataclass
class Schema:
    version: int
    unciv_version_text: str
    unciv_version_number: int
    ruleset_fingerprint: str
    caps: dict[str, Any]
    layout: list[dict[str, Any]]
    spatial_channels: list[str]
    raw: dict[str, Any]

    def expect_compatible(self) -> None:
        """Raise unless this schema's VERSION matches the reader's. Provenance lives in the
        shards themselves; the sidecar is the dataset-level declaration."""
        if self.version != SCHEMA_VERSION:
            raise SchemaError(
                f"schema.json version {self.version} != reader SCHEMA_VERSION {SCHEMA_VERSION}; "
                "regenerate the dataset against the pinned game version."
            )


def load_schema(path: str | Path) -> Schema:
    raw = json.loads(Path(path).read_text("utf-8"))
    return Schema(
        version=int(raw["schemaVersion"]),
        unciv_version_text=str(raw.get("uncivVersionText", "")),
        unciv_version_number=int(raw.get("uncivVersionNumber", -1)),
        ruleset_fingerprint=str(raw["rulesetFingerprint"]),
        caps=dict(raw.get("caps", {})),
        layout=list(raw["layout"]),
        spatial_channels=list(raw.get("spatialChannels", [])),
        raw=raw,
    )
