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
SCHEMA_VERSION = 1


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
