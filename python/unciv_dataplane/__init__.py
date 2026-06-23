"""Pure-Python reader for Unciv self-play trajectory shards (data plane only — no NN/training).

Datasets are PERISHABLE: scoped to one pinned game version. The reader REFUSES a shard whose
``SampleSchema.VERSION`` mismatches and WARNS on a ruleset-fingerprint mismatch within a dataset.
Regenerate against the pinned checkout — never migrate.
"""

from .reader import (
    MAGIC,
    Provenance,
    Shard,
    ShardError,
    Step,
    load,
    load_dataset,
)
from .schema import SCHEMA_VERSION, Schema, SchemaError, load_schema

__all__ = [
    "MAGIC",
    "Provenance",
    "Shard",
    "ShardError",
    "Step",
    "load",
    "load_dataset",
    "SCHEMA_VERSION",
    "Schema",
    "SchemaError",
    "load_schema",
]
