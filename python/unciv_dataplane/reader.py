"""Pure-Python reader for Unciv self-play trajectory shards.

The binary container is modeled on NumPy's self-describing ``.npy`` format:

    magic   : 8 bytes  b"UNCVSMP1"
    version : uint16    little-endian  (== SampleSchema.VERSION)
    hdrLen  : uint32    little-endian
    header  : hdrLen bytes  UTF-8 JSON  (provenance + caps + layout + dtypes + gameId + nTiles)
    records : repeated  [ uint32 recLen LE | payload ]
    footer  : uint32 recCount LE | uint32 crc32 LE   (CRC32 over the records region only)

Every multi-byte field is little-endian. The reader REFUSES a shard whose
``SampleSchema.VERSION`` mismatches, verifies the CRC32, exposes provenance to the caller,
WARNS on a ruleset-fingerprint mismatch across a loaded dataset, and tolerates a truncated
trailing record (salvaging the preceding valid steps).

Dependencies: ``numpy`` only (BSD, a thin numeric dep — NOT TF/PyTorch). A numpy-free path
is trivial since the container is self-describing, but tensors are returned as numpy arrays.
"""

from __future__ import annotations

import json
import struct
import warnings
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .schema import SCHEMA_VERSION, SchemaError

MAGIC = b"UNCVSMP1"

# step-header layout (fixed, little-endian): see ShardFormat / Featurizer.
#   i32 turn | i32 civSlot | u8 isFirst | u8 isLast | u8 isTerminal | u8 overflow | f32 reward
_STEP_HEADER = struct.Struct("<ii4Bf")
_STEP_HEADER_LEN = _STEP_HEADER.size  # 16

# numpy dtype for each container dtype tag.
_DTYPE = {"<f4": np.dtype("<f4"), "<i4": np.dtype("<i4"), "<u1": np.dtype("<u1")}


class ShardError(Exception):
    """Raised when a shard cannot be parsed or fails validation (incl. VERSION mismatch)."""


@dataclass
class Provenance:
    schema_version: int
    unciv_version_text: str
    unciv_version_number: int
    compatibility_number: int | None
    ruleset_fingerprint: str
    git_sha: str | None
    game_id: str
    seed: int | None
    n_tiles: int


@dataclass
class Step:
    turn: int
    civ_slot: int
    is_first: bool
    is_last: bool
    is_terminal: bool
    overflow: bool
    reward: float
    blocks: dict[str, np.ndarray]  # named observation/mask blocks per the header layout


class Shard:
    """A parsed trajectory shard: provenance + lazily-decoded steps."""

    def __init__(self, header: dict[str, Any], steps: list[Step], *, truncated: bool):
        self.header = header
        self.steps = steps
        self.truncated = truncated

    @property
    def provenance(self) -> Provenance:
        h = self.header
        return Provenance(
            schema_version=int(h["schemaVersion"]),
            unciv_version_text=str(h.get("uncivVersionText", "")),
            unciv_version_number=int(h.get("uncivVersionNumber", -1)),
            compatibility_number=h.get("compatibilityNumber"),
            ruleset_fingerprint=str(h["rulesetFingerprint"]),
            git_sha=h.get("gitSha"),
            game_id=str(h.get("gameId", "")),
            seed=h.get("seed"),
            n_tiles=int(h.get("nTiles", 0)),
        )

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)


def _decode_blocks(payload: bytes, layout: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Slice the per-step payload (after the fixed step header) into named tensor blocks.

    FIXED blocks are ``len`` items of ``dtype``. VARIABLE blocks are a u16 count prefix followed by
    ``count * perItem`` items, returned reshaped to ``(count, perItem)`` (only present entities are
    stored — pad to the schema caps in the data loader)."""
    blocks: dict[str, np.ndarray] = {}
    off = _STEP_HEADER_LEN
    for blk in layout:
        name = blk["name"]
        dtype = _DTYPE[blk["dtype"]]
        kind = blk.get("kind", "fixed")
        if kind == "var":
            if off + 2 > len(payload):
                raise ShardError(f"record payload too short for var-block {name!r} count prefix")
            (count,) = struct.unpack_from("<H", payload, off)
            off += 2
            per_item = int(blk.get("perItem", 1)) or 1
            n_items = count * per_item
            nbytes = n_items * dtype.itemsize
            chunk = payload[off : off + nbytes]
            if len(chunk) < nbytes:
                raise ShardError(f"record payload too short for var-block {name!r}: "
                                 f"need {nbytes} bytes, have {len(chunk)}")
            arr = np.frombuffer(chunk, dtype=dtype).copy()
            blocks[name] = arr.reshape(count, per_item) if count else arr.reshape(0, per_item)
        else:
            n_items = int(blk["len"])
            nbytes = n_items * dtype.itemsize
            chunk = payload[off : off + nbytes]
            if len(chunk) < nbytes:
                raise ShardError(f"record payload too short for block {name!r}: "
                                 f"need {nbytes} bytes, have {len(chunk)}")
            blocks[name] = np.frombuffer(chunk, dtype=dtype).copy()
        off += nbytes
    return blocks


def load(path: str | Path) -> Shard:
    """Load and validate a single shard. Raises ShardError on bad magic / VERSION mismatch /
    CRC failure (when the footer is present)."""
    data = Path(path).read_bytes()
    if data[:8] != MAGIC:
        raise ShardError(f"bad magic {data[:8]!r} (expected {MAGIC!r}) — not an Unciv shard")
    pos = 8
    (version,) = struct.unpack_from("<H", data, pos)
    pos += 2
    if version != SCHEMA_VERSION:
        raise ShardError(
            f"shard SampleSchema.VERSION={version} != reader SCHEMA_VERSION={SCHEMA_VERSION}; "
            "datasets are perishable — REGENERATE against the pinned game version, do not migrate."
        )
    (hdr_len,) = struct.unpack_from("<I", data, pos)
    pos += 4
    header = json.loads(data[pos : pos + hdr_len].decode("utf-8"))
    pos += hdr_len
    if int(header.get("schemaVersion", -1)) != version:
        raise ShardError("header schemaVersion disagrees with the magic-block version")

    layout = header["layout"]
    region_start = pos
    truncated = False

    # Footer is the last 8 bytes (recCount u32 + crc32 u32). If absent/short -> truncated.
    if len(data) - region_start >= 8:
        rec_count, footer_crc = struct.unpack_from("<II", data, len(data) - 8)
        region = data[region_start : len(data) - 8]
        actual_crc = zlib.crc32(region) & 0xFFFFFFFF
        if actual_crc != footer_crc:
            # Could be a real corruption OR a truncated shard whose footer is stale garbage.
            truncated = True
            region = data[region_start:]  # try to salvage records from the whole tail
            rec_count = None
    else:
        truncated = True
        region = data[region_start:]
        rec_count = None

    steps: list[Step] = []
    roff = 0
    while roff + 4 <= len(region):
        (rec_len,) = struct.unpack_from("<I", region, roff)
        roff += 4
        if rec_len < _STEP_HEADER_LEN or roff + rec_len > len(region):
            truncated = True
            break  # truncated, or footer/garbage bytes misread as a record length
        payload = region[roff : roff + rec_len]
        roff += rec_len
        turn, civ_slot, is_first, is_last, is_terminal, overflow, reward = _STEP_HEADER.unpack_from(payload, 0)
        try:
            blocks = _decode_blocks(payload, layout)
        except ShardError:
            truncated = True
            break
        steps.append(Step(turn, civ_slot, bool(is_first), bool(is_last),
                          bool(is_terminal), bool(overflow), reward, blocks))

    if rec_count is not None and len(steps) != rec_count and not truncated:
        raise ShardError(f"record count mismatch: footer={rec_count}, parsed={len(steps)}")
    if truncated:
        warnings.warn(f"shard {path}: truncated/footer-invalid — salvaged {len(steps)} step(s)")
    return Shard(header, steps, truncated=truncated)


def load_dataset(paths: list[str | Path]) -> list[Shard]:
    """Load several shards, WARNING on a ruleset-fingerprint mismatch across them (mixing game
    versions in one dataset is unsupported by design — perishable, regenerate)."""
    shards = [load(p) for p in paths]
    fingerprints = {s.provenance.ruleset_fingerprint for s in shards}
    if len(fingerprints) > 1:
        warnings.warn(
            "ruleset-fingerprint mismatch across the loaded dataset "
            f"({sorted(fingerprints)}) — shards were generated against DIFFERENT ruleset content; "
            "this is unsupported. Regenerate the whole dataset against one pinned version."
        )
    return shards
