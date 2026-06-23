"""Self-contained reader tests: build synthetic shards matching the binary contract and assert
parse / VERSION-refuse / CRC / truncation-tolerance / fingerprint-warn. A committed Kotlin-emitted
golden fixture (if present at tests/fixtures/golden.bin) is also cross-checked."""

import json
import struct
import warnings
import zlib
from pathlib import Path

import pytest

from unciv_dataplane import SCHEMA_VERSION, ShardError, load, load_dataset

MAGIC = b"UNCVSMP1"
_STEP = struct.Struct("<ii4Bf")  # turn, civSlot, isFirst,isLast,isTerminal,overflow, reward
FIXTURE = Path(__file__).parent / "fixtures" / "golden.bin"


def _build_shard(*, version=SCHEMA_VERSION, fingerprint="abc123", n_steps=3,
                 layout=None, good_footer=True) -> bytes:
    if layout is None:
        layout = [
            {"name": "global", "dtype": "<f4", "kind": "fixed", "perItem": 0, "len": 4},
            {"name": "mask_tech", "dtype": "<u1", "kind": "fixed", "perItem": 0, "len": 8},
            {"name": "civ_tokens", "dtype": "<f4", "kind": "var", "perItem": 3, "len": 0},
        ]
    header = {
        "schemaVersion": version,
        "uncivVersionText": "4.20.15",
        "uncivVersionNumber": 1229,
        "compatibilityNumber": 4,
        "rulesetFingerprint": fingerprint,
        "gitSha": None,
        "gameId": "fixed-game-id",
        "seed": 42,
        "nTiles": 91,
        "caps": {"maxMajorCivs": 16},
        "spatialChannels": ["visibility_state"],
        "layout": layout,
    }
    hdr = json.dumps(header).encode("utf-8")

    records = bytearray()
    for i in range(n_steps):
        payload = bytearray()
        payload += _STEP.pack(i, 0, 1 if i == 0 else 0, 1 if i == n_steps - 1 else 0,
                              1 if i == n_steps - 1 else 0, 0, 0.0)
        for blk in layout:
            if blk.get("kind", "fixed") == "var":
                rows = 2  # two present entities
                payload += struct.pack("<H", rows)
                n = rows * blk["perItem"]
                payload += struct.pack(f"<{n}f", *([0.0] * n)) if blk["dtype"] == "<f4" else bytes(n)
            else:
                n = blk["len"]
                payload += struct.pack(f"<{n}f", *([0.0] * n)) if blk["dtype"] == "<f4" else bytes(n)
        records += struct.pack("<I", len(payload)) + payload

    out = bytearray()
    out += MAGIC
    out += struct.pack("<H", version)
    out += struct.pack("<I", len(hdr))
    out += hdr
    out += records
    crc = zlib.crc32(bytes(records)) & 0xFFFFFFFF
    if not good_footer:
        crc ^= 0xFFFFFFFF  # corrupt the footer CRC
    out += struct.pack("<II", n_steps, crc)
    return bytes(out)


def test_load_valid_shard(tmp_path):
    p = tmp_path / "s.bin"
    p.write_bytes(_build_shard(n_steps=3))
    shard = load(p)
    assert len(shard) == 3
    assert not shard.truncated
    prov = shard.provenance
    assert prov.schema_version == SCHEMA_VERSION
    assert prov.unciv_version_text == "4.20.15"
    assert prov.ruleset_fingerprint == "abc123"
    assert prov.game_id == "fixed-game-id"
    s0 = shard.steps[0]
    assert s0.is_first and not s0.is_last
    assert set(s0.blocks) == {"global", "mask_tech", "civ_tokens"}
    assert s0.blocks["global"].shape == (4,)
    assert s0.blocks["mask_tech"].shape == (8,)
    assert s0.blocks["civ_tokens"].shape == (2, 3)  # 2 present entities, perItem=3


def test_refuses_version_mismatch(tmp_path):
    p = tmp_path / "old.bin"
    p.write_bytes(_build_shard(version=SCHEMA_VERSION + 1))
    with pytest.raises(ShardError, match="VERSION"):
        load(p)


def test_refuses_bad_magic(tmp_path):
    p = tmp_path / "bad.bin"
    p.write_bytes(b"NOTASHARD" + bytes(50))
    with pytest.raises(ShardError, match="magic"):
        load(p)


def test_truncation_tolerated(tmp_path):
    # Drop the footer + part of the last record → reader salvages preceding steps + warns.
    full = _build_shard(n_steps=4)
    p = tmp_path / "trunc.bin"
    p.write_bytes(full[:-30])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        shard = load(p)
    assert shard.truncated
    assert len(shard) >= 1  # at least the early steps survive
    assert any("truncated" in str(x.message) for x in w)


def test_corrupt_footer_crc_salvages(tmp_path):
    p = tmp_path / "corrupt.bin"
    p.write_bytes(_build_shard(n_steps=3, good_footer=False))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        shard = load(p)
    assert shard.truncated  # CRC mismatch flips to salvage mode


def test_dataset_fingerprint_mismatch_warns(tmp_path):
    a = tmp_path / "a.bin"; a.write_bytes(_build_shard(fingerprint="aaa"))
    b = tmp_path / "b.bin"; b.write_bytes(_build_shard(fingerprint="bbb"))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_dataset([a, b])
    assert any("fingerprint mismatch" in str(x.message) for x in w)


@pytest.mark.skipif(not FIXTURE.exists(), reason="golden fixture not generated yet")
def test_golden_fixture_from_kotlin():
    shard = load(FIXTURE)
    assert shard.provenance.schema_version == SCHEMA_VERSION
    assert len(shard) > 0
    assert shard.provenance.ruleset_fingerprint  # non-empty
