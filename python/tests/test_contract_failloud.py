"""Contract fail-loud + version tests (FND-0007/0011/0022 + the plural→singular seam).

Schema-driven widths/cardinalities MUST come from the schema; a missing field RAISES rather than
silently using a stale fallback (which would drift the Kotlin↔Python contract). Fast, no JVM.
"""
from __future__ import annotations

import json

import pytest

from unciv_train import contract


def _write(tmp_path, obj) -> str:
    p = tmp_path / "schema.json"
    p.write_text(json.dumps(obj))
    return str(p)


def test_contract_versions_distinct():
    assert contract.CONTRACT_VERSION_RICH == 2
    assert contract.CONTRACT_VERSION_STRUCTURED == 3
    assert contract.NEIGHBOR_INPUT_NAMES == ["neighbor_index", "neighbor_mask"]


def test_vocab_counts_missing_raises(tmp_path):
    path = _write(tmp_path, {"schemaVersion": 3, "spatialChannels": ["a", "b"]})  # no vocabCounts
    with pytest.raises(ValueError, match="vocabCounts"):
        contract.vocab_counts_from_schema(path)


def test_vocab_counts_plural_keys_map_to_singular(tmp_path):
    # The Kotlin header emits PLURAL keys; the model embeds off SINGULAR names. The reader bridges.
    path = _write(tmp_path, {"vocabCounts": {
        "terrains": 6, "resources": 5, "improvements": 4, "religions": 3, "eras": 4,
        "buildings": 7, "units": 8, "promotions": 3, "nations": 2,
    }})
    vc = contract.vocab_counts_from_schema(path)
    for singular, expected in [("terrain", 6), ("resource", 5), ("improvement", 4),
                               ("religion", 3), ("era", 4), ("building", 7), ("unit", 8),
                               ("promotion", 3), ("nation", 2)]:
        assert vc[singular] == expected, f"{singular} not bridged from plural key"


def test_token_specs_failloud_missing_perItem(tmp_path):
    # spatialChannels present but an entity perItem missing → RAISE (no silent stale-width fallback).
    path = _write(tmp_path, {
        "spatialChannels": ["c"] * 13,
        "layout": [{"name": "own_units", "perItem": 9}],  # opp_units/... missing
    })
    with pytest.raises(ValueError, match="perItem"):
        contract.token_specs_from_schema(path)


def test_token_specs_failloud_missing_spatialchannels(tmp_path):
    path = _write(tmp_path, {"layout": [
        {"name": n, "perItem": w} for n, w in
        [("own_units", 9), ("opp_units", 9), ("own_cities", 17), ("opp_cities", 17), ("civ_tokens", 84)]
    ]})  # no spatialChannels
    with pytest.raises((ValueError, KeyError)):
        contract.token_specs_from_schema(path)
