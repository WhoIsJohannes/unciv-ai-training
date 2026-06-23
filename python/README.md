# Unciv Self-Play Data Plane — Python reader

A pure-Python (numpy-only) reader for the binary trajectory shards emitted by the Unciv
self-play data plane (`com.unciv.logic.simulation.dataplane`). **Data plane only — no NN, no
training.** This package just loads + validates shards and exposes their tensors + provenance.

## PIN-ONE-VERSION DISCIPLINE (read this first)

Training data here is **perishable**: it is scoped to ONE pinned Unciv checkout. We make version
and ruleset-content changes **detectable**, and rely on cheap **regeneration** — there is **no
migration tooling and there will never be**.

- Generate every shard of a training campaign against a **single pinned Unciv commit**.
- **Mixing versions in one dataset is unsupported by design.** `load_dataset()` WARNS on a
  ruleset-fingerprint mismatch across shards; the reader REFUSES a shard whose
  `SampleSchema.VERSION` differs from this package's `SCHEMA_VERSION`.
- If the game version or ruleset content changed: **regenerate**, do not patch old shards.

Two independent provenance signals are carried in every shard + the `schema.json` sidecar:
- **`SampleSchema.VERSION`** — the feature/mask *layout* version. A layout change bumps it
  (mirroring the engine's `CURRENT_COMPATIBILITY_NUMBER` discipline); old shards are refused.
- **`RulesetFingerprint`** — a content hash over the loaded GnK ruleset (all entity ids + the
  enum/vocab order). Catches *content* drift (a balance/entity change) even when VERSION is
  unchanged.

## Binary format (self-describing, modeled on NumPy `.npy`)

```
magic   : 8 bytes  b"UNCVSMP1"
version : uint16    little-endian   (== SampleSchema.VERSION)
hdrLen  : uint32    little-endian
header  : hdrLen bytes  UTF-8 JSON  { schemaVersion, uncivVersionText/Number, compatibilityNumber,
                                       gitSha?, rulesetFingerprint, gameId, seed, nTiles, caps,
                                       spatialChannels, layout:[{name,dtype,len}, ...] }
records : repeated  [ uint32 recLen LE | payload ]
            payload = stepHeader(16B) + blocks-in-`layout`-order
            stepHeader = i32 turn | i32 civSlot | u8 isFirst | u8 isLast | u8 isTerminal | u8 overflow | f32 reward
footer  : uint32 recCount LE | uint32 crc32 LE   (CRC32 over the records region ONLY)
```

All multi-byte fields are **little-endian** (`<f4`/`<i4`/`<u1` dtype tags). The checksum covers
only the records region — the header (with wall-clock / hostname / git-SHA) is excluded, so two
deterministic runs produce byte-identical *records* regardless of header timestamps.

## Usage

```python
from unciv_dataplane import load, load_dataset

shard = load("shards/shard-simulation-3-0.bin")
print(shard.provenance)            # Unciv version + ruleset fingerprint + gameId + nTiles
print(len(shard), "steps")
for step in shard:                 # Step: turn, civ_slot, is_first/last/terminal, overflow, reward, blocks
    obs = step.blocks["global"]    # numpy array; block names + dtypes come from the header layout
    mask = step.blocks["mask_tech"]

shards = load_dataset(["a.bin", "b.bin"])   # WARNS if the shards' ruleset fingerprints differ
```

`schema.json` (one per output dir) carries the dataset-level declaration; `load_schema()` reads
it and `.expect_compatible()` checks the version.

## Install / test

```bash
pip install numpy pytest
python -m pytest python/tests/test_reader.py
```

The reader tolerates a **truncated trailing record** (a crashed/interrupted gen run): it salvages
the preceding valid steps and sets `shard.truncated = True` with a warning.
