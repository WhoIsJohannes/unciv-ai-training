# Test Spec — Self-Play Data Plane (test_mode = integration)

Framework: JUnit + `@RunWith(GdxTestRunner::class)` + `com.unciv.testing.TestGame` (headless GnK).
Location: `tests/src/com/unciv/logic/simulation/dataplane/`. Run: `JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home ./gradlew tests:test --tests 'com.unciv.logic.simulation.dataplane.*'`.
Python: `python -m pytest python/tests/test_reader.py`.

Each test maps to an Acceptance Criterion (AC#). RED-first: these reference the planned `com.unciv.logic.simulation.dataplane.*` API which does not exist yet ⇒ the tests module fails to compile = RED until Phase 3 builds the API.

## Functional
- **AC1 RunToCompletion** (`RunToCompletionTests`): `ScenarioGenerator` + `Simulation` with `SampleConfig(enabled=true)` + `RandomPolicy` drives a small (Tiny map, 3 civs) game to completion; a shard file is written; the Python reader (or a Kotlin re-read) loads it; `schema.json` validates.
- **AC2 MaskParity** (`MaskParityTests`): on a sampled turn, each factored head's mask == the engine's candidate enumeration: Tech vs `TechManager.canBeResearched`, Policy vs `PolicyManager.isAdoptable`, Construction vs `CityConstructions.getBuildableBuildings`+`getConstructableUnits`, Promotion vs `UnitPromotions.getAvailablePromotions`, Great-person vs `GreatPersonManager.getGreatPeople`, Diplomatic-vote vs `getKnownCivsSorted`+abstain. Assert set-equality (bit i set ⟺ candidate i legal).
- **AC3 Determinism** (`DeterminismTests`): two runs of the same scenario `(seed≠0, fixed gameId, deterministicShuffle=true)` ⇒ byte-identical shards (`calculateChecksum` equal). Golden parity: a committed fixture scenario reproduces a committed CRC32 with `omniscientOpponents=false`.

## Fairness (must pass with omniscientOpponents=false)
- **AC4 Leakage** (`LeakageTests`): two games identical from X's vantage, differing ONLY in a rival's HIDDEN state — (a) rival's exact gold when X has no trade access/spy; (b) WHICH techs the rival holds (same COUNT, different SET) — MUST produce byte-identical `Featurizer.observe(X).bytes`. (Implies tech LIST never enters; gold only via trade-mask path.)
- **AC5 Unmet** (`UnmetTests`): an unmet rival's entire CIV token = zeros + every mask 0; contributes no CITY/UNIT tokens.
- **AC6 DownGate** (`DownGateTests`): for might/pop/production/GNP/land/happiness/culture, the observation contains only rank + bucket integers and identity-free best/avg/worst aggregates — assert NO raw per-civ float for these categories appears (e.g. mutate a met rival's exact stat by a non-bucket-crossing delta ⇒ observation bytes unchanged except possibly rank/bucket).
- **AC7 TileGate** (`TileGateTests`): an opponent city/unit whose center/unit tile ∉ `X.viewableTiles` ⇒ no token; making the tile visible adds the token with surface fields (and spy-gated interior incl. stealable-tech multi-hot when a set-up spy is present).
- **AC8 OmniscientAblation** (`OmniscientAblationTests`): with `omniscientOpponents=true`, AC4–AC7 invariants FAIL by design (raw values present / tokens present without vision); determinism (AC3) still passes. Confirms the flag is the ONLY switch that changes observations.

## Provenance (must pass)
- **AC9 ShardProvenance** (`ProvenanceTests`): every shard header + `schema.json` carry `SampleSchema.VERSION`, `UncivGame.VERSION.{text,number}`, and `RulesetFingerprint`; the reader surfaces them and REFUSES a shard whose `SampleSchema.VERSION` mismatches.
- **AC10 FingerprintDrift** (`ProvenanceTests`): mutating loaded ruleset content changes `RulesetFingerprint`; with `strictVersioning=true` and a stale `expectedRulesetFingerprint`, startup REFUSES; with it off, WARNS. `RulesetFingerprint` is deterministic across runs for the same ruleset.

## Python reader (`python/tests/test_reader.py`)
- Loads a committed golden fixture shard: validates magic + `SampleSchema.VERSION`, verifies CRC32, exposes provenance, reshapes tensors per `schema.json`; asserts refusal on a VERSION-mismatched fixture and a warning on a fingerprint-mismatched second shard.
