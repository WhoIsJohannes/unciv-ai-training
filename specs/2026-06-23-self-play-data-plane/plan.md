# Plan — Self-Play Data Plane for Unciv

**Mode:** BUILD · **Size:** L · **Branch:** `self-play-data-plane` · **Base:** `master` (`4debb125f`)
**Ruleset:** Gods & Kings (`Civ V - Gods & Kings`) · **Scope:** data plane ONLY — no NN, no training, no UI, no gameplay/balance change.

## Why (the ask, restated)
Build a policy-agnostic, deterministic **data plane** so an external RL/NN stack can later train on self-play trajectories from the Unciv engine. It must (a) featurize each deciding civ's state **fog-correctly** under a principled **fair opponent-information model** (the policy sees only what a human controlling that civ could obtain), (b) emit **factored legal-action masks** straight from the engine's existing candidate enumeration, (c) write **thread-safe binary trajectory shards** with **version + ruleset provenance** (datasets self-identifying, perishable/regenerable — never migrated), (d) generate randomized GnK scenarios with guardrails, and (e) ship a **pure-Python reader** that validates schema/provenance. A `RandomPolicy` stub drives full games to prove the plumbing.

## How it solves it (design → ask bridge)
The observation follows the **AlphaStar tri-partite shape** (scalar + fixed-capacity-padded entity-lists + spatial planes) — which is exactly the prompt's TOKEN GROUPS + `Tile.zeroBasedIndex` spatial layer. Fairness is enforced at the single point where features are *read* from the engine (`FairOpponentModel`), gated by the engine's own visibility/met/spy APIs, with one ablation switch (`omniscientOpponents`). Determinism reuses Unciv's existing `GameContext.stateBasedRandom(label)` (already keyed on `gameId`/`turn`/`civID`); the only engine change is the **gated** `GameStarter:322` shuffle fix (off for interactive play). Provenance pins on `UncivGame.VERSION` + a content-hash `RulesetFingerprint`, written into every shard header + a `schema.json` sidecar, modeled on the self-describing `.npy` container. All shard bytes are explicit **little-endian** (the JVM default is big-endian — a deliberate, tested choice).

---

## Architecture

### New Kotlin package: `core/src/com/unciv/logic/simulation/dataplane/`
(No collision with existing `com.unciv.logic.simulation.*`.)

| File | Responsibility | Tag |
|---|---|---|
| `SampleSchema.kt` | `VERSION: Int` (=1), all layout constants (caps, channel counts, per-head sizes), the field-inventory enum, dtype constants. **Single source of truth for layout.** Bumping `VERSION` mirrors `CURRENT_COMPATIBILITY_NUMBER` discipline. | [AI_CODE] |
| `SampleConfig.kt` | `data class SampleConfig(enabled, outputDir, omniscientOpponents=false, strictVersioning=false, expectedRulesetFingerprint:String?=null, expectedSchemaVersion:Int?=null, caps: Caps = Caps.DEFAULT, seedBase: Long)`. `omniscientOpponents` defaults OFF. | [AI_CODE] |
| `Vocab.kt` | Programmatic string-id↔index adapters built from the loaded GnK ruleset in canonical order: `technologies, units, buildings, policies, policyBranches, tileResources, unitPromotions, terrains, nations, eras` (LinkedHashMap iteration order) + `religions` **sorted alphabetically** (the lone `ArrayList`) + enums (`RankingType, ResourceType, TerrainType`). | [AI_CODE] |
| `RulesetFingerprint.kt` | `fun compute(ruleset): String` — SHA-256 hex over the canonical-ordered concatenation of every entity id + enum/vocab order (same order `Vocab` uses). Deterministic across runs. `java.security.MessageDigest` (stdlib). | [AI_CODE] |
| `FairOpponentModel.kt` | Per-attribute fair encoding + availability masks for each opponent civ O relative to deciding civ X (the prompt's encoding table + D4). The ONE place opponent intel is read; `omniscientOpponents=true` bypasses to raw. | [AI_CODE] |
| `Featurizer.kt` | Builds the full observation for deciding civ X: GLOBAL + ACTING-CIV scalars, CIV tokens (via `FairOpponentModel`), DIPLO edges, CITY tokens (own EXACT; opp tile+spy gated), UNIT tokens (own EXACT; opp tile gated), spatial planes keyed by `zeroBasedIndex`. Returns a flat little-endian-ready buffer + per-attribute availability masks. | [AI_CODE] |
| `LegalActionMasks.kt` | Factored heads from engine enumeration ONLY: Tech (`TechManager.canBeResearched`), Policy (`PolicyManager.isAdoptable`), City-construction (`CityConstructions.getBuildableBuildings`+`getConstructableUnits`), Promotion (`UnitPromotions.getAvailablePromotions`), Great-person (`GreatPersonManager.getGreatPeople`), Diplomatic-vote (`diplomacyFunctions.getKnownCivsSorted`+abstain), Unit-intent+target (from `UnitAutomation` candidate sets). | [AI_CODE] |
| `PolicyProvider.kt` | `interface PolicyProvider { chooseTech/choosePolicy/chooseConstruction/choosePromotion/chooseGreatPerson/chooseDiplomaticVote(...); fun actUnit(unit) }`. `class RandomPolicy(rngFor: (Civilization,Int)->Random)` picks uniformly among legal candidates per non-unit head and delegates `actUnit` to existing `UnitAutomation`. | [AI_CODE] |
| `ShardFormat.kt` | Format constants: magic `UNCVSMP1` (8B), framing, dtype tags. CRC32 (`java.util.zip.CRC32`, mirrored by Python `zlib.crc32`) for `calculateChecksum`. | [AI_CODE] |
| `TrajectoryEmitter.kt` | Thread-safe by construction: **one shard file per worker**, keyed by `CoroutineName` (read via `coroutineContext[CoroutineName]`). Writes LE header (provenance) + per-step records + footer checksum. No shared writer ⇒ no locks. | [AI_CODE] |
| `ScenarioGenerator.kt` | Randomized `GameParameters`+`MapParameters` over the GnK envelope (map size Tiny–Huge, shape/type, 2–`maxMajorCivs`, 0–`maxCityStates`), guardrails: land-tiles/civ ≥ 80 (retry), `MapRegions` "too many players" retry, seed≠0, deterministic `gameId`. Emits a per-episode log (params+seed+gameId+version+fingerprint). | [AI_CODE] |
| `DataPlaneHooks.kt` | Glue: open/record/finalize emitter around the `Simulation` turn loop; featurize the deciding civ at the policy decision points; honors `SampleConfig`. Startup fingerprint/version check (warn/refuse per `strictVersioning`). | [AI_CODE] |

### Engine touch-points (minimal, mostly additive; gated so interactive play is byte-identical)
| File | Change | Tag |
|---|---|---|
| `logic/automation/civilization/NextTurnAutomation.kt:38` | Add optional `policyProvider: PolicyProvider? = null` param to `automateCivMoves`; at the factored decision points, when `policyProvider != null` consult it (else existing behavior). Default callers pass `null` ⇒ **unchanged**. | [AI_CODE] |
| `logic/GameStarter.kt:322` | `if (shufflePlayerOrder) { if (gameSetupInfo... deterministicShuffle) otherPlayers.shuffle(rng) else otherPlayers.shuffle() }`. `rng` already exists at :48. | [AI_CODE] |
| `models/metadata/GameParameters.kt` (or `GameSetupInfo`) | Add `var deterministicShuffle: Boolean = false` (default off ⇒ interactive play unchanged). | [AI_CODE] |
| `logic/simulation/Simulation.kt` (~99/101-104/108-114/118-130) | Accept a `SampleConfig?` + `PolicyProvider`; open a shard per `CoroutineName` (101-104), record per deciding-civ-turn (108-114), finalize/close (118-130). Fix the `seed = 0` bug on the data-plane path (seed≠0). All behind `config.enabled`; default null ⇒ existing Simulation behavior. | [AI_CODE] |

### New desktop entrypoint
| File | Responsibility | Tag |
|---|---|---|
| `desktop/src/com/unciv/app/desktop/DataPlaneGen.kt` | Headless `main` (mirrors the user's `SimBenchmark`): `UncivGame(true)`, `RulesetCache.loadRulesets(consoleMode=true)`, build GnK, run N episodes via `ScenarioGenerator`+`Simulation` with the emitter enabled, write shards + `schema.json`. | [AI_CODE] |
| `desktop/build.gradle.kts` | Register `:desktop:dataGen` `JavaExec` task (mirrors the `simBench` pattern). | [AI_CODE] |

### Python reader
| File | Responsibility | Tag |
|---|---|---|
| `python/unciv_dataplane/reader.py` | Pure-Python (numpy + stdlib) reader: parse shard header + records via `np.frombuffer`, validate magic + `SampleSchema.VERSION` (**REFUSE** on mismatch), verify CRC32, expose provenance (Unciv version + fingerprint), WARN on fingerprint mismatch within/across shards. | [AI_CODE] |
| `python/unciv_dataplane/schema.py` | Load + validate `schema.json`; `SampleSchema.VERSION` mirror constant. | [AI_CODE] |
| `python/README.md` | The **pin-one-version discipline** doc + format spec + usage. | [AI_RESEARCH] |
| `python/tests/test_reader.py` | Loads a committed golden fixture shard; asserts shapes/provenance/refusal-on-version-mismatch. | [AI_CODE] |

### Kotlin tests (`tests/`, JUnit + `@RunWith(GdxTestRunner::class)`, reuse `TestGame`)
| File | Covers (acceptance criteria) | Tag |
|---|---|---|
| `tests/.../dataplane/MaskParityTests.kt` | #2 masks == engine enumeration per head (sampled turn). | [AI_CODE] |
| `tests/.../dataplane/DeterminismTests.kt` | #3 `calculateChecksum` replay byte-identical; golden parity (omniscient=false). | [AI_CODE] |
| `tests/.../dataplane/LeakageTests.kt` | #4 hidden rival gold (no trade/spy) + rival tech SET (same count, diff set) ⇒ byte-identical X obs. | [AI_CODE] |
| `tests/.../dataplane/UnmetTests.kt` | #5 unmet rival ⇒ zeros+masks0, no city/unit tokens. | [AI_CODE] |
| `tests/.../dataplane/DownGateTests.kt` | #6 no raw per-civ float for demographic categories; rank/bucket + identity-free best/avg/worst only. | [AI_CODE] |
| `tests/.../dataplane/TileGateTests.kt` | #7 opp city/unit token absent when tile not visible; appears (with spy-gated interior) when visible. | [AI_CODE] |
| `tests/.../dataplane/OmniscientAblationTests.kt` | #8 with omniscient=true tests 4–7 FAIL by design; determinism still passes. | [AI_CODE] |
| `tests/.../dataplane/ProvenanceTests.kt` | #9 shard+schema carry VERSION+version+fingerprint; reader refuses VERSION mismatch. #10 fingerprint-drift: alter ruleset ⇒ fingerprint changes; strict+stale ⇒ refuse, else warn; deterministic across runs. | [AI_CODE] |
| `tests/.../dataplane/RunToCompletionTests.kt` | #1 RandomPolicy drives a small game to completion; emitter writes a shard; reader (via a thin Kotlin re-read or the golden) validates. | [AI_CODE] |

---

## Observation layout (SampleSchema v1)

**Scalar groups** (fixed): GLOBAL (turn, era-of-X index, map-size index, #players bucket, enabled-victory-type bits, speed index, ruleset-fingerprint-tag), ACTING-CIV X (EXACT own state: gold, gpt, science/culture/faith per-turn, happiness, era, #cities, #units, own techs multi-hot, own policies multi-hot + branch bits, own victory progress).

**Entity lists** (fixed capacity, padded, each row carries a `present`/`met` bit = the AlphaStar "slot populated" mask):
- **CIV tokens** `[MAX_MAJOR_CIVS + MAX_CITY_STATES]`: per the fair model — met-bit; era; policy-branch bits + count; wonders multi-hot; total-score (EXACT, met); tech-COUNT (EXACT, met; **never the list**); victory numerators (SS-parts COUNT only, original-capitals-owned [derived or denom-mask=0], completed branches, religion conversions) with denom-mask via `shouldHideCivCount`; opinion-of-X + O→X diplo modifiers; diplo flags vs X; **rank + 5-bucket** per demographic category (might/pop/prod/GNP/land/happiness/culture) — **NO raw float**; trade slots (gold, gpt, tradeable-resources multi-hot) behind trade-mask (met∧major∧¬CS); #cities-of-O-seen (from tile vision). Unmet ⇒ all zeros + every mask 0.
- **DIPLO edges**: relationship booleans among met civs; third-party identity masked exactly as `getCivName` does.
- **CITY tokens** own `[MAX_OWN_CITIES]` EXACT + opponent `[MAX_VIS_OPP_CITIES]` (included ONLY if center tile ∈ `X.viewableTiles`): name-hash bucket, owner-slot, pop SIZE, defensive strength, health (if damaged), air-unit count, majority religion, resistance/puppet/razing flags; **spy-gated interior** (buildings multi-hot, production id+turns, exact yields, **stealable-tech multi-hot** per D4) only with a set-up spy in that city. Own-city tokens also carry the **chosen-construction index + construction mask** (the per-city construction action).
- **UNIT tokens** own `[MAX_OWN_UNITS]` EXACT + opponent `[MAX_VIS_OPP_UNITS]` (included ONLY if unit's tile ∈ `X.viewableTiles`, excluding invisible units): relative coords, type category, health bucket, owner-slot, promotions. Own-unit tokens carry the **chosen-intent index + target offset + per-unit action mask**.

**Spatial planes** keyed by `Tile.zeroBasedIndex`, length = **this game's `tileList` size** (recorded in header; per-game, not a fixed Huge tensor):
- visibility-state channel (0 never-explored / 1 explored-not-visible / 2 currently-visible);
- PERSISTENT (for explored): terrain-type, terrain-features, resource (if revealed), road/river, is-city-center;
- TRANSIENT (currently-visible only): owner-slot, improvement, unit-present + unit-owner-slot + unit-type-category + unit-health-bucket.
`omniscientOpponents=true` ⇒ all tiles treated visible; all civ tokens raw EXACT.

**Masks emitted alongside** the tensor: the 7 factored legal-action heads (boolean per candidate) + the fair-info availability masks (met / trade / spy / tile-visible / denominator).

---

## Shard binary format (modeled on `.npy`; explicit little-endian)
```
magic   : 8 bytes  "UNCVSMP1"
version : uint16   = SampleSchema.VERSION
hdrLen  : uint32   ; header JSON length
header  : UTF-8 JSON { uncivVersion{text,number}, compatibilityNumber, gitSha?(best-effort),
                       rulesetFingerprint, schemaVersion, caps, channelLayout, dtypes,
                       gameId, seed, mapParameters, gameParameters, nTiles }
records : repeated  [ uint32 recLen | LE payload ]   ; payload = stepHeader + tensor blocks + mask blocks
footer  : uint32 recCount | uint64 crc32-of-all-records
```
- `stepHeader`: turn, civSlot, flags(is_first/is_last/is_terminal), reward placeholder (f32), overflow flag.
- `calculateChecksum` = CRC32 over the concatenated record bytes (Kotlin `java.util.zip.CRC32` ↔ Python `zlib.crc32`).
- `schema.json` sidecar (one per output dir) duplicates the layout + provenance for the reader.

## Determinism contract
Given identical `(scenario seed≠0, deterministic gameId, deterministicShuffle=true, ruleset)`, two runs produce **byte-identical** shards + identical header (modulo a wall-clock field excluded from the checksum). Policy RNG = `civ.state.stateBasedRandom("dataplane-policy-$turn-$head")` ⇒ derived from `(gameId, turn, civID)`. The `calculateChecksum` replay test re-runs a recorded scenario and asserts equality.

## Provenance & versioning (perishable — detect, regenerate, never migrate)
- `RulesetFingerprint` = deterministic SHA-256 over canonical entity ids + enum/vocab order.
- Every shard header + `schema.json` carry `SampleSchema.VERSION`, `UncivGame.VERSION.{text,number}`, `compatibilityNumber`, optional git SHA (best-effort `git rev-parse` at gen-time; **no build constant exists**), and `RulesetFingerprint`.
- Startup check: if `expectedRulesetFingerprint`/`expectedSchemaVersion` set and live values differ → WARN, or REFUSE when `strictVersioning=true`.
- Reader REFUSES a shard whose `SampleSchema.VERSION` mismatches; WARNS on fingerprint mismatch within/across shards.
- VERSION bump rule: any layout change bumps `VERSION` (mirrors `CURRENT_COMPATIBILITY_NUMBER`); layout-affecting ruleset changes are caught by the fingerprint even at the same VERSION.

## Fixed-width caps (configurable in `SampleConfig.Caps`; defaults)
`maxMajorCivs=16`, `maxCityStates=24`, `maxOwnCities=64`, `maxVisOppCities=64`, `maxOwnUnits=192`, `maxVisOppUnits=192`. **Overflow** (entities exceed a cap mid-game) ⇒ clamp to cap + set the step `overflow` flag + increment an overflow tally in the run log; the reader validates caps against the schema. Never silent truncation-without-signal. Spatial length is per-game (no cap needed).

## Council-driven refinements folded in (from `decisions.md` triage)
Gated-shuffle wording (architect); PolicyProvider as optional param (architect); subset-field serialization not full Json (architect); one-file-per-shard structural thread-safety (architect); unmet civs masked from ranks not UI `shouldHideCivCount` (security); `religions` sorted (security); partial-visibility fog channels (domain_fidelity); city-state indirect-ally via engine masking (domain_fidelity); spy-gated stealable-tech multi-hot + spaceship count-only (D4); lightweight per-shard logging + per-game try/catch isolation (practitioner, scoped down).

## Deferred / out of scope (explicit)
No NN/training/inference; no Prometheus/`/healthz`/separate-JVM ops; no schema-migration tooling; no mod-portable vocab; no arbitrary-mod support (GnK only); spaceship part *identities* never encoded; engine-wide RNG-label audit (labels don't flow into the fog-gated featurizer).

---

## Walkthrough
One representative input — **deciding civ X = "Rome" on turn 42, mid-game, has met "Greece" (O₁) but not "Egypt" (O₂); O₁ has a city Athens whose center tile is currently in Rome's vision; Rome has a set-up spy in Athens** — traced through the data plane. *Illustrative mock values — not measured. LLM-authored; cross-check against the architecture above.*

| Hop | Contract (A → B) | Mock data shape |
|---|---|---|
| 1. Turn loop | `Simulation` (worker `simulation-3`) → `DataPlaneHooks.onCivTurn(X, turn=42)` | open shard `shard-simulation-3.bin` if first; CoroutineName="simulation-3" |
| 2. Featurize | `Featurizer.build(X)` → reads engine via fair gates | GLOBAL=[turn=42, eraIdx=3(Medieval), mapSizeIdx=2, …]; ACTING-CIV X EXACT=[gold=210, sci=38.0, …, ownTechs=multi-hot(31)] |
| 3a. CIV token O₁ (Greece, met) | `FairOpponentModel.encode(O₁)` | met=1; era=3; score=540 (EXACT); techCount=29 (EXACT, **no list**); mightRank=2, mightBucket=3 (**no raw float**); tradeMask=1→gold=80, gpt=4, luxuries=multi-hot; #citiesSeen=1; SS-parts count=0 |
| 3b. CIV token O₂ (Egypt, UNMET) | `FairOpponentModel.encode(O₂)` | **all zeros**, met-mask=0, every per-attr mask=0 |
| 4. CITY token Athens (tile visible + spy) | `Featurizer` city loop, `X.viewableTiles.contains(centerTile)`=true, `getSpiesInCity(Athens)` set-up | surface: ownerSlot=O₁, popSize=7, defStr=24, religion=idx5, puppet=0; **spy interior**: buildings=multi-hot, production="Library"#3turns, stealableTech=multi-hot{Mathematics,Currency} (D4) |
| 5. Masks | `LegalActionMasks.build(X)` | techHead=bool[ |techVocab| ] from `canBeResearched`; constructionHead per own city; voteHead from `getKnownCivsSorted`+abstain |
| 6. Policy acts + record | `RandomPolicy.chooseTech(...)` picks uniform-legal; `DataPlaneHooks.record(step)` | step record: obs blocks + masks + chosenTech=idx17, flags{is_first=0,is_last=0}, reward=0.0(placeholder); CRC32 updated |
| 7. Finalize | game ends turn 240 → `DataPlaneHooks.close()` | footer: recCount=240, crc32=0x9F3A…; `schema.json` + episode log written |
| 8. Read (Python) | `reader.load("shard-simulation-3.bin")` | validates magic+VERSION(==1)+CRC; exposes provenance {uncivVersion 4.20.15, fingerprint "a3f1…"}; yields per-step numpy arrays |

**Leakage invariant check on this trace:** if O₁'s exact gold differed but Rome had no trade access, or O₂ (unmet) held a different tech *set* with the same count, **bytes 2–7 for Rome are identical** — because gold only enters via the trade-mask path and the tech list never enters at all.

---

## Plan-review council refinements (BINDING — folded in after Step 11 council, 40 findings)
The plan-review council (8 reviewers, 40 findings) surfaced no architectural change — all are detail-tightenings. Binding amendments:

### Fairness / covert-channel hardening
- **R1 (FND-0025, crit): canonical entity ordering.** `Featurizer` fills every entity list (CIV/CITY/UNIT) in a **deterministic canonical order** — civs by `civID`, cities by city `id`, units by unit `id` (ascending) — so slot position carries no hidden signal.
- **R2 (FND-0025/0027/0028, crit/major): nothing meta enters the observation tensor.** The `overflow` flag, `RulesetFingerprint`, `gameId`, and `seed` live ONLY in the step-record metadata / shard header — they are NOT features fed to the policy. Removes the GLOBAL `ruleset-fingerprint-tag` from the observation (it stays in the header).
- **R3 (FND-0029): third-party DIPLO edges are FAIR and KEPT** — the prompt's fairness principle explicitly lists "the global politics/diplomacy web" as a freely-openable screen for met civs (mirrors `GlobalPoliticsOverviewTable`). O₁↔O₂ among MET civs is encoded with the engine's `getCivName` third-party identity masking. (Counter to the reviewer's over-strict reading; documented.)
- **R4 (FND-0026): city name** is a TILE-GATED VISIBLE attribute (the human reads the city banner) and self-play names come from the ruleset's fixed lists — so encode the name as a ruleset-name index (no salt needed); anonymization is moot here.
- **R5 (FND-0036): stealable-tech multi-hot is the D4 spy-gated EXCEPTION, not a contradiction.** The passive/met path encodes tech **COUNT only, never the list**; the stealable-tech multi-hot is encoded ONLY under a set-up spy (earned), per user decision D4. The two coexist — different gates. (Note: this reviewer's lens contradicted its own intake-round position; D4 is the resolution.)

### Determinism tightening
- **R6 (FND-0020, crit): `calculateChecksum` covers RECORD bytes ONLY** — the header (which carries wall-clock / hostname / git-SHA) is excluded entirely, so determinism = identical record bytes regardless of header timestamps. No per-field exclusion list needed.
- **R7 (FND-0006/0030, crit): explicit little-endian** via `ByteBuffer.order(ByteOrder.LITTLE_ENDIAN)` for every binary field; dtype endianness markers (`<f4`,`<i4`) in the header. (Already intended; now mandated.)
- **R8 (FND-0021, major): deterministic overflow drop policy** — on overflow, keep the first N entities in the R1 canonical order; the rest are dropped + `overflow` flag set + tallied in the run log.
- **R9 (FND-0039, major): demographics rank/bucket algorithm** — rank = 1-based position among met majors (+ self), ties broken by `civID`; bucket = 5 equal-width buckets over the met-major range; best/avg/worst computed over all alive majors but identity-stripped (the only cross-civ aggregate, anonymous).

### Emitter robustness (lightweight — offline batch, not a service)
- **R10 (FND-0010, crit): shard naming collision guard** — filename = `shard-{coroutineName}-{workerId}.bin` where `workerId` is a per-emitter monotonic counter; if `CoroutineName` is absent (non-coroutine caller), fall back to `workerId` alone. No serialized-writer/bounded-queue (rejected as over-engineered for offline gen).
- **R11 (FND-0011/0013, major→scoped): crash safety** — write to `*.tmp` via `use{}`/try-with-resources, atomic rename to `*.bin` on success, delete partial on unrecoverable error; the Python reader tolerates a truncated trailing record (length-prefixed framing) and salvages preceding valid steps. REJECT bounded-async-queue/back-pressure/Prometheus (FND-0012 over-engineered).
- **R12 (FND-0014, minor): three episode-log statuses** — `OK_SHARD_WRITTEN` / `GAME_OK_SHARD_FAILED` / `GAME_FAILED` (machine-readable line per episode).

### Scenario generator
- **R13 (FND-0022, major): bounded retries** — `ScenarioGenerator` retries guardrails up to `maxAttempts=100` (configurable), then throws `IllegalStateException`. Add `ScenarioGeneratorGuardrailTests` (impossible config ⇒ exception).

### Victory encoding (FND-0035, crit — explicit)
- Domination: derive "original major capitals owned" by counting original-capital cities owned by O (if a clean `isOriginalCapital`/founder signal exists), else emit numerator with denom-mask=0. Spaceship: COUNT only (never part identities). Diplomatic: vote count / `hasEverWonDiplomaticVote`. Culture/Science/Religion: completed-branches count / SS-count / conversions count.

### Spec completeness (FND-0024/0036, qa/domain)
- **R14: the complete fair-info encoding-status table** (below) is the self-contained test oracle for the leakage/down-gate/tile-gate tests.

### Build sequencing (FND-0019, accepted)
- **R15: validate determinism on a Tiny single-thread scenario FIRST** (emitter + checksum) before wiring multi-thread/Huge — de-risks the determinism pipeline early.

### Rejected (with reason)
- FND-0001 remove `deterministicShuffle` flag → REJECT: default-deterministic shuffle WOULD change interactive player order for a given seed (violates D1's byte-identical promise).
- FND-0003 remove `PolicyProvider` interface → REJECT: the feature is *policy-agnostic*; the interface is the required RL seam (prompt mandates it).
- FND-0002 hardcode `omniscientOpponents`/`strictVersioning` → REJECT: both are required (acceptance #8 / #10).
- FND-0017 "VP+architect approval for fairness deviation" → REJECT: corporate ceremony, N/A for an OSS PR.
- FND-0012 bounded-async write queue / back-pressure / throughput SLOs / `/healthz` → REJECT: over-engineered for offline batch data-gen (skeptic concurs). Soft throughput note only (from `SimBenchmark`).
- Cost-metric/"cost per trajectory" findings → note a soft throughput target; no cloud-cost machinery (no cloud, no LLM).

## Complete fair-info encoding-status table (R14 — test oracle)
| Opponent attribute | Encoding | Gate |
|---|---|---|
| met flag / identity | present bit | `X.knows(O)` |
| era index | EXACT | met |
| policy branches + count | EXACT (bits) | met |
| wonders owned | EXACT (multi-hot, broadcast) | met; location only if `X.hasExplored` |
| total score | EXACT | met |
| tech COUNT | EXACT | met |
| tech LIST | **HIDDEN — never** | — |
| victory numerators (SS count, orig-capitals, branches, conversions) | EXACT numerator; denom via `shouldHideCivCount` | met; denom-mask |
| opinion-of-X + O→X modifiers | EXACT | met |
| diplo flags vs X + among met civs | EXACT (3rd-party id masked like `getCivName`) | met |
| might/pop/prod/GNP/land/happiness/culture | **RANK + 5-bucket only; NO raw float** | met (rank among met) |
| global best/avg/worst per category | EXACT but **identity-free** | always (anonymous aggregate) |
| gold / GPT / tradeable resources | EXACT | trade-mask (met ∧ major ∧ ¬CS) |
| #cities of O seen | EXACT (from tile vision) | tile-derived |
| opp CITY surface (pop size, def str, health-if-damaged, religion, flags, air count) | EXACT | center tile ∈ `viewableTiles` |
| opp CITY interior (buildings, production, yields, **stealable-tech multi-hot**) | EXACT | **set-up spy** in that city (D4) |
| opp UNIT (coords, type, health, owner) | EXACT | unit tile ∈ `viewableTiles` (visible units only) |
| spaceship PART identities, per-turn sci/culture/faith, foreign production w/o spy, rival↔rival trades X isn't party to, current research/in-progress | **HIDDEN — never** | — |
