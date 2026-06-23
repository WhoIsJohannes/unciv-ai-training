# Codebase Scan (Light) — ONNX policy bridge + self-play loop

Base branch: `self-play-data-plane` (the data plane this feature builds on). Worktree:
`/Users/j/Unciv-onnx-selfplay-loop`. Source-verified; **3 Explore-agent claims were wrong and
are corrected below** (each verified by reading primary source — noted ⚠️).

## The integration contract (PolicyProvider)
`core/.../dataplane/PolicyProvider.kt:13-37`
```kotlin
interface PolicyProvider {
    fun chooseIndex(head: String, civ: Civilization, legalMask: BooleanArray, turn: Int): Int  // -1 = no legal/unmodeled
    fun actUnit(unit: MapUnit)
}
class RandomPolicy(private val rngFor: (Civilization, Int) -> Random) : PolicyProvider { ... }
```
- `RandomPolicy` is the template: filter legal indices, sample via `rngFor(civ,turn)`, return `-1` on empty.
- `actUnit` → `UnitAutomation.automateUnitMoves(unit)` (OnnxPolicy keeps this unchanged in v1).
- RNG factory: `DataPlaneHooks.defaultRngFor()` → `GameContext(civInfo=civ).stateBasedRandom("dataplane-policy-$turn")` — deterministic per (game, civ, turn).

## Action heads / schema (source of truth = SampleSchema.kt)
- `SampleSchema.VERSION = 1` (mirrored in `python/unciv_dataplane/schema.py:16` `SCHEMA_VERSION = 1`).
- Civ-level heads (`recordCivTurn`, DataPlaneHooks.kt:119): **order = `["tech","policy","greatPerson","diplomaticVote"]`**.
- Mask widths are **runtime, vocab-derived** (NOT hardcoded): tech=`vocab.techCount`, policy=`vocab.policyCount`.
  Plan claims tech≈80, policy≈70 for GnK — **must confirm empirically** (generate one shard, read `schema.json` layout / block `len`). The ONNX contract MUST use the real widths, not literal 80/70.

## Observation / Featurizer (the ONE input path)
`Featurizer.observe(x: Civilization): Observation` — pure/deterministic given game state (no RNG).
- `obs.block("global")` width = **26 floats** (5 scalars + 7 demographics × 3).
- `obs.block("acting_civ")` width = **13 + techCount + policyCount + policyBranchCount floats** (GnK ≈ 13+80+70+~30 ≈ 193). Confirm empirically.
- **Net input = concat(global, acting_civ)** — fixed once widths are pinned for GnK. Both blocks are FIXED f32.
- The legal masks are ALSO in the obs (`mask_tech`, `mask_policy` FIXED u8) — but at inference `chooseIndex` already receives the authoritative `legalMask` param; use the param.

## Legal masks
`LegalActionMasks.techMask(civ,vocab)`, `policyMask(civ,vocab,ruleset)` → `BooleanArray` (width = vocab count). The hook converts the obs `mask_*` block to a `BooleanArray` and passes it to `chooseIndex`.

## ⚠️ CORRECTION 1 — actions ARE recorded (Explore said they weren't)
`DataPlaneHooks.kt:119-136` `recordCivTurn` samples `policy.chooseIndex(head,...)` for all 4 heads and writes an
`"actions"` block (FIXED f32, 4 values = chosen index per head, order tech/policy/greatPerson/diplomaticVote).
→ Python: `step.blocks["actions"]` (the task's `step.actions`). Available today. ✓

## ⚠️ CORRECTION 2 — reward + terminal NOT wired (THE linchpin gap)
`DataPlaneHooks.kt:147-148`: `isLast=0, isTerminal=0, reward=0f` hardcoded. Comment (:122): "terminal flag left
false in v1." → **No learning signal in shards.** REINFORCE needs return-to-go. **This feature must wire terminal
±1 reward** (per-civ at game end) and **bump SampleSchema.VERSION 1→2** (mirror python). Sanctioned by the task's
NON-GOAL escape hatch ("VERSION bump if a new field is recorded"). The reader REFUSES version mismatch
(`reader.py:152`) → old VERSION-1 shards correctly rejected; regenerate.

### Wiring seam (clean, append-only — no binary backfill)
`Simulation.kt:149-161`: at game end `step.winner` (civ name) is known, or null = draw at turn cap. Right before
`recorder.close()`, call a new `recorder.recordTerminal(perCivReward)` that emits ONE terminal record per recorded
civ (`is_terminal=1`, `reward=±1`). `ShardRecorder` already tracks `seenCivs`. Python dataset.py groups steps by
`civ_slot`, reads the terminal reward, broadcasts it as the undiscounted return for that civ's steps.

## ⚠️ CORRECTION 3 — binomialTest already EXISTS (Explore said it didn't)
`Simulation.kt:320 binomialTest(successes,trials,p,alternative)` (normal approx via `normalCdf`/`erf`), used in
`text()` (:266-280) to print `winRate%` + one-tail p-value. `numWins` (`var`, public) + `steps` public. EVAL reuses
this — I'll EXTRACT `binomialTest`/`normalCdf`/`erf` into a public `SimStats` util for reuse + unit-testing.

## Policy injection is ONE policy per run (not per-civ)
`Simulation.kt:102 DataPlaneHooks.install(it.policy)` → `NextTurnAutomation.onCivTurn` calls the single policy for
EVERY civ. For "learner vs RandomPolicy", wrap with a **RoutingPolicy(learnerCivId, onnx, random)** that dispatches
in `chooseIndex`/`actUnit` by civ identity. **No data-plane modification** (honors "don't rebuild the data plane").
Learner identity must be stable across games for win-rate aggregation → pin fixed nations in the Tiny config; route
by nation; aggregate `numWins[learnerNation]`.

## Headless harness + Tiny config
- GENERATE template: `desktop/.../DataPlaneGen.kt:30-74` — builds UncivGame(headless), loads rulesets, builds a
  scenario, `Simulation(..., dataPlane = DataPlaneContext(config, vocab, policy, fingerprint))`, `sim.start()`.
  CLI args today: outputDir, maxTurns, episodes, seedBase, maxMapRadius.
- `DataPlaneGen` uses `ScenarioGenerator` (RANDOM size/civ-count). Self-play needs a **FIXED Tiny 2-civ GnK config**
  for fast, comparable rounds + stable learner identity. Tiny map params modeled on `ConsoleLauncher.kt:70-79`
  (`MapSize.Tiny`, GnK, Prince). New `SelfPlayRunner` (GENERATE + EVAL modes) rather than overloading DataPlaneGen.
- Determinism: data-plane path forces non-zero seed + `deterministicShuffle` + deterministic `gameId`
  (`Simulation.kt:113-124`) → replay is byte-identical (DETERMINISM criterion).
- Shard provenance header (`buildHeaderJson`) carries `schemaVersion` + `rulesetFingerprint` (`RulesetFingerprint.compute`
  = SHA-256 over `Vocab.canonicalSections`). `schema.json` sidecar written alongside shards.

## Python side (`python/unciv_dataplane/`)
- `reader.load(path) -> Shard`; `Shard.steps: list[Step]`; `Step(turn, civ_slot, is_first, is_last, is_terminal,
  overflow, reward, blocks: dict[str,np.ndarray])`. `step.blocks["global"|"acting_civ"|"actions"|"mask_tech"|"mask_policy"]`.
- `Shard.provenance` → `schema_version`, `ruleset_fingerprint`, `game_id`, etc. `load_dataset` WARNS on fingerprint
  mismatch; the new `unciv_train.dataset` must **REFUSE** (strict provenance gate).
- No helper concats blocks into a flat obs vector → training code builds `concat(global, acting_civ)`.

## onnxruntime dependency placement (build-arch decision for the plan)
`onnxruntime` is JVM-only with bundled natives (~100s MB) — must NOT land in the Android APK. Plan: keep OnnxPolicy
in `core/.../dataplane/` (per task) with onnxruntime `compileOnly` in `core`, and a runtime `implementation` dep in
`desktop` (where self-play actually runs). Verify Unciv module graph (android→core) before finalizing; fallback =
place OnnxPolicy in `desktop`.

## SEAM SUMMARY
| File | Role | Key symbol |
|---|---|---|
| `core/.../dataplane/PolicyProvider.kt` | interface + RandomPolicy template | `chooseIndex`, `actUnit` |
| `core/.../dataplane/SampleSchema.kt` | VERSION + head order/widths | `VERSION`, `MASK_HEADS` |
| `core/.../dataplane/Featurizer.kt` | the ONE obs path | `observe(x)` → global/acting_civ |
| `core/.../dataplane/LegalActionMasks.kt` | legality | `techMask`, `policyMask` |
| `core/.../dataplane/DataPlaneHooks.kt` | recorder + injection + header | `recordCivTurn`, `install`, `buildHeaderJson` |
| `core/.../simulation/Simulation.kt` | headless loop, winner, stats | `start()`, `binomialTest`, `numWins` |
| `desktop/.../DataPlaneGen.kt` | GENERATE template | `main()` |
| `python/unciv_dataplane/reader.py` | shard loader | `load`, `Step`, `Provenance` |
| `python/unciv_dataplane/schema.py` | VERSION mirror + validator | `SCHEMA_VERSION` |

## OPEN DECISIONS for the plan (most are technical; 2 go to the user at Step 3)
1. **[USER] Win definition at turn cap** — victory-only (sparse on Tiny) vs victory-or-score-leader-at-cap (denser
   signal). Must be consistent between training reward AND EVAL win-rate.
2. **[USER] Loop compute budget** — K rounds × N gen games × M eval games (wall-clock vs statistical power of curve).
3. [decide] VERSION bump 1→2 for terminal reward; mirror python; regen each round.
4. [decide] RoutingPolicy wrapper; pin nations in Tiny config; route by nation.
5. [decide] Extract `binomialTest`→public `SimStats`; EVAL emits one machine-readable line.
6. [decide] onnxruntime: `compileOnly` core + `implementation` desktop (verify module graph).
7. [decide] ONNX I/O contract widths read from vocab/schema at build time; version-stamped; PARITY test JVM vs Python.
