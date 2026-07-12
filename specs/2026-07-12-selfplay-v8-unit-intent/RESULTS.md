# v8 — Unit control via a per-unit INTENT head — RESULTS

Branch `selfplay-v8-unit-intent`. Mirrors the v7 per-city construction control (per-entity categorical head +
BC-clone + KL-leash) applied per-unit. The net picks, for each controlled land-military unit, WHICH existing
`UnitAutomation` behaviour to run; pathfinding stays 100% heuristic. Base config = the v7.4 construction win
(tech+policy+construction, BC 120ep + KL 0.5).

## Status

**Implementation COMPLETE; the parity/no-op/legality/throughput GATE is GREEN. The 8-seed paired EFFECT
experiment (AC#4) is the remaining step** (a multi-hour compute run — `run_v8kl.sh`).

## Design — deferred (turn-end) frame emission

The user chose "record the REALIZED executed intent" (AC#1 literal). Unlike `construction_current` (a readable
turn-start property), the heuristic's chosen rung and the realized executed intent are only known AFTER the
`automateUnits` ladder runs. So the per-civ-turn frame is emitted at **turn-END**:

- `onCivTurn` → `DataPlaneHooks.handleCivTurn`: featurize (snapshot obs incl. `mask_unit_intent`), decide+apply
  tech/policy/construction, **sample each controlled land-military unit's intent** (+ keep the masked
  log-prob vector), snapshot Φ, stash a `PendingCivTurn`. No emit.
- act-time inside `UnitAutomation.automateUnitMoves`: dispatch the unit's decided intent (run only its `tryX`);
  on failure fall through to the instrumented ladder. `noteUnitIntent` captures the realized rung (and the
  heuristic first-firing rung = `unit_intent_current`, the BC target — for every land-military unit).
- turn-END (lazy-flush at the next `handleCivTurn` + at `finalizeGame`): build the three per-unit VARIABLE
  blocks aligned to the turn-start `orderedOwnUnits` snapshot (matched by the STABLE `MapUnit.id`), emit.

`own_units` was re-sorted by the stable `MapUnit.id` (was `currentTile.zeroBasedIndex`, which moves as units
move) so a unit keeps its row from turn-start to act-time; own_units feeds permutation-invariant aggregation,
so the civ heads are unchanged. Emission timing shifted for construction/tech/policy too, so Φ is snapshotted
at turn-start (else the recorded Φ would drift and break the PBRS no-op).

## Acceptance criteria — gate results

| AC | Result |
|----|--------|
| **#1 LEGALITY** | ✅ end-to-end gen (control-unit-intent ON, RandomPolicy): **0 illegal / 11,172 decisions**; recorded intent == realized executed intent (fallback → realized rung, or −1 if the realized rung was not an offered legal intent); OFF arm = 0 unit decisions (units heuristic); `unit_intent_current` BC labels captured in BOTH arms. `UnitIntentCodeTest` (mask width/decode/land-military-only) + `OnnxPolicyLegalityTest` green. |
| **#2 NO-OP** | ✅ Python zero-summard oracle (`test_v8_unit_intent`): `--control-unit-intent off` vs on with all-−1 unit actions → shared weights identical, max\|Δw\|<1e-6 (also under v6 replay). OFF-arm gen: units 100% heuristic (0 decisions). All 27 Kotlin dataplane tests + 81 Python tests green. |
| **#3 PARITY** | ✅ exported `unit_intent_logits` (ORT) == the torch net's unit-intent head, atol 1e-4. JVM↔Python `tech`/`policy` structured parity still holds after the `own_units` reorder (the JVM `OnnxPolicy` reads the same ORT output — ORT==torch is the cross-boundary guarantee, as for construction). |
| **#4 EFFECT** | ⏳ PENDING the 8-seed paired run (`run_v8kl.sh`). |
| **#5 SCHEMA** | ✅ `SampleSchema.VERSION` 8→9 lockstep with `schema.py SCHEMA_VERSION`; old (v8) shards refused; ONNX contract grows by EXACTLY `unit_intent_logits` (`test_ac6_warm_net_matches_exported_onnx` asserts the exact output set); intent enum enters the ruleset fingerprint (`Vocab enum:UnitIntent`). Reader unchanged (layout-generic). |
| **#6 THROUGHPUT** | ✅ ON vs OFF gen wall-clock **60.8s vs 60.75s (~100%)** — the dispatch + deferred-emission overhead is negligible; the net forward is memoized per civ-turn (one inference, per-unit is an index). |
| **#7 Determinism** | ✅ single RNG draw/decision via `MaskedChoice` (golden-tested); terminal-only ±1 reward; tech/policy/construction heads unchanged (no-op oracle). |

Full pipeline validated by a pilot (BC both heads → 2 RL rounds → export → OnnxPolicy net-driven eval,
diverged=0): `bc_acc=0.415`, **`bc_unit_acc=0.690`**, onnx_decisions climbing (unit intents dispatched by
the net), no crashes.

## The intent enum (14 land-military intents)

`HEAL, UPGRADE(human-only⇒inert for AI), ACCOMPANY, GO_TO_RUIN, DEFEND_SIEGED_CITY, ATTACK, RETAKE_CITY,
ADVANCE_ENEMY_CITY, ATTACK_ENCAMPMENT, GARRISON, ADVANCE_CLOSE_ENEMY, PREPARE, EXPLORE, FOG_BUST` — the
`automateUnitMoves` ladder order; heal variants folded into one HEAL. Cheap mask predicates where the `tryX`
early-out is cheap; unconditional-include (dispatch falls back) for the 4 that pathfind immediately
(ACCOMPANY/ATTACK/ADVANCE_CLOSE_ENEMY/EXPLORE), guaranteeing ≥1 legal intent per unit (the head never abstains).

## Experiment (to run) — `run_v8kl.sh`

Base (both arms): construction control ON + BC(120ep, both heads) + construction-KL 0.5 (the v7.4 winner).
Paired variable, 8 gen-seeds, Medium, 16 rounds, rw1/mb0/small/ent0.02, 200-game ceiling @ eval-seed 4242424:

- `off`: `--control-unit-intent off --unit-kl-coef 0`
- `on` : `--control-unit-intent on  --unit-kl-coef 0.5`

Report (to fill after the run): the arm×seed ceiling table with mean±SE, the paired `on − off` Δpp with t/p
(`analyze_v8.py`, df=7), and each arm's absolute win-rate vs the 50% break-even (net-intent vs random-intent).

| arm | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 | s7000 | s8000 | mean±SE | vs-50% |
|-----|-------|-------|-------|-------|-------|-------|-------|-------|---------|--------|
| off |  |  |  |  |  |  |  |  |  |  |
| on  |  |  |  |  |  |  |  |  |  |  |

**Paired on − off = ___ pp ± ___ (t=___, p=___), ___/8 seeds positive.** Ship criterion (v7 D-C5): the head
ships if it beats OFF at p<0.05 in ≥1 configuration (crossing 50% is a reported milestone, not a gate). If it
does NOT beat off, that is a real negative (like v7's first pass) — report it plainly and whether BC/KL tuning
(`run_v8bc.sh` ablation: unit-KL 0 vs 0.5; more `--bc-epochs`) recovers it.

## Files changed

Kotlin: `UnitIntent.kt` (new enum), `Vocab.kt` (enum:UnitIntent section + `unitIntentId`/`unitIntentCount`),
`SampleSchema.kt` (VERSION 9 + OUTPUT_UNIT_INTENT + block consts), `SampleConfig.kt` (`controlUnitIntent`),
`LegalActionMasks.kt` + `UnitAutomation.kt` (`unitIntentMask` + instrumented ladder + `dispatchUnitIntent`),
`Featurizer.kt` (`orderedOwnUnits` + `mask_unit_intent`), `PolicyProvider.kt`/`RandomPolicy`/`RoutingPolicy`
(`chooseUnitIntentWithLogp`), `MaskedChoice.kt` (`maskedLogSoftmax`), `DataPlaneHooks.kt` (deferred emission),
`OnnxPolicy.kt` (unit-intent head), `SelfPlayRunner.kt` (CLI positional). Python: `schema.py`, `contract.py`,
`model.py` (6-tuple head), `export_onnx.py`, `dataset.py`, `train.py` (joint PG + BC both heads + unit-KL +
stackers), `run_loop.py`/`analyze_v5.py` (flags). Tests: `UnitIntentCodeTest.kt`, `MaskedLogSoftmaxTest.kt`,
`test_v8_unit_intent.py`. Drivers: `run_v8kl.sh`, `run_v8bc.sh`, `analyze_v8.py`.
