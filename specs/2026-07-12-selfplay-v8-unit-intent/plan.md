# v8 — Unit control via a per-unit INTENT head

Mirrors the v7 per-**city** construction control (per-entity categorical head + BC-clone + KL-leash),
applied per-**unit**. The net picks, for each controlled land-military unit, **which existing automation
behaviour to run** — not raw tile movement. Pathfinding stays 100% heuristic (intents dispatch to the
existing `tryX` sub-routines). v1 scope = **land military units only**.

Base config = the best v7 arm (tech+policy+construction, BC 120ep + KL 0.5).

## Prerequisite self-check (master) — GREEN, no drift

- Schema lockstep: `SampleSchema.VERSION = 8` ↔ `schema.py SCHEMA_VERSION = 8`.
- Per-entity construction control + recording: `DataPlaneHooks.chooseAndApply` per-city loop; `recordStep`
  VARIABLE blocks (`construction_action/_logp/_current`, `econ_city`).
- BC-clone: `bc_pretrain_construction`, `--bc-pretrain-dir/--bc-epochs`, `run_v74bc.sh`.
- KL-leash: `--construction-kl-coef`, `_construction_kl_loss`, frozen `bc_ref`, `run_v74kl.sh` (the +14.8pp win).
- `MODELED_HEADS = ["tech","policy"]` in both Kotlin + Python (construction is per-entity, NOT here — unit-intent follows the same rule).
- Unit seam: `NextTurnAutomation.automateUnits` → `UnitAutomation.automateUnitMoves`; `DataPlaneHooks.controls(civ)`.
- `own_units` tokens already emitted; `mask_promotion` is an existing per-unit VARIABLE mask precedent.

## The intent enum (14 land-military intents, stable order)

Derived from the `automateUnitMoves` land-military ladder (`UnitAutomation.kt:74-116`), first-firing rung =
the executed intent. Heal variants (rungs 38/78/81/100/105 + `tryRetreat`) fold into one `HEAL`.

| idx | intent | rung / sub-routine | mask precondition (cheap unless noted) |
|----|--------|--------------------|-----------------------------------------|
| 0 | HEAL | tryHealUnit / tryRetreat / wait-heal | `unit.health < 100` |
| 1 | UPGRADE | tryUpgradeUnit (human-only) | `getUnitsToUpgradeTo(unit).any()` (⇒ 0 for AI civs — inert) |
| 2 | ACCOMPANY | tryAccompanySettlerOrGreatPerson | unconditional-include (pathfinds immediately) |
| 3 | GO_TO_RUIN | tryGoToRuin | major civ && a ruin tile in `viewableTiles` |
| 4 | DEFEND_SIEGED_CITY | tryHeadTowardsOurSiegedCity | any own city with `health < maxHealth` |
| 5 | ATTACK | tryAttacking / tryDisembark…Attack | enemy military unit within ~5 tiles (cheap proxy) |
| 6 | RETAKE_CITY | tryTakeBackCapturedCity | a just-captured own city exists |
| 7 | ADVANCE_ENEMY_CITY | HeadTowardsEnemyCityAutomation | at war && has cities && an enemy city known |
| 8 | ATTACK_ENCAMPMENT | tryHeadTowardsEncampment | known barbarian encampment near cities && !SelfDestructs |
| 9 | GARRISON | tryGarrisoningLandUnit | land unit && a garrisonable own city near |
| 10 | ADVANCE_CLOSE_ENEMY | tryAdvanceTowardsCloseEnemy | unconditional-include (3-turn pathfind) |
| 11 | PREPARE | tryPrepare | a close hostile city (getNeighboringCitiesOfOtherCivs) |
| 12 | EXPLORE | tryExplore | unconditional-include (pathfinds immediately) |
| 13 | FOG_BUST | tryFogBust | `Automation.afraidOfBarbarians(civ)` && invisible tiles near |

Sentinel `-1` = not a controlled land-military unit / no rung fired (city-state `wander`).
"Unconditional-include" intents rely on dispatch fallback when not actually doable (kept cheap per spec).

## Architecture — DEFERRED per-civ-turn frame emission (the key deviation from construction)

The user chose **record the realized executed intent** (AC1 literal: `recorded == executed`, incl. fallback).
Unlike `construction_current` (a readable turn-start property), the heuristic's chosen rung and the realized
executed intent are only known **after** the ladder runs during `automateUnits` (act-time). So the per-civ-turn
frame must be emitted at **turn-end**, not turn-start.

```
onCivTurn (turn-start)  →  DataPlaneHooks.handleCivTurn(civ):
    obs = featurize(civ)                       # snapshot (incl. mask_unit_intent @ turn-start legality)
    decide + APPLY tech / policy / construction (unchanged, turn-start)
    for u in orderedOwnUnits(civ):             # controlled land-military only
        (X, logpVec) = policy.chooseUnitIntentWithLogp(...)   # sample intent, keep full masked log-prob vec
        pending.decided[u.id] = X ; pending.logpVec[u.id] = logpVec
    pending.obs / actions / construction* = snapshot          # DO NOT emit yet
    games[civ].pending[civ.id] = pending

automateUnitMoves(u)  (act-time, per unit):
    if controls(u.civ) && u is land-military && pending.decided[u.id] != null:
        if dispatchIntent(u, X): notifyRealized(u, X); return   # ran only tryX, succeeded
        # else: fall through to the instrumented full ladder (fallback)
    runInstrumentedLadder(u)                    # records first-firing rung via notifyRealized(u, Y)

onCivTurnEnd (turn-end)  →  DataPlaneHooks.handleCivTurnEnd(civ):
    for i, u in enumerate(pending.snapshotUnits):   # turn-start orderedOwnUnits, matched by MapUnit.id
        y = pending.realized[u.id]                  # X if tryX succeeded, else ladder rung, else -1
        unit_intent_action[i] = (y if controlled else -1)
        unit_intent_logp[i]   = (pending.logpVec[u.id][y] if y is legal in mask else 0)
        unit_intent_current[i]= heuristic first-firing rung when the full ladder ran, else -1  # BC label
    recordStep(... + unit_intent_action/_logp/_current)         # emit the frame now
    clear pending
```

- **Alignment**: `Featurizer.orderedOwnUnits(civ, cap)` sorts by the **stable** `MapUnit.id` (mirrors
  `orderedOwnCities` `sortedBy{it.id}`), replacing the current `sortedBy{currentTile.zeroBasedIndex}` (which
  moves as units move). own_units feeds permutation-invariant aggregation, so reorder is a no-op for the civ
  heads. Live units matched to their row by `id` at act-time; a unit truncated by cap or created mid-turn ⇒
  not in snapshot ⇒ abstain / not recorded.
- **No-op (AC2)**: OFF ⇒ the intercept branch is skipped; the ladder runs byte-identical (instrumentation is a
  side-effect only, never alters control flow); `unit_intent_action=-1` ⇒ zero PG summand ⇒ shared-weight
  oracle holds. `unit_intent_current` (heuristic label) is still captured in the OFF arm — that IS the BC gen.
- **Fallback (AC1)**: `unit_intent_action` = the realized executed intent; its logp = `log π(realized)` from
  the turn-start masked logits, or `-1/0` if the realized rung was not an offered legal intent. Fallback count
  is telemetry (contamination check), like construction's `constructionFallbacks`.
- **Throughput (AC6)**: one memoized forward per civ-turn already yields `[1, Nunits, nIntents]` for all units;
  per-unit is an index, no extra inference. Expect ≥70% baseline comfortably.

## Recorded blocks (all VARIABLE, aligned to own_units, perItem as noted)

- `mask_unit_intent` (u8, perItem=nIntents) — turn-start legal mask (Featurizer).
- `unit_intent_action` (f32, perItem=1) — realized executed intent idx; -1 if not a controlled land-military unit.
- `unit_intent_logp` (f32, perItem=1) — log π(realized); 0 where no decision.
- `unit_intent_current` (f32, perItem=1) — heuristic first-firing rung (BC target); -1 if none.

Schema bumped `VERSION 8→9` lockstep (Kotlin + Python); old shards refused. Reader is layout-generic (no change).

## ONNX contract growth (exactly one output)

`unit_intent_logits` `[batch, n_own_units(dynamic), nIntents]`, bound to the existing `n_own_units` input axis.
Value heads stay train-only (dropped). `META_UNIT_INTENT_WIDTH` for the JVM dim cross-check. Bridge otherwise
unchanged.

## Training (mirror construction verbatim)

- PG: `+ Σ_units _masked_logp(unit_intent)` (only controlled units; -1 → 0), sharing the per-step GAE advantage.
- BC: `unit_intent_current` is the clone target; `bc_pretrain` includes the unit-intent head (legal-masked CE).
- KL-leash: `--unit-kl-coef` (default 0.5) penalizes `KL(current ‖ frozen BC clone)` on the unit-intent head.
- Replay (v6) stored-old_logp includes unit-intent logp. PPO clip/value/entropy/GAE math otherwise verbatim.

## Control flags

`--control-unit-intent {on,off}` (default off) composes with `--control-construction`, `--bc-pretrain-dir`,
`--bc-epochs`, `--unit-kl-coef`. OFF ⇒ units fully heuristic, head inert, reproduces v7 within fp tolerance.

## Experiment (multi-seed — variance is law)

Medium, ≥8 gen-seeds, PAIRED per seed (control-units off vs on-with-BC+KL), 200-game ceilings, paired t-test
(`analyze_v73rep.py` clone). Base = best v7 (tech+policy+construction BC+KL). Report unit-control Δpp with t/p
and absolute win-rate vs 50%. Single-seed is not acceptable.

## Acceptance criteria

1. LEGALITY — every executed intent ∈ that unit's legal mask; `unit_intent_action[i]` == the intent actually
   applied to `orderedOwnUnits[i]` (incl. fallback = realized rung); 0 illegal across eval; uncontrolled unit
   types stay heuristic (asserted). Fallback count reported.
2. NO-OP — `--control-unit-intent off` reproduces v7 within fp tolerance (head inert; ladder byte-identical).
3. PARITY — JVM per-unit intent logits == Python reference (atol 1e-4).
4. EFFECT — control-units on (BC+KL) beats off, paired t p<0.05 over ≥8 seeds; report Δpp + absolute vs 50%.
   If not, report the negative plainly + whether BC/KL tuning recovers it.
5. SCHEMA — VERSION bumped lockstep; old shards refuse; ONNX grows by exactly `unit_intent_logits`; bridge else unchanged.
6. THROUGHPUT — ms/decision + turns/s ≥70% of heuristic baseline (bench-onnx).
7. Determinism (single RNG draw/decision via MaskedChoice), terminal-only ±1 reward, tech/policy/construction heads unchanged.

Gate: parity + no-op + legality green BEFORE the 8-seed run.

## Non-goals

Pathfinding stays heuristic (no raw movement/tile-target logic). No civilian/air/nuke intent heads (land-military
v1). No encoder rewrite. No reward shaping. tech/policy/construction heads + PPO math frozen. Multi-seed mandatory.
