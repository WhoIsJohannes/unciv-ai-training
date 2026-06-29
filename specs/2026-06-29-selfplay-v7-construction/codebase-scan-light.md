# Light Codebase Scan — selfplay-v7-construction (prereq self-check)

**Result: NO DRIFT.** All 15 file:line claims from the prompt verified against the worktree
(branch `selfplay-v7-construction` @ 27feebe59). Minor line offsets noted; substance confirmed.

## A — v6/v5 machinery (present)
- `SampleSchema.VERSION = 4` at **SampleSchema.kt:27** (prompt said :23 — offset only). `MODELED_HEADS = listOf("tech","policy")` at :68. `BLOCK_BEHAVIOR_LOGP = "behavior_logp"` at :133. Blocks tagged `BlockKind.FIXED|VARIABLE`.
- PPO stored old_logp: `train.py:137-210`; importance ratio `logratio = (logp - old_logp[...]).clamp(-20,20)` at :210/:247. Stored old_logp = per-step SUM of head behavior logps.
- `--replay-window` (default **4**) at `run_loop.py:251`; `replay = deque(maxlen=...)` at :289.
- v5 continual: `_load_warm()` (net+opt from `ckpt_round_{n-1}.pt`/`opt_round_{n-1}.pt`) at `run_loop.py:80-109`; `warm_net,warm_opt` carried at :285/:354. Micro-batch in `train.py:120-167` (`forward_chunk_fn`, `micro_batch_steps`).

## B — control seam
- `DataPlaneHooks.handleCivTurn` :102 → `chooseAndApply` :107 → returns `Pair<FloatArray,FloatArray>` (actions, behaviorLogp). Tech decision :120-125 (`chooseIndexWithLogp("tech",...)` → `civ.tech.techsToResearch.add`); policy :128-135. `recordStep(civ,obs,actions,behaviorLogp,turn)` at :108; recorder body `ShardRecorder` :219-236 (uses `BlockKind.FIXED`).
- `onCivTurn?.invoke(civInfo)` at **NextTurnAutomation.kt:53**, BEFORE `automateCities` :105. `adoptPolicy` guarded `if (DataPlaneHooks.controls(civInfo)) return` at :302.
- `DataPlaneHooks.controls(civ)` :99 = `installedPolicy != null && civ.isMajorCiv() && !civ.isSpectator() && games.containsKey(civ.gameInfo)`.

## C — per-city construction mask
- `LegalActionMasks.constructionMask(city,vocab)` **:45-56** → `BooleanArray(vocab.buildingCount + vocab.unitCount)`; buildings at idx, units at `buildingCount + unitIdx`.
- `Featurizer.kt:107-110`: `constrW = vocab.buildingCount + vocab.unitCount`; `construction = FloatArray(ownCityList.size * constrW)`; row i = mask for `ownCityList[i]`.
- **own_cities order (Featurizer.kt:61): `val ownCityList = x.cities.sortedBy { it.id }`** ← per-city decision loop MUST reuse this exact order.

## D — heuristic to override
- `ConstructionAutomation.chooseNextConstruction()` **:116-157**; early guard `if (getCurrentConstruction() !is PerpetualConstruction) return`; ends with `cityConstructions.setCurrentConstruction(chosenConstruction.name)`. ← add controlled-civ + pre-filled guard here.
- `CityConstructions.kt:74-85`: `setCurrentConstruction(value)` → `if (queue.isEmpty()) add(value) else queue[0]=value`; `var constructionQueue = ArrayList<String>(...)`. **Pre-fill = `setCurrentConstruction(name)` (or queue[0]=name).**
- `Vocab.constructionId(idx)` **DOES NOT EXIST** — Vocab has forward `constructionCode(name)` only; must add inverse (build reverse map: idx<buildingCount → buildingName(idx); else unitName(idx-buildingCount)).
- No turn-start clobber: `validateConstructionQueue()` runs in `constructIfEnough()`/`endTurn()`, not before automation → pre-fill respected.

## E — net + python
- `RichPolicyValueNet` (model.py:79-113) — prompt called it `StructuredPolicyValueNet`; actual class is **`RichPolicyValueNet`** (the "structured"/"rich" variant). `_TokenEncoder` per token type in `encoders` ModuleDict; own_cities encoded → masked_pool into trunk. **Un-pooled per-city embeddings exist inside the encoder but are NOT currently exposed** → construction head must read them out.
- `export_onnx.py`: blind `_PolicyOnly` (:44), rich `_RichPolicyOnly` (:59) — both return `(tech, policy)`, value dropped. Dynamic batch axis (:83 blind, :135-143 rich). ← add `construction_logits` output + dynamic city axis.
- `OnnxPolicy.kt` (desktop/src/com/unciv/app/desktop) — `logitsFor()` memoized per `(gameId|civID|turn)`; reads `OUTPUT_TECH`/`OUTPUT_POLICY` (:130-131). ← add `OUTPUT_CONSTRUCTION` + per-city indexing.
- `dataset.py:189-198`: actions + `behavior_logp[0]`(tech)/`[1]`(policy) per step. ← add construction_action/logp variable load.
- `python/unciv_dataplane/schema.py:20`: `SCHEMA_VERSION = 4`. Reader `reader.py:105-140` `_decode_blocks()` is descriptor-generic (`kind="var"` reads `<H` count then perItem items) → **two new VARIABLE f32 blocks round-trip with no reader change.**

## Tests to extend
- `tests/.../FairnessAndDeterminismTests.kt` — `determinism_sameStateSameBytes()`, `maskParity_techMatchesEngine()`, `maskParity_policyMatchesEngine()` → add per-city construction parity.
- `tests/.../OnnxPolicyLegalityTest.kt` — masked-choice legality → add per-city construction legality.
- `tests/.../MaskedChoiceLogpTest.kt` — logp recording.

## Naming note for plan
- prompt's `StructuredPolicyValueNet` == actual `RichPolicyValueNet`; prompt's `schema.py` == `python/unciv_dataplane/schema.py` (NOT unciv_train).
