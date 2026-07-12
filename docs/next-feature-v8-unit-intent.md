# Next feature — v8: unit control via a per-unit INTENT head

Paste the block below into `/feature` on the new box. Design summary: the net picks, per unit, **which
existing `UnitAutomation` behavior to run** (an "intent"), not raw tile movement — so pathfinding stays
heuristic (throughput preserved) and it reuses v7's per-entity + BC-clone + KL-leash machinery.

**Two things to watch** (this is the hardest feature; the prompt was grounded by direct reads, not a
deep multi-agent scan): (1) the per-unit applicability **mask** is the fiddly part — some intents'
true preconditions involve pathfinding, which can't go in the mask, so use cheap predicates + a
dispatch fallback and let the `/feature` scan pin the predicate set down; (2) **expect a first-pass
negative like v7** — a bigger action space collapses harder, so BC-clone + KL-leash (and `--unit-kl-coef`
tuning) are load-bearing, and acceptance #4 treats a negative-then-recovered arc as the expected path.
If you want the maximally-grounded version, ask Claude to "use a workflow" to deep-scan it first.

```
/feature Unit control via a per-unit INTENT head (v8). Unciv self-play RL. Repo /home/johannes/projects/unciv-ai-training, branch master (construction win is now merged in: per-entity heads + BC-clone + KL-leash all present).

== CONTEXT / GOAL ==
The learned policy controls tech + policy + construction and beats random (v7: construction-control 52.3% vs 37.5% off, +14.8pp, n=8, crosses 50%). The last big high-leverage frontier is UNITS — combat/expansion, the core of Civ, currently 100% heuristic for both learner and opponent. v8 gives the net a per-unit INTENT head: for each of a controlled civ's units, choose WHICH existing automation behavior to run — NOT raw tile movement. The chosen behavior's sub-routine executes (pathfinding stays heuristic → throughput preserved). This is another per-entity categorical head (per UNIT, like construction is per CITY) and MUST use the v7 recipe (BC-clone the heuristic + KL-leash), because a bigger action space mode-collapses from scratch even worse than construction did.

== THE INTENT VOCABULARY (from UnitAutomation.automateUnitMoves, UnitAutomation.kt:34-117) ==
The heuristic is an ordered `if (tryX(unit)) return` ladder. Each rung is a candidate INTENT; the FIRST rung that returns true is the intent the unit executed. Define a stable MILITARY intent enum from the rungs (dedupe the repeated tryHealUnit health-threshold calls into one HEAL): HEAL(tryHealUnit/tryRetreat), UPGRADE(tryUpgradeUnit), ACCOMPANY(tryAccompanySettlerOrGreatPerson), GO_TO_RUIN(tryGoToRuin), DEFEND_SIEGED_CITY(tryHeadTowardsOurSiegedCity), ATTACK(tryAttacking/BattleHelper.tryDisembarkUnitToAttackPosition), RETAKE_CITY(tryTakeBackCapturedCity), ADVANCE_ENEMY_CITY(HeadTowardsEnemyCityAutomation.tryHeadTowardsEnemyCity), ATTACK_ENCAMPMENT(tryHeadTowardsEncampment), GARRISON(tryGarrisoningLandUnit), ADVANCE_CLOSE_ENEMY(tryAdvanceTowardsCloseEnemy), PREPARE(tryPrepare), EXPLORE(tryExplore), FOG_BUST(tryFogBust). v1 scope = LAND MILITARY units only. Civilian units (CivilianUnitAutomation, UnitAutomation.kt:41-56), air/nuke units (:59-71), and the civilian-ability-military branch stay 100% heuristic (recorded intent = HEURISTIC/none, head = -1) — do NOT model them in v8.

== PREREQUISITE SELF-CHECK (print evidence; abort on drift) ==
- v7 on master: per-entity construction control + recording in DataPlaneHooks (chooseAndApply per-city loop + per-entity variable blocks); BC-clone (--bc-pretrain-dir, --bc-epochs, run_v74bc.sh) + KL-to-clone leash (--construction-kl-coef) in model.py/train.py/run_loop.py; SampleSchema.VERSION (current) + MODELED_HEADS incl construction.
- Unit seam: NextTurnAutomation.automateUnits (NextTurnAutomation.kt:427) → UnitAutomation.automateUnitMoves (UnitAutomation.kt:34). DataPlaneHooks.controls(civ) guard exists.
- own_units tokens already emitted by the featurizer (per-unit token set) — the intent head conditions on these, like the construction head conditions on own_cities.

== DESIGN DECISIONS (decisive — MIRROR construction's per-entity + BC+KL pattern, applied to units) ==

(A) INTENT ENUM + per-unit MASK. Define the military intent enum above (stable order, in SampleSchema). Build a per-unit applicability mask: an intent is legal only when cheap preconditions pass (ATTACK ⇐ an attackable enemy in range; HEAL ⇐ damaged; ADVANCE_ENEMY_CITY ⇐ at war & an enemy city known; GARRISON ⇐ near own city; EXPLORE/FOG_BUST ⇐ unexplored nearby; UPGRADE ⇐ upgrade available; etc.). Keep the predicates CHEAP (no full pathfinding in the mask — reuse the sub-routines' own early-out checks where cheap; if a precondition is expensive, include the intent unconditionally and let dispatch fall back). Featurizer emits mask_unit_intent as a VARIABLE block, one row per own_units token, aligned to own_units order (mirror mask_construction / Featurizer per-city construction).

(B) DISPATCH. Intercept automateUnitMoves for controlled land-military units (guard: DataPlaneHooks.controls(unit.civ) && land military). The net picks an intent from that unit's legal mask → invoke ONLY that intent's sub-routine (tryX). If it returns false (situation changed / not actually doable), FALL BACK to the full heuristic ladder and record the fallback intent (recorded == executed). Civilian/air/nuke units → unchanged heuristic path. Pathfinding lives INSIDE the sub-routines — v8 adds NO movement/target logic.

(C) BC LABEL (instrument the heuristic). Wrap the automateUnitMoves ladder so that, for each unit, the index of the FIRST tryX that returns true is captured as the heuristic's intent (a single instrumentation of the ladder; keep behavior byte-identical when uncontrolled). This is the BC-clone target — mirror how construction BC-clones the heuristic's city choice. (The v7 finding: from-scratch collapses; cloning the ~random-level heuristic then leashed-finetuning is what works. Expect the same here, harder.)

(D) NET + ONNX. Add a per-unit intent head on the own_units token embeddings (mirror the per-city construction head on own_cities tokens): logits [B, Nunits, nIntents], masked by mask_unit_intent. Export adds unit_intent_logits (dynamic unit axis; value head still training-only/dropped). OnnxPolicy: one memoized inference per civ-turn → index the row for each unit; new PolicyProvider.chooseUnitIntentWithLogp(civ, unit, unitRow, mask, turn) (default uniform-legal for RandomPolicy; single RNG draw via MaskedChoice, byte-identical replay).

(E) RECORD. Per-civ-turn step gains two VARIABLE blocks aligned to own_units: unit_intent_action (chosen intent idx, -1 = not a controlled land-military unit / no decision) and unit_intent_logp. Reuse construction's per-entity recording machinery. Bump SampleSchema.VERSION + schema.py in lockstep (old shards refuse; perishable). Reader is layout-generic — no reader change.

(F) TRAIN. The per-step policy-gradient logp adds Σ_units _masked_logp(unit_intent) (only controlled units; -1 → 0), sharing the per-step GAE advantage (mirror construction). BC-clone pretrain includes the unit-intent head (target = heuristic intent from C). KL-to-clone leash extends to the unit-intent head via a new --unit-kl-coef (analog of --construction-kl-coef, default 0.5). v6 replay stored-old_logp includes unit-intent logp. Keep the PPO clip/value/entropy/GAE math otherwise verbatim.

(G) CONTROL FLAGS. --control-units {on,off} (default off; on for v8 arms). off ⇒ units fully heuristic, head inert, reproduces v7 within fp tolerance (assert). Compose with --control-construction, --bc-pretrain-dir, --bc-epochs, --unit-kl-coef.

== EXPERIMENT (multi-seed — the variance lesson is now law) ==
Medium, ≥8 gen-seeds, PAIRED per seed (same seed, control-units off vs on-with-BC+KL), 200-game ceilings, paired t-test (reuse the v7 n=8 paired-diff analysis). Best v7 config as the base (tech+policy+construction BC+KL). Report: unit-control uplift Δpp with t/p, and absolute win-rate vs 50%. A single-seed number is NOT acceptable (identical code swings 8.8↔41.7% by seed).

== ACCEPTANCE CRITERIA ==
1. LEGALITY: every executed intent is in that unit's legal mask; recorded intent == executed (incl. fallback); zero illegal across eval. Uncontrolled unit types stay heuristic (asserted).
2. NO-OP: --control-units off reproduces v7 within fp tolerance (head inert; ladder byte-identical when uncontrolled).
3. PARITY: JVM per-unit intent logits == Python reference (atol 1e-4).
4. EFFECT (multi-seed): control-units on (BC+KL) beats off, paired t-test p<0.05 over ≥8 seeds; report Δpp and absolute vs 50%. If it does NOT beat off, report plainly (a real negative, like v7's first pass) and whether BC/KL tuning (epochs, --unit-kl-coef) recovers it.
5. SCHEMA VERSION bumped lockstep; old shards refuse; ONNX MODEL contract grows by exactly unit_intent_logits (documented); bridge otherwise unchanged.
6. THROUGHPUT: the net now runs per-UNIT (many/turn) → more inference; measure ms/decision + turns/s; stay ≥70% of heuristic baseline (bench-onnx). If per-unit inference blows the budget, batch all a civ's units into one forward.
7. Determinism (single RNG draw/decision), terminal-only ±1 reward, tech/policy/construction heads unchanged.

== FROZEN / NON-GOALS ==
Pathfinding stays 100% heuristic (intents dispatch to existing sub-routines; v8 adds NO raw movement/tile-target logic — that's a far harder future step). No civilian/air/nuke intent heads (v1 land-military only). No encoder rewrite (intent head reads existing own_units embeddings). No reward shaping. tech/policy/construction heads + the PPO math frozen. Multi-seed is mandatory — no single-seed ACs.

== DELIVERABLES ==
- Intent enum + per-unit mask (SampleSchema + LegalActionMasks + Featurizer mask_unit_intent, aligned to own_units).
- automateUnitMoves instrumentation (BC label = first-firing rung) + controlled-civ dispatch (net intent → sub-routine, fallback recorded).
- PolicyProvider.chooseUnitIntentWithLogp + RandomPolicy default + OnnxPolicy per-unit impl.
- model.py per-unit intent head + export_onnx unit_intent_logits; dataset.py + train.py intent logp + BC target + --unit-kl-coef; run_loop --control-units.
- run_v8 driver (paired off vs on-BC+KL, ≥8 seeds) + paired-t analysis; RESULTS.md with Δpp/t/p + vs-50%.
- parity + no-op + legality tests. Build (gradle + python) green; no-op/parity/legality hold before the multi-seed run.
```
