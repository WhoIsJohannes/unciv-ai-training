# Test Spec — selfplay-v7-construction (test_mode=integration)

Highest-fidelity path for this RL infra: JVM gradle tests (engine parity/legality/determinism) +
pytest (schema round-trip, no-op, dataset). No agentic/browser tests (pure backend). RED-first: the
Python `test_v7_construction.py` is RED now (SCHEMA_VERSION==4); the Kotlin/trainer tests below are
written in the build against the new APIs.

## AC#1 — LEGALITY (Kotlin: OnnxPolicyLegalityTest / FairnessAndDeterminismTests)
- GIVEN a controlled civ (controlConstruction=ON) with ≥1 city that would choose this turn,
  WHEN chooseAndApply runs, THEN every pre-filled `city.cityConstructions.currentConstructionName()`
  is in `LegalActionMasks.constructionMask(city,vocab)` (idx mapped via `vocab.constructionId`).
- GIVEN the recorded step, THEN `construction_action[i]` (0-indexed mask idx) maps via
  `vocab.constructionId(idx)` to EXACTLY the construction queued for `orderedOwnCities(civ)[i]`
  (recorded == applied), per city; cities not deciding record −1 and the heuristic is untouched.
- ACROSS an eval run: zero illegal constructions applied.

## AC#2 — PARITY (JVM per-city construction logits == Python reference, atol 1e-4)
- GIVEN a fixed Observation (committed fixture), WHEN the JVM construction head (via ONNX) and the
  Python `StructuredPolicyValueNet` construction head run on it, THEN per-city `construction_logits`
  match elementwise within atol 1e-4 (extend the existing tech/policy logit-parity harness to the
  per-city output). PLUS `constructionId(idx)` round-trips: for every legal mask idx, the name maps
  back to that same idx (0-indexed mask space, NOT 1-indexed constructionCode).

## AC#3 — EFFECT (the experiment; analyze_v7.py)
- Per rung: two-proportion ONE-SIDED z-test H1: winrate(ON) > winrate(OFF), p<0.05 = directional
  proof (ship criterion D-C5). Draws/timeouts/crashes = non-win; 200 fixed denominator.
- Each arm's winrate vs the 50% break-even (z/p), stated plainly (crosses or not — milestone, not gate).

## AC#4 — SCHEMA (pytest: test_v7_construction.py — RED now)
- `SCHEMA_VERSION == 5` (Python) lockstep with Kotlin `SampleSchema.VERSION == 5`.
- A v5 shard carrying `construction_action`/`construction_logp` (perItem=1, var f32) round-trips with
  NO reader.py change (descriptor-generic). A v4 shard refuses (`expect_compatible` raises).

## AC#5 — NO-OP (pytest trainer test, added in build; mirrors test_replay_noop.py)
- **DETERMINISTIC ORACLE (the gate — PR1):** GIVEN a trajectory whose construction actions are all −1
  (OFF / no decision), WHEN `train_actor_critic_structured(..., behavior_logp=True)` runs with vs without
  the construction summand wired, THEN the construction logp summand ≡ 0 and the resulting weights are
  bit-identical (max|Δw| < 1e-6) ⇒ OFF reproduces v6 (and v5 at K=1).
- **CONFIRMATORY (not a gate):** the OFF-arm win-rate curve matches v6 within fp tolerance — a weaker
  stochastic signal, reported but not the pass/fail oracle.

## AC#1b — ON-ARM VALIDITY (PR2 — analyze_v7 + recorder counters)
- The recorder counts construction-fallback events (idx<0 / illegal-after-mask / NaN / missing ONNX
  output). The ON arms MUST report ≈0 fallbacks; analyze_v7 asserts/flags a contaminated ON arm
  (heuristic silently substituting for the policy invalidates the ON-vs-OFF delta).

## AC#2b — ONNX/Vocab DIM CROSS-CHECK (PR3 — fail-loud)
- At ONNX load / first inference, assert `construction_logits` last-dim == `vocab.buildingCount +
  vocab.unitCount`; mismatch raises loudly (mirror `test_contract_failloud`) — no silent corruption.

## AC#6 — THROUGHPUT (bench-onnx)
- Per-city construction head inference cost measured via the existing bench-onnx; ON-arm sample/sec
  ≥ 70% of the heuristic baseline (single memoized forward per civ-turn emits all city rows — no
  O(cities) inference).

## Determinism (FairnessAndDeterminismTests)
- Single RNG draw per construction decision (MaskedChoice.chooseWithLogp); identical order
  (orderedOwnCities); shards WITH construction blocks byte-identical on replay of identical state.

## Invariants asserted (NON-GOALS)
- unit movement, promotion, great-person, diplomatic-vote remain heuristic (no new modeled head;
  MASK_HEADS unchanged; actions[2],[3] stay −1).
