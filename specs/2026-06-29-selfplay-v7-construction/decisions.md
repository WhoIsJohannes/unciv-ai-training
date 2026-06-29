# Decisions — selfplay-v7-construction

## Discovery findings (Phase 1)

### D-disc-1: Routed invariant docs are for a different project (noted, not blocking)
`~/.claude/feature-workflow-internal/docs/invariants/*` describe the **Mentiora** stack
(Next.js + FastAPI + Prisma + Stagehand agent tests). This repo is **Unciv** — a Kotlin/libGDX
game engine + a Python RL training package (`python/unciv_train/`). The generic invariant docs
do **not** govern this feature. The authoritative constraints are:
- the user's explicit **FROZEN / NON-GOALS** and **ACCEPTANCE CRITERIA** in the prompt, and
- the established v5 (continual) / v6 (replay-buffer) code patterns in the repo.
Testing philosophy ("test what matters") maps here to **parity / no-op / legality / determinism**
assertions, not browser/agent tests.

### D-disc-2: The Medium experiment is a long, resumable background batch
`python/run_v6.sh`: 4 arms run **sequentially** (CPU oversubscription if parallel), each = v5 continual
config (structured, Medium, 16 rounds, gen 16 / eval 80, turn-cap 250), `--resume`-able per round, then
a per-arm **200-game ceiling eval** @ eval-seed 4242424 + z-tests vs the fixed blind baseline (58/200 =
28.9%). This is multi-hour compute. v7 collapses to **2 arms** (construction OFF == v6 tech+policy; ON ==
v7) at `--replay-window 4`, the only changed axis being `--control-construction`.
Acceptance ordering (from the prompt): **parity + no-op + legality green BEFORE the Medium run.**

### D-disc-3: Fork topology — do NOT `git pull`
`origin` = yairm210/Unciv (upstream public repo); `fork` = WhoIsJohannes/unciv-ai-training (self-play
work). Local `master` is 48 ahead / 1 behind origin. The feature branch was cut from **local master**
with no pull (pulling would merge unrelated upstream Unciv changes mid-feature).

## Design decisions (from the prompt — marked "decisive", carried verbatim into the plan)
A) Control: pre-fill each city's `constructionQueue[0]` at the civ-turn hook, after tech/policy.
B) Seam: `PolicyProvider.chooseConstructionWithLogp(civ, city, cityRow, legalMask, turn)`.
C) Net: per-city construction head on UN-pooled own_cities token embeddings → logits [B,Ncities,constrW].
D) Record: two VARIABLE blocks (`construction_action`, `construction_logp`) aligned to own_cities; SCHEMA 4→5.
E) Train: per-step logp += Σ_cities masked_logp(construction); shares the per-step GAE advantage; replay old_logp includes construction.
F) No-op: `--control-construction {on,off}`, default ON; OFF reproduces v6 (and v5 at K=1).

## Phase 2 JOIN — intake council triage (31 findings, 1 round, research-blind)

Branches: research ✓ · council ✓ (8 lenses) · deep-scan ✓. research_seen_by_rounds = none (R1 was research-blind; stopped at R1 — substantive, diminishing returns).

**🔴 Critical (10) — disposition:**
- C1 logp summing vs stored old_logp scalar / C10 replay city-set frozen → **FOLD (design E)**: at generation, stored `behavior_logp`/old_logp = sum over heads + Σ_cities construction; live logp recomputes over the SAME recorded `construction_action` per city → ratio is a well-defined per-step scalar. Recorded city set+order is frozen in the sample.
- C2 own_cities ordering cross-system contract → **FOLD**: centralize `x.cities.sortedBy{it.id}` (capped at maxOwnCities) in ONE shared fn reused by featurizer + decision loop + recorder; parity/determinism test asserts row alignment.
- C3 schema v5 rollback / mixed replay → **FOLD**: v4 shards refuse to load (perishable, by design); v7 starts replay FRESH (no v6 carryover). Documented.
- C4 ONNX fallback when construction output missing/bad → **FOLD**: OnnxPolicy → heuristic when OUTPUT_CONSTRUCTION absent / NaN / wrong dim / no legal after mask (record −1). This IS the OFF/no-op path too.
- C5 **ship decision on NEGATIVE experiment result → USER DECISION (asked at JOIN).**
- C6 parity/no-op/legality not defined → **FOLD**: Step 11 test spec defines exact names/assertions/tolerances.
- C7 variable city-count edge cases → **FOLD**: 0 cities → empty block; decide only when city would choose (queue empty / on Perpetual) else −1; empty mask → −1; capture-between-obs-apply impossible (same synchronous handleCivTurn); invalid reverse → guard canConstruct.
- C8 O(N) ONNX inference → **FOLD**: ONE memoized forward per (game,civ,turn) emits all city rows.
- C9 GAE advantage scales with city count → **FOLD (deliberate)**: shared per-step advantage (research-confirmed standard A2C/AlphaStar); watch ratio/clip-fraction telemetry on the ON arm.

**🟡 Worth considering (majors) — FOLD:** M11 keep PolicyProvider seam minimal (one method, default uniform-legal, no class hierarchy — per-entity signature genuinely differs from civ-level, so justified); M12 schema bump required (layout changed); M13 construction is a per-entity block, not a MASK_HEADS slot; M14 no-op = zero-summand equivalence proof (OFF records 0 construction decisions → summand≡0 → identical to v6); M15/M18 run_v7.sh mirrors v6 (resumable, --resume, heartbeat, per-round ckpt); M16/M26/M27 head is a small MLP, bench-onnx measures latency (AC#6), maxOwnCities bounds it; M17 add per-city debug logs (chosen id, legality); M19 copy FROZEN/NON-GOALS into plan explicitly; M21 two-proportion z-test, one-sided ON>OFF, draws/timeouts=non-win, 200 fixed denom (analyze_v5/v6 convention); M22 throughput via bench-onnx (samples/sec vs heuristic ≥70%); M23 fixtures via TestGame; M28 OFF==v6 via zero-summand; M29 decide-when-idle mirrors heuristic cadence (correct, intentional); M30 covered by C7.

**⚪ Low / OUT OF SCOPE — single-trust local training (no adversary):** M24 ONNX file signing, M25 replay data-poisoning, m31 reader u16 bound. Self-generated data, local box; documented as out-of-scope threat model (the `<H` bound is already buffer-safe).

**Net:** 30/31 folded into the plan; ONE genuine user decision (C5).

### D-C5 (USER): Ship criterion = directional proof, not 50%
Ship the per-entity construction machinery + RESULTS as long as we can PROVE it moves the
win-rate in the right direction — i.e. **construction-ON beats construction-OFF at p<0.05**
within at least one rung — EVEN IF it doesn't cross the 50% break-even. Crossing 50% is a
**reported milestone, not a ship gate**. Default `--control-construction = ON` if the directional
win is proven (else OFF; the infra still merges for reuse by the next promotion/GP/vote heads, but
without a "beneficial" claim). A truly null/negative result (ON ≤ OFF, not significant) is reported
honestly and does not get the directional-success claim.

## Step 11 — plan council triage (31 findings: 5 critical, 21 major)
**Folded refinements (6 actionable — sharpen plan + tests; none invalidate design):**
- PR1 (crit "no-op not a deterministic oracle" + major "conflates 2 claims): AC#5 gate = the
  DETERMINISTIC zero-summand bit-identical-weights test (max|Δw|<1e-6); the v6 training-CURVE match is
  CONFIRMATORY only, not a gate.
- PR2 (major "ONNX fallback silently invalidates ON arm" + "heuristic fallback mixes policies"): COUNT
  construction-fallback events; the ON arms must report ~0 fallbacks (assert), else the ON-vs-OFF
  comparison is contaminated → invalid. Guard-failure records −1 (policy abstains; heuristic builds; train
  contributes 0 — no gradient mixing).
- PR3 (major "no cross-validation ONNX dim vs constrW"): fail-loud assert at model load/first inference
  that `construction_logits` width == `vocab.buildingCount+vocab.unitCount` (mirror test_contract_failloud).
- PR4 (major "null-pointer: constructionId null but canConstruct unconditional"): explicit null-guard —
  `constructionId(idx) ?: record −1` BEFORE any `canConstruct` call.
- PR5 (crit "unbudgeted 42% compute from throughput regression"): bench-onnx is a PRE-run GATE (measure
  ON-arm throughput BEFORE the multi-hour batch; a <70% head is fixed before launch, not discovered mid-run).
- PR6 (crit "schema v5 rollback/operating plan"): v7 runs in a FRESH OUT_ROOT, replay starts EMPTY, NO
  cross-schema ckpt resume from v6 dirs; rollback = revert branch (v5 shards/replay are perishable & separate).

**Already-covered / intentional / out-of-scope:**
- crit "PyTorch pickle RCE via torch.load" → OUT OF SCOPE (single-trust local training; pre-existing v5/v6
  behavior, v7 adds no new load path). Note: `weights_only=True` is a cheap future hardening, not v7 scope.
- "f32 storage for discrete actions" → intentional (matches existing f32 `actions`/`behavior_logp` FIXED blocks).
- "padding to max Ncities wastes compute" → acceptable for Medium; mask zeros padded rows (could pack later).
- "decide-only-when-idle limits expressiveness" → deliberate first-cut (mirrors heuristic cadence, won't
  interrupt a mid-build wonder); preemption is a future extension, not v7.
- "shared advantage over-credits idle cities" → false: idle/heuristic cities record −1 → contribute 0 logp.
- "city-count log-ratio variance biases the comparison" → training-stability risk (watch clip-fraction), NOT a
  measurement bias (the 200-game eval is deterministic given the trained net; both arms use identical eval).
- SLO/incident/timebox majors → run_v7.sh mirrors v6 (resumable --resume, heartbeat, per-round ckpt); user
  chose drive-to-completion.
- "golden analyzer / bench contract / fallback / ordering fixture tests" → defined in agent-test-spec.md; written in build.

## Build correction (D-build-1): per-turn construction control (the perpetual-gate was inert)
The plan's "decide only when the city is idle on a PerpetualConstruction" gate (mirroring the heuristic
cadence) fired **~never** at the onCivTurn hook — the heuristic keeps every city mid-build, so a gen-ON
smoke recorded **0 construction decisions** (the lever was inert; exactly council M29's concern, now
proven). FIX: the policy DRIVES each non-puppet city's production EVERY turn (pre-fills queue[0]). Safe
because Unciv stores accumulated production per construction (`inProgressConstructions`), so re-picking is
non-destructive and the reward shapes coherent choices. After the fix: a gen-ON smoke recorded 1106
decisions across 474 steps, **0 illegal** (AC#1). OFF unchanged (loop gated by controlConstruction → all −1
→ no-op intact). This is a 🔄 REFRAMED deviation from design (A)'s "mirror heuristic cadence" — necessary
for the lever to function; the per-entity infra + everything else is unchanged.
