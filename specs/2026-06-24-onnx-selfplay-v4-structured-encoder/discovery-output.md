# Phase 1 Discovery — Output

**Feature:** v4 — Structured Encoder (hex GNN + categorical embeddings + self/cross-attention)
replaces the v3 rich-critic's masked mean+max pool, to fix the Medium regression (v3 rich-critic
14.7% vs blind-critic 28.9%, z=−3.48).

- **Mode:** BUILD · **Size:** L · **Model:** Opus 4.8 (passes Opus-minimum gate)
- **Worktree:** `/Users/j/Unciv-onnx-selfplay-loop` · **Branch:** `onnx-selfplay-loop` · **Base:** master `5ac0a3cf6`
- **Context category:** backend (cross-language contract bridge + architecture-heavy ML pipeline)
- **Domain preset (Phase 2 hint):** data-pipeline → `practitioner` (contract/numerical correctness),
  `cost_efficiency` (D8 throughput gate). `data_privacy_legal` N/A (offline RL, no user data).
- **Invariant docs:** loaded code-quality/architecture/testing/features/security — **noted these
  describe a different project (Mentiora)**; carrying forward only transferable principles
  (small focused files, Python type hints, behavior-over-coverage tests, council review). Real
  conventions come from the existing v2/v3 Unciv code.

## Pre-feature decisions (resolved by user)

1. **v4 vs recorded diagnosis** — my saved memory + v2 RESULTS.md say the Medium regression is
   undertraining (from-scratch-per-round training), and that adding encoder capacity before fixing
   that risks making Medium worse. **User chose: proceed with v4 as written**, accepting the
   training-regime confound. Weight-carryover remains the recommended follow-up if v4's Medium
   result is ambiguous. (Memory updated: do not re-flag on this branch.)
2. **Worktree loss** — the spec's named worktree + branch (and `self-play-data-plane`) were
   `git worktree remove`d by an external process mid-session. No code lost (master holds the v3
   baseline). **User chose: recreate from master and continue.** Cause unknown — watch for
   recurrence; commit early.

## Light scan summary

See `codebase-scan-light.md`. Pipeline mapped end-to-end (Kotlin emit → shard → Python read →
model → ONNX export → JVM bridge); plan's cited line numbers verified (no material drift). Key
correction: **`SimBenchmark` already exists** (`:desktop:simBench`, measures heuristic baseline
turns/s) → D8 extends it rather than creating it.

## Open questions for Phase 2 design (Claude-resolvable, no user input needed)

- Signed-coord encoding for D1.1: i16 side-tensor vs biased-u8 (spec prefers a signed side-tensor;
  `coerceAtMost(255)` would corrupt negatives). Decide in design.
- GNN message-passing op choice for ONNX opset-17 exportability (gather/index_select/scatter_add) —
  validate export on the *small* rung before scaling (pre-registered export risk).
- Confirm `maxCivTokens` (slot table size = `maxCivTokens+2` assumed = 42).

## Round 1 Q&A

Iterations: 0 (scope exhaustively defined by prompt + scan; the only real forks were the two
pre-feature decisions above, already resolved). Remaining unknowns are design-internal, not scope.
