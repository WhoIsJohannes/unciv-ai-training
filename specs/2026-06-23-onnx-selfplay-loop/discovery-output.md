# Discovery Output — ONNX policy bridge + minimal self-play training loop

- **Mode**: BUILD  **Size**: L
- **Base branch**: `self-play-data-plane` (the committed data plane; backed up on `fork`). PR target = base.
- **Feature (1-line)**: Close the self-play loop — a JVM `OnnxPolicy` that picks TECH/POLICY from the legal mask via an in-process ONNX net, plus headless GENERATE/EVAL entrypoints and a Python `unciv_train` package + round driver that trains on emitted trajectories and produces a win-rate-vs-RandomPolicy learning curve.
- **Context category**: backend + tooling (cross-language: Kotlin engine + Python training). Invariant docs are the generic install templates (an unrelated Next.js/FastAPI "Eval Platform") → only the transferable principles apply: small focused files (<300 lines), explicit-over-implicit, Python type hints, critical-path explainability. Repo specifics come from the data plane's own committed spec docs (`specs/2026-06-23-self-play-data-plane/`).
- **Domain preset**: data-pipeline (practitioner always-on; cost_efficiency) for the Phase 2 council bias.
- **Prior-work recall**: no hits (fresh in_repo event-log backend) — continued.

## Light scan summary
Full detail in `codebase-scan-light.md`. Key, source-verified facts:
1. `PolicyProvider.chooseIndex(head, civ, legalMask, turn): Int` + `actUnit(unit)`; `RandomPolicy` is the template (RNG via `defaultRngFor()` = `stateBasedRandom("dataplane-policy-$turn")`).
2. Heads order = `["tech","policy","greatPerson","diplomaticVote"]`; widths runtime/vocab-derived (confirm GnK tech/policy empirically — not hardcoded 80/70).
3. Net input = `concat(obs.block("global"), obs.block("acting_civ"))` from `Featurizer.observe(civ)` — pure/deterministic; the ONE input path for JVM + Python.
4. **Actions ARE recorded** today (`step.blocks["actions"]`, 4 floats). ✓
5. **Reward + terminal are NOT wired** (hardcoded 0). → must add per-civ terminal ±1 at game end + bump `SampleSchema.VERSION` 1→2 (mirror python; reader refuses old shards → regen). The linchpin new work.
6. **`binomialTest` already exists** in `Simulation.kt`; `numWins`/`steps` public → EVAL reuses it (extract to a public `SimStats` util).
7. Injection is ONE policy/run → add a **RoutingPolicy** wrapper (learner→onnx, others→random), no data-plane change. Pin fixed nations in the Tiny config so the learner identity (and `numWins[learner]`) is stable.
8. GENERATE/EVAL = new `SelfPlayRunner` desktop entrypoints using a FIXED Tiny 2-civ GnK config (not the randomized `ScenarioGenerator`); modeled on `DataPlaneGen.kt` + `ConsoleLauncher.kt` Tiny params.

## Invariant docs loaded
features.md, code-quality.md, architecture.md (generic templates — transferable principles only). testing.md noted for the test-heavy surface (parity/legality/determinism/provenance).

## Round-1 Q&A (decisions from the user)
1. **Win rule** → Raise the turn cap so games play to a **real victory**; score-leader is only an extreme-case tiebreaker at a hard cap. SAME definition for training reward (±1) and EVAL win-rate. → High default `maxTurns` (configurable); win = victory winner, else score-leader at hard cap.
2. **Loop budget** → **Make it configurable** (K rounds, gen-games, eval-games, turn-cap, threads, seed, learner nation, eval/sample mode all CLI+config knobs) with sensible balanced defaults chosen by me. First run measures throughput; defaults tunable without code changes.

## Decided by default-rigor (not user questions)
- VERSION bump 1→2 for terminal reward; lockstep python `SCHEMA_VERSION`; regen per round.
- RoutingPolicy wrapper; route by learner nation; pin nations in Tiny config.
- Extract `binomialTest`/`normalCdf`/`erf` → public `SimStats`; EVAL prints one machine-readable result line.
- onnxruntime: `compileOnly` in `core` + runtime `implementation` in `desktop` (verify android→core module graph; fallback = OnnxPolicy in desktop).
- ONNX I/O contract: input = concat(global, acting_civ) (fixed GnK width); outputs `tech_logits[techCount]`, `policy_logits[policyCount]`; widths + VERSION + ruleset fingerprint stamped as ONNX metadata and shared constants; PARITY test JVM-session vs Python-reference on a fixed observation.

## Open questions remaining
None blocking. Empirical width confirmation (GnK tech/policy counts) happens early in build by generating one shard and reading `schema.json`.
