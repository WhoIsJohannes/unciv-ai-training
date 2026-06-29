# Discovery Output — selfplay-v7-construction

- **Mode**: BUILD
- **Size**: L (cross-cutting: Kotlin engine + Python training, schema lockstep bump, new net head, new per-entity control seam; ~12-15 files)
- **Feature (1-line)**: First PER-ENTITY control head — per-city PRODUCTION (construction). The learned policy gains its first high-leverage, variable-count, per-city lever on top of the civ-global tech/policy heads.

## Context category & docs
- **Category**: backend (cross-system pipeline; correctness-critical). NOTE: the routed invariant docs (`~/.claude/.../docs/invariants/*`) describe the **Mentiora** Next.js/FastAPI stack and do **not** apply to this Unciv Kotlin+Python RL repo. Authoritative constraints = the prompt's FROZEN/NON-GOALS + ACCEPTANCE CRITERIA + the v5/v6 code patterns. (See decisions.md D-disc-1.)
- **Domain preset**: data-pipeline (schema round-trip, dataset/replay training pipeline, throughput criterion). Phase-2 roster bias: practitioner (always-on, reliability/determinism), cost_efficiency (throughput ≥70%), plus an ML-correctness lens for the PPO/GAE/logp math.

## Prereq self-check
**NO DRIFT** — all 15 file:line anchors verified (see codebase-scan-light.md). Key facts:
- own_cities order = `x.cities.sortedBy { it.id }` (Featurizer.kt:61) — per-city loop must reuse exactly.
- constrW = `vocab.buildingCount + vocab.unitCount` (units offset by buildingCount).
- `Vocab.constructionId(idx)` absent → must add inverse.
- Two schema files lockstep: `SampleSchema.kt:27` (VERSION=4) + `python/unciv_dataplane/schema.py:20` (SCHEMA_VERSION=4); reader is descriptor-generic → new VARIABLE blocks round-trip free.
- Net class = `RichPolicyValueNet` (prompt's "StructuredPolicyValueNet"); per-city embeddings exist un-pooled inside the encoder but are not currently exposed.
- No turn-start queue clobber → pre-fill is safe.

## Open questions — RESOLVED (Round 1)
1. **Net rung for the v7 comparison** → **both rungs (4 arms)**: small + medium, each construction OFF vs ON, all at `--replay-window 4`. Construction is the only changed variable within each rung pair. (run_v7 mirrors v6's matrix but swaps the K-axis for the construction-axis.)
2. **Experiment execution** → **Build → validate → run to completion**: get parity/no-op/legality green, then launch the resumable run_v7 batch and drive it to completion; report 200-game OFF-vs-ON win-rates with z/p vs the 50% break-even when done (async across re-invocations).

## Experiment design (locked)
- 4 arms, sequential (CPU): `small-OFF`, `small-ON`, `medium-OFF`, `medium-ON`; all `--replay-window 4`, `--continual`, `--micro-batch-steps 256`, structured/Medium map, 16 rounds, gen 16 / eval 80, turn-cap 250, seed 4242424 for the 200-game ceiling eval.
- AC ordering: parity + no-op + legality green BEFORE the Medium run.
- Acceptance #3 framing: construction-ON beats construction-OFF (p<0.05) within each rung, AND report win-rate vs the 50% break-even (state plainly whether it crosses 50%).

## Invariant docs loaded
features.md, architecture.md, testing.md (read; found to describe a different stack — see decisions.md D-disc-1). code-quality.md / security.md skipped (wrong stack).
