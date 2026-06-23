# Council Plan Review — verdict

**Roster:** 8 (Core 6 + domain_fidelity + cost_efficiency), vendor-diverse. **Rounds:** 1 (adaptive stop). **Findings:** 40 (8/8 reviewers succeeded). **Cost est.** ~$1.31.

**Verdict: APPROVE (after folding refinements R1–R15 into plan.md).**
No finding requires an architectural change — every critical/major is an under-specified detail now tightened in `plan.md` § "Plan-review council refinements (BINDING)". Criticals are therefore **mitigated, not unmitigated** → no REQUEST_CHANGES re-run needed (matches the intake-council handling; cap=2 adaptive stop).

## Critical findings → resolution
- FND-0006 endianness → R7 explicit `ByteOrder.LITTLE_ENDIAN` + dtype markers.
- FND-0010 CoroutineName collision → R10 `workerId` in filename + fallback.
- FND-0015 success metrics → the 10 acceptance tests ARE the metrics (mask-parity 100%, leakage 0%, byte-identical replay, fingerprint-drift).
- FND-0020 wall-clock-in-checksum → R6 checksum covers RECORD bytes only (header excluded).
- FND-0025 entity-ordering + overflow-flag covert channel → R1 canonical ordering + R2 (overflow flag/fingerprint/gameId/seed NOT in observation tensor).
- FND-0030 unseeded shuffle → already the gated R7/D1 fix.
- FND-0035 victory-encoding rules → R-victory (derive orig-capitals or denom-mask; SS count only; vote count).
- FND-0036 stealable-tech "contradiction" → R5: it's the D4 spy-gated exception (passive path stays count-only). Reviewer contradicted its own intake position; D4 is the resolution.

## Major (folded): FND-0007 (PolicyProvider optional param), FND-0008/0037 (structural thread-safety), FND-0009 (subset serialization), FND-0011/0013 (atomic .tmp + reader tolerates truncated tail), FND-0021 (deterministic overflow drop order), FND-0022 (maxAttempts=100 + throw + test), FND-0027 (fingerprint out of observation), FND-0038 (religions sorted), FND-0039 (rank/bucket algorithm + tie-break), FND-0040 (CityCombatant for def-strength), FND-0031/0032/0033/0034 (cost note / git-SHA best-effort / overflow / unmet — already in plan).

## Rejected (documented in plan.md)
FND-0001 (keep deterministicShuffle flag — default-deterministic would change interactive play), FND-0003 (keep PolicyProvider interface — required RL seam), FND-0002 (keep omniscient/strict — required), FND-0012 (no async-queue/back-pressure/SLO — over-engineered for offline gen), FND-0017 (no VP-approval ceremony), FND-0029 (third-party politics web IS fair per prompt's freely-openable-screen principle).
