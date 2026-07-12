---
name: selfplay-baseline-variance
description: Self-play eval is HIGH-VARIANCE (identical code swings 8.8%↔41.7% ceiling by gen-seed) — single-seed experiments are unreliable; replicate every arm
metadata:
  node_type: memory
  type: project
  originSessionId: f6780e61-0544-41d1-ba19-6b764050f0f5
---

**The self-play ceiling is HIGH-VARIANCE across gen-seeds — single-seed experiments are unreliable.**
Discovered while chasing a "weak baseline" that turned out to be NOISE, not a regression. IDENTICAL code
(current v7.3, structured/small/Medium/16-round/mb0/rw1/gen16) produced ceilings of **8.8% (seed 2000) vs
41.7% (seed 3000)** — a ~33pp swing from the seed alone. OFF across 3 seeds: 17.2%, 8.8%, 41.7%.

**Cause: nondeterministic self-play GEN.** The policy RNG is deterministic (`DataPlaneHooks.defaultRngFor` =
`stateBasedRandom` per civ+turn) and per-game map seeds are fixed (`seedBase+(threadId<<20)+iteration`), but
the ~250-turn game simulation itself is nondeterministic run-to-run (engine/heuristic-AI collection-iteration
order + residual unseeded randomness) — same-seed gen gives different step counts (e.g. 800 vs 777). The v5
RESULTS already warned "read the trajectory, not single points" and noted arms "oscillate" 17–63%. Not
cheaply fixable (upstream engine level).

**The compounding climb happens in ROUNDS 8–15, not before** (v5 ARM A curve: `11 12 11 12 18 25 14 19 | 24
40 42 38 36 31 49 38`). Diagnostics that stop at 8 rounds see only the flat pre-climb (~15%) and look broken.
ALWAYS run the full 16 rounds before judging.

**Implications:**
- Any single-seed result — v5 40.7%, the v7 "47% OFF vs 14% ON" negative, v7.2 PBRS 37% — is ONE draw from a
  wide distribution and is individually UNRELIABLE. The [[selfplay-v7-construction-negative]] verdict is being
  re-tested under replication (`python/run_v73rep.sh`: 3 arms × 4 seeds, paired per-seed diffs).
- Going forward: REPLICATE every arm across ≥4 seeds; compare MEAN ceilings + PAIRED per-seed differences
  (same gen-seed controls the shared variance → far tighter effect estimate). `analyze_v73rep.py` does this.
- micro-batch 0 vs 256 is mathematically EQUIVALENT (net is per-sample: LayerNorm + per-sample masked_pool,
  no BatchNorm) — earlier mb256-looks-worse was also just seed variance. Use mb (256) for memory, mb0 fits
  small-rung construction on Medium (no OOM).

**NO regression exists in v6/v7/v7.2/v7.3** — every changed line is byte-equivalent for the OFF/rw1 path
(verified by full v5→v6 diff + v5-commit reproduces ~37% on current torch 2.8.0). The bisection that seemed to
localize a regression to v6 was comparing single noisy samples. See [[selfplay-roadmap-bottleneck]].
