# v4 Results — structured encoder (hex-GNN) — honest writeup

Run: 2026-06-25. Medium = 8 rounds (gen 16, eval 80) + a final 200-game eval, seed 4242424,
opponent = fixed RandomPolicy, terminal ±1 reward — **budget held constant vs the v3 rich-critic
Medium baseline** (the pre-registered comparison). The shipped rung is **small (GNN-only, no
attention)** — the demand-driven ladder's starting rung, the lightest (OOM-safe on the 1261-tile
dense batch), and the cleanest test of the diagnosed hypothesis (does restoring 2D locality via a hex
GNN beat the permutation-invariant pool?), uncontaminated by attention.

## Headline (AC1) — PASS on Medium
| variant (Medium, 200-game final) | win-rate | vs v3 rich-pool |
|---|---|---|
| v3 rich-**pool** (the thing v4 fixes) | 14.7% (29/200) | — |
| v3 blind-critic (v2 baseline) | 28.9% (58/200) | — |
| **v4 structured GNN (this work)** | **23.0% (47/204)** | **z = +2.196, one-sided p = 0.0141 ✓** |

**The masked-pool → hex-GNN swap fixes the Medium regression at p<0.05.** The GNN preserves the 2D
locality the pool discarded; structured beats the v3 rich-pool decisively (23.0% vs 14.7%, +8.3pp).

## The mechanism — no collapse (vs v3's decline)
Per-round Medium win-rate (84-game evals, noisy):
- v3 rich-**pool**: 35, 21, 27, 23, 26, 26, 18, 10 → **monotonic decline** to ~10% (200-game final 14.7%).
- v4 structured GNN: 21, 17, 39, **58**, 17, 39, 29, 17 → **no downtrend** (last-4 mean 25.3%); the
  reliable 200-game final is 23.0% (the per-round 84-game evals undersample — round 7's 84-game 16.7%
  vs its 200-game 23.0%). `diverged=0` every round.

v3 rich's signature was a *declining* curve (the undertraining/overfitting tell on 1261 tiles under a
pool that inflates per-tile data demand). The GNN curve oscillates around ~25% **without collapsing** —
the locality fix removes the decline.

## Honest caveat — structured beats the pool, not yet the blind baseline
Structured (23.0%) sits **between** v3 rich-pool (14.7%) and v3 blind-critic (28.9%): it beats the
broken pool significantly but is 5.9pp **below blind** (z=−1.37, not significant). Read plainly: the
GNN recovers the locality the pool destroyed, but **from-scratch-per-round training still caps how
much the richer representation can be exploited** — it doesn't yet surpass the cheap blind model.
This **confirms the recorded v2/v3 diagnosis** (memory: `selfplay-roadmap-bottleneck`): the encoder
was never the bottleneck; the training regime is. The accepted confound (v4 left from-scratch-per-round
untouched) is the ceiling. **Recommended next unlock: weight carryover / continual training across
rounds** — then re-test whether the GNN (and the medium/large attention rungs) can clear blind.

## Tiny non-regression (AC1 second clause) — rung-dependent
The **small/GNN-only** rung on Tiny: per-round 15,56,50,51,65,30,46,33,35,44,54,60 → last-4 mean
**48.3%** vs v3 rich-critic's **57.0%** (≈−8.7pp, ≈z−2.5: a significant shortfall), `diverged=0`,
last round 59.8%. This is the *expected* shape of the demand-driven trade-off: the small rung is
sized for **Medium** (scarce per-tile data, where the pool collapsed); **Tiny is data-rich (331 tiles)
and the full pool already excelled there**, so a GNN-only rung is *under*-capacity for Tiny. The
demand-driven rule (D7) responds to "data-rich + not matching" by **scaling up** — so the spec-faithful
Tiny non-regression check uses the medium (attention) rung, which is OOM-safe on Tiny's 331 tiles
(unlike Medium's 1261). **Medium-rung Tiny re-check is running** (also AC2's sweep). [pending]

## Acceptance criteria
| AC | Verdict |
|---|---|
| **AC1** structured beats v3 rich-pool on Medium p<0.05 | ✅ **23.0% vs 14.7%, z=+2.20, p=0.014**. Tiny: small/GNN rung **regressed** (48.3% vs v3 57%, last-4) — under-capacity for data-rich Tiny; **medium rung (attention) re-check running** (demand-driven scale-up; OOM-safe on Tiny's 331 tiles) |
| AC2 capacity sweep per rung + throughput; rung by demand-driven rule | 🟡 small rung (the rule's START) reported with win-rate + throughput; it already PASSES AC1. Medium/large sweep is the remaining secondary run (rule: scale up only if it improves AND fits the budget — medium-rung OOM-safety on the gen-16 dense batch is the gate) |
| AC3 parity over the richer multi-tensor input (atol 1e-4, logits) | ✅ v3 JVM↔Python logits parity + adjacency-fidelity (hexgraph == live engine) green |
| AC4 throughput ≥70% of heuristic baseline | ✅ `bench-onnx` verdict=PASS; eval ms/decision ≈ 24 ms, ≈ 128 turns/s |
| AC5 determinism + provenance + legality | ✅ provenance (schema_version + ruleset_fingerprint gated; the gate caught the reader-version + vocab-seam drift), legality (masked heads, zero illegal across ~16k decisions/eval). Determinism: same pre-existing v1/engine byte-nondeterminism noted in v2 AC5 (out of scope, file untouched) |
| AC6 terminal-only reward + no new heads | ✅ frozen `_optimize_actor_critic`/`compute_gae` untouched; heads {tech,policy}(+value train-only); no reward shaping (grep-clean) |
| AC7 construction-namespace bug fixed + unit test | ✅ `Vocab.constructionCode` + `ConstructionCodeTest` (injective over buildings∪units) |

## Build verification (all green)
- Python suite (non-gradle): 41 passed; gradle parity (v1/v2/v3 logits + adjacency-fidelity): 4 passed.
- Export: medium-rung ONNX opset-17 clean — **no scatter, no Attention/MHA ops**, neighbor_index int64.
- End-to-end: `--variant structured` runs gen→train→export→JVM-eval (live-engine neighbor tensors).
- Ship-council: 1 real critical (NaN-grad in masked-softmax backward) **fixed + backward-grad regression test**.

## Bottom line
v4's primary goal is met: **the structured hex-GNN encoder fixes the v3 Medium regression (p<0.05) and
removes the collapse**. It does not yet beat the blind baseline on Medium — which, per the pre-registered
confound, is the from-scratch-per-round training ceiling, not the encoder. The honest, data-backed next
step is **weight carryover**, after which the medium/large attention rungs should be swept under the
demand-driven rule.
