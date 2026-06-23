# Plan Council Triage (Phase 2, Step 11) — 42 findings, round 1

Roster: Core 6 + domain_fidelity + ethics_responsible_ai + cost_efficiency (9). Severity: 7 critical, 29 major, 6 minor.
Verdict: **REQUEST_CHANGES → plan revised** (one fundamental gap + many refinements folded). No KILL/BLOCK.

## 🔴 THE fundamental gap (plan revised — new P0)
- **FND-0002 (+ the whole learning premise)**: The data plane's `onCivTurn` is a RECORDING hook only; the engine's `chooseTechToResearch`/`adoptPolicy` heuristics make the REAL decisions. As drafted, the policy would only LABEL while the heuristic plays → REINFORCE on non-causal actions, and "OnnxPolicy vs RandomPolicy" = heuristic-vs-heuristic (~50% flat, wrong reason). **FIX (D14): add a CONTROL seam** — the installed policy DRIVES tech+policy for the civs it controls (decision-gated: tech when `techsToResearch.isEmpty()`, policy when `canAdoptPolicy()`), recorded action == applied action, control runs in BOTH gen and eval (separate from emission). Pre-fill `techsToResearch` (heuristic respects it) + guard `adoptPolicy` to skip the heuristic for controlled civs.

## 🟢 ADOPT — folded into the revised plan
| FND | Finding | Action |
|---|---|---|
| 0001, 0026, 0023 | Concurrent games corrupt the per-(civ,turn) memo / state aliasing / cross-thread bleed | **D15**: ThreadLocal memo keyed by (gameId, civID, turn), holding only the most recent — no cross-thread/cross-game sharing. |
| 0024 | Negative-index gather bug (`logp[-1]`) | **D16**: skip a head's term when its action `< 0`; assert `a>=0` before gather. |
| 0025 | Whole-step dropping discards valid data | **D16**: per-head masking — keep a step if ANY head acted; loss only over acted heads. |
| 0010, 0018 | No go/no-go rule / vague "plateau" | **D17**: explicit rule — GO if final-round winrate≥60% AND p<0.05; PLATEAU if winrate∈[45,55]% with no upward trend over last ≥3 rounds; driver emits a verdict line + the curve. |
| 0015 | Parity obs source undefined | **D18**: `selfPlay parity-dump <seed>` writes a committed fixture obs; both JVM + Python parity read the SAME fixture; assert within 1e-4. |
| 0005, 0017, 0006, 0020, 0021, 0019 | Driver crash/timeout handling; per-game error isolation; subprocess arg-smuggling; EVAL_RESULT/model-load trust | **D19**: driver uses subprocess **arg-list** (no shell), checks returncode + per-call timeout, fails the round loudly; OnnxPolicy inference error → abort that GAME (isolation), not the run; validate model path exists; (local-trust context noted — driver+JVM are the same trusted user). |
| 0011, 0033, 0036, 0007, 0035 | Defaults/compute-budget/eval-power/threads unspecified | **D20**: concrete balanced defaults — K=10, gen-games=24, eval-games=100 (60% vs 50% → p≈0.023), turn-cap=325, threads=cores−1; ORT intra-op=1; phases (gen→train→eval) are sequential so no JVM/ORT/torch thread contention; a hard total-games budget cap. |
| 0027, 0004 | Reward broadcast mixes games / nondeterministic shard order | dataset groups by (shard, civ_slot) (one shard = one game); training sorts shards by name. |
| 0034, 0039, 0013 | Shard retention is not optional | retention moved to **P0** (per-round regen fills disk over K rounds): fresh per-round dir + keep-last-N. |
| 0003 | Softmax before sampling | OnnxPolicy: gen = sample from softmax(masked logits); eval = argmax(masked logits). Explicit. |
| 0022 | Mask-source divergence (infer param vs recorded block) | the unified control path calls chooseIndex ONCE at turn-start with the mask built from the obs → recorded mask == inference mask by construction. |
| 0038 | Synthetic terminal samples = schema debt | terminal records are reward-carriers (is_terminal=1, zero obs); dataset EXCLUDES them from training inputs (uses only their reward). Documented. |
| 0008, 0014 | Observability/baseline not committed | round-0 measures+logs throughput; driver logs per-round (winrate, pval, baseline, loss); P0-lite. |
| 0009 | Thread-safety asserted not verified | add a concurrent EVAL smoke assertion (multi-thread, no crash/corruption). |
| 0042, 0041 | Contract duplicated / SimStats bag | single name-constants source (Kotlin object + python contract.py) guarded by the parity + a name test; SimStats = binomialTest + scoreLeader only. |
| 0012, 0040, 0039 | P0 too fat / config creep | P0 tightened (below); 6 knobs only. |

## 🟡 ACKNOWLEDGE — short note, not engineered (over-reach for a Civ-game AI)
- **FND-0028/0029/0030/0031/0032** (vulnerable-populations / dual-use / red-team / harm-reporting / hallucination): a tech/policy picker for a turn-based strategy game has negligible real-world harm or dual-use. Plan adds a one-paragraph **Responsible-use/scope** note (game-AI research artifact; no real-world decisions, no PII, no users); no harm-reporting machinery. **FND-0037** (ML deps CI): pyproject pins versions; noted.

## Revised P0 (tightened)
Control wiring (D14) · terminal reward + VERSION 2 · RoutingPolicy · OnnxPolicy (ThreadLocal memo, masked sample/argmax, error isolation) · SimStats · SelfPlayRunner (gen/eval/parity-dump) + gradle task + ORT dep · unciv_train (contract/model/dataset/train/export/run_loop with crash+timeout handling, go/no-go rule, shard retention) · tests (legality, determinism, parity, provenance, concurrent-eval smoke) · curve.csv + plot.
