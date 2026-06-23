# Intake Council Triage (Phase 2, Step 5) — 40 findings, round 1

Roster: Core 6 + domain_fidelity + ethics_responsible_ai + cost_efficiency (9). Severity: 5 critical, 29 major, 6 minor.

## 🟢 ADOPT — folds into the plan
| FND | Finding | Action |
|---|---|---|
| 0002, 0006 | ORT in `core` (even compileOnly) is dependency inversion | **Move OnnxPolicy to `desktop`** (implements core `PolicyProvider`); onnxruntime = `desktop` dep only. Reverses D4. Deviation from task's "core/.../dataplane" path — justified + noted. |
| 0037 | OnnxPolicy needs per-civ-turn inference caching | **One forward pass per (civ,turn)** producing BOTH tech+policy logits; memo keyed by (civID, turn). Second head reuses it. |
| 0007 | Unmanaged native memory | Close `OnnxTensor`/`OrtSession.Result` (try-with-resources); single shared session closed at run end. |
| 0022 | OnnxPolicy error states | Model-load failure / inference exception → **fail fast** (loud error), never silent-fallback (a broken model must not masquerade as a learning signal). |
| 0008, 0021 | Implicit JVM↔Python handoff + EVAL line format | Specify: driver invokes `./gradlew selfPlayGen/selfPlayEval` with explicit args; EVAL prints ONE line `EVAL_RESULT {json:{games,wins,winrate,pval,learner,seed}}` parsed by the driver. |
| 0023, 0027, 0009 | Score-leader tiebreaker ambiguity / reward-hacking / exact ties | **Win = formal victory; else (hard cap) higher `calculateTotalScore()`; exact tie = draw (reward 0, excluded from win numerator)**. Cap is high so this is the rare path (user decision). |
| 0030 | Pinned nations → matchup overfitting | Keep fixed nations for v1 clean stationary signal, but **map seed varies per game** (already true). Document overfitting-to-matchup as an explicit v1 limitation + v2 todo (randomize nations, track learner by slot). |
| 0033 | Undiscounted reward over long horizons = high variance | Honor non-goal (terminal ±1, no shaping): default γ=1.0 undiscounted; baseline reduces variance; expose γ as an optional knob. Note variance as expected. |
| 0001, 0003, 0017, 0005, 0004 | Config theater / framework creep / generic head abstraction / SimStats bag | **Lean**: ~6 knobs (K, gen-games, eval-games, turn-cap, threads, seed) with baked balanced defaults (the "default operating contract"); plain-script driver; hardcode 2 heads; SimStats = just the binomial test. Reconciles skeptic with the user's "configurable" ask. |
| 0039, 0013 | Shard disk/retention + VERSION-bump rollback | Driver uses a fresh per-round shard dir + retention (keep last N, configurable); VERSION-bump rollback = bump back + regen (perishable-by-design, documented). |
| 0011 | No hang detection | The **turn cap bounds every game** (no infinite loop). Add a generous per-process timeout in the driver as a backstop. |
| 0016, 0019 | No P0/P1 split / out-of-scope boundary | Plan gets a P0/P1 split + explicit Out-of-Scope (= the task non-goals). |
| 0024 | Parity fp tolerance undefined | Define `atol=1e-4` (f32) for the PARITY test. |
| 0028 | No check routing actually held | EVAL asserts the learner civ actually invoked OnnxPolicy (decision counter > 0). |
| 0014, 0040 | RNG thread-safety / CPU oversubscription | ORT `intraOpNumThreads=1`; sim threads default = cores−1; determinism via existing per-(game,civ,turn) seeding. |
| 0010, 0012 | No throughput numbers / observability | Round 0 measures + logs turns/s; driver writes a per-round structured log line + `curve.csv`. |
| 0015, 0020, 0018 | Success not measurable / criteria / bite-tests | Restate acceptance criteria in plan; map each (legality, determinism, parity, provenance, curve) to a concrete test. |
| 0025 | Model path code/deserialization risk | Model is self-produced in the loop from a trusted path (not third-party). Note as a trust assumption; no untrusted-model loading. |
| 0031, 0032 | Output misses 2 heads / masking | Masking explicit (illegal logits → −inf before softmax/argmax). The 2 other heads (greatPerson, diplomaticVote) stay heuristic via RoutingPolicy fallback — intentional v1 scope. |

## 🟡 INTENTIONAL / DEFERRED — note, don't act (respects task non-goals)
- **FND-0029** "routing precludes self-play": BY DESIGN — v1 is learner-vs-RandomPolicy (a stationary opponent for a clean signal), per the task non-goal. "Loop" = generate→train→eval, not opponent co-learning. v2.
- **FND-0038** adaptive eval stop (SPRT): v2; v1 uses fixed configurable M for reproducibility (determinism criterion).
- **FND-0034, 0035** adversarial/dual-use of the policy: a Civ tech/policy picker is not a weapons system; negligible dual-use. Out of scope.
- **FND-0026** research-notes fence: framework-level, not this feature's concern.

## Verdict
APPROVE-with-revisions: no blocking finding; the adopt-list materially improves the plan (esp. OnnxPolicy→desktop, per-turn caching, win/tiebreaker precision, lean knob set). No user decision required — all adopts are within the agreed scope + user decisions. Will reflect in `plan.md` Deviations table with `[intake council FND-NNNN]` origins.
