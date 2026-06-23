# Plan — ONNX-in-JVM policy bridge + minimal self-play training loop

**Mode**: BUILD · **Size**: L · **Base**: `self-play-data-plane` · **PR target**: `self-play-data-plane`

## The ask
Close the self-play loop on top of the existing data plane: a JVM `OnnxPolicy` that runs a trained net **inside the JVM** to choose TECH + POLICY from the legal mask, plus headless GENERATE/EVAL entrypoints and a Python `unciv_train` package + round driver that trains on emitted trajectories, exports ONNX, and measures win-rate vs RandomPolicy. **The deliverable is a working end-to-end LOOP and an honest LEARNING CURVE** — the go/no-go evidence for from-scratch self-play. Reuse the data plane; don't rebuild it.

## How this plan solves it
There are **two linchpins** (the second was caught by the plan council, FND-0002):

1. **The policy must actually DRIVE the game, not just label it (D14, new P0).** Today `NextTurnAutomation.onCivTurn` is a *recording* hook — it records what `chooseIndex` would pick, but the engine's `chooseTechToResearch`/`adoptPolicy` heuristics make the real decisions. Training on those labels would REINFORCE non-causal actions, and "OnnxPolicy vs RandomPolicy" would secretly be heuristic-vs-heuristic (~50%, flat, wrong reason). The fix: a **control seam** so the installed policy drives the learner's (and opponent's) tech+policy — pre-fill `techsToResearch` (the heuristic respects a non-empty queue) and adopt the chosen policy + guard the heuristic. Control runs in gen **and** eval; recorded action == applied action.
2. **The shards record no reward (hardcoded 0) and no terminal flag** → no learning signal. Wire a per-civ **terminal ±1 reward** into the shards (a sanctioned `SampleSchema.VERSION` 1→2 bump).

With both, REINFORCE-with-baseline trains a tiny MLP on `concat(global, acting_civ)` → `{tech_logits, policy_logits}`; the same net runs in-JVM via onnxruntime through a `RoutingPolicy` that sends only the learner civ to ONNX (opponents stay RandomPolicy). A plain-script driver loops generate→train→export→eval for K rounds, writes `curve.csv` + a plot, and emits an explicit **GO / PLATEAU / INCONCLUSIVE** verdict. A golden PARITY test pins the one observation code-path so train-time and infer-time never drift.

## Why (rationale)
- **Reuse over rebuild**: every seam (Featurizer, masks, emitter, reader, Simulation harness, `binomialTest`) already exists and is verified. We add the *missing* reward signal and the *new* policy/runner/training pieces only.
- **One observation path**: `concat(global, acting_civ)` is produced identically on JVM (`Featurizer.observe`) and Python (`step.blocks`), golden-tested — the #1 risk (drift) is closed by construction.
- **Stationary opponent**: learner-vs-RandomPolicy gives a clean, low-variance learning signal for a true go/no-go (per the task's non-goals — no co-learning yet).

---

## Architecture: the loop
```
            ┌────────────────────────── run_loop.py (driver, K rounds) ──────────────────────────┐
 round 0:   │  gen (RandomPolicy)                                                                 │
 round r≥1: │  gen ──► ./gradlew selfPlay --args="gen <model> <out_r> ..."  → shards (VERSION 2)  │
            │   │                                                                                 │
            │   ▼                                                                                 │
            │  train ── unciv_train.dataset(shards) → REINFORCE-w-baseline → state_dict           │
            │   │                                                                                 │
            │   ▼                                                                                 │
            │  export ── export_onnx → policy.onnx  (fixed names + metadata: ver, fingerprint)    │
            │   │                                                                                 │
            │   ▼                                                                                 │
            │  eval ── ./gradlew selfPlay --args="eval policy.onnx <M> ..." → EVAL_RESULT {json}  │
            │   │                                                                                 │
            │   ▼  append curve.csv (round, games, winrate, pval) ; replot curve.png              │
            └─────────────────────────────────────────────────────────────────────────────────────┘
```
Inside the JVM (gen + eval): `Simulation` runs N/M Tiny 2-civ GnK games; `NextTurnAutomation.onCivTurn` → the existing single-policy `install()` → a `RoutingPolicy` that dispatches the **learner** civ to `OnnxPolicy` and **opponents** to `RandomPolicy`. `OnnxPolicy` builds `concat(global, acting_civ)` via `Featurizer`, runs one ONNX forward per (civ,turn), masks illegal logits to −inf, then samples (gen) or argmaxes (eval).

---

## P0 / P1 split
**P0 — required for a *meaningful* loop + all 6 acceptance criteria**
- **Control wiring (D14)**: the installed policy drives tech+policy for controlled civs (decision-gated; recorded==applied; gen+eval); `adoptPolicy` guard. *Without this the loop cannot learn.*
- Schema VERSION 2 + per-civ terminal reward (Kotlin emitter + Simulation hook; Python `SCHEMA_VERSION` lockstep).
- `RoutingPolicy` (core), `OnnxPolicy` (desktop: ThreadLocal per-(game,civ,turn) memo, masked sample/argmax, per-game error isolation), `SimStats` (core, extracted `binomialTest`).
- `SelfPlayRunner` (desktop: `gen`/`eval`/`parity-dump` modes) + gradle `selfPlay` task + onnxruntime dep.
- `unciv_train`: `contract`, `model`, `dataset` (provenance gate; per-head masking; group by (shard,civ_slot)), `train` (no negative-index gather), `export_onnx`, `run_loop` (subprocess arg-list + returncode/timeout checks; GO/PLATEAU/INCONCLUSIVE verdict; per-round shard retention).
- Tests: legality, determinism, parity, provenance, concurrent-eval smoke. `curve.csv` + plot + verdict.

**P1 — polish (do if cheap; never blocks the curve)**
- Structured per-round log line beyond curve.csv; plot styling; optional `γ` discount knob (default 1.0); entropy-bonus knob. *(Adaptive eval-stop, nation randomization, co-learning, heads beyond tech+policy = OUT, v2.)*

---

## Change inventory (file-by-file)

### Kotlin — `core`
| File | Change | What |
|---|---|---|
| `core/.../dataplane/SampleSchema.kt` | edit | `VERSION = 1` → `2`. Add ONNX-contract constants: `OBS_INPUT_NAME="obs"`, `HEAD_TECH="tech"`, `HEAD_POLICY="policy"`, output names `tech_logits`/`policy_logits`, `CONTRACT_VERSION=1`. |
| `core/.../dataplane/RoutingPolicy.kt` | **new** | `class RoutingPolicy(learnerCivId: String, learner: PolicyProvider, opponent: PolicyProvider) : PolicyProvider` — `chooseIndex`/`actUnit` dispatch by `civ.civID`/`unit.civInfo.civID`. No ORT dep. |
| `core/.../dataplane/DataPlaneHooks.kt` | edit | **(D14 CONTROL)** Split per-civ-turn handling into **controlApply** (always, when a policy is installed) + **emit** (only if a recorder is registered). `controlApply(civ, policy)`: TECH — if `civ.tech.techsToResearch.isEmpty()`, `idx=chooseIndex("tech",…)`; if `idx>=0` set `techsToResearch=[techName]`; else record −1. POLICY — if `civ.policies.canAdoptPolicy()`, `idx=chooseIndex("policy",…)`; if `idx>=0` `civ.policies.adopt(policy)`; else −1. `chooseIndex` called ONCE per head ⇒ recorded==applied. `install(policy)` wires control for the run; recorders stay per-game/optional (eval = control, no emit). **(D1 reward)** Add `ShardRecorder.recordTerminal(perCivReward)`: one record per `seenCivs` civ — `isTerminal=1, reward=value`, same `civSlot`, zero-filled obs (terminal obs unused for training; dataset excludes it from inputs). |
| `core/.../automation/.../NextTurnAutomation.kt` | edit | **(D14)** Guard `adoptPolicy(civInfo)` to `return` early when the civ is policy-controlled (control already adopted the chosen policy). `chooseTechToResearch` needs no change — it respects a non-empty `techsToResearch`. |
| `core/.../simulation/Simulation.kt` | edit | (1) Add ctor flag `scoreLeaderOnTimeout: Boolean=false`. At game end: if `victoryType==null && scoreLeaderOnTimeout` → `step.winner = SimStats.scoreLeader(gameInfo)` (null on exact tie). (2) Before `recorder.close()`, compute per-civ reward (win=+1, lose=−1, draw=0) from `step.winner` and call `recorder.recordTerminal(...)`. (3) Replace private `binomialTest`/`normalCdf`/`erf` with `SimStats`. Pass `scoreLeaderOnTimeout` from the runner (gen+eval). |
| `core/.../simulation/SimStats.kt` | **new** | Public `binomialTest(successes,trials,p,alt)`, `normalCdf`, `erf` (moved verbatim), `scoreLeader(gameInfo): Civilization?` (max `calculateTotalScore()` among alive majors; null on tie). |

### Kotlin — `desktop`
| File | Change | What |
|---|---|---|
| `desktop/.../OnnxPolicy.kt` | **new** | `class OnnxPolicy(modelPath, vocab, config, rngFor, eval: Boolean) : PolicyProvider`. Lazy shared `OrtEnvironment`+`OrtSession` (intra-op threads=1; ORT supports concurrent `run`). `chooseIndex`: **ThreadLocal memo keyed by (gameId,civID,turn)** (one forward per decision; no cross-thread/cross-game bleed) → `concat(global,acting_civ)` → `tech_logits`/`policy_logits`; non-{tech,policy} head → −1; mask illegal → −inf; empty legal → −1; gen = **sample from softmax(masked)** via `rngFor(civ,turn)`, eval = **argmax(masked)**. `actUnit` → `UnitAutomation.automateUnitMoves`. Reads ONNX `getCustomMetadata()` → fail fast on schema/fingerprint/contract mismatch. Inference error → abort THIS game (isolation), not the run. Close tensors/Result; close session at run end. |
| `desktop/.../SelfPlayRunner.kt` | **new** | `main`: arg[0]=mode. `gen <model|random> <out> <nGames> <maxTurns> <threads> <seed> <learnerNation> <oppNation>`; `eval <model> <M> <maxTurns> <threads> <seed> <learnerNation> <oppNation>` → prints `EVAL_RESULT <json>{games,wins,winrate,pval,learner,seed,onnx_decisions}`; `parity <model> <obsFile> <outFile>` → writes JVM logits for a fixed obs. Shared `buildTinyGameInfo(seed, learnerNation, oppNation)` (MapSize.Tiny, GnK, pinned `Player(Nation)`, noBarbarians/noRuins/noNaturalWonders). `eval`+`gen` set `scoreLeaderOnTimeout=true`. |
| `desktop/build.gradle.kts` | edit | Add `implementation(libs.onnxruntime)`; add `selfPlay` `JavaExec` task (mainClass `SelfPlayRunner`, `-Xmx8G`). |
| `gradle/libs.versions.toml` | edit | Add `onnxruntime = "<pinned>"` + `[libraries] onnxruntime = { module = "com.microsoft.onnxruntime:onnxruntime", version.ref = "onnxruntime" }`. |

### Python — `python/unciv_train/`
| File | Change | What |
|---|---|---|
| `__init__.py` | **new** | package marker |
| `contract.py` | **new** | Shared ONNX I/O contract: input name `"obs"`, outputs `["tech_logits","policy_logits"]`, head order, `CONTRACT_VERSION=1`; helpers to read widths + fingerprint from a generated `schema.json`. |
| `model.py` | **new** | `PolicyNet(in_dim, tech_w, policy_w)`: shared MLP trunk (2×128 ReLU) → two linear heads. `forward(obs)->(tech_logits, policy_logits)`. |
| `dataset.py` | **new** | Load shards via `unciv_dataplane.reader`. **Provenance gate**: refuse if shard `schema_version != expected` or `ruleset_fingerprint != expected` (strict, not warn). Filter to learner `civ_slot`; per (shard,civ_slot) read terminal reward → return-to-go broadcast; extract `(obs=concat(global,acting_civ), a_tech, a_policy, mask_tech, mask_policy, R)`. Skip steps where a head's action `<0`. |
| `train.py` | **new** | REINFORCE-w-baseline: masked `log_softmax` (illegal→−1e9), gather chosen logp per head, advantage `R−running_mean`, loss `−A·Σ_h logp`, Adam, CPU. Optional `γ` (default 1.0), entropy bonus (default 0). Saves `state_dict` + resolved widths + fingerprint. |
| `export_onnx.py` | **new** | `torch.onnx.export(..., input_names=["obs"], output_names=["tech_logits","policy_logits"], dynamic_axes={batch})`; then `onnx.helper.set_model_props` with `schema_version`, `ruleset_fingerprint`, `contract_version`, `input_width`, `tech_width`, `policy_width`. |
| `run_loop.py` | **new** | The driver (D8 lean knobs + baked defaults). Round 0 = RandomPolicy gen. Each round: gen→train→export→eval (subprocess `./gradlew selfPlay --args=...`, parse `EVAL_RESULT`), append `curve.csv` (round,games,winrate,pval), replot `curve.png`. Fresh per-round shard dir + retention; per-process timeout backstop. |
| `python/pyproject.toml` | **new** | Package `unciv_train` (+ expose `unciv_dataplane`); deps `torch`, `onnxruntime`, `numpy`, `matplotlib`, `onnx`. |

### Tests
| File | Change | Criterion |
|---|---|---|
| `tests/.../dataplane/OnnxPolicyLegalityTest.kt` | **new** | **AC2 LEGALITY** — mirror `maskParity_*`: across many states/turns + a tiny fixture model, assert `chooseIndex` returns a legal index or −1 (never illegal); empty-legal→−1; unmodeled head→−1. |
| `tests/.../dataplane/SelfPlayDeterminismTest.kt` | **new** | **AC3 DETERMINISM** — fixed model + seed → identical trajectory `calculateChecksum` across two runs. |
| `tests/.../dataplane/SimStatsTest.kt` | **new** | binomialTest known-value unit test (regression guard for the extraction). |
| `tests/.../resources/policy-test.onnx` | **new (force-add)** | tiny random-weights fixture for the Kotlin tests (legality is weight-independent). |
| `python/tests/test_train_dataset.py` | **new** | **AC6 PROVENANCE** — refuse mismatched VERSION/fingerprint; correct action/return extraction from a synthetic VERSION-2 shard. |
| `python/tests/test_parity.py` | **new** | **AC4 PARITY** — fixed obs+model → JVM logits (via `selfPlay parity`) == Python reference (onnxruntime) within `atol=1e-4`. |

### Docs / misc
| File | Change |
|---|---|
| `python/unciv_train/CONTRACT.md` | **new** — the ONNX I/O contract doc (names, shapes, dtypes, metadata keys, version, parity rule). |
| `.gitignore` | edit — add `*.onnx`, `selfplay-output*/`, `curve.csv`, `curve.png`, `python/**/__pycache__/`, `.pytest_cache/`, `*.egg-info/`, `training-runs/`. |
| `specs/.../onnx-io-contract.md` | **new** — copy of the contract for the spec record. |

---

## Walkthrough
One learner civ-turn → one training step → one inference, with illustrative mock shapes (illustrative — not measured):

| Hop | Contract | Mock payload |
|---|---|---|
| 1. Civ-turn fires | `NextTurnAutomation.onCivTurn(civ)` → `RoutingPolicy.chooseIndex("tech", BabylonCiv, mask, turn=42)` | learner=`Babylon`; `mask_tech` legal at idx {3,7,9} |
| 2. OnnxPolicy featurize | `Featurizer.observe(civ)` → `concat(global[26], acting_civ[206])` = `obs[232] f32` | `[12.0, 3.0, …, 0,1,0,…]` |
| 3. ONNX forward (memoized for civ42) | `OrtSession.run({"obs": [1,232]})` → `tech_logits[80]`, `policy_logits[~80]` | tech_logits idx7=2.1 (max legal) |
| 4. Mask + choose + **APPLY** | illegal→−inf; sample over {3,7,9}; **set `Babylon.tech.techsToResearch=[tech7]`** (control) + record `actions[0]=7` | applied==recorded; Babylon now researches the net's pick (not the heuristic's) |
| 5. Game ends (turn 188, Babylon domination win) | `Simulation` → `recorder.recordTerminal({Babylon:+1, Sumeria:−1})` | terminal step `civ_slot=0, reward=+1` |
| 6. Train (next round) | `dataset`: Babylon's step42 gets `R=+1`; masked logp(tech=7); `A=+1−baseline` | loss `−A·logp` → net nudges tech7 up |

(LLM-authored — sanity-check against the architecture detail above. The widths 232/80/~80 are illustrative; the build pins them from the generated `schema.json`.)

---

## Tests → acceptance criteria
1. **Loop K≥10 rounds headless** → `run_loop.py` end-to-end (gen→train→export→eval), no manual steps; `curve.csv` has K rows.
2. **Legality** → `OnnxPolicyLegalityTest` (mirrors RandomPolicy legality): zero illegal indices ever.
3. **Determinism** → `SelfPlayDeterminismTest`: fixed model+seed ⇒ identical `calculateChecksum`; eval reproducible.
4. **Parity** → `test_parity.py`: JVM vs Python logits within 1e-4 on a fixed obs.
5. **Learning curve** → `curve.csv` + `curve.png`; honest reporting (≥60% @ p<0.05 target, or a plain-stated plateau).
6. **Provenance** → `dataset.py` refuses mismatched VERSION/fingerprint; `policy.onnx` carries matching metadata; EVAL refuses a contract-mismatched model.

---

## Risks
| # | Risk | L×S | Mitigation |
|---|---|---|---|
| 1 | **Control wiring incorrect** — policy doesn't truly drive tech/policy ⇒ a meaningless ~50% curve (FND-0002) | M×H | D14 control seam (pre-fill `techsToResearch` + adopt + `adoptPolicy` guard); a test asserts a controlled civ's applied tech == the policy's chosen tech; recorded==applied by single-`chooseIndex` design. |
| 2 | Observation drift JVM↔Python (the classic #1 risk) | M×H | Single `concat(global,acting_civ)` path; golden PARITY test @1e-4 from a committed fixture; widths runtime-derived from `schema.json`, never hardcoded. |
| 3 | Terminal-reward wiring wrong (no/incorrect signal) | M×H | VERSION 2 + explicit `recordTerminal`; unit test reward extraction; per-head no-negative-index gather; determinism test guards bytes. |
| 4 | Concurrent-game memo corruption (FND-0001) | M×H | ThreadLocal memo keyed by (gameId,civID,turn); concurrent-eval smoke test. |
| 5 | onnxruntime JVM thread-safety / native mem | L×M | One shared read-only session; intra-op=1; close tensors/Result; concurrent-eval smoke; verify on pinned Javadoc. |
| 6 | Fixed-matchup overfit + undiscounted variance (FND-0030/0033) | M×L | Accepted for a v1 stationary-opponent go/no-go; map seed varies per game; baseline; optional `γ`; documented limitations; honest curve. |

## Out of scope (task non-goals — enforced)
Self-play co-learning; full entity/spatial net input (v1 = global+acting_civ only); heads beyond tech+policy (greatPerson/diplomaticVote/construction/promotion stay heuristic via RoutingPolicy fallback); GPU/distributed/Ray; PPO/advanced RL; reward shaping beyond terminal ±1; hyperparameter sweeps; production model; gameplay/balance changes; fairness-model/featurizer-output changes.

## Human actions
- None required to run the loop. (Optional: `pip install -e python/` once for the training deps.)

## Rollback
Revert the branch; `master` untouched. Schema rollback = bump `VERSION` back to 1 + regenerate (datasets are perishable by design).

## Loop defaults + go/no-go (D17, D20)
Baked balanced defaults (all overridable; the "default operating contract"): **K=10** rounds, **gen-games=24**, **eval-games=100** (60% vs 50% ⇒ p≈0.023 < 0.05), **turn-cap=325** (high, so games play to a real victory per your decision), **threads=cores−1**, ORT intra-op=1, **shard retention=keep-last-2 rounds**, hard total-games budget cap. gen→train→eval are sequential phases (no JVM/ORT/torch thread contention).
**Verdict rule** (emitted with the curve): **GO** if final-round winrate ≥60% AND p<0.05; **PLATEAU** if winrate ∈ [45,55]% with no upward trend over the last ≥3 rounds; **INCONCLUSIVE** otherwise. The infra + an honest curve is the deliverable — not a guaranteed number.

## Responsible-use scope (D21)
This is a game-AI research artifact: it chooses **techs and policies inside Unciv** (a turn-based strategy game) to study whether from-scratch self-play can learn. No real-world decisions, no PII, no end users, no networked service. The trained `policy.onnx` is a game-balance experiment with negligible dual-use; no harm-reporting machinery is warranted (the council's vulnerable-population / red-team findings are acknowledged as over-reach for this domain).

## Open questions
None blocking. The exact GnK `policyCount` (70 vs ~80) is resolved empirically in the first build step (generate one shard, read `schema.json`); the contract reads it at runtime regardless.

See `decisions.md` (D1–D13) for the full rationale and `council-intake-triage.md` for the 40-finding disposition.
