# Plan — v2: rich representation + value critic (onnx-selfplay-v2)

## The two questions this answers
1. **Will it converge?** A learned value critic (actor-critic + GAE) should give per-state credit and **reduce late-round win-rate variance** vs v1 REINFORCE — banked as an isolated `blind-critic` curve on the SAME blind input (so only the algorithm changes).
2. **Does seeing the board help?** A model that consumes the FULL observation (map tiles + units + cities + opponents) should reach a **higher win-rate where position matters** (Medium map) — banked as `rich-critic` vs `blind-critic`.

Reward stays **terminal-only ±1** end-to-end; the critic is the only new credit mechanism (learned from the terminal outcome, not a shaped proxy).

## Non-goals (unchanged from brief)
No reward shaping; no self-play/league opponent (RandomPolicy stays); no new action heads (tech+policy only); no new engine-emitted data; no distributed training / giant transformer.

## Delivery in three stages (each independently verifiable)

### Stage A — Value critic on the blind input (python-only; banks the convergence curve)
Reuses the v1 blind 199-dim input and the EXISTING ONNX contract v1, so the JVM bridge is untouched in this stage.

1. **`model.py`**: add `value_head: Linear(hidden→1)`; `forward(obs) → (tech_logits, policy_logits, value)`. (Blind trunk unchanged otherwise.)
2. **`dataset.py`**: add a trajectory loader returning **ordered per-game learner-step sequences** (`TrainTrajectory`: ALL non-terminal learner steps in turn order — do NOT drop no-action steps as v1 did; that would break GAE's temporal sequence and risk dropping the reward-bearing step — R1). reward 0 except the terminal ±1 at the final transition. Keep `TrainStep` fields per step; **drop the v1 return-broadcast** (GAE computes returns). Provenance gates unchanged.
3. **`train.py`**: replace REINFORCE with **actor-critic + GAE**:
   - per trajectory compute δ_t = r_t + γV(s_{t+1}) − V(s_t) (V(terminal)=0), advantage A_t = Σ_l (γλ)^l δ_{t+l}, return R_t = A_t + V(s_t). γ=0.99, λ=0.95. (`compute_gae(rewards, values, γ, λ)` — pinned by `test_gae.py`.)
   - **batch-level** advantage standardization (all steps from all trajectories pooled, so winning+losing games coexist — NEVER per-trajectory, which would zero the signal — R2); `policy_loss` PPO-clipped; `value_loss = MSE(V,R)`; `entropy` bonus; `total = policy + c_v·value − c_ent·entropy` (c_v=0.5, c_ent=0.01).
   - **AS-BUILT (R12, supersedes the earlier "single-epoch" note):** standard PPO — advantages + GAE returns computed ONCE per round from a V-snapshot (fixed target), **PPO-clip ε=0.2 default, K=8 epochs**, `clamp(logp−old_logp,±20)`, **tanh-bounded value head** + small value-head init (calibration showed recompute-each-epoch degenerates value_loss and unnormalized features blow up an unbounded value). Value head dropped at export ⇒ zero parity impact. Attribution preserved: v1-reinforce ignores the value head; blind/rich share the identical PPO+GAE algo. Reuse `_masked_logp` + −1/no-action handling VERBATIM.
   - **divergence guard:** abort a round on non-finite loss (NaN/Inf), log the signal, resume from last good checkpoint (R8).
   - emit per-round metrics: policy_loss, value_loss, entropy, mean_adv, mean_value, grad_norm.
4. **`contract.py` / `export_onnx.py`**: value head is **training-only** — export drops it; exported ONNX stays policy-only contract v1 (`obs`→`tech_logits`,`policy_logits`). **Negative test**: exported model has no `value` output.
5. **`run_loop.py`**: add `--variant {v1-reinforce, blind-critic, rich-critic}` (selects algo+model+contract), per-round `.pt` checkpoint + `--resume`, and a metrics column set. `v1-reinforce` preserves the exact v1 path for the attributable baseline.
6. **Produce curves**: `v1-reinforce` + `blind-critic` on Tiny, overlaid `curve.png` + per-variant `curve.csv`. Convergence stat: last-K=4 win-rate stddev (blind-critic < v1-reinforce).

### Stage B — Rich representation (map + entities; the contract grows)
7. **`features.py` (new)** + `dataset.py`: assemble the full per-step multi-tensor input from `Step.blocks`:
   - `spatial` nTiles×13 → **per-tile token set** (the 13 channels only — NO positional feature; zeroBasedIndex is insertion order not 2D position, including it risks spurious-order overfit — R6) + a padding mask (first nTiles valid, padding beyond batch-max nTiles invalid); padded to batch-max nTiles.
   - entity blocks (own_units, opp_units, own_cities, opp_cities, civ_tokens, diplo_edges) → per-type padded token sets + presence masks (present-only on disk → padded to batch-max).
8. **`model.py`**: `RichPolicyValueNet` — per-tile token MLP → masked mean+max pool; per-entity-type token MLP → masked mean+max pool; concat with global+acting_civ → trunk MLP → {tech, policy, value}. Modest (hidden≈256). **Masked-pool NaN guard:** mean = sum/clamp(count,min=1); max over an empty set → 0; an entity type with 0 present → zero vector (R3, unit-tested).
9. **Contract v2** (`contract.py` + Kotlin `SampleSchema.OnnxContract`, lockstep): named multi-tensor inputs + presence/validity masks, dynamic axes (batch / nTiles / entity counts). `CONTRACT_VERSION`→2. `export_onnx` exports the multi-tensor policy-only model (value dropped).
10. **`OnnxPolicy.kt`**: build the SAME multi-tensor input from the live `Observation` (reuse `block(name)` + VARIABLE counts; feed u8 spatial as f32), `session.run(Map<name,OnnxTensor>)`, read logits, mask + sample (seeded `MaskedChoice` unchanged). Heads not modeled → −1 (heuristic fallback unchanged). Provenance gate validates contract v2 + tensor inventory.
11. **`SelfPlayRunner.kt` parity-dump/parity-run**: dump/run the multi-tensor obs (JSON of named arrays) instead of the single 199-vector.

### Stage C — Eval
12. **Map-size CLI param** (`SelfPlayRunner.mapParameters(seed, mapSize)`), threaded through `gen`/`eval`. Keep Tiny win-rate-vs-RandomPolicy + binomial harness (comparability). Add **Medium** training+eval for `blind-critic` vs `rich-critic`; two-proportion z-test sized for p<0.05 (~200 eval games).

## File change inventory
- **Edit (python):** `model.py`, `train.py`, `dataset.py`, `contract.py`, `export_onnx.py`, `run_loop.py`; **new** `features.py`.
- **Edit (kotlin):** `desktop/.../OnnxPolicy.kt`, `desktop/.../SelfPlayRunner.kt` (map-size arg + parity dump/run), `core/.../dataplane/SampleSchema.kt` (OnnxContract v2).
- **Tests:** `python/tests/test_parity.py` (multi-tensor), new `test_gae.py` (GAE math + terminal-only-reward + no-value-output export), keep `test_determinism.py`/`test_train_dataset.py`; Kotlin `OnnxPolicyLegalityTest` stays green.
- **No-touch:** Featurizer/Observation/emitter/shard format/masks/RulesetFingerprint/reader provenance/RandomPolicy.

## Operational
- **Budget (D13, HARD):** Tiny: 3 variants × 12 rounds × (24 gen + 100 eval). Medium: 2 variants × bounded rounds + ~200-game eval. Smoke-measure per-round wall-clock, set a ceiling, STOP at budget (report honestly even if borderline).
- **Checkpoint/resume + observability (D14):** per-round `.pt` + resumable loop; richer metrics file.
- **Env (D16):** create python venv + `pip install -e ./python` before training.

## Acceptance (operationalized — see decisions.md D12 + R5)
**Definition of done = the experiment is executed correctly and reported honestly** (the scientific results — convergence stddev, Medium ceiling p-value — are reported as-is, including "not significant"; p<0.05 is the hypothesis under test, NOT a ship-blocking gate — R5; the brief's AC3 explicitly permits this).
AC1 attributable curves (shared harness/opponent/seeds, only named axis differs; both critic variants use plain A2C+GAE so the axis is clean) · AC2 report last-K=4 win-rate stddev AND mean late win-rate for blind-critic vs v1-reinforce (steadier is meaningful only with the level — R9) · AC3 rich vs blind on Medium, two-proportion test + p-value reported (significant or stated plainly) · AC4 multi-tensor parity atol=1e-4 · AC5 determinism+provenance+contract bump · AC6 legality green · AC7 terminal-only reward (grep/assert).

## Security hardening (council 🟡 — R7)
Checkpoint load = `torch.load(weights_only=True)` (no arbitrary-code path); every JVM `OnnxTensor` closed via try/finally over the tensor map (no native leak across many decisions); parity JSON shape-validated before ORT; mask tensors included in ONNX `dynamic_axes`. Self-generated local data, but cheap guards.

## Walkthrough
One learner decision → trained credit, traced with illustrative mock shapes (sanity-check against the design above; not measured):

1. **JVM decision (rich-critic play):** `Featurizer.observe(civ)` builds `Observation` on a Medium map (nTiles≈1261). `OnnxPolicy` reads blocks → tensors: `spatial`[1,1261,14], `own_units`[1,7,8]+mask[1,7], `opp_units`[1,3,8]+mask, `own_cities`[1,2,16], …, `global`[1,26], `acting_civ`[1,173]. `session.run(map)` → `tech_logits`[1,80], `policy_logits`[1,70]. Contract: v2 named tensors.
2. **Mask+sample:** `MaskedChoice.choose(tech_logits, legalTechMask, eval, rng)` → e.g. tech index 12 (Bronze Working); illegal indices were −1e9. Seeded RNG ⇒ replayable.
3. **Emit step:** `TrajectoryEmitter` writes the step (blocks + `actions=[12, 4, -1, -1]`, reward 0) to the shard. Contract: shard schema v2.
4. **Game ends:** learner wins on turn 137 → terminal step reward `+1.0` (only there). Contract: terminal-only ±1.
5. **Python train:** `dataset.load_trajectories` returns the ordered 137-step learner sequence (rewards all 0 except +1 at end). `train`: V predicts e.g. [0.1,0.12,…,0.9]; δ_t, GAE(0.99,0.95) → advantages e.g. A_12-step ≈ +0.4 (this early Bronze-Working state credited toward the eventual win); `policy_loss = −A·logp`, `value_loss = MSE(V,R)`.
6. **Export+eval:** value head dropped → policy-only v2 ONNX; eval 200 Medium games vs RandomPolicy → win-rate + binomial p; appended to `curve.csv`; compared rich vs blind (AC3).

## Risks
| # | Risk | L×S | Mitigation |
|---|---|---|---|
| 1 | Full-run wall-clock blows up (Medium games long) | M×H | Hard budget + per-round checkpoint/resume; smoke-measure first; STOP at ceiling, report honestly |
| 2 | Token-pool spatial doesn't beat blind on Medium (no 2D locality) | M×M | Honest AC3 reporting allowed; rich still sees entities/terrain blind never gets; positional scalar added |
| 3 | Multi-tensor parity drift JVM↔Python | M×H | Extended parity test (atol=1e-4) is the anti-drift gate; feed u8→f32 consistently both sides |
| 4 | GAE/value instability (bootstrap) | L×M | Advantage normalization, c_v=0.5, optional PPO clip when epochs>1, entropy bonus, grad-norm logging |
| 5 | Contract bump breaks v1 path / baseline | L×H | v1-reinforce variant preserves the exact v1 contract-v1 path; bump only affects rich-critic |
