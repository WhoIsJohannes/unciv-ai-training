# Decisions — onnx-selfplay-v2

## Setup / scope (user-answered Phase 1)
- **D1 — Worktree:** v1 was uncommitted WIP; per user, committed v1 (`8e0e4ba0a`), fast-forwarded `master`, branched `onnx-selfplay-v2` worktree off it. Clean committed base.
- **D2 — Training scope:** "Build + full run to acceptance." BOUNDED with a hard compute budget + stopping rule + per-round checkpoint/resume (council 🔴/🟡 on unbounded runs). Measured per-round wall-clock will be surfaced before the long run; budget adjustable at the Step 11 gate.

## Council intake roster rationale
Reused v1's intake roster (identical RL/ML self-play domain): Core 6 + `domain_fidelity` (RL/ML correctness), `ethics_responsible_ai` (ML failure-mode / value-bootstrap instability), `cost_efficiency` (full-run compute). 9 roles, within L's ≤14. No user-facing/auth/PII surface ⇒ no end_user/data-privacy roles beyond core security_red_team.

## Architecture decisions

- **D3 — Spatial encoder = per-tile-token + positional feature + masked pooling (NOT grid-CNN).**
  Rationale: deep scan confirms per-tile (col,row) is NOT recoverable from shard data (`zeroBasedIndex` is tile-insertion order, positions not emitted, maps hexagonal); a grid-CNN would require emitting tile positions — forbidden by the "no new engine-emitted data" non-goal — or fragile generator-order replay. Council 🔴 ("2D grid reconstruction impossible") + 🟡 ("prefer simple pooling") concur, and the task explicitly sanctions this fallback. Each tile → token of its 13 channels + a cheap positional scalar (normalized zeroBasedIndex); masked mean+max pool over present (explored) tiles.
  **Consequence (honest, vs AC3 ceiling):** 2D conv locality is sacrificed. The rich model still sees the full board as a *set* of tiles (terrain/resource/units/owner) + all entity tokens that the blind 199-dim never gets; the ceiling test asks "does seeing the map+entities help," which token-pool delivers. AC3 permits reporting the curves and "stating plainly if not."

- **D4 — Entity encoder = per-type token MLP → masked mean+max pool, concatenated.**
  Types: own_units(8), opp_units(8), own_cities(16), opp_cities(16), civ_tokens(84), diplo_edges. Each type: shared small MLP per token → masked pool (padded entities excluded). Permutation-invariant (Deep Sets); attention is a deferred upgrade (start simple — council 🟡 + research).

- **D5 — Trunk + heads.** Concat[ global(26), acting_civ(173), spatial_pool, per-type entity pools ] → 2-layer MLP (hidden≈256) → shared trunk → heads {tech_logits(80), policy_logits(70), value(1)}. Modest sizes (CPU-trainable). Blind variant skips the spatial+entity pools (input = global+acting_civ only) but keeps the value head.

- **D6 — Actor-critic + GAE.** γ=0.99, λ=0.95. **Dataset restructured** to emit ordered per-game learner-step trajectories with reward = 0 except terminal ±1 placed at the last learner step (NO v1-style broadcast — the 🔴 incompatibility). GAE computed per-trajectory: δ_t = r_t + γV(s_{t+1}) − V(s_t), V(terminal)=0 bootstrap; advantage = Σ(γλ)^l δ_{t+l}; return target = advantage + V(s_t). Reward stays terminal-only ±1 (AC7) — the critic is the only new credit mechanism, learned from terminal outcome.

- **D7 — Loss.** `policy = −(adv.detach() · Σ_head masked_logp)`; `value = MSE(V(s), return)`; `total = policy + c_v·value − c_ent·entropy`. Defaults c_v=0.5, c_ent=0.01. **PPO clip:** when inner epochs > 1, clipping is REQUIRED (council 🟡 + research — stale-ratio blowup otherwise); default = PPO-clip ε=0.2, K=4 epochs; single-epoch A2C also supported (no clip). Reuse v1 `_masked_logp` (illegal→−1e9, gather chosen, 0 where action<0) and the −1/no-action handling VERBATIM.

- **D8 — Variant scheme `--variant {v1-reinforce, blind-critic, rich-critic}`** selects (algorithm, model, contract):
  - `v1-reinforce`: exact preserved v1 path (REINFORCE + mean baseline, blind MLP, contract v1) — the attributable baseline. NOT broken by Stage A.
  - `blind-critic`: A2C+GAE on the blind 199-dim input (contract v1, value head dropped at export → JVM bridge UNCHANGED).
  - `rich-critic`: A2C+GAE on the multi-tensor rich input (contract v2).

- **D9 — Value head is training-only.** `export_onnx` drops it; exported play-model is policy-only (`tech_logits`,`policy_logits`). Negative test asserts the exported ONNX has NO value output (council 🟡 "provably dropped").

- **D10 — Contract v2 = named multi-tensor input** (global, acting_civ, spatial + its valid-mask, own_units/opp_units/own_cities/opp_cities/civ_tokens/diplo_edges each with a presence mask) with dynamic axes (batch, nTiles, entity counts). `CONTRACT_VERSION`→2 in lockstep on `contract.py` + `SampleSchema.OnnxContract`. Provenance gate keeps META_SCHEMA_VERSION/META_CONTRACT_VERSION/META_RULESET_FINGERPRINT and validates v2 + the tensor inventory (council 🟡 "gate coverage"). Hard break vs v1 .onnx is expected (perishable artifacts). u8 spatial fed as f32 on the JVM.

- **D11 — Map-size CLI param** threaded `gen`/`eval` → `mapParameters(seed, mapSize)`. Tiny kept for comparability (AC1/AC2); Medium added for the ceiling test (AC3). nTiles dynamic ⇒ Medium shards work unchanged.

## Operationalized acceptance (council 🔴 "cited not operationalized")
- **D12a — Attribution protocol (AC1):** v1-reinforce, blind-critic, rich-critic run through the SAME harness, SAME opponent (RandomPolicy), SAME eval seeds + game counts + map; ONLY the named axis (algorithm / representation) differs. curve.csv per variant + overlaid plot.
- **D12b — Convergence (AC2):** ≥12 Tiny rounds per variant; metric = stddev of eval win-rate over the last K=4 rounds. PASS = blind-critic last-K stddev < v1-reinforce last-K stddev (report both; "measurably steadier"). Also report mean late win-rate.
- **D12c — Ceiling (AC3):** blind-critic vs rich-critic trained AND evaluated on Medium; two-proportion z-test (and per-variant binomial vs 0.5) on a fixed eval-game count sized for p<0.05 power (≈200 games). PASS = rich win-rate > blind at p<0.05; else report curves + state plainly.
- **D12d — Parity (AC4):** `test_parity.py` extended to the full multi-tensor input; JVM-built tensors + logits match Python ORT within atol=1e-4.
- **D12e — Determinism/provenance (AC5):** same seed → byte-identical shards (existing test); contract version bumped; fingerprint stamped + gated.
- **D12f — Legality (AC6):** `OnnxPolicyLegalityTest` green; zero illegal-action exceptions across the runs.
- **D12g — Terminal-only reward (AC7):** grep/assert no hand-authored intermediate reward term; reward placed only at terminal step in dataset + emitter unchanged.

## Operational hardening (council 🔴/🟡)
- **D13 — Compute budget + stopping rule (HARD):** per-variant Tiny = 12 rounds × (24 gen + 100 eval) games; Medium ceiling = 2 critic variants × bounded rounds + ~200-game eval. Smoke-measure per-round wall-clock first; set a wall-clock ceiling; STOP at budget even if a criterion is borderline (report honestly). Surfaced for approval at Step 11.
- **D14 — Checkpoint/resume + observability:** per-round `.pt` checkpoint + resumable `run_loop`; per-round metrics (policy_loss, value_loss, entropy, mean_adv, mean_value, ret_pos, grad_norm) appended to curve.csv / a metrics file.
- **D15 — Dual-use (security_red_team 🟡):** research artifact; opponent is RandomPolicy; learner sees only fairness-gated observations (leakage tests already green); Tiny/Medium maps. Negligible real-world dual-use; documented, not a blocker.
- **D16 — Env setup prerequisite:** no python venv present; create one + `pip install -e ./python` before any training (Phase 3 first step).

## Open for user at Step 11 gate
- Compute budget (D13) magnitude — concrete numbers proposed; user may dial rounds/games at approval.

## Plan council resolutions (Step 11) — folded into the plan
- **R1 (🔴 no-action filtering breaks GAE / drops reward):** dataset emits the FULL ordered sequence of non-terminal learner steps (do NOT drop both-heads=−1 steps as v1 did). Value + GAE run over every step; policy loss is masked to acting heads (logp=0 where a head didn't act — v1 machinery). Terminal ±1 placed at the final transition (bootstrap V(terminal)=0). Amends D6.
- **R2 (🔴 per-trajectory adv-norm destroys signal):** advantage standardization is BATCH-level (all steps from all trajectories pooled, so winning and losing games coexist) — never per-trajectory. Flag-controlled; default on. Amends D7.
- **R3 (🔴 NaN masked pooling on empty sets):** masked mean = sum / clamp(count, min=1); masked max over an empty set → 0 (not −inf). Entity type with 0 present → zero vector. Covered by a unit test. Amends D4.
- **R4 (🔴 PPO conflates critic vs PPO in isolated curves):** the attributable curves use **plain A2C+GAE, single inner epoch, NO PPO clip** so v1-reinforce→blind-critic isolates ONLY the credit mechanism and blind-critic→rich-critic isolates ONLY the representation. PPO-clip (ε=0.2, K=4) is an OPTIONAL non-default knob, not used for headline curves. Amends D7/D8.
- **R5 (🔴 p<0.05 vs bounded budget):** ACCEPTANCE = execute the operationalized experiment correctly + report results honestly (curves, last-K stddev, Medium two-proportion test with its p-value, "states plainly if not significant"). The p<0.05 is the hypothesis under test, NOT a ship-blocking gate (brief AC3 itself permits this). Amends D12c.
- **R6 (🟡 positional feature spurious):** DROP the zeroBasedIndex positional scalar — it is insertion order, not 2D position; including it risks overfitting to ordering. Spatial encoder = pure permutation-invariant masked pool over the multiset of per-tile 13-channel vectors. Amends D3.
- **R7 (🟡 security hardening):** checkpoint load uses `torch.load(weights_only=True)`; every JVM `OnnxTensor` is closed (try/finally over the tensor map); parity JSON is shape-validated before ORT; mask tensors included in `dynamic_axes`. 
- **R8 (🟡 divergence detection):** abort a round on non-finite loss (NaN/Inf) with a logged signal; resume from last good checkpoint. gradle gen/eval already has `--gradle-timeout`.
- **R9 (🟡 AC2 stable-but-bad):** report BOTH last-K stddev AND mean late win-rate; "steadier" is only meaningful alongside the level (a stable 50% is not a win).
- **R10 (🟡 padding/cost):** pad to batch-max (not global cap) per batch; entity counts already capped by SampleCaps; Medium high-N eval is FINAL-gated (run once at the end), not per-round.
- **R11 (🟡 Stage B big-bang):** Stage B validated incrementally behind the parity test (spatial pool, then entity pools) but shipped together; parity atol=1e-4 is the correctness guard.

## R12 — Final A2C/GAE recipe (supersedes R4 + progress.md's "recompute-each-epoch" note)
Calibration on real (large-magnitude, unnormalized) game features surfaced two failures that drove
the final recipe; the as-built `_optimize_actor_critic` is the **standard PPO/A2C** form:
- **Advantages + GAE returns computed ONCE per round from a V-snapshot** (fixed regression target ≈
  bounded discounted-terminal return at λ≈0.95). Recompute-each-epoch (the R4 attempt) degenerated
  `value_loss` to `mean(adv²)` (≈477) because the target chased V — rejected.
- **PPO clip ε=0.2 default ON, K=8 inner epochs** (a fresh net per round needs K steps to fit the
  critic; multi-epoch reuse of fixed advantages ⇒ clipping required, exactly the council's point).
  `clamp(logp−old_logp, ±20)` guards `exp()` overflow on large fresh-net policy shifts.
- **tanh-bounded value head** (true value = expected discounted terminal reward ∈ [−1,1]) + **small
  value-head init** (V≈0 at start) so the critic stays bounded despite unnormalized inputs. The value
  head is dropped at export ⇒ zero parity/contract impact.
Result: value_loss ~1, diverged=0, healthy entropy. **Attribution preserved:** v1-reinforce ignores
the value head (baseline byte-unchanged); blind-critic and rich-critic share the identical PPO+GAE
algorithm so blind→rich isolates representation, v1→blind isolates the credit mechanism.
