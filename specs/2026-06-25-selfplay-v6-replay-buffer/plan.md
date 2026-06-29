# Plan — selfplay-v6: PPO-correct cross-round replay buffer

## How this solves the ask
v5 (continual warm-start + micro-batch) was the first config to clear the blind baseline on Medium
(40.7% vs 28.9%, p=0.0069). v5 trains each round on ONLY that round's ~16 games and **deferred**
cross-round reuse because trajectories store no behavior-policy log-probs — so reusing them would
anchor PPO's importance ratio to the wrong policy → silently biased gradient. v6 closes that gap:
**record per-head behavior log π_b(a|s) at the sampling point**, **use the stored π_b as `old_logp`
for replayed (off-policy) steps**, and add a **small recent-round-window replay buffer** so each
update sees up to K rounds of games (K=4 ≈ 64 vs 16). With the real π_b in hand, PPO's existing clip
becomes correct off-policy importance sampling instead of an always-≈1 no-op. Then re-run the Medium
comparison to answer: does replay reach v5's 40.7% in FEWER generation rounds, or a HIGHER 200-game
ceiling at equal rounds?

## Correctness invariant (headline — 2 council reviewers raised it)
**The value/critic side stays on-policy-fresh.** `val0` is ALWAYS a current-net forward (never
sourced from storage); GAE (`compute_gae`) is recomputed per-trajectory under the CURRENT critic
every round, episodically per game (traj_lens preserve boundaries). Terminal-only ±1 rewards are
fixed. ONLY the policy-ratio `old_logp` source switches to the stored π_b for replayed steps. This
hybrid — fresh critic + off-policy-corrected policy ratio — is the textbook treatment (research-notes).

## Pillar A — record behavior logp (Kotlin) + shard schema VERSION 3→4
- `core/.../MaskedChoice.kt`: add `fun chooseWithLogp(logits, mask, eval, rng): Pair<Int, Float>` —
  the chosen index AND its masked-softmax log-prob `ln(exps[pos]/sum)` (= `log_softmax(masked
  logits)[chosen]`, the exact quantity `_masked_logp` scores). Empty support / idx<0 → `(-1, 0f)`.
  Eval (argmax) logp is unused (eval games never recorded) — documented, not special-cased.
  **`choose` is re-expressed as `chooseWithLogp(...).first`** so there is exactly ONE code path and
  ONE `rng.nextDouble()` draw in identical order (generation replay must stay byte-identical).
- `core/.../PolicyProvider.kt`: default `fun chooseIndexWithLogp(head, civ, legalMask, turn):
  Pair<Int,Float>` delegating to `chooseIndex`, returning uniform-over-legal `ln(1/nLegal)`. **KDoc
  WARNING (plan council):** this uniform logp is correct ONLY for a uniform policy (the RandomPolicy
  stub) AND the value is discarded by the learner-slot filter (round-0 RandomPolicy data is excluded
  from replay); ANY non-uniform PolicyProvider MUST override this to record its true sampling logp.
  `chooseIndex` signature unchanged (RandomPolicy non-breaking).
- `desktop/.../OnnxPolicy.kt`: override `chooseIndexWithLogp` — reuse memoized `logitsFor`, call
  `MaskedChoice.chooseWithLogp(logits, legalMask, eval, rngFor(civ,turn))`, increment `decisions` on
  idx≥0. `chooseIndex` delegates to `chooseIndexWithLogp(...).first` (single draw / single source).
- `core/.../DataPlaneHooks.kt`: in `chooseAndApply`, build `behaviorLogp = FloatArray(MASK_HEADS.size){0f}`
  alongside `actions`; at the tech + policy sites call `chooseIndexWithLogp(...)` and write
  `actions[0/1]=idx`, `behaviorLogp[0/1]=logp`. Return `(actions, behaviorLogp)`; `handleCivTurn`
  destructures and passes `behaviorLogp` into `recordStep`.
- `core/.../DataPlaneHooks.kt` `ShardRecorder.recordStep`: after the existing `actions` block append
  `Observation.Block(SampleSchema.BLOCK_BEHAVIOR_LOGP, DT_F32, FIXED, 0, behaviorLogp)` — same width
  (MASK_HEADS.size), same perItem=0. **Appended at EMIT time** (scan finding #3) right after actions;
  framePayload / buildHeaderJson layout / recordTerminal zero-fill all flow generically (no other change).
- `core/.../SampleSchema.kt`: add `const val BLOCK_BEHAVIOR_LOGP = "behavior_logp"` (KDoc: per-head
  log π_b at sampling time, MASK_HEADS order {tech,policy}, shard-only, NOT an ONNX I/O). Bump
  `VERSION 3 → 4` + update the version KDoc.
- `python/unciv_dataplane/schema.py`: `SCHEMA_VERSION 3 → 4` (the single Python gate). Old v3 shards
  refuse to load (perishable discipline — intended). `reader.py` `_decode_blocks` is layout-driven →
  decodes the new FIXED f32 block generically; only version-message text touched if needed.

## Pillar B — PPO uses stored behavior logp as old_logp for REPLAYED steps (Python)
- `dataset.py` `TrainTrajectory`: APPEND `b_logp_tech: np.ndarray`, `b_logp_policy: np.ndarray`
  ([T] f32, 0 where the head did not act) at the END of the dataclass; `load_trajectories` reads
  `np.array([float(s.blocks["behavior_logp"][0]) for s in steps])` (and `[1]`), passed positionally at
  the END of the ctor call. No `.get()` fallback (SCHEMA_VERSION=4 guarantees the block; fail-loud).
  `load_training_steps`/`TrainStep` (v1 REINFORCE) untouched.
- `train.py` `_stack_traj`: concatenate `b_logp_tech` / `b_logp_policy` in the SAME order as `a_tech`
  (flat-batch alignment); return them.
- `train.py` `_optimize_actor_critic`: add `stored_old_logp: torch.Tensor | None = None`. THE change at
  the old_logp source (lines ~150-164): keep running the forward for `val0` ALWAYS; only switch the
  policy-logp SOURCE —
  - micro branch: keep accumulating `v_parts` always; append to `lp_parts` ONLY when `stored_old_logp
    is None`; `val0 = torch.cat(v_parts)`; `old_logp = stored_old_logp.detach() if stored_old_logp is
    not None else torch.cat(lp_parts).detach()`.
  - whole branch: `val0` from forward as today; `old_logp = stored_old_logp.detach() if … else
    (_masked_logp(...)+_masked_logp(...)).detach()`.
  EVERYTHING ELSE verbatim (ratio+clip, value MSE, entropy, objective, grad-clip 10.0, divergence
  guard + safe_state/safe_opt rollback, micro weight). **GUARD:** assert `clip_eps` truthy whenever
  `stored_old_logp` is set AND replay-window>1 (the clip_eps-falsy branch ignores old_logp → would
  silently apply replayed advantages on-policy = biased). Fail loud with a clear message.
- The three trainers (blind/_rich/_structured) gain a `behavior_logp` plumb-through param (after
  `micro_batch_steps`); each sums the two head arrays → `stored = b_logp_tech + b_logp_policy` and
  passes `stored_old_logp=stored` into `_optimize_actor_critic`.

## Pillar C — recent-window replay buffer in the round loop (run_loop.py)
- argparse `--replay-window` (int, default 4): K rounds back; K=1 ⇒ no replay (== v5).
- Before the round loop: `from collections import deque; replay = deque(maxlen=max(1, args.replay_window))`.
- After `data = ds.load_trajectories(...)`: for r≥1 `replay.append(data)` then `train_data =
  [t for round_trajs in replay for t in round_trajs]`; for r==0 train on `data` directly (round 0 is
  RandomPolicy-generated → maximally off the current net → EXCLUDED from the window). Pass `train_data`
  into `train_round`. Warm-start + micro-batching untouched (orthogonal; mb now chunks the larger n).
- **Window-gated source switch (NO-OP SAFETY):** when `args.replay_window <= 1`, pass
  `behavior_logp=None` into the trainers → `_optimize_actor_critic` takes the ORIGINAL recompute branch
  (literal v5 path; zero numerical drift); train_data==data. K=1 is bit-identical to v5.
- `--resume` persistence: floor `keep_shards = max(args.keep_shards, args.replay_window - 1)` so the
  last K rounds' `round_*/*.bin` survive the prune; on `--resume`, before the loop, refill the deque by
  `ds.load_trajectories` over rounds **`max(1, start-(K-1)) .. start-1`** — **never round 0** (D7: the
  in-process loop excludes round 0 from the window, so the resume refill must too; globbing
  `[start-K..start-1]` would re-admit RandomPolicy data). Missing kept dir → refill what exists + warn,
  don't crash. Covered by `test_replay_resume.py`.

## Pillar D — off-policy variance via EXISTING knobs only (no new machinery)
small K window (deque maxlen=4) · PPO clip ε=0.2 · `logratio.clamp(±20)` · `norm_adv` · grad-clip 10.0
· divergence guard + safe_opt rollback. NO V-trace / truncated-IS / soft-clip. Lever if variance is
empirically high = lower `--replay-window` or `--clip-eps` (existing knobs).

## ➕ ADDED (council) — replay-health diagnostics (D3)
In `_optimize_actor_critic`, add to the existing `stats.update(...)` (NOT the verbatim math lines):
`mean_ratio` and `clip_frac` (computed from the already-present `ratio` tensor in the clip branch),
and `frac_replayed` (threaded from the trainer = replayed_steps / total_steps). Emit into
`metrics.jsonl`. Read-only diagnostic → K=1 no-op stays bit-identical. Lets the multi-hour run be
watched (mean_ratio≈1 confirms near-on-policy; clip_frac shows how often the trust region binds).

## Test plan (RED-first; all FAST, the merge gate)
1. **Kotlin** `MaskedChoiceLogpTest` (new, mirrors OnnxPolicyLegalityTest): `chooseWithLogp` returns a
   legal idx + logp == `ln(softmax_over_legal)[idx]`; empty → `(-1,0f)`; `choose == chooseWithLogp.first`
   (same idx for same seed — single-draw equivalence). Existing OnnxPolicyLegalityTest must still pass.
2. **Python** `test_replay_noop.py`: K=1 (behavior_logp=None) trainer output is bit-identical to the
   pre-v6 recompute path on synthetic trajectories (mirrors `test_microbatch_noop_when_K_ge_N`).
3. **Python** `test_offpolicy_equivalence.py`: stored gen-logp ≈ recomputed warm-net logp within 1e-4
   ⇒ ratio = exp(logp − old_logp) ≈ 1.0 at epoch 0 (the gen net IS the round's exported warm net).
4. **Python** `test_behavior_logp_shard.py`: a v4 shard with a `behavior_logp` FIXED f32 block
   round-trips through reader → `load_trajectories` → `TrainTrajectory.b_logp_*`; a v3 shard refuses.
5. **Python** `test_clip_eps_guard.py`: `_optimize_actor_critic(stored_old_logp=…, clip_eps=0)` with
   replay raises a clear error; `clip_eps=0` WITHOUT replay (plain A2C) is still allowed.
7. **Python** `test_replay_resume.py` (plan council): on `--resume` the deque refill EXCLUDES round 0
   (refills `max(1,start-(K-1))..start-1`), and a missing kept round dir → warn + refill-what-exists,
   no crash. Non-acting-head logp 0f covered (Kotlin empty-support + shard terminal 0,0).
8. **Tiny smoke (merge-gate, D-experiment):** `run_loop --map-size Tiny --rounds 2 --gen-games 2
   --eval-games 2 --variant structured --rung small --replay-window 4`. PASS = runs to completion;
   round 0 trained alone (excluded from window); round 1 trains on current∪round-0-excluded; v4 shards
   written + reloaded; `frac_replayed`>0 at round≥… (with K=4 and round-0 excluded, replay engages once
   ≥2 non-zero rounds exist); no crash/NaN.
6. **Fixture updates (scan finding #6, REQUIRED for pytest green):** `test_train_dataset.py`
   `_build_v2_shard` default version → SCHEMA_VERSION + behavior_logp block where it feeds
   load_trajectories; `test_v2_units.py` `_shard_with_steps` schemaVersion 2→4 + behavior_logp block;
   verify `test_parity.py` / `test_structured_smoke.py` / `test_continual_resume.py` export calls still
   pass (parametrize schema_version from the live value where it's a real version, not the contract const).

## Experiment driver (D4) — `python/run_v6.sh` (mirrors run_v5.sh), 4 arms
structured, Medium, 16 rounds, gen-games 16, eval-games 80, turn-cap 250, epochs 8, continual,
gen-seed 1000 / eval-seed 999000, then 200-game ceiling @ eval-seed 4242424 + z-tests (analyze_v5):
- small-rung K=1 (`--rung small --micro-batch-steps 0 --replay-window 1`) — must reproduce v5 40.7%.
- small-rung K=4 (`--rung small --micro-batch-steps 256 --replay-window 4`).
- medium-rung K=1 (`--rung medium --micro-batch-steps 256 --replay-window 1`) — reference v5 46.6%.
- medium-rung K=4 (`--rung medium --micro-batch-steps 256 --replay-window 4`).
D8: **K=4 arms set `--micro-batch-steps 256`** — the 4× assembled batch would otherwise hit the
whole-batch optimizer path and risk OOM; micro-batching is math-identical (v5 AC6) so it does NOT
confound the K=1-vs-K=4 comparison. Resumable via `--resume`; launched in the background this session;
AC1/AC2 reported when complete.

## Acceptance (reframed — D5)
- **AC1 (no-op):** determinism gate = unit bit-identity (test_replay_noop) + gen byte-identity
  (single-draw Kotlin test) GREEN ⇒ end-to-end K=1 reproduces v5's exact curve; **acceptance** = K=1
  200-game ceiling within the 95% binomial CI of v5's 40.7% (n=200 ⇒ ≈±6.8%), else investigate.
  "1e-4 fp tolerance" → ONLY the unit stored≈recomputed test (test_offpolicy_equivalence).
- **AC2 (sample efficiency):** headline win = K=4 **200-game ceiling** z-test vs the K=1 arm with
  p<0.05 (NOT the noisy 80-game per-round eval); round-to-≥40.7%/≥46.6% in fewer GENERATED rounds is
  directional/secondary; else NULL (valid result). Sample efficiency = fewer GENERATED games (gen
  dominates wall-clock); K=4 ≈4× training compute/round acknowledged. Both framings + z/p.
- **AC3 correctness:** stored old_logp for replayed (asserted); GAE under current critic each round;
  clip_eps-truthy guard. **AC4 schema:** 3→4 lockstep (Kotlin+Python+fixtures); v3 refuses; ONNX
  contract names/widths/keys/CONTRACT_VERSION* byte-unchanged (schema_version value moves 3→4 in
  lockstep — intended). **AC5 determinism:** one rng draw/sample, byte-identical gen.
  **AC6:** terminal-only ±1 + heads {tech,policy} unchanged.

## Out of scope / NOT this (council A5)
deque(maxlen=K) ONLY — NO sampler/priority/persistence framework, NO head registry, NO configurable
layouts. Two known heads {tech,policy}. NO control heads, NO encoder/reward/matchup change, NO ONNX
contract change, NO V-trace/truncated-IS/soft-clip, NO prioritized replay, NO cross-ruleset replay,
NO shard-tamper validation (local trusted pipeline; logratio.clamp is defense-in-depth). Merge gate =
code + unit assertions + Tiny smoke GREEN; the Medium AC1/AC2 numbers are reported evidence, not a blocker.

## Walkthrough
See `## Walkthrough` below in the rendered plan (one decision step traced gen→shard→train).

## Walkthrough (canonical)
Trace one learner tech-decision through the system at round r (K=4):
1. **Generate (JVM, round r):** `OnnxPolicy.chooseIndexWithLogp("tech", civ, mask, turn)` → reuse
   memoized logits → `MaskedChoice.chooseWithLogp(logits, mask, eval=false, rngFor(civ,turn))` draws
   ONE `nextDouble()`, returns `(idx=42, logp=-2.31)` where −2.31 = `ln(softmax_over_legal)[42]`.
2. **Apply + record:** `chooseAndApply` writes `actions[0]=42f`, `behaviorLogp[0]=-2.31f`; `recordStep`
   emits the step with an `actions` block `[42,-1,-1,-1]` and a `behavior_logp` block `[-2.31, 0]`.
3. **Shard (v4):** header layout lists `…,"actions",…,"behavior_logp"`; reader decodes both by name.
4. **Load (Python, round r):** `load_trajectories` → `TrainTrajectory(… a_tech=[…,42,…],
   b_logp_tech=[…,-2.31,…] …)`. The deque holds rounds r−3..r ⇒ `train_data` flattens ~64 games.
5. **Train:** `_stack_traj` concatenates `a_tech`/`b_logp_tech`; `stored = b_logp_tech + b_logp_policy`.
   `_optimize_actor_critic(stored_old_logp=stored)`: `val0`=current-net forward (fresh) → GAE per game;
   for this step `old_logp=-2.31` (its behavior logp), current-net `logp=-2.10` ⇒ `ratio=exp(-2.10−
   (−2.31))=exp(0.21)=1.23`; clip(1.23, 0.8..1.2)=1.2 → surrogate uses min(1.23·A, 1.2·A). At K=1 the
   same step's `old_logp` would instead be the current-net recompute (≈ratio 1.0, the v5 no-op).
   *Illustrative mock values — not measured; sanity-check against the design.*
