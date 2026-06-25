# Plan — selfplay-v5: Continual Training

## WHAT
Make self-play RL training **continual**: keep a persistent `(net, optimizer)` pair alive across the
round loop so each round warm-starts from the previous round's weights AND carries Adam moment state,
instead of re-initializing a fresh net + fresh optimizer every round. Add **micro-batching** to the
dense whole-round forward/backward so the structured encoder's *medium* rung can train on Medium
(maxn≈1261) without the v4 OOM. Then re-run the v4 Medium comparison.

## WHY (the ask)
v4 proved the structured GNN encoder is NOT the bottleneck (it cleared the rich-pool baseline,
p=0.014), yet the correctly-built structured net still sat at 23.0% — BELOW the blind baseline
(28.9%). The remaining suspect is the **training regime**: every round trains a brand-new net on ~16
games and discards all prior learning. v5 tests whether *not resetting* (warm-start weights + carried
Adam state, over more rounds) finally clears blind. Research confirms carrying the full Adam triple
across rounds is a validated continual-RL technique (faster, smoother) and is NOT washed out at our
~128 total update steps.

## HOW (design, all in `python/unciv_train/`)

### 1. Lift the optimizer out of `_optimize_actor_critic` + inject it (train.py)
- Delete the inline `opt = torch.optim.Adam(net.parameters(), lr=lr)` (train.py:136). Add a required
  `optimizer` parameter; use it directly. The per-update math (advantage snapshot, PPO clip, value MSE,
  entropy, combined objective, grad-clip max_norm=10.0, `opt.step()`) stays IDENTICAL.
- **Opt-aware divergence guard** (FND-0010/0012): at init also `safe_opt = copy.deepcopy(optimizer.state_dict())`
  (alongside safe_state @139); on the NaN branch (@180) also `optimizer.load_state_dict(safe_opt)`;
  re-checkpoint `safe_opt` after each successful epoch (@188). Guard logic otherwise unchanged.

### 2. Micro-batch the dense traversal (train.py) — math frozen, traversal chunked
- **R2 (council FND-0002): NO loss helper.** Keep the whole-batch per-update body (train.py:144-192)
  BYTE-VERBATIM, wrapped in an `else:` branch (the wrap is the permitted "forward TRAVERSAL change").
  The micro-batch path is NEW code (a new `if micro_batch_steps and micro_batch_steps < n:` branch)
  that re-expresses the SAME loss arithmetic on slices. Numerical equivalence is guarded by
  `test_microbatch.py`, not by sharing code.
- Add `forward_chunk_fn(lo, hi)` (slice-aware forward) + `micro_batch_steps` params to
  `_optimize_actor_critic`. When `micro_batch_steps` is falsy or ≥ n → the **whole-batch path runs
  exactly as today** (no-op; byte-identical for Tiny/small arms). When set < n → chunked path:
  - **Snapshot pass**: under `no_grad`, loop K-step chunks computing `val0`/`old_logp` per chunk and
    `torch.cat` them. Detached → memory relief, math-identical.
  - **GAE/adv/returns**: computed ONCE on the full flattened `val0` (UNCHANGED). adv is normalized
    batch-wide (train.py:159) — so it is computed once and INDEXED per chunk, never recomputed.
  - **Epoch pass**: `optimizer.zero_grad()`; for each chunk forward→`_ac_loss_terms` on the chunk's
    slices→`loss_c = (policy_loss_c + value_coef*value_loss_c - entropy_coef*ent_c) * (n_c/N)`→
    `loss_c.backward()` (accumulate, frees the chunk graph). After all chunks: check the accumulated
    detached loss is finite (divergence guard, same semantics — no step + restore net+opt on NaN);
    else `clip_grad_norm_(net.parameters(), 10.0)` once on the accumulated grad → `optimizer.step()`.
  - Size-weighting `n_c/N` makes the summed per-chunk `.mean()`s equal the whole-batch `.mean()`
    (research: correct gradient-accumulation weighting); equivalence holds within fp tolerance.

### 3. Warm net+opt in the trainers (train.py) — symmetry across structured/_rich/_blind
**R3 (council FND-0001): collapsed API** — each trainer takes only `net=None, optimizer=None,
micro_batch_steps=None` (NO resume_ckpt/resume_opt; disk-resume lives in run_loop, below):
- `net is None` → **fresh**: `torch.manual_seed(seed)` + construct net.
- `net not None` → **warm**: reuse the passed net, skip seed + construction (FND-0006 determinism).
  (Disk-resumed rounds arrive here as a warm net that run_loop already built + loaded.)
- Optimizer: `optimizer is None` → construct `Adam(net.parameters(), lr)` (once, round 0). Else reuse.
- Build `forward_chunk_fn`: structured/rich `lambda lo,hi: net({k: v[lo:hi] for k,v in inputs.items()})`
  (every build_rich_batch tensor is `[B,…]` on axis 0 — confirmed per-row-independent, no cross-step
  op); blind `lambda lo,hi: net(obs[lo:hi])`. Return `(net, stats, optimizer)`.

### 4. Persist + carry across the loop (run_loop.py)
- `train_round(...)` gains `net=None, optimizer=None`; threads them through; returns
  `(net, stats, mode, optimizer)`. Single caller @226 updated.
- **R1 (council FND-0006/0009): `--continual` flag (default True).** When True (the v5 mode): `run()`
  holds `warm_net=warm_opt=None` before the loop; each iteration calls `train_round(..., seed=r,
  net=warm_net, optimizer=warm_opt)` then `warm_net, warm_opt = net, opt` (carry forward). Round 0
  fresh; r>0 warm. When `--continual false`: pass `net=None, optimizer=None` every round → per-round
  fresh net+opt (v4 from-scratch), for rollback + a clean regime A/B.
- **Atomic saves** (FND-0011): new `_atomic_torch_save(obj, path)` (tmp→`os.replace`). Save BOTH
  `ckpt_round_{r}.pt` (replaces the bare save @229) AND new `opt_round_{r}.pt` each round. Prune
  `ckpt/opt` .pt older than the keep window (FND-0010; small files, low-pri).
- **Resume (R3, run_loop.py:194-200):** when `start_round>0` & continual, a small helper
  `_load_warm(variant, dims, token_specs, vocab_counts, rung, out, start, lr)` builds the arch +
  `load_state_dict(torch.load(ckpt_round_{start-1}.pt, weights_only=True))` (FND-0025) + builds the
  optimizer + loads `opt_round_{start-1}.pt`. Requires BOTH sidecars present & rung/dims-compatible
  (**fail-fast** with a clear message; load_state_dict errors on shape mismatch — FND-0003/0007/0008).
  The warm (net,opt) seed the loop; the trainer just sees a warm net.
- `--micro-batch-steps` arg (default 0 = no-op). metrics.jsonl write (@261-266) gains
  `continual`, `warm_start=(continual and r>0)`, `micro_batch_steps`, plus existing throughput — NOT
  curve.csv (CURVE_COLS frozen).

### 5. Experiment driver `python/run_v5.sh` (mirrors run_acceptance.sh)
Two arms, run SEQUENTIALLY (14 cores; 12 gradle threads saturate the box):
- **Arm A — primary (AC1/AC2)**: `--variant structured --map-size Medium --rounds 16 --gen-games 16
  --eval-games 80 --turn-cap 250 --threads 12 --rung small --epochs 8 --lr 1e-3 --gamma 0.99 --lam 0.95
  --value-coef 0.5 --entropy-coef 0.01 --clip-eps 0.2 --gen-seed 1000 --eval-seed 999000
  --micro-batch-steps 0 --out runs/structured-Medium --resume`. (small rung ran clean on Medium in v4,
  so micro-batch is a no-op here → only the regime changed.)
- **Arm B — medium-rung (AC3)**: same but `--rung medium --micro-batch-steps 256 --out
  runs/structured-medium-Medium --resume` (micro-batching newly enables this; would OOM in v4).
- **Ceiling eval** (both arms): `python -m unciv_train.analyze --root runs/<arm> --ceiling-games 200
  --turn-cap 250 --threads 12 --eval-seed 4242424` → uses `_two_proportion_z` (one-sided, p<0.05).
- **bench-onnx**: `./gradlew selfPlay --args="bench-onnx runs/structured-Medium/policy_round_15.onnx 200
  Medium 12 777000"` → gate ratio≥0.70.

### 6. Tests (pytest, python/tests/)
- `test_microbatch.py` — **numerical equivalence**: same net (cloned init) + same injected optimizer,
  one optimize pass whole-batch vs micro-batched (small K) → loss + grad-norm + post-step weights match
  within atol≈1e-5. Oracle = the whole-batch path (FND-0021).
- `test_continual_resume.py` — **warm-from-memory == warm-from-disk**: train round 0, save ckpt+opt
  (atomic), then (a) continue round 1 warm in-memory vs (b) reload ckpt+opt from disk → continue round 1;
  assert identical resulting weights + opt state (FND-0022, no gradle needed). Plus the **AC6 parity**
  check: the warm net at the start of round r produces logits matching the round r-1 *exported ONNX*
  within atol (proves warm net == gen net → on-policy carryover; FND-0030).
- Assert invariants (AC7): terminal-only ±1 reward unchanged; MODELED_HEADS == {tech,policy}; export
  outputs exactly {OUTPUT_TECH, OUTPUT_POLICY} (extend/lean on existing test_gae/test_parity).

## Walkthrough
See `## Walkthrough` in the plan.html body. One round through the continual loop + one micro-batch
chunk, with illustrative shapes.

## FROZEN / not touched
`compute_gae`, the per-update algebra (PPO clip, value MSE, entropy, objective, grad-clip 10.0),
`_masked_logp`/`_entropy`, the encoder + heads + RUNGS, terminal-only ±1 reward, the ONNX export
contract + Kotlin bridge (contract_version=3, byte-unchanged), CURVE_COLS.

## Out of scope (NON-GOALS)
No new action heads, no self-play (opponent stays RandomPolicy), no reward shaping, no encoder changes,
no replay buffer (DEFERRED — gated on behavior-logp recording; one-line note in RESULTS).

## Plan items
- [AI_CODE] train.py: inject optimizer, opt-aware guard, `_ac_loss_terms` helper, micro-batch path.
- [AI_CODE] train.py: warm/resume in the 3 trainers; return optimizer.
- [AI_CODE] run_loop.py: thread net/opt through train_round; carry across loop; atomic ckpt+opt saves;
  resume net+opt with fail-fast; `--micro-batch-steps`; metrics.jsonl fields.
- [AI_CODE] python/run_v5.sh experiment driver.
- [AI_CODE] tests: test_microbatch.py, test_continual_resume.py (+ AC6 parity), invariant asserts.
- [AI_RESEARCH] Run both arms (16 rounds each), ceiling evals (200 games @4242424), bench-onnx.
- [AI_CODE] RESULTS.md: continual curve (16 rounds), z-tests (vs v4 23.0%, vs blind 28.9%),
  medium-rung result, bench verdict, replay-deferred note. **R4 (council FND-0029/0027):** also report
  v5's round-8 winrate vs v4-from-scratch@8 (23.0%) to disentangle regime from round-count, and note
  warm-start makes `old_logp` == the gen policy (more on-policy-correct than v4's random-init anchor).
  Report a null/negative result PLAINLY if it occurs (the experiment answers the question either way).
