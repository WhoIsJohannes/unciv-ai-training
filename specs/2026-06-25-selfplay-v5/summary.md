# selfplay-v5 — Continual Training — SHIPPED

**One-liner:** Made self-play training continual (warm-start net + carried Adam state across rounds,
16 rounds) + micro-batched the dense Medium path — and it works: the structured GNN goes 23.0% →
**40.7%**, clearing the blind baseline (28.9%) for the first time.

**Verdict: SHIPPED** — all 7 acceptance criteria met.

## Result (200-game Medium ceiling, seed 4242424)
- **AC1 ✅** continual (small rung) 40.7% (83/204) beats v4 from-scratch 23.0% — z=3.83, p=6.5e-05.
- **AC2 ✅ (headline)** clears blind 28.9% — z=2.46, p=0.0069. *Not resetting clears blind.*
- **AC3 ✅** medium rung trains on Medium with micro-batching, no OOM; strongest arm at 46.6% (95/204).
- **AC4 ✅** bench-onnx ratio=1.49 ≥ 0.70.
- **AC5/6/7 ✅** contract+bridge byte-unchanged; frozen PPO math verbatim + micro-batch numerically
  equivalent (tested); warm-start on-policy (warm net == exported gen net, tested); terminal-only ±1
  reward + {tech,policy} heads unchanged. Zero divergence over 32 rounds.
- Key nuance: continual wins by **accumulating** (rounds 8–15), not per-round-early — at v4's matched
  8-round mark v5 was ~19%, below v4's 23%. The regime *unlocks* compounding v4 structurally couldn't use.

## PR / merge
No GitHub PR (origin is the public upstream). Ships via **local fast-forward to master**, the v4 pattern.

## Councils
Intake (34 findings) + plan-review (34 → 4 revisions: `--continual` flag, byte-verbatim chunk path,
collapsed resume API, round-8 disentangling) + ship-review (33 → 5 fixes: real AC6 logit comparison,
driver failure-tracking, prune try/except, per-round GC, micro-batch guard).

## Files touched
- `python/unciv_train/train.py` — inject optimizer; opt-aware divergence guard; micro-batch chunk
  branch (+ whole-batch path byte-verbatim); warm/fresh net+opt in 3 trainers; return optimizer.
- `python/unciv_train/run_loop.py` — `_atomic_torch_save`, `_load_warm`; carry (net,opt) across loop;
  `--continual`/`--micro-batch-steps`; metrics fields; sidecar pruning.
- `python/unciv_train/analyze_v5.py` (new) — 200-game ceiling eval + z-tests vs fixed v4 baselines.
- `python/run_v5.sh` (new) — two-arm driver + ceiling evals + bench-onnx (per-arm failure-tracking).
- `python/tests/test_microbatch.py`, `test_continual_resume.py` (new); `test_structured_train.py` (3-tuple).

## Open issues
- 5 pre-existing test failures (stale VERSION=2 fixtures + a gradle-determinism test) — untouched code.
- Replay buffer DEFERRED (needs behavior-logp recording). 1 cleanup item filed (deliberate freeze-driven
  loss-expression duplication in `_optimize_actor_critic`; deferred for when the freeze lifts).

## Next
Record behavior log-probs → correct cross-round replay; and/or add heuristic-only action heads.
