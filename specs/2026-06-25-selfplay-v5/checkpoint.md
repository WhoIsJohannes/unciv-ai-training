# Checkpoint — selfplay-v5 (end of Phase 3 build)

## What was built
Continual training for the self-play RL trainer: persistent `(net, optimizer)` carried across the
round loop (warm-start weights + Adam moments), micro-batched dense traversal (size-weighted chunk
accumulation, math frozen), opt-aware divergence guard, atomic ckpt+opt sidecars, fail-fast multi-
artifact `--resume`, a `--continual` flag (default True), `--micro-batch-steps`, the `analyze_v5`
ceiling-eval + z-test tool, and the two-arm `run_v5.sh` driver.

## Key files
- `python/unciv_train/train.py` — optimizer injection, opt-aware guard, micro-batch branch, warm/fresh in 3 trainers (returns optimizer).
- `python/unciv_train/run_loop.py` — `_atomic_torch_save`, `_load_warm`, train_round 4-tuple, continual carry, argparse `--continual`/`--micro-batch-steps`, metrics fields, sidecar pruning.
- `python/unciv_train/analyze_v5.py` — 200-game ceiling eval (seed 4242424) + z-tests vs fixed v4 baselines (47/204, 58/200) + round-7 disentangling.
- `python/run_v5.sh` — ARM A (small rung, micro-batch no-op) + ARM B (medium rung, --micro-batch-steps 256) + ceiling evals + bench-onnx.
- `python/tests/test_microbatch.py`, `python/tests/test_continual_resume.py` — GREEN.

## Gate status
- pytest (my paths): GREEN. Full suite: 45 passed, 1 skipped, 5 pre-existing failures (4× stale VERSION=2 fixtures from v4's contract bump; 1× gradle gen determinism — all in code I never touched).
- ruff: clean on my files except non-enforced E702 (repo's committed code uses it); F401 dead-import fixed.
- py_compile OK; imports OK.

## Open issues / next (Phase 4)
- Run the experiment: `cd python && THREADS=12 OUT_ROOT=../training-runs/v5 ./run_v5.sh` (~6–10h, sequential arms, --resume restart-safe). Babysit.
- Produce RESULTS.md: 16-round curve, z-tests (vs v4 23.0%, vs blind 28.9%), round-7-vs-final disentangling, medium-rung result, bench-onnx verdict, replay-deferred note.

## Test spec: `agent-test-spec.md` (test_mode=integration). Council plan review: `council-plan-review.json`.
