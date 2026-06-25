# Build Output — selfplay-v5 (Phase 3)

## Summary
Implemented continual training (warm-start net + carried Adam state across rounds), micro-batched
dense traversal (size-weighted chunk accumulation, frozen math), opt-aware divergence guard, atomic
ckpt+opt sidecars, fail-fast `--resume`, `--continual` (default on) + `--micro-batch-steps` knobs, the
`analyze_v5` ceiling-eval/z-test tool, and the two-arm `run_v5.sh` driver. All new tests GREEN; no
regressions.

## Files changed
- **Edited** `python/unciv_train/train.py` — inject optimizer (drop inline Adam @136); opt-aware
  divergence guard (snapshot/restore `opt.state_dict()`); micro-batch chunk branch + whole-batch
  byte-verbatim `else:`; warm/fresh net+opt in structured/rich/blind; return `(net, stats, optimizer)`.
- **Edited** `python/unciv_train/run_loop.py` — `_atomic_torch_save`, `_load_warm`; `train_round`
  4-tuple + net/opt threading; continual carry across loop; `--continual`/`--micro-batch-steps`;
  metrics.jsonl fields (continual/warm_start/micro_batch_steps); sidecar pruning; removed dead `import torch`.
- **Edited** `python/tests/test_structured_train.py` — 2-tuple → 3-tuple unpack.
- **New** `python/unciv_train/analyze_v5.py` — ceiling eval (200 games, seed 4242424) + z-tests vs fixed
  v4 baselines (47/204, 58/200) + round-7 disentangling. Reuses frozen `run_loop.evaluate` + `analyze._two_proportion_z`.
- **New** `python/run_v5.sh` — ARM A (small rung, micro-batch no-op) + ARM B (medium rung, --micro-batch-steps 256) + ceiling evals + bench-onnx.
- **New** `python/tests/test_microbatch.py`, `python/tests/test_continual_resume.py`.

## Gate status
- **pytest (changed paths): GREEN** — test_microbatch (2), test_continual_resume (3), test_structured_train (3), test_gae all pass.
- **Full suite: 45 passed, 1 skipped, 5 FAILED — all pre-existing, NONE in changed code:**
  - 4× `ShardError VERSION=2 != SCHEMA_VERSION=3` (test_train_dataset ×2, test_v2_units ×2) — stale
    fixtures from v4's contract bump to v3 (files unchanged by v5; confirmed via git diff).
  - 1× `test_determinism::test_same_seed_byte_identical_shards` — gradle/JVM gen byte-identity; Kotlin
    untouched by Python-only edits → cannot be a v5 regression.
- **ruff:** F401 dead-import fixed; remaining E702 are non-enforced (repo's committed `plot()` uses semicolons; no ruff config).
- **py_compile / imports:** OK.

## Traceability
`traceability.md` — all 13 plan items → file:line → covering test. No code gaps; driver artifacts
(`analyze_v5`, `run_v5.sh`) smoke-verified (import/help/`bash -n`), full exercise in Phase 4 run.

## Test results
`test_microbatch` proves micro-batch == whole-batch within atol (loss 1e-5, grad-norm 1e-4, weights
1e-5). `test_continual_resume` proves warm-from-memory == warm-from-disk (1e-6) + AC6 export-head parity.

## Plan fidelity
13/13 items DONE (100%). No deviations beyond the council-approved revisions (--continual flag,
byte-verbatim else-branch, collapsed resume API, RESULTS round-7 disentangling).

## Security checklist
- No new secrets (diff adds none). New inputs: `--micro-batch-steps` (int) + ckpt/opt files loaded
  with `torch.load(weights_only=True)` (pickle-RCE mitigated, FND-0025). No PII logging. No new
  endpoints/auth surface (local research trainer).

## Sync-with-base: SKIPPED (documented)
`origin/master` is the unrelated PUBLIC upstream (yairm210/Unciv), 39 commits behind our local-only
research master. The true base is LOCAL master (branched ~1h ago, unchanged). Merging public
origin/master would violate the prompt's "single source of truth" and risk a large spurious merge.
No work landed on the true base in flight → sync is a genuine no-op. (Same rationale as skipping the
initial `git pull` at Setup.)

## Open issues / next (Phase 4)
Run `cd python && THREADS=12 OUT_ROOT=../training-runs/v5 ./run_v5.sh` (~6–10h, sequential arms,
--resume restart-safe), babysit, then write RESULTS.md.
