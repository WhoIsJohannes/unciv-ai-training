# Build progress — selfplay-v5

## Plan items (completion oracle: `[x] verified in <file>:<line>`)
- [x] Inject optimizer into `_optimize_actor_critic` (remove inline Adam) — verified `train.py:136` (`opt = optimizer`), sig `train.py:119-121`.
- [x] Opt-aware divergence guard (snapshot+restore `opt.state_dict()`) — verified `train.py:140` (safe_opt), `:208-210` & `:251-253` (restore + re-checkpoint, both paths).
- [x] Micro-batch path (chunked snapshot + chunked epoch, size-weighted `n_c/N`, single step) — verified `train.py:145-158` (snapshot), `:200-228` (epoch chunk branch); whole-batch path byte-verbatim in `else:` `:230-262`.
- [x] Warm/fresh net+opt in 3 trainers; return `(net, stats, optimizer)` — verified structured `train.py:331-345`, rich, blind (all `net=None/optimizer=None/micro_batch_steps=None`, manual_seed only on fresh branch).
- [x] `train_round` threads net/opt, returns 4-tuple — verified `run_loop.py:108-145`.
- [x] Persist + carry `(net,opt)` across loop; `--continual` default True — verified `run_loop.py:219` (warm init), `:241-250` (load/carry), argparse `:188`.
- [x] Atomic ckpt + `opt_round_{r}.pt` sidecar — verified `_atomic_torch_save` `run_loop.py:73-80`, save `:252-255`, prune `:300-302`.
- [x] Resume net+opt fail-fast, `weights_only=True` — verified `_load_warm` `run_loop.py:83-110`.
- [x] `--micro-batch-steps` arg + metrics fields (continual/warm_start/micro_batch_steps) — verified argparse `:192`, metrics `:283-285`.
- [x] `analyze_v5.py` ceiling eval + z-tests vs fixed v4 baselines + round-7 disentangling — verified `analyze_v5.py` (reuses run_loop.evaluate + analyze._two_proportion_z).
- [x] `run_v5.sh` two-arm driver + ceiling evals + bench-onnx — verified `python/run_v5.sh` (bash -n OK).
- [x] Tests: `test_microbatch.py` (numerical equiv), `test_continual_resume.py` (resume equiv + AC6 parity) — 5/5 GREEN.
- [x] Update existing `test_structured_train.py` 2-tuple→3-tuple unpack — verified `:79`.

No MISSING items.

## Gate status
- pytest (my code paths): GREEN — test_microbatch (2), test_continual_resume (3), test_structured_train (3), test_gae all pass.
- ruff: F401 fixed (removed dead `import torch`); remaining E702 are pre-existing (`plot()`) + test fixtures — not enforced in this repo (committed code uses semicolons; no ruff config).
- py_compile: all changed/new files OK.
- Pre-existing failures (NOT regressions, code I never touched): 4× `ShardError VERSION=2≠3` (stale fixtures from v4's v3 contract bump) + 1× gradle gen byte-identity determinism test (Kotlin/JVM unchanged by Python-only edits).

## Codebase patterns reused
- Atomic write mirrors `export_onnx.py` tmp→`os.replace`.
- `_two_proportion_z` + `run_loop.evaluate` reused verbatim (frozen) by `analyze_v5`.
- Driver mirrors `run_acceptance.sh` (PY/THREADS/OUT_ROOT env, `--resume`, per-arm subdirs).
