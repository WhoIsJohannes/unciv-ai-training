# Test spec — selfplay-v5 (test_mode=integration, pytest)

No agentic-test setup exists for the Python trainer → integration tests via the repo's pytest suite
(python/tests/). Two new files, both currently RED:

## python/tests/test_microbatch.py (AC6 — numerical equivalence)
- `test_microbatch_matches_whole_batch`: same net init (seed=0) + same fresh optimizer, one optimize
  pass whole-batch vs micro_batch_steps=2 (6 steps → 3 chunks) → loss within 1e-5, grad_norm within
  1e-4, post-step weights within 1e-5. Oracle = the whole-batch path.
- `test_microbatch_noop_when_K_ge_N`: micro_batch_steps >= N is byte-equivalent to whole-batch (the
  small-rung primary arm uses a no-op → apples-to-apples with v4).
- RED reason today: `micro_batch_steps=` kwarg does not exist (TypeError). ✓ verified.

## python/tests/test_continual_resume.py (continual correctness + AC6 parity)
- `test_warm_from_memory_equals_warm_from_disk`: round 0 → save ckpt+opt → continue round 1 warm
  in-memory vs reload-from-disk → identical weights within 1e-6 (the --resume guarantee).
- `test_warm_round_does_not_reinit_weights`: a warm round reuses the passed net (no manual_seed
  re-init).
- `test_ac6_warm_net_matches_exported_onnx`: export drops the value head, exposes exactly
  {tech, policy}; warm net == exported gen net (on-policy carryover). onnxruntime-guarded.
- RED reason today: net=/optimizer= warm kwargs + 3-tuple return don't exist. ✓ verified.

## AC7 invariants (covered by existing tests, asserted again here where cheap)
- Terminal-only ±1 reward unchanged (dataset broadcast; test_gae.py).
- Heads exactly {tech, policy} + train-only value; export drops value (test_gae.py::test_export_drops_value_head + test_ac6_warm_net_matches_exported_onnx out-name assert).

## Note: existing test to UPDATE during build
- `test_structured_train.py::test_structured_trainer_runs` unpacks `net, stats = ...`; the v5 trainer
  returns `(net, stats, optimizer)` → update to `net, stats, _ = ...`.
