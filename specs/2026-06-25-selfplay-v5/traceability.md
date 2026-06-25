# Traceability — selfplay-v5

| Plan item | Implementing file:line | Covering test |
|---|---|---|
| Inject optimizer into `_optimize_actor_critic` | train.py:115-136 (sig + `opt = optimizer`) | test_microbatch, test_continual_resume (pass optimizer=) |
| Opt-aware divergence guard | train.py:140, ~208-210, ~251-253 | test_continual_resume (warm-from-disk==memory exercises opt carry); guard path indirectly via existing finite-loss tests |
| Micro-batch chunked path (size-weighted) | train.py:145-158, 200-228 | **test_microbatch::test_microbatch_matches_whole_batch** (loss/grad/weights within atol) + `test_microbatch_noop_when_K_ge_N` |
| Whole-batch path byte-verbatim (else) | train.py:230-262 | test_microbatch noop case + test_structured_train (unchanged values) |
| Warm/fresh net+opt in 3 trainers; 3-tuple return | train.py structured/rich/blind | test_continual_resume (warm net=/opt=), test_structured_train (fresh 3-tuple) |
| `train_round` threads + 4-tuple | run_loop.py:108-145 | (driver-level; exercised by run_v5.sh smoke) |
| Carry across loop + `--continual` default | run_loop.py:219, 241-250; argparse | test_continual_resume (warm-from-memory mirrors the carry) |
| Atomic ckpt + opt sidecar | run_loop.py `_atomic_torch_save` 73-80; save 252-255 | test_continual_resume (torch.save/load roundtrip == in-memory) |
| Resume net+opt fail-fast, weights_only | run_loop.py `_load_warm` 83-110 | test_continual_resume (load_state_dict(weights_only=True) path) |
| metrics fields (continual/warm/micro) | run_loop.py ~283-285 | (metrics.jsonl; visual-inspect in run) |
| `analyze_v5` ceiling eval + z-tests | analyze_v5.py | import/help smoke; full exercise in Phase 4 run |
| `run_v5.sh` driver + bench-onnx | python/run_v5.sh | `bash -n` OK; full exercise in Phase 4 run |
| Numerical-equivalence (AC6) | — | test_microbatch (GREEN) |
| Resume equivalence + AC6 parity | — | test_continual_resume (GREEN) |
| AC7 invariants (terminal reward; heads {tech,policy}; export drops value) | unchanged (frozen) | test_gae::test_export_drops_value_head + test_continual_resume::test_ac6_warm_net_matches_exported_onnx (out_names assert) |

**Gaps:** none for the code. The two driver artifacts (`analyze_v5`, `run_v5.sh`) are smoke-verified (import/help/`bash -n`); their full behavior is exercised by the Phase 4 experiment run (the AC1/AC2/AC3 numbers).
