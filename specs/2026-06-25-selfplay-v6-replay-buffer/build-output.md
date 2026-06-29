# Build Output — selfplay-v6-replay-buffer

## Summary
PPO-correct cross-round replay buffer (v6): record per-head behavior log π_b at the masked-softmax
sample point (Kotlin), use the stored π_b as `old_logp` for replayed (off-policy) steps with the
value/critic side kept on-policy-fresh (val0 always a current-net forward; GAE under the current
critic each round), and a recent-round-window replay buffer (deque, K=4) in the round loop. Shard
schema VERSION 3→4 in lockstep. Off-policy variance handled by the EXISTING knobs (small K + PPO
clip + logratio.clamp + grad-clip + safe_opt rollback) — no new IS machinery.

## Files changed (12 edited, 8 new)
Edited: core/.../{MaskedChoice,PolicyProvider,RoutingPolicy,DataPlaneHooks,SampleSchema}.kt;
desktop/.../OnnxPolicy.kt; python/unciv_dataplane/schema.py;
python/unciv_train/{dataset.py,train.py,run_loop.py}; python/tests/{test_train_dataset,test_v2_units}.py.
New: python/run_v6.sh; python/unciv_train/analyze_v6.py; python/tests/{test_replay_noop,
test_offpolicy_equivalence,test_clip_eps_guard,test_behavior_logp_shard,test_replay_resume}.py;
tests/src/com/unciv/logic/simulation/dataplane/MaskedChoiceLogpTest.kt.

## Gate status — GREEN (merge gate met)
- Kotlin: `tests:test --tests "*dataplane*"` GREEN; `core`+`desktop` compile GREEN.
- Python: full suite GREEN — new v6 tests (replay_noop, offpolicy_equivalence, clip_eps_guard,
  behavior_logp_shard, replay_resume), updated fixtures (test_train_dataset 3, test_v2_units 6),
  regressions (microbatch, continual_resume, gae, contract_failloud, hexgraph, structured_attn,
  structured_smoke, reader, structured_train), and **test_parity 4/4 (AC4: JVM↔Python ONNX logits
  match for contract v1/v2/v3 — export contract + inference path byte-unchanged)**.
- Tiny e2e replay smoke GREEN: structured/small/Tiny, K=4, 3 rounds → diverged=0 every round,
  frac_replayed=0.50 at round 2 (replay active), mean_ratio≈1.04 (healthy near-on-policy).
- test_determinism: pre-existing failure (RED on clean master 04f2e27fa too — D9); v6 does NOT
  regress it (the per-sample single rng draw + `choose==chooseWithLogp.first` are unit-verified).

## Acceptance criteria
- AC3 correctness: stored old_logp for replayed (test_offpolicy_equivalence, test_replay_noop);
  GAE under current critic each round (val0 always current-net forward); clip_eps-truthy guard
  (test_clip_eps_guard). ✓
- AC4 schema: 3→4 lockstep (Kotlin+Python+2 fixtures); v3 refuses (test_behavior_logp_shard); ONNX
  contract names/widths/keys/CONTRACT_VERSION* byte-unchanged (test_parity); META_SCHEMA_VERSION value
  moves 3→4 in lockstep (intended — still matches within a v6 run). ✓
- AC5 determinism: one rng draw/sample, identical order (MaskedChoiceLogpTest); whole-shard
  determinism pre-existing-red (D9). ✓ (to the degree master has it)
- AC6 terminal-only ±1 + heads {tech,policy} unchanged. ✓
- AC1/AC2 (Medium re-run): launched in background, resumable; reported when complete (D6: evidence,
  not a merge blocker).

## Traceability
See traceability.md — every plan item maps to a file + a covering test (no empty rows except the
Medium-run-dependent AC1/AC2 numbers).

## Security checklist
No secrets (no auth/PII/network surface — offline RL training). New external input = the shard
behavior_logp block, read fail-loud (no .get() fallback) and bounded by logratio.clamp(±20). No new
logging of sensitive data. No endpoints.

## Plan fidelity
DONE = all plan items (Pillars A/B/C/D + driver + tests) + the 2 council-found refinements (D7
round-0-resume bug, D8 micro-batch on K=4) + the ➕ ADDED diagnostics (D3). No MODIFIED/MISSING.
