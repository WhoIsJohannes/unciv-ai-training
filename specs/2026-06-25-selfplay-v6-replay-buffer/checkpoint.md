# Checkpoint — selfplay-v6-replay-buffer (end of Phase 3)

## Built
Pillar A (Kotlin): MaskedChoice.chooseWithLogp (+choose delegates, single rng draw); PolicyProvider
default chooseIndexWithLogp (uniform, override-warned); OnnxPolicy override; RoutingPolicy delegates;
DataPlaneHooks records behavior_logp block at emit time; SampleSchema.BLOCK_BEHAVIOR_LOGP + VERSION 3→4.
schema.py SCHEMA_VERSION 3→4.
Pillar B (Python): TrainTrajectory.b_logp_*; loader reads behavior_logp; _stack_traj concat (None→zeros);
_optimize_actor_critic stored_old_logp source switch (val0 ALWAYS current-net forward) + clip_eps guard
+ mean_ratio/clip_frac diagnostics; 3 trainers behavior_logp plumb-through.
Pillar C (run_loop): --replay-window (4); deque; round-0 exclusion + current∪K-1 assembly; window-gated
behavior_logp; keep_shards floor; _replay_refill_rounds + resume refill (round-0 excluded); frac_replayed
+ replay_window + mean_ratio + clip_frac → metrics.jsonl.
Driver: run_v6.sh (4 arms, D8); analyze_v6.py (rounds-to-target + K=1-vs-K=4 z-test).

## Gate status (merge gate = code + unit assertions + Tiny smoke GREEN — MET)
- Kotlin compile (core+desktop) GREEN; dataplane JUnit (MaskedChoiceLogpTest + OnnxPolicyLegalityTest) GREEN.
- Python: ALL modules GREEN incl. 5 new v6 tests, 2 updated fixtures, parity (AC4), structured_train.
- Tiny e2e smoke GREEN: 3 rounds, diverged=0, frac_replayed=0.50 at round 2, mean_ratio≈1.04.
- test_determinism: pre-existing failure (red on clean master too, D9) — NOT a v6 regression.

## Open items
- The 4-arm Medium re-run (AC1/AC2) launches in the background (resumable); results reported when done.
  Per D6 this is experimental evidence, NOT a merge blocker.

## Key files
core/.../{MaskedChoice,PolicyProvider,RoutingPolicy,DataPlaneHooks,SampleSchema}.kt;
desktop/.../OnnxPolicy.kt; python/unciv_{dataplane/schema.py,train/{dataset,train,run_loop}.py};
python/{run_v6.sh,unciv_train/analyze_v6.py}; tests (5 new + 2 fixtures + MaskedChoiceLogpTest.kt).

## Test spec / council
Plan council ran at Step 11 (D7 🔴 round-0-resume bug + D8 micro-batch caught & fixed); intake council
at Step 5. agent-test-spec = the 5 new pytest modules + MaskedChoiceLogpTest.kt (integration mode).
