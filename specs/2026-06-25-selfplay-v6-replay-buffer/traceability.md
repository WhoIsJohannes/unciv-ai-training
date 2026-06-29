# Traceability — selfplay-v6-replay-buffer

| Plan item | Implemented in | Covering test |
|---|---|---|
| A1 `MaskedChoice.chooseWithLogp` + `choose` delegates (single rng draw) | MaskedChoice.kt:14-49 | MaskedChoiceLogpTest.kt (logp == log_softmax(masked)[idx]; choose==chooseWithLogp.first; empty→(-1,0f)); OnnxPolicyLegalityTest.kt (legality unchanged) — both GREEN |
| A2 `PolicyProvider.chooseIndexWithLogp` default (uniform ln(1/nLegal) + override warning) | PolicyProvider.kt:18-33 | exercised via OnnxPolicy override + RandomPolicy default; gradle tests:test GREEN |
| A3 `OnnxPolicy.chooseIndexWithLogp` override; `chooseIndex` delegates | OnnxPolicy.kt:102-114 | desktop:compileKotlin GREEN; test_parity (inference path unchanged) — running |
| A3b `RoutingPolicy` delegates chooseIndexWithLogp (no uniform-default leak) | RoutingPolicy.kt:25-29 | gradle compile GREEN |
| A4 `chooseAndApply` builds behaviorLogp; `recordStep` emits behavior_logp block at EMIT time | DataPlaneHooks.kt:114-136, 217-226 | Tiny smoke (writes v4 shards) — running; test_behavior_logp_shard (round-trip) GREEN |
| A5 `BLOCK_BEHAVIOR_LOGP` + `VERSION 3→4` | SampleSchema.kt:23,121-129 | test_behavior_logp_shard (v3 refuses, v4 loads) GREEN |
| A6 `SCHEMA_VERSION 3→4` | schema.py:18 | reader/dataset provenance tests GREEN |
| B1 `TrainTrajectory.b_logp_tech/policy` + loader reads | dataset.py:58-62, 195-200 | test_v2_units (load_trajectories) GREEN; test_behavior_logp_shard GREEN |
| B2 `_stack_traj` concat (None→zeros guard) | train.py:284-300 | test_microbatch + test_continual_resume (synthetic None) GREEN |
| B3 `_optimize_actor_critic` stored_old_logp source switch (val0 always current-net) | train.py:135-181 | test_replay_noop (K=1 bit-identical) + test_offpolicy_equivalence (ratio≈1) GREEN |
| B4 clip_eps-truthy guard under replay | train.py:144-152 | test_clip_eps_guard GREEN |
| B5 3 trainers behavior_logp plumb-through | train.py:303-... (blind/rich/structured) | test_structured_train — running; test_replay_noop GREEN |
| B6 ➕ replay-health diagnostics (mean_ratio/clip_frac) | train.py micro+whole stats.update | Tiny smoke metrics.jsonl — running |
| C1 `--replay-window` argparse (default 4) | run_loop.py:234-241 | Tiny smoke (K=4) — running |
| C2 deque + round-0 exclusion + current∪K-1 assembly | run_loop.py:261-281, ~300-312 | Tiny smoke (frac_replayed>0 at round 2) — running |
| C3 window-gated behavior_logp into train_round | run_loop.py:117-121 | test_replay_noop (K=1 path) GREEN |
| C4 keep_shards floor + `_replay_refill_rounds` + resume refill (round-0 excluded) | run_loop.py:243-245, _replay_refill_rounds, 263-281 | test_replay_resume (4 cases) GREEN |
| C5 frac_replayed/replay_window/mean_ratio/clip_frac → metrics.jsonl | run_loop.py metrics block | Tiny smoke — running |
| Experiment driver (4 arms, D8) + analyze_v6 | run_v6.sh; analyze_v6.py | bash -n GREEN; analyze_v6 import GREEN; runs in the Medium re-run |
| AC5 determinism (per-sample 1 draw; whole-shard nondeterminism pre-existing) | MaskedChoice single draw | MaskedChoiceLogpTest GREEN; test_determinism RED on master too (D9 — not a v6 regression) |

No empty rows except the still-running JVM tests (Tiny smoke / test_parity / test_structured_train) — results folded in at Step 15.
