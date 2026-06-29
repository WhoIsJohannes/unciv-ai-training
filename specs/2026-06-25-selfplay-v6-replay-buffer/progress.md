# Build progress — selfplay-v6-replay-buffer

## Codebase patterns (reuse)
- Shard block append at EMIT time (DataPlaneHooks.recordStep), name-keyed everywhere.
- `_masked_logp` = log_softmax(masked)[chosen], 0 for non-acting head — mirror for chooseWithLogp.
- Trainer core `_optimize_actor_critic` is encoder-agnostic; 3 trainers call it via _stack_traj.
- Fail-loud provenance (no .get() fallback); SSOT lockstep SampleSchema.VERSION ↔ schema.py.

## Pillar A — record behavior logp (Kotlin) + schema 3→4
- [ ] MaskedChoice.chooseWithLogp + choose routed through it (single draw)
- [ ] PolicyProvider.chooseIndexWithLogp default (+ override-warning KDoc)
- [ ] OnnxPolicy.chooseIndexWithLogp override; chooseIndex delegates
- [ ] DataPlaneHooks.chooseAndApply builds behaviorLogp; handleCivTurn; recordStep emits block
- [ ] SampleSchema.BLOCK_BEHAVIOR_LOGP + VERSION 3→4 + KDoc
- [ ] schema.py SCHEMA_VERSION 3→4 (+ reader message text if needed)

## Pillar B — PPO stored old_logp for replayed steps (Python)
- [ ] dataset.TrainTrajectory b_logp_tech/policy (defaults) + load_trajectories reads
- [ ] train._stack_traj concat (None→zeros guard) + return
- [ ] _optimize_actor_critic stored_old_logp param + source switch (val0 always current-net)
- [ ] clip_eps-truthy guard (stored set ⇒ clip_eps truthy)
- [ ] 3 trainers behavior_logp plumb-through (sum heads → stored)
- [ ] diagnostics: mean_ratio + clip_frac in stats

## Pillar C — replay buffer (run_loop)
- [ ] --replay-window K (default 4); deque(maxlen)
- [ ] round 0 excluded; current∪last K-1 assembly; train_data into train_round
- [ ] window-gated behavior_logp=None when K<=1
- [ ] keep_shards floor; _replay_refill_rounds(start,K) (round-0 excluded); resume refill + warn
- [ ] frac_replayed into metrics.jsonl

## Pillar D / experiment
- [ ] run_v6.sh (4 arms; K=4 micro-batch 256)

## Tests
- [x] MaskedChoiceLogpTest.kt GREEN (gradle tests:test, KOTLIN_TESTS_RC=0; OnnxPolicyLegalityTest still green)
- [x] test_replay_noop / test_offpolicy_equivalence / test_behavior_logp_shard / test_clip_eps_guard / test_replay_resume GREEN
- [x] fixtures test_train_dataset.py (3) + test_v2_units.py (6) updated (v4 + behavior_logp) GREEN
- [x] regression: microbatch, continual_resume, gae, contract_failloud, hexgraph, structured_attn, structured_smoke, reader GREEN
- [x] desktop:compileKotlin GREEN (OnnxPolicy changes compile)
- [~] test_determinism (AC5 byte-identity): FAILS on my branch — investigating regression-vs-preexisting.
      My diff does NOT touch Featurizer/Observation/TrajectoryEmitter/ShardFormat; `gen random` never calls
      MaskedChoice; behavior_logp for RandomPolicy = ln(1/nLegal) (deterministic). ⇒ nondeterminism is in
      unchanged code ⇒ predicted PRE-EXISTING. Definitive check: test_determinism on clean master (running)
      + byte-offset diff of two same-seed branch shards (running, job bpi5iko17).
- [ ] test_parity / test_structured_train (JVM/heavy) — run after the determinism probe frees the gradle daemon.
