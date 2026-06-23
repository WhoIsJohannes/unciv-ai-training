# Light Codebase Scan — v1 infra map (consolidated from 3 Explore agents)

Worktree: `/Users/j/Unciv-onnx-selfplay-v2`. All v1 infra is committed on `master` (8e0e4ba0a).
This is the substrate v2 builds on; everything below is REUSED, not rebuilt.

## Contract & provenance (python/unciv_train/contract.py, CONTRACT.md)
- `CONTRACT_VERSION = 1`. Schema `VERSION = 2` (`unciv_dataplane/schema.py` mirrors Kotlin `SampleSchema.VERSION`).
- `Dims(global_w, acting_w, tech_w, policy_w)`; `input_w = global_w+acting_w`. Loaded from `schema.json` at runtime — never hardcoded. GnK: global=26, acting=173, **input=199**, tech=80, policy=70.
- ONNX I/O names: `INPUT_NAME="obs"`, `OUTPUT_TECH="tech_logits"`, `OUTPUT_POLICY="policy_logits"`.
- Metadata keys stamped on the .onnx: `schema_version`, `ruleset_fingerprint`, `contract_version`, `input_width`, `tech_width`, `policy_width`. JVM `OnnxPolicy.init` REFUSES a model whose schema/contract/fingerprint mismatch the live ruleset (provenance gate).
- `LEARNER_CIV_ID = "SimulationCiv1"`.

## Reader / shard format (python/unciv_dataplane/reader.py, Kotlin TrajectoryEmitter/ShardFormat/SampleSchema)
- Shard: `MAGIC "UNCVSMP1"` | u16 version | u32 hdrLen | header JSON | records `[u32 len|payload]` | u32 recCount | u32 crc32. CRC covers records only (determinism-safe).
- `Step(turn, civ_slot, is_first, is_last, is_terminal, overflow, reward, blocks: dict[str,np.ndarray])`. Header struct `<ii4Bf` (16B).
- Block kinds: FIXED → 1D `(len,)`; VARIABLE → u16 count prefix then `(count, perItem)` — **present-only on disk, padding is the loader's job.**
- Block table (per step): `global`[26 f32], `acting_civ`[173 f32], `civ_tokens`[N×84 f32], `diplo_edges`[N×N u8], `own_cities`[M×16 f32], `opp_cities`[K×16 f32], `own_units`[U×8 f32], `opp_units`[V×8 f32], `spatial`[nTiles×13 u8, FIXED per-run], plus masks `mask_tech`[80 u8], `mask_policy`[70 u8], `mask_greatPerson`, `mask_construction`(var), `mask_promotion`(var), `mask_diplomaticVote`(var), `actions`[f32, a_tech@0, a_policy@1; -1 = head did not act].
- Caps (`SampleCaps.DEFAULT`): maxMajorCivs 16, cityStates 24, ownCities 64, oppCities 64, ownUnits 192, oppUnits 192. `overflow` flag is diagnostic, never fed to policy.

## Spatial block — the 13 channels (Featurizer.buildSpatial, SampleSchema.SPATIAL_CHANNELS)
Indexed linearly by `tile.zeroBasedIndex * 13`. Channels: [0]visibility_state(0/1/2) [1]terrain_base [2]terrain_feature [3]resource [4]road [5]river [6]is_city_center | transient (visible only): [7]owner_slot [8]improvement [9]unit_present [10]unit_owner_slot [11]unit_type_cat(1civ/2land/3water/4air) [12]unit_health_bucket(0-4). Never-explored tiles all-zero. u8 values.
- **OPEN: how `zeroBasedIndex` → 2D offset coords.** nTiles is in provenance; per-tile (col,row) and map W/H are NOT separately emitted. Hex/round maps ⇒ nTiles ≠ W×H. → Phase-2 design fork (see decisions).

## Entity token layouts (Featurizer)
- unit token (8): [0]presence [1]is_own [2]owner_slot [3]type_cat [4]health_bucket [5]dx_from_capital [6]dy_from_capital [7]avail_promotions.
- city token (16): presence/is_own/owner_slot/pop/defense/health_bucket/air_units/religion/resistance/puppet/razed/has_spy/current_construction/built_count (+2 zero-pad).
- civ_token (84): met/era/policy counts/branch bits/wonders multi-hot/score/techCount/victory numerators/opinion/diplo flags/demographics rank+bucket/trade. (FairOpponentModel) — fairness-gated (no raw rival floats when unmet).

## model.py PolicyNet
`trunk = Linear(199,128)->ReLU->Linear(128,128)->ReLU`; `tech_head Linear(128,80)`, `policy_head Linear(128,70)`. `forward(obs)->(tech_logits, policy_logits)`. hidden=128.

## train.py — REINFORCE (the algo v2 Stage A replaces)
- `_masked_logp(logits, actions, mask)`: illegal→ -1e9, `log_softmax`, gather chosen, **0 where action<0** (head didn't act). KEEP this machinery verbatim.
- Batch full: `obs[N,199]`, `rets[N]` (terminal ±1 broadcast undiscounted to all of that civ's non-terminal steps), `baseline=rets.mean()`, `adv=rets-baseline`. Loss `-(adv*logp).mean()`, optional entropy. Adam lr=1e-3, epochs=8, seed→`torch.manual_seed`.
- dataset.py `TrainStep(obs, a_tech, a_policy, mask_tech, mask_policy, ret)`; `load_training_steps` filters learner civ, broadcasts terminal reward, **provenance-gates** schema_version + fingerprint. Drops steps where both heads = -1.

## export_onnx.py
`torch.onnx.export(net, dummy[1,199], input_names=["obs"], output_names=["tech_logits","policy_logits"], dynamic_axes={..:{0:"batch"}}, opset=17)`, then stamps metadata props. (Value head must be DROPPED at export for Stage A.)

## run_loop.py — round driver
Per round r: gradle `selfPlay gen <model|random> <dir> <n> <maxTurns> <threads> <seed>` → load schema.json (Dims+fp+ver) → `load_training_steps` → `train` → `export_onnx` → gradle `selfPlay eval <model> <m> <maxTurns> <threads> <seed>` → append `curve.csv` + plot `curve.png` → drop old shards. CLI: `--rounds 10 --gen-games 24 --eval-games 100 --turn-cap 1000 --threads --gen-seed 1000 --eval-seed 999000 --out training-runs/run --keep-shards 2 --epochs 8 --lr 1e-3 --gradle-timeout 1800`. curve.csv cols: `round,games,winrate,pval,n_steps,loss,ret_pos,onnx_decisions`. **No `--variant` flag yet.** matplotlib plot exists. `verdict()` → GO/PLATEAU/INCONCLUSIVE.

## OnnxPolicy.kt (desktop) — inference bridge (the one JVM file to grow)
- `ai.onnxruntime.*`. init: open session (intraOp=1), READ metadata, provenance-gate (schema/contract/fingerprint).
- `logitsFor`: `input = obs.block("global") + obs.block("acting_civ")` (single float[199]); `forward`: `OnnxTensor.createTensor(env, FloatBuffer.wrap(input), [1, len])`, `session.run(mapOf("obs" to t))`, read `tech_logits`/`policy_logits` rows. ThreadLocal memo per (gameId,civID,turn).
- `chooseIndex(head,...)`: head∉MODELED_HEADS(["tech","policy"]) → **-1 (heuristic fallback)**. `MaskedChoice.choose(logits, legalMask, eval, rng)`: eval→argmax over legal; else stable softmax sample over legal support; empty legal→ -1. RNG = `GameContext(civ).stateBasedRandom("dataplane-policy-$turn")` (deterministic, replayable).

## SelfPlayRunner.kt — modes & config
- `gen|eval|parity-dump|parity-run|trace`. Learner=SimulationCiv1, Opponent=SimulationCiv2 (RandomPolicy), +spectator. RoutingPolicy routes learner heads to OnnxPolicy, everything else RandomPolicy.
- `mapParameters()` HARDCODES `mapSize = MapSize.Tiny` (radius10 ~331 tiles hex), noRuins, noNaturalWonders, legendaryStart, strategicBalance, mirroring, seeded. Difficulty King, GnK ruleset, 0 city-states, no barbarians. → **Medium eval = MapSize.Medium (r20, 44×29); must parametrize map size by CLI (currently not factored).**
- eval: winrate = wins(LEARNER)/games; `pval = SimStats.binomialTest(wins, games, 0.5, "greater")` (Kotlin normal-approx, no scipy). Emits `EVAL_RESULT {json}`. parity-dump writes 199-dim obs CSV; parity-run runs ORT on a CSV obs → tech/policy logits JSON.

## Tests (reuse + extend)
- `test_parity.py`: synthetic Dims, fresh random net+obs, export, JVM `parity-run` vs Python ORT, `np.allclose(atol=1e-4)`. No golden fixture. → **extend to multi-tensor input.**
- `test_determinism.py`: same seed → byte-identical shards (SHA-256). Kotlin `FairnessAndDeterminismTests` (mask parity, leakage, fingerprint), `OnnxPolicyLegalityTest` (choice always legal or -1), `SimStatsTest` (60/100 → p<0.05).

## Build / deps
- gradle tasks: `selfPlay` (SelfPlayRunner), `dataGen` (DataPlaneGen), `simBench`. onnxruntime **1.19.2** pinned JVM (libs.versions.toml) + py (`onnxruntime>=1.17`). JAVA_HOME default `/opt/homebrew/opt/openjdk@21`.
- python deps (pyproject): numpy, torch≥2.0, onnx≥1.15, onnxruntime≥1.17, matplotlib≥3.7; test: pytest. **No scipy** (binomial in Kotlin).

## Top design implications for Phase 2
1. **Spatial encoder approach (THE fork):** grid CNN needs per-tile (col,row) + map W/H, which are NOT in the shard and "no new emitted data" is a hard non-goal. Task explicitly permits the documented fallback: per-tile tokens + positional features + masked pooling/attention. Resolve with deep-scan evidence on `zeroBasedIndex`/`HexMath`; pick + document in decisions.md.
2. **Contract bump** to v2: multi-tensor ONNX input (spatial + entity token sets + masks + global + acting_civ) with dynamic axes; `CONTRACT_VERSION` → 2; export drops value head; JVM `OnnxPolicy` builds the same multi-tensor input from the live `Observation`.
3. **Stage A is python-only** (value head + GAE) — JVM/export unchanged, banks an isolated convergence curve before touching the bridge.
4. **Map-size parametrization** for the Medium eval (CLI arg threaded run_loop → SelfPlayRunner).
5. **Compute reality:** user chose full-run-to-acceptance; Medium games are far longer than Tiny — budget rounds/threads accordingly.
