# Traceability — onnx-selfplay-v2

Verified against the actual code on disk in `/Users/j/Unciv-onnx-selfplay-v2`. File:line references are to the
current working tree. "Covering test" lists the test(s) of record that exercise each deliverable.

## Stage A — value critic on the blind input (python-only)

| Plan item | Implementing file:symbol | Covering test |
|---|---|---|
| `model.py`: `value_head: Linear(hidden→1)`; `forward → (tech, policy, value)` | `python/unciv_train/model.py:40` `PolicyNet.value_head`; `model.py:43-47` `PolicyNet.forward` (returns 3-tuple, `tanh`-bounded value) | `python/tests/test_gae.py:26` `test_model_forward_returns_value` |
| `dataset.py`: trajectory loader — ordered per-game learner steps, keep no-action steps, terminal-only reward, drop v1 return-broadcast | `python/unciv_train/dataset.py:124` `load_trajectories` (keeps all non-terminal steps `dataset.py:150`; `rewards[-1]=term_r` else 0 `dataset.py:161-162`); `TrainTrajectory` `dataset.py:40-54` | **GAP** — no direct test. `test_train_dataset.py` only covers the v1 `load_training_steps`. Indirectly exercised via the run loop only. |
| `train.py`: `compute_gae(rewards, values, γ, λ)` (V(terminal)=0) | `python/unciv_train/train.py:45` `compute_gae` | `test_gae.py:39` `test_compute_gae_terminal_only_reward`; `test_gae.py:58` `test_gae_loss_reward_is_terminal_only` |
| `train.py`: replace REINFORCE with actor-critic + GAE; batch-level adv standardization; policy/value/entropy losses; divergence guard; per-round metrics | `train.py:204` `train_actor_critic_blind`; `train.py:115` `_optimize_actor_critic` (batch-norm adv `train.py:157-158`; `policy_loss`/`value_loss`/`entropy` `train.py:170-175`; divergence guard `train.py:177-180`; metrics `train.py:185-188`); `_masked_logp` reused verbatim `train.py:28` | **GAP (direct)** — `compute_gae` math is covered (`test_gae.py`), but `train_actor_critic_blind`/`_optimize_actor_critic` have no unit test; exercised only through the run loop. |
| `train.py`: keep `train_reinforce` (v1 baseline path) | `train.py:70` `train_reinforce` (running-mean baseline, value head ignored `train.py:97`) | No direct test (loop-only). v1 dataset path covered by `python/tests/test_train_dataset.py`. |
| `contract.py`/`export_onnx.py`: value head training-only — export drops it; ONNX stays policy-only contract v1; negative test | `python/unciv_train/export_onnx.py:19` `_PolicyOnly` (drops value, `export_onnx.py:27-29`); `export_onnx.py:47` `export` (outputs only `tech_logits`,`policy_logits`); `contract.CONTRACT_VERSION=1` `contract.py:17` | `test_gae.py:70` `test_export_drops_value_head` (asserts outputs == {tech_logits, policy_logits}) |
| `run_loop.py`: `--variant {v1-reinforce, blind-critic, rich-critic}`; per-round `.pt` + `--resume`; metrics columns | `python/unciv_train/run_loop.py:142` `--variant`; `run_loop.py:163` `--resume`; checkpoint `run_loop.py:209` (`torch.save(state_dict)`); resume `run_loop.py:177-183`; `train_round` dispatch `run_loop.py:74`; `CURVE_COLS` metrics `run_loop.py:109-110` | No automated test (operational driver). |

## Stage B — rich representation (map + entities)

| Plan item | Implementing file:symbol | Covering test |
|---|---|---|
| `features.py` (new) + `dataset.py`: multi-tensor assembly — spatial per-tile token set + entity token sets + presence masks, padded to batch-max | `python/unciv_train/features.py:33` `build_rich_batch`; `features.py:51` `build_rich_single`; `features.py:17` `_pad_token_set` (pad + mask). Per-step block extraction `dataset.py:106` `_rich_step_blocks`; `load_trajectories(rich=True)` `dataset.py:163` | `test_parity.py:78` `test_jvm_python_rich_logits_match` uses `build_rich_single` (incl. empty entity sets). No standalone `features.py` unit test (`build_rich_batch` is loop/export-only). |
| `model.py`: `RichPolicyValueNet` — per-tile/per-entity token MLP → masked mean+max pool → trunk → {tech,policy,value}; masked-pool NaN guard (R3) | `python/unciv_train/model.py:79` `RichPolicyValueNet`; `model.py:50` `masked_pool` (mean=sum/clamp(count,1) `model.py:58-59`; empty-set max→0 `model.py:60-62`); `_TokenEncoder` `model.py:66` | `test_parity.py:78` `test_jvm_python_rich_logits_match` (fixture includes empty `opp_units`/`opp_cities` → drives the NaN-guard path). **PARTIAL** — no dedicated `masked_pool` unit test asserting the all-padding→zero invariant in isolation; only validated end-to-end via parity. |
| Contract v2: `contract.py` + Kotlin `SampleSchema.OnnxContract` lockstep — named multi-tensor inputs + masks, dynamic axes, `CONTRACT_VERSION→2` | `contract.py:18` `CONTRACT_VERSION_RICH=2`, `RICH_TOKEN_NAMES` `contract.py:29`, `INPUT_GLOBAL/INPUT_ACTING` `contract.py:26-27`, `META_INPUT_NAMES` `contract.py:38`, `token_specs_from_schema` `contract.py:85`; Kotlin `SampleSchema.kt:42` `CONTRACT_VERSION_RICH=2`, `RICH_TOKEN_NAMES` `SampleSchema.kt:48`, `MASK_SUFFIX` `SampleSchema.kt:49`, `META_INPUT_NAMES` `SampleSchema.kt:61` | `test_parity.py:78` (lockstep validated cross-boundary). Names/constants otherwise unverified by a dedicated lockstep assertion. |
| `export_onnx`: multi-tensor policy-only export (value dropped), dynamic axes incl. masks | `export_onnx.py:82` `export_rich`; `_RichPolicyOnly` `export_onnx.py:32` (drops value `export_onnx.py:41-44`); dynamic axes incl. `<name>_mask` `export_onnx.py:107-115`; stamps `CONTRACT_VERSION_RICH` `export_onnx.py:132` | `test_parity.py:78` uses `export_rich`. (Note: `test_export_drops_value_head` asserts value-drop only for the *blind* `export`, not `export_rich`.) |
| `OnnxPolicy.kt`: build multi-tensor input from live `Observation`, `session.run(map)`, mask+sample, contract-v2 provenance gate, close all tensors | `desktop/.../OnnxPolicy.kt:116` `forwardRich`; `OnnxPolicy.kt:136` `buildRichTensors`; `OnnxPolicy.kt:148` `richTensorsFromArrays`; `tokenTensors` `OnnxPolicy.kt:168` (empty-set→N=1,mask=0); v2 provenance gate `OnnxPolicy.kt:59-63`; tensors closed in `finally` `OnnxPolicy.kt:124-126` | `test_parity.py:78` `test_jvm_python_rich_logits_match` (drives `richTensorsFromArrays`). Legality of choice: `OnnxPolicyLegalityTest.kt` (mask-choice invariant, weight-independent). |
| `SelfPlayRunner.kt` parity-dump/parity-run multi-tensor | `SelfPlayRunner.kt:323` `parityDumpRich`; `SelfPlayRunner.kt:349` `parityRunRich` (mode wiring `SelfPlayRunner.kt:72-73`) | `test_parity.py:78` invokes `selfPlay parity-run-rich` |
| `test_parity.py`: multi-tensor parity atol=1e-4 | `python/tests/test_parity.py:78` `test_jvm_python_rich_logits_match` (`ATOL=1e-4` `test_parity.py:29`) | self (plus `test_parity.py:37` `test_jvm_python_logits_match` blind, atol=1e-4) |

## Stage C — eval

| Plan item | Implementing file:symbol | Covering test |
|---|---|---|
| Map-size CLI param `SelfPlayRunner.mapParameters(seed, mapSize)`, threaded through gen/eval | `SelfPlayRunner.kt:122` `mapParameters(seed, mapSizeName)`; `resolveMapSize` `SelfPlayRunner.kt:114`; threaded into `gen` (arg[7] `SelfPlayRunner.kt:151`, `SELFPLAY_GEN_DONE … mapSize` `SelfPlayRunner.kt:173`) and `eval` (arg[6] `SelfPlayRunner.kt:183`); `buildBaseGameInfo(ruleset, mapSizeName)` `SelfPlayRunner.kt:135` | No automated test (operational). |
| `run_loop.py`: `--map-size`; Medium train+eval; two-proportion test rich vs blind | `run_loop.py:144` `--map-size`; passed to `generate`/`evaluate` `run_loop.py:192-193,224-225`; two-proportion z-test `python/unciv_train/analyze.py:50` `_two_proportion_z` + AC3 driver `analyze.py:120-138` | No automated test (operational; AC3 analysis is reporting code). |

## Tests of record (status)

| Test | File:symbol | Covers |
|---|---|---|
| value head | `test_gae.py:26` `test_model_forward_returns_value` | `PolicyNet.value_head` + 3-tuple forward |
| compute_gae | `test_gae.py:39` `test_compute_gae_terminal_only_reward` | hand-computed GAE, V(terminal)=0 |
| terminal-only reward | `test_gae.py:58` `test_gae_loss_reward_is_terminal_only` | AC7 reward vector zero except last step |
| export-drops-value | `test_gae.py:70` `test_export_drops_value_head` | policy-only blind export |
| blind parity | `test_parity.py:37` `test_jvm_python_logits_match` | JVM↔Python contract-v1 logits, atol=1e-4 |
| rich parity | `test_parity.py:78` `test_jvm_python_rich_logits_match` | JVM↔Python contract-v2 multi-tensor logits incl. empty-set NaN guard, atol=1e-4 |
| legality | `tests/.../OnnxPolicyLegalityTest.kt` | MaskedChoice never returns an illegal index (4 @Test) |
| fairness/determinism | `tests/.../FairnessAndDeterminismTests.kt` | (pre-existing, reported green) |

## Gaps

Every plan deliverable has an implementing symbol on disk — **no plan item is unimplemented**. The gaps are
in **test coverage** and **plan-vs-shipped defaults**, not missing code.

1. **`dataset.load_trajectories` has NO covering test (true gap).** The new trajectory loader
   (`dataset.py:124`) — ordered steps, keep-no-action-steps, terminal-only reward placement, `rich=True`
   block attachment — is exercised only through the live run loop. `python/tests/test_train_dataset.py`
   tests only the v1 `load_training_steps`. The behaviors most at risk (no-action steps NOT dropped — R1;
   reward 0 except terminal) are unguarded by any unit test. The terminal-only-reward *math* is guarded in
   `compute_gae` (`test_gae.py:58`), but the *loader* producing that reward vector is not.

2. **Actor-critic trainers have no direct unit test.** `train_actor_critic_blind` /
   `train_actor_critic_rich` / `_optimize_actor_critic` (batch-level advantage standardization — R2;
   divergence guard — R8; per-round metrics) are covered only transitively via the run loop. `compute_gae`
   alone is unit-tested. No assertion pins batch-level (vs per-trajectory) normalization or the NaN-abort
   path.

3. **`masked_pool` NaN guard (R3) is only end-to-end-tested, not unit-tested (partial gap).** The
   plan calls it "unit-tested." On disk the empty-set/all-padding→zero invariant (`model.py:50-63`) is
   validated only indirectly through `test_jvm_python_rich_logits_match` (whose fixture includes empty
   `opp_units`/`opp_cities`). There is no standalone test asserting `masked_pool` returns a zero vector for
   an all-padding row.

4. **`export_rich` value-drop is not negative-tested.** `test_export_drops_value_head` asserts the
   policy-only output set for the *blind* `export` only. `export_rich` (`export_onnx.py:82`) also drops the
   value head (`_RichPolicyOnly`), but no test asserts the v2 export's output set == {tech_logits,
   policy_logits}.

5. **`features.build_rich_batch` (the training-time padder) is untested.** Only `build_rich_single`
   (the single-step parity reference) is exercised (`test_parity.py`). The batch padder used in every
   rich-critic training round has no covering test.

6. **Contract v1/v2 lockstep is asserted only via parity, not directly.** No test cross-checks the
   Python `contract.py` constants against the Kotlin `SampleSchema.OnnxContract` names/versions; lockstep is
   only implied by the rich parity test running successfully end-to-end.

## Plan-vs-shipped notes (not gaps — documented amendments)

- **Default A2C config differs from `plan.md` but matches `progress.md`'s logged amendment.** `plan.md`
  §A.3 specifies "Default = plain A2C+GAE, single inner epoch, NO PPO clip"; the shipped defaults are
  `--epochs 8` (`run_loop.py:154`) and `--clip-eps 0.2` (PPO clip ON, `run_loop.py:160`). This is the
  "R4 amended" decision recorded in `progress.md` (K=8 epochs, advantages recomputed... — note the code
  actually computes advantages ONCE per round from a V-snapshot, `train.py:139-159`, with a comment
  explaining why recompute-each-epoch was rejected; this further diverges from progress.md's stated
  "recomputed each epoch"). Worth a reviewer's eye: the as-built advantage/target computation matches
  neither plan.md (single-epoch plain A2C) nor progress.md (recompute-each-epoch) verbatim — it is a
  fixed-target K-epoch PPO-clip variant. The attribution-baseline cleanliness argued in the plan depends on
  this default; flagged for the experiment writeup, not as a code defect.

- **All `progress.md` checkboxes are still `[ ]` (unchecked)** even though every listed deliverable is
  implemented on disk and the tests of record exist. The progress file does not reflect the completed
  state.
