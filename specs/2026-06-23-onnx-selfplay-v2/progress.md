# Build progress — onnx-selfplay-v2

## Implementation refinement (logged at Step 13)
- **R4 amended:** critic variants use **A2C+GAE with K=8 epochs, advantages/returns recomputed each epoch from the current critic** — NOT single-epoch. The v1 loop trains a fresh net per round, so the value head needs K gradient steps to fit. Recompute-each-epoch ⇒ no frozen importance ratio ⇒ no stale-ratio instability ⇒ no PPO clip needed (council's stability concern satisfied). Attribution stays clean: blind-critic = v1's exact loop with `advantage = GAE(reward, learned V)` + a value-loss term, instead of `advantage = ret − mean_baseline`. PPO clip remains an unused optional knob. The from-scratch-per-round structure (v1's) is preserved unchanged for all variants.

## Stage A — value critic on blind input (python-only)
- [ ] model.py: value_head; forward → (tech, policy, value)
- [ ] train.py: compute_gae; train_actor_critic (A2C+GAE, recompute-each-epoch); keep train_reinforce (v1)
- [ ] dataset.py: load_trajectories (ordered, terminal-only reward, keep no-action steps); keep load_training_steps
- [ ] export_onnx.py: policy-only wrapper drops value head
- [ ] test_gae.py GREEN
- [ ] run_loop.py: --variant {v1-reinforce, blind-critic, rich-critic}, --resume, --map-size, metrics, checkpoints
- [ ] smoke: v1-reinforce + blind-critic loop runs end-to-end (JVM gen/eval)

## Stage B — rich representation (map + entities)
- [ ] features.py: multi-tensor assembly from Step.blocks (spatial tile-set + entity token sets + masks)
- [ ] model.py: RichPolicyValueNet (masked mean+max pool, NaN-guarded)
- [ ] contract.py + SampleSchema.kt: CONTRACT_VERSION 2, multi-tensor names, lockstep
- [ ] export_onnx.py: multi-tensor export, dynamic axes incl. masks
- [ ] OnnxPolicy.kt: build multi-tensor input from Observation; close all tensors
- [ ] SelfPlayRunner.kt: parity-dump/parity-run multi-tensor JSON
- [ ] test_parity.py: multi-tensor parity atol=1e-4

## Stage C — eval
- [ ] SelfPlayRunner.kt: map-size CLI arg threaded gen/eval
- [ ] run_loop.py: --map-size; Medium train+eval; two-proportion test rich vs blind

## Full run to acceptance
- [ ] smoke-measure per-round wall-clock; set budget
- [ ] Tiny: v1-reinforce, blind-critic, rich-critic curves (12 rounds)
- [ ] Medium: blind-critic vs rich-critic (ceiling)
- [ ] report: last-K=4 stddev (AC2), Medium two-proportion p (AC3)

## Codebase patterns
- Fresh PolicyNet per round (no carryover); 8 full-batch Adam epochs; seed=round.
- Masked-logp: illegal→-1e9, log_softmax, gather chosen, 0 where action<0. REUSE verbatim.
- Provenance gates in dataset (schema_version + fingerprint) + OnnxPolicy.init. PRESERVE.
- Shard blocks present-only (VARIABLE = count-prefixed); pad in loader.
