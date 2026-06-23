# Build progress — onnx-selfplay-v2

## Implementation refinement (logged at Step 13)
- **R4 amended:** critic variants use **A2C+GAE with K=8 epochs, advantages/returns recomputed each epoch from the current critic** — NOT single-epoch. The v1 loop trains a fresh net per round, so the value head needs K gradient steps to fit. Recompute-each-epoch ⇒ no frozen importance ratio ⇒ no stale-ratio instability ⇒ no PPO clip needed (council's stability concern satisfied). Attribution stays clean: blind-critic = v1's exact loop with `advantage = GAE(reward, learned V)` + a value-loss term, instead of `advantage = ret − mean_baseline`. PPO clip remains an unused optional knob. The from-scratch-per-round structure (v1's) is preserved unchanged for all variants.

## Stage A — value critic on blind input (python-only)
- [x] model.py: value_head; forward → (tech, policy, value)
- [x] train.py: compute_gae; train_actor_critic (A2C+GAE, recompute-each-epoch); keep train_reinforce (v1)
- [x] dataset.py: load_trajectories (ordered, terminal-only reward, keep no-action steps); keep load_training_steps
- [x] export_onnx.py: policy-only wrapper drops value head
- [x] test_gae.py GREEN
- [x] run_loop.py: --variant {v1-reinforce, blind-critic, rich-critic}, --resume, --map-size, metrics, checkpoints
- [x] smoke: v1-reinforce + blind-critic loop runs end-to-end (JVM gen/eval)

## Stage B — rich representation (map + entities)
- [x] features.py: multi-tensor assembly from Step.blocks (spatial tile-set + entity token sets + masks)
- [x] model.py: RichPolicyValueNet (masked mean+max pool, NaN-guarded)
- [x] contract.py + SampleSchema.kt: CONTRACT_VERSION 2, multi-tensor names, lockstep
- [x] export_onnx.py: multi-tensor export, dynamic axes incl. masks
- [x] OnnxPolicy.kt: build multi-tensor input from Observation; close all tensors
- [x] SelfPlayRunner.kt: parity-dump/parity-run multi-tensor JSON
- [x] test_parity.py: multi-tensor parity atol=1e-4

## Stage C — eval
- [x] SelfPlayRunner.kt: map-size CLI arg threaded gen/eval
- [x] run_loop.py: --map-size; Medium train+eval; two-proportion test rich vs blind

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

## Bookkeeping (Step 14/16)
- [x] cleanup-opportunities.md written (2 items: obs normalization, hyperparam magic constants)
- [x] traceability.md written (sub-agent) — all plan items implemented; test gaps closed below
- [x] test gaps closed: test_v2_units.py (load_trajectories R1, masked_pool R3, build_rich_batch, export_rich value-drop)
- [x] doc drift reconciled: decisions.md R12 + plan.md amended to the as-built standard-PPO recipe
- [ ] full training runs to acceptance (background) + analyze.py — IN PROGRESS
