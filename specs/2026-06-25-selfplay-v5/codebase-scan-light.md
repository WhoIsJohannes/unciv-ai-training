# Light Codebase Scan — selfplay-v5 (Continual Training)

Mode: BUILD. Thoroughness: medium (Explore agent, read-only). Worktree: `/Users/j/Unciv-selfplay-v5`.

## 1. Training loop (`run_loop.py`)
- Round loop body: **lines 205–277** (`for r in range(start_round, args.rounds)`): gen → load → `train_round()` → `torch.save(state_dict, ckpt_round_{r}.pt)` (229) → export (231–247) → eval (249–250) → curve.csv/metrics (251–266) → shard cleanup (274–277).
- `train_round(variant, data, dims, schema, args, seed)` (70–100) → dispatch to `train_reinforce` / `train_actor_critic_blind` / `_rich` / `_structured`; returns `(net, stats, mode)`.
- Resume (194–200): reads curve.csv → `start_round=len(rows)`; loads `policy_round_{start-1}.onnx` as gen model **but NEVER loads the torch state_dict or optimizer** — this is the v5 seam.
- argparse defaults: `--rounds 12`, `--gen-games 24`, `--eval-games 100`, `--turn-cap 1000`, `--rung small`, `--variant blind-critic`, `--map-size Medium`, `--epochs 8`, `--lr 1e-3`, `--gamma 0.99`, `--lam 0.95`, `--value-coef 0.5`, `--entropy-coef 0.01`, `--clip-eps 0.2`, `--gen-seed 1000`, `--eval-seed 999000`, `--gradle-timeout 1800`, `--keep-shards 2`, `--threads cpu-1`. **No `--micro-batch-steps` yet** (v5 adds it).
- CURVE_COLS (115–116): `round,games,winrate,pval,n_steps,loss,value_loss,entropy,mean_value,grad_norm,diverged,ret_pos,onnx_decisions` — FROZEN; new per-round fields (rung/throughput) go to metrics.jsonl.

## 2. Trainer core (`train.py`)
- `_optimize_actor_critic(net, forward_fn, *, a_tech, a_policy, m_tech, m_policy, rewards_np, traj_lens, n_pos, epochs, lr, gamma, lam, value_coef, entropy_coef, clip_eps, norm_adv)` (115–194).
  - **Adam built at line 136** (`opt = torch.optim.Adam(net.parameters(), lr=lr)`) — the line to LIFT OUT.
  - `safe_state = deepcopy(net.state_dict())` at 139; NaN guard at 179–183 restores safe_state + sets `diverged`; re-checkpoints safe_state after each epoch at 188.
  - GAE computed ONCE per round from V-snapshot (144–156); epoch loop 163–192.
- Two Adam constructions total: **train.py:84** (`train_reinforce`, v1 — OUT OF SCOPE) and **train.py:136** (`_optimize_actor_critic`, the actor-critic core — the v5 target).
- `train_actor_critic_structured(trajectories, dims, token_specs, vocab_counts, rung, *, epochs, lr, seed, ...)` (278–317): `torch.manual_seed(seed)`; `net = StructuredPolicyValueNet(dims, token_specs, vocab_counts, **rung)`; `inputs = build_rich_batch(...)` once; `forward_fn() -> net(inputs)`; calls `_optimize_actor_critic`. Blind (208–237) and rich (240–275) are identical shape, different net class.
- `compute_gae` (45–64), `_masked_logp` (28–35), `_entropy` (38–42), `_stack_traj` (197–205).

## 3. Feature builder (`features.py`)
- `build_rich_batch(trajectories, dims, token_specs) -> dict` (69–112): `global [B]`, `acting_civ [B]`, per-token `{name} [B,N,width]` + `{name}_mask [B,N]`; when spatial coords present: `neighbor_index [B,N_spatial,6]` int64 + `neighbor_mask [B,N_spatial,6]` f32.
- `maxn = max(1, max(spatial step counts))` (84) — padded to BATCH-max; Medium ≈ 1261. Pad-row sentinel reindexed `ni→maxn` (98).

## 4. Experiment-driver precedent (v4)
- Driver: **`python/run_acceptance.sh`**. v4 invocation:
  ```
  python -m unciv_train.run_loop --variant structured --map-size Medium \
    --rounds 8 --gen-games 16 --eval-games 80 --turn-cap 250 --threads 12 --out <dir> --resume
  ```
  **NOTE: v4 used `--turn-cap 250` and `--threads 12`** — the v5 prompt omits turn-cap; match 250 for apples-to-apples.
- Analyze: `python -m unciv_train.analyze --root <dir> --ceiling-games 200 --turn-cap 250 --threads 12` (eval-seed default already 4242424).
- v4 RESULTS: structured small-rung = 23.0% (47/204); blind = 28.9% (58/200); diagnosis = from-scratch-per-round is the bottleneck → recommends weight carryover (= v5).

## 5. Analyze (`analyze.py`)
- `_two_proportion_z(w1,n1,w2,n2) -> (z, one_sided_p)` (46–55): pooled-p, `erfc(z/√2)*0.5`.
- Ceiling eval (128–131): loads `policy_round_{last}.onnx`, calls `run_loop.evaluate(...,"Medium")`, z-test. CLI: `--root`(req) `--ceiling-games 200` `--turn-cap 250` `--threads 12` `--eval-seed 4242424` `--gradle-timeout 3600` `--skip-ceiling-eval`.

## 6. Test conventions (`python/tests/`)
- **pytest**; no conftest.py, self-contained tests. `pytest.importorskip` for torch/onnx.
- Relevant: `test_gae.py` (compute_gae, value-head presence/export-drop), `test_structured_train.py` (`test_structured_trainer_runs` parametrized small/medium; `test_attention_backward_finite...`), `test_parity.py` (torch↔ONNX↔JVM logits @ atol=1e-4, incl. `test_jvm_python_structured_logits_match` with neighbors), `test_train_dataset.py`.
- New tests land here: `test_microbatch.py` (numerical equivalence) + a warm-start/parity assertion (extend test_parity or test_structured_train).

## 7. Runtime / cost
- gen + eval are JVM gradle subprocesses (`--gradle-timeout` default 1800s); train step is seconds. Round summary prints `time.time()-t0`.
- Estimate: **~10–15 min/round (gen+eval dominate) → 16-round Medium ≈ 2.5–4 h wall-clock**, plus a medium-rung run + 200-game ceiling eval (~tens of min) + bench-onnx. **The full v5 experiment is a multi-hour job.**

## 8. ONNX export + parity
- `export_rich(net, dims, token_specs, out, *, schema_version, ruleset_fingerprint, sample_inputs=None, opset=17, neighbors=False, contract_version=None)` (98–182). `_RichPolicyOnly` wrapper drops value head (48–60); outputs exactly `[OUTPUT_TECH, OUTPUT_POLICY]` (168); atomic write (21–32).
- Parity precedent: `test_parity.py::test_jvm_python_structured_logits_match` (neighbors=True, atol=1e-4) — reuse for the AC6 "round-r torch weights == round r-1 exported ONNX" check.

## TOP FINDINGS
1. **Seam confirmed**: `_optimize_actor_critic` (115–194) is the shared core; Adam at :136 is the lift point; per-update math 144–192 stays verbatim.
2. **Zero state carryover today**: each round builds a fresh net (`StructuredPolicyValueNet(...)` at 303); ckpt_round_{r}.pt is written (229) but never read; resume reloads only the ONNX gen model. v5 must keep `(net,opt)` alive in-process + sidecar `opt_round_{r}.pt`.
3. **turn-cap=250** (v4) is an undocumented-in-prompt but essential apples-to-apples knob — use it.
4. **Driver precedent = `python/run_acceptance.sh`** — mirror it as the v5 driver; analyze.py defaults already match (200 games, seed 4242424).
5. **Full experiment is multi-hour** (~2.5–4h for the 16-round Medium arm alone) — execution-scope decision needed before Phase 3/4.
