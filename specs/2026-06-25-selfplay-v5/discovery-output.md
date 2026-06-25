# Discovery Output — selfplay-v5 (Continual Training)

- **Mode**: BUILD
- **Size**: L (cross-cutting trainer change + multi-hour experiment; all stops enforced)
- **Feature (1-line)**: Make self-play training CONTINUAL — warm-start net weights + carry Adam optimizer state across rounds, micro-batch the dense Medium forward/backward to lift the v4 OOM gate, and re-test whether the structured GNN clears the blind baseline (28.9%) on Medium.
- **Context category**: backend (Python ML trainer; RL correctness + experiment orchestration). Kotlin ONNX bridge is FROZEN/do-not-touch.
- **Domain preset**: data-pipeline (bias). Real council lenses needed: RL/PPO correctness, numerical-equivalence rigor, determinism/reproducibility, compute/throughput. `data_privacy_legal` N/A.
- **Invariant docs loaded**: features, code-quality, architecture, testing — all are the user's *eval-platform* product docs (Next.js/FastAPI), NOT this Unciv repo. Transferable only: Python type hints on signatures, simple/explicit code, small focused files, pytest unit tests for pure logic. Repo conventions come from `python/unciv_train` itself.
- **Prior-work recall**: no hits (in_repo backend, no prior events).

## Light scan summary
See `codebase-scan-light.md`. Key seams (all in `python/unciv_train/`):
- `run_loop.run()` round loop 205–277; `train_round()` 70–100 returns `(net, stats, mode)`; resume 194–200 loads only the ONNX gen model, never torch state/opt. ckpt_round_{r}.pt written (229) but never read.
- `_optimize_actor_critic` 115–194; **Adam at train.py:136** (lift point); divergence guard 139/179–183/188; per-update math 144–192 (FROZEN verbatim).
- `train_actor_critic_structured` 278–317 (manual_seed + `StructuredPolicyValueNet(...,**rung)` + `build_rich_batch` once + `forward_fn`).
- `build_rich_batch` 69–112 (Medium maxn≈1261; spatial + neighbor_index/mask).
- Driver precedent `python/run_acceptance.sh`; analyze.py `_two_proportion_z` 46–55, ceiling eval defaults already 200 games / seed 4242424.
- **turn-cap=250** (v4) — undocumented in prompt, required for apples-to-apples.
- Tests: pytest under `python/tests/`; `test_parity.py`, `test_structured_train.py`, `test_gae.py` are the precedents.

## Open questions — RESOLVED (Round 1)
1. **Execution scope** → **Build + run it all autonomously.** Implement, run the full experiment in background (~3–5h/arm), babysit with `--resume`, write RESULTS.md in-session; ACs closed in-session.
2. **Medium-rung scope (AC3)** → **Full 16-round medium-rung run** (capacity comparison vs small-rung primary arm).

## Implications for design/build
- Two full 16-round Medium arms: small rung (primary, AC1/AC2) + medium rung (AC3). Each ~3–5h. Plus 200-game ceiling evals + bench-onnx. Total ~8–12h autonomous compute.
- Must keep `(net, opt)` alive in-process across rounds; sidecar `opt_round_{r}.pt`; reload net+opt on `--resume`.
- Micro-batch traversal only; per-update math byte-verbatim; numerical-equivalence + warm-start parity tests required (AC6).
- Use `--turn-cap 250 --threads 12` to match v4.
