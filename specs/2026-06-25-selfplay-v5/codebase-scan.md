# Deep Codebase Scan — selfplay-v5 (Continual Training)

Explore agent, very-thorough, read-only. Full transcript in task output. All claims file:line in `python/unciv_train/`.

## 1. `_optimize_actor_critic` body (train.py 115-194)
- Sig 115-134 (`net, forward_fn, *, a_tech,a_policy,m_tech,m_policy,rewards_np,traj_lens,n_pos,epochs,lr,gamma,lam,value_coef,entropy_coef,clip_eps,norm_adv`).
- 136: `opt = torch.optim.Adam(net.parameters(), lr=lr)` — **LIFT TARGET**.
- 137 `n=a_tech.shape[0]` (flat step count); 139 `safe_state=deepcopy(net.state_dict())` — divergence checkpoint (net only).
- 144 `with no_grad:` 145 `tl0,pl0,val0 = forward_fn()` (snapshot); 147 `old_logp=(_masked_logp(tl0..)+_masked_logp(pl0..)).detach()`.
- 149-156 per-trajectory GAE over `traj_lens` slices `[off:off+L]`; 157-161 to tensors, **adv normalized BATCH-WIDE** (159: `(adv-adv.mean())/(adv.std()+1e-8)`).
- 163 `for ep in range(epochs)`; 164 `opt.zero_grad()`; 165 `tl,pl,val=forward_fn()` (epoch fwd); 167 `logp=...`; 168-174 policy_loss (PPO clip if clip_eps else `-(adv*logp).mean()`); 175 `value_loss=F.mse_loss(val,ret)`; 176 `ent=(_entropy(tl..)+_entropy(pl..)).mean()`; 177 `loss=policy_loss+value_coef*value_loss-entropy_coef*ent`.
- 179-183 NaN guard: `if not isfinite(loss): net.load_state_dict(safe_state); diverged=True; return` (**net only — no opt restore**).
- 185 `loss.backward()`; 186 `clip_grad_norm_(...,max_norm=10.0)`; 187 `opt.step()`; 188 re-`safe_state=deepcopy(net.state_dict())`.
- 193-194 return `(net, stats)` — opt discarded.

## 2. forward_fn per trainer — all return `(tech[B,tech_w], policy[B,policy_w], value[B,1])`, B=flat step axis
- blind 229-230 `net(obs)` (obs [B,input_w] @227); rich 267-268 `net(inputs)` (build_rich_batch @265); structured 309-310 `net(inputs)` (build_rich_batch @307, net @303).

## 3. CHUNKABILITY — CONFIRMED per-row independent
- All inputs are `[B,...]` on axis 0 (global[B], acting_civ[B], spatial[B,N,w], `{ent}`[B,M,w]+mask, neighbor_index/mask[B,N,6]). StructuredPolicyValueNet.forward (model.py 454-524): GNN/attention are **within-step** (over tiles/tokens), `_masked_mean` per step, single-query cross-attn per step. **No cross-STEP op, no batchnorm-over-batch, no global pool across steps.** Slicing `inputs[name][lo:hi]` → `net(sub)` equals `net(full)[lo:hi]`.
- ⚠️ **CAVEAT**: max-N padding is BATCH-WIDE (features.py:84). Must slice the ALREADY-PADDED full-batch tensors — do NOT recompute padding per chunk.
- ⚠️ adv normalization is batch-wide (159) → compute GAE/adv/ret/old_logp ONCE on full batch, INDEX per chunk.

## 4. train_round (run_loop.py 70-100) → `(net, stats, mode)`; mode ∈ {"blind", ("rich",ts), ("structured",ts)} (76/89/99). Single caller @226; mode consumed only for export 232-247.

## 5. Resume (run_loop.py 194-200): `start_round=len(rows)` from curve.csv; loads ONLY `policy_round_{start-1}.onnx` for gen. **net/opt never reloaded.** ckpt_round_{r}.pt saved @229 but NEVER loaded. generate()@209 + evaluate()@249 take ONNX paths — **zero in-memory net carryover today.**

## 6. Collision/reuse: NO existing chunked-forward / grad-accum / `opt.state_dict()` save anywhere. Exactly **3** `_optimize_actor_critic` call sites (232/270/312). No "fresh net per round" assertion.

## 7. Determinism: `torch.manual_seed(seed)` in each trainer (79/222/260/302), `seed=r` @226. gen seed `gen_seed+r*1000`@210, eval seed @249 — both independent of trainer seed. → **Skip manual_seed on warm rounds (r>round0)** else weights re-init.

## 8. Tests (python/tests/): `test_structured_trainer_runs` checks `isfinite(loss)` + `not diverged` ONLY (safe under refactor). `test_structured_attn` finite-grad (safe). `test_gae`/`test_parity`/`test_train_dataset`/`test_v2_units` no trainer calls (safe). `test_determinism` tests GENERATION only (safe). **No test pins optimizer-inside or exact loss values.**

## 9. Metrics: CURVE_COLS frozen (115-116). metrics.jsonl write @261-266 already carries variant/map_size/rung/turns_per_sec/ms_per_decision/mean_adv — **add warm-start/throughput fields HERE, not curve.csv.**

## 10. bench-onnx: SelfPlayRunner.kt benchOnnx (471-511), gate `ratio>=0.70`@504, dispatched `"bench-onnx"`@77; CLI `./gradlew selfPlay --args="bench-onnx <onnx> [turns] [map] [threads] [seed]"`. **NOT wired into run_loop/analyze** — invoked manually.

## Top surprises
1. **No warm-start infra**: train_round has no net/opt params; optimizer reborn @136 each round; manual_seed re-inits weights every round.
2. **Divergence guard restores net only** (180) — persistent opt would keep poisoned moments → must snapshot/restore `opt.state_dict()`.
3. **adv normalized batch-wide** (159) → micro-batch MUST index a once-computed adv, never recompute per chunk.
4. **No seed-skip for warm rounds** — must guard manual_seed on r>round0 or weights re-init (determinism break).
5. **forward_fn closes over full-batch tensors by reference** — chunking needs a slice-aware forward (pass inputs/obs into the optimizer, or build a `forward_chunk_fn(lo,hi)`), not the zero-arg closure.
6. **ckpt_round_{r}.pt saved but never loaded** — dead infra v5 will finally use; saves are NOT atomic today (council FND-0011: make them atomic).
7. **bench-onnx not wired in** — must invoke via gradle subprocess after export (like generate/evaluate).
8. **metrics.jsonl is append-only** but the round loop only runs not-yet-done rounds on resume, so no duplication (scan's dup worry is moot).
