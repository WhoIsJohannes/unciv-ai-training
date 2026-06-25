# Decisions — selfplay-v5 (Continual Training)

## Phase 1 (discovery)
- **D-Q1 Execution scope** → Build + run the full experiment autonomously (background, ~3–5h/arm, babysit with `--resume`), write RESULTS.md in-session. ACs closed in-session. [user clarification, Step 3]
- **D-Q2 Medium-rung scope (AC3)** → Full 16-round medium-rung run as a capacity comparison vs the small-rung primary arm. [user clarification, Step 3]
- **D-turncap** turn-cap=250 + threads=12 to match v4 apples-to-apples (undocumented in prompt; sourced from `python/run_acceptance.sh`). [Claude interpretation, scan finding]

## Phase 2 (design)

### Step 5 intake council roster rationale
- **Core 6** (always-on): skeptic, architect, practitioner, product_manager, qa_testing, security_red_team.
- **+domain_fidelity**: RL/PPO is a specialist domain where the wrong abstraction is a silent failure (biased importance ratio, broken on-policy assumption, Adam-state poisoning). Load-bearing. Modeled on Opus (deep).
- **+cost_efficiency**: multi-hour compute, micro-batching to fit memory, throughput gate (bench-onnx ≥0.70). Relevant.
- **Skipped**: end_user/power_user/new_user_onboarding/support_agent (no UI; "user" = researcher), data_privacy_legal/compliance (no PII/regulated data), ethics_responsible_ai (game-AI research, no user-impacting harm), finance/b2b_buyer/marketing/investor (no monetization/GTM), accessibility/i18n (no frontend).
- Roster size 8 (L cap ≤14). Vendor-diverse: Opus / Gemini / GPT / DeepSeek spread across roles.

### Step 5 intake council triage (34 findings: 11 critical, 21 major, 2 minor)
Council reviewed `discovery-output.md` only (not the full decisive prompt), so the scope/PM cluster is largely artifact.

**A. ADOPT into plan (genuine build-improvers, default-rigor):**
- FND-0011 → **atomic checkpoint saves** (ckpt + opt sidecar via tmp→os.replace, mirroring export_onnx atomic write).
- FND-0012/0010/0028 → **snapshot/restore `opt.state_dict()` in the divergence guard** (already deliverable #2/#5).
- FND-0006 → **skip `manual_seed` on warm rounds** (already; confirmed by scan #4/#7).
- FND-0007/0003 → **resume consistency invariant**: require curve.csv + ckpt_round_{N-1}.pt + opt_round_{N-1}.pt all present & rung/dims-compatible; fail-fast with clear error; if opt sidecar absent (pre-v5 run) fail loudly (no silent fresh-optimizer).
- FND-0025 → **`torch.load(weights_only=True)`** on resume (already mandated by prompt; enforce).
- FND-0021/0009/0029 → micro-batch **size-weighted accumulation** (chunk loss × n_c/N) + GAE/adv/old_logp computed ONCE & indexed per chunk; **numerical-equivalence test** with the whole-batch path as oracle (atol≈1e-5 on loss + grad-norm).
- FND-0022 → **interruption/resume-equivalence test** (interrupt at round k + --resume == uninterrupted, identical curve).
- FND-0030 → reinterpret AC6 parity as **warm net == prior-round exported gen net** (tech/policy/embedding initializers) — proves on-policy carryover.
- FND-0001/0002/0004 → **YAGNI**: minimal plumbing (plain (net,opt) locals threaded through), ONE chunk-forward helper, no abstraction/framework.
- FND-0013 → **disk hygiene**: prune old ckpt/opt .pt beyond a keep window (small files; low-pri).
- FND-0034 → throughput/timing already → metrics.jsonl (cost telemetry).

**B. DISMISS — already defined in the full prompt (council saw only the summary):**
- FND-0015/0023 success metric & thresholds: defined — beat 23.0% at p<0.05 (one-sided z) + report vs 28.9% (AC1/AC2).
- FND-0020 ACs: 7 ACs defined. FND-0018 out-of-scope: NON-GOALS section. FND-0017 non-negotiables: FROZEN section. FND-0016 prioritization: ACs are ordered.
- FND-0031 apples-to-apples: fixed seed 4242424 / same map / same RandomPolicy opponent / turn-cap 250 — it IS apples-to-apples.

**C. DISMISS — refuted by research/design:**
- FND-0027 "Adam carryover violates on-policy": FALSE — `old_logp` is recomputed from the current (warm) net, whose weights == the gen weights; carryover changes only init. (Research-confirmed.)
- FND-0019 Adam stability: handled by the divergence guard (now opt-aware).
- FND-0026 `--out` arbitrary write: operator-controlled local research tool; not a threat surface. Note only.
- FND-0032/0033 circuit-breaker for wasted compute: babysat run + `diverged` flag in curve.csv; no auto-abort framework (FND-0001 YAGNI). Note only.
- FND-0014 in-process memory: small/medium-rung net+opt are tiny; the big alloc is the batch (micro-batching handles it).

### Step 11 plan-council triage (34 findings: 5 critical, 23 major, 6 minor) → 4 plan revisions
**REVISIONS applied to plan.md (council-driven):**
- **R1 (FND-0006/0009/0029)** — add `--continual` flag (default True). Default = v5 continual mode;
  `--continual false` restores per-round fresh net+opt (v4 from-scratch) for rollback + clean A/B.
  Reverses the earlier D1 "no toggle".
- **R2 (FND-0002)** — NO `_ac_loss_terms` helper. Keep train.py:144-192 whole-batch path BYTE-VERBATIM
  inside an `else:` branch (wrapping in else = the permitted "traversal change"); the micro-batch path
  is NEW code with the same expressions, size-weighted. Equivalence guarded by test_microbatch.py.
  Reverses the earlier D3 helper.
- **R3 (FND-0001)** — collapse the resume API: trainers take only `net=None, optimizer=None,
  micro_batch_steps=None` (3 added params). run_loop does the disk-load (build arch + load_state_dict,
  weights_only=True) via a small helper and passes warm (net,opt) objects — no resume_ckpt/resume_opt
  in the trainer signature. load_state_dict naturally errors on rung/dims mismatch (FND-0008) → wrap
  with a clear message.
- **R4 (FND-0029/0027)** — RESULTS must (a) report v5 round-8 winrate vs v4-from-scratch@8=23.0% to
  disentangle regime from round-count, and (b) note that warm-start makes `old_logp` == the gen policy
  (more on-policy-correct than v4's random-init anchor) — refuting FND-0027.

**DISMISS (refuted / out-of-threat-model / prompt-defined):**
- FND-0027 on-policy break: REFUTED (warm net at round start == gen policy == old_logp anchor).
- FND-0028 opt-rollback desync: export happens AFTER the round with the (restored) final net, which IS
  the next round's gen policy → anchor stays consistent. No desync.
- FND-0030 adv normalization: per-round-batch normalization is identical to v4; micro-batch computes
  adv once on the full batch and indexes → unchanged.
- FND-0004 AC6 vs dropped value head: parity is on tech+policy (the exported, generation-relevant
  heads) — value head is train-only; the test already asserts out_names == {tech,policy}. Correct as-is.
- FND-0024/0025/0026 security: local single-user research tool, no adversary on the run dir;
  weights_only=True covers pickle-RCE/model-substitution on resume. Note only.
- FND-0011/0012/0033 ops: small nets (tiny RSS), --gradle-timeout bounds hangs, sequential arms babysat.
- FND-0013/0014/0015/0018 scope: success metric/ACs/out-of-scope defined in prompt; AC2 reports a
  null/negative result PLAINLY (the experiment answers the question either way).
- FND-0016/0031 replay: DEFERRED per prompt (gated on behavior-logp recording); 16 rounds stay
  on-policy (each round trains only on its own freshly-generated data).
**Adopt-lite:** quote run_v5.sh args (FND-0023); prune old ckpt/opt .pt beyond keep window (FND-0010);
document K=256 choice (FND-0034); report divergence status before ceiling eval (FND-0032).
**Verdict: APPROVE with revisions applied** (no unmitigated blocker; criticals either adopted as the
`--continual` flag or dismissed with rationale).

### Step 18 ship-council triage (33 findings: 3 critical, 22 major, 8 minor) — smoke PASSED first
Continual pipeline smoke (Tiny, 2 rounds) confirmed end-to-end: round 0 cold → round 1 warm_start=True, ckpt+opt sidecars + export + eval all work.

**FIX before the long run (5 genuine items):**
- **FND-0016 (major)** — AC6 ONNX parity test only checked out_names, never compared logits → strengthen to run the ONNX session and assert warm-net torch logits == ONNX logits (atol 1e-4), the real on-policy invariant.
- **FND-0010/0003/0015/0022/0031** — run_v5.sh swallowed arm failures, ran ceiling eval on a broken arm, reported DONE → track per-arm success, skip eval for a failed arm, report accurate final status (robustness for the unattended multi-hour run).
- **FND-0012 (major)** — sidecar prune unlink could crash the loop → wrap in try/except (best-effort, never crash training).
- **FND-0013/0033 (major)** — add `gc.collect()` at end of each round (memory insurance for the 16-round Medium run).
- **FND-0004/0019 (minor)** — negative `--micro-batch-steps` crashed `range()` → require `micro_batch_steps > 0` for the chunk path (else no-op).

**DISMISS (false positive / already-handled / approved design):**
- FND-0005 resume-load-once: CORRECT by design (load at first resumed round via `warm_net is None and r>0`, carry in-memory after). Not a bug.
- FND-0006 torn curve write: SAFE — curve.csv is the completion marker (written last); a crash mid-round leaves an orphan ckpt that re-running the round overwrites; resume keys off curve len.
- FND-0007/0026 pruning vs resume: SAFE — prune only `r-3`; resume needs `start-1` = last completed = within keep-3.
- FND-0027 _load_warm lr: load_state_dict restores saved param_groups (incl lr); lr is constant anyway (no scheduler). Non-issue.
- FND-0028 round-7 off-by-one: CORRECT — round index 7 (0-based) IS the 8th round, matched to v4's 8-round run; 80g-vs-200g caveat already documented.
- FND-0002 2-tuple callers: all call sites updated (verified: run_loop ×3 + all tests).
- FND-0025/0008/0030 diverged-carry/circuit-breaker: divergence is handled (guard restores last-good net+opt, exports last-good, `diverged` flag logged in curve.csv) + babysat; no auto-abort framework (YAGNI, approved).
- FND-0001/0017 continual default: approved plan-council R1 decision; `--no-continual` reproduces v4.
- FND-0018 resume config validation: load_state_dict errors loudly on rung/dims mismatch (the validation).
- FND-0011 .tmp leak: fixed tmp name, overwritten next save; disk-full correctly raises.
- FND-0009/0029/0021/0024/0014/0020/0023/0032 minors: stats are diagnostic; fixed eval seed is intended (v4 parity); idempotent overwrite is fine; bench/analyze exit handled by babysitting.
