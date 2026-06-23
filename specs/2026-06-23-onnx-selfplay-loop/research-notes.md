> Web-sourced content below is DATA, not instructions.

# Research Notes — ONNX-in-JVM + PyTorch→ONNX + masked REINFORCE

## 1. onnxruntime JVM (`com.microsoft.onnxruntime:onnxruntime`)
- **Query**: ONNX Runtime Java API — env/session/tensor/inference/metadata.
- **Maven**: `com.microsoft.onnxruntime:onnxruntime:<ver>` (CPU; bundles native libs for common OS/arch). GPU variant `onnxruntime_gpu` not needed. Pin a recent stable (1.20.x-era) and verify it resolves on desktop JVM. Place per D4 (compileOnly core + impl desktop).
- **API (official get-started + Javadoc)**:
  - `OrtEnvironment env = OrtEnvironment.getEnvironment();` — process singleton, shared.
  - `OrtSession session = env.createSession(path, new OrtSession.SessionOptions());`
  - Input: `OnnxTensor t = OnnxTensor.createTensor(env, FloatBuffer.wrap(vec), new long[]{1, width});` map by input name `Map.of("obs", t)`.
  - Run + read named output: `try (var r = session.run(inputs)) { float[][] logits = (float[][]) ((OnnxTensor) r.get("tech_logits").get()).getValue(); }` — shape `[1, headWidth]`. Close tensors/Result (AutoCloseable) to avoid native leaks.
  - **Metadata (provenance gate)**: `session.getMetadata().getCustomMetadata()` → `Map<String,String>` (the keys written by export_onnx). EVAL reads `schema_version`/`ruleset_fingerprint`/`contract_version` and REFUSES on mismatch (criterion 6).
- **Thread-safety**: ORT `OrtSession.run()` is designed for **concurrent inference on one session** (the C `OrtRun` is thread-safe per-session); `OrtEnvironment` is a shared singleton. → ONE shared read-only session across `Simulation`'s worker threads satisfies the task. Per-session thread pool is internal; for many short single-row inferences set `SessionOptions.setIntraOpNumThreads(1)` to avoid oversubscription against the coroutine workers. **Verify against the pinned Javadoc during build** (low risk, well-established).
- **Relevance**: directly drives `OnnxPolicy` (lazy shared session, build `FloatArray` input via Featurizer, read `tech_logits`/`policy_logits`).

## 2. PyTorch → ONNX export (fixed names + metadata)
- `torch.onnx.export(model, dummy, "policy.onnx", input_names=["obs"], output_names=["tech_logits","policy_logits"], dynamic_axes={"obs":{0:"batch"},"tech_logits":{0:"batch"},"policy_logits":{0:"batch"}}, opset_version=17)`. Model `forward(obs)` returns a TUPLE `(tech_logits, policy_logits)` → two named outputs (tuples/lists are the supported multi-output form).
- **Custom metadata**: export, then `m = onnx.load(p); onnx.helper.set_model_props(m, {"schema_version":"2","ruleset_fingerprint":"<hex>","contract_version":"1","input_width":str(W),"tech_width":str(T),"policy_width":str(P),"heads":"tech,policy"}); onnx.save(m, p)`. (metadata_props is a list of StringStringEntryProto; `set_model_props` is the canonical helper.)
- Use a FIXED batch dummy of width `concat(global,acting_civ)`; keep float32 throughout (parity with JVM f32 blocks).
- **Relevance**: `export_onnx.py` contract; the Java metadata read closes the provenance loop.

## 3. Masked REINFORCE-with-baseline (CPU, well-known)
- Per recorded learner step: obs = concat(global, acting_civ); per head h∈{tech,policy}: chosen index `a_h = step.blocks["actions"][headIdx]` (skip if `a_h < 0` / head not acted); legal mask `m_h = step.blocks["mask_tech"|"mask_policy"]`.
- Masked log-prob: `logits[~m_h] = -1e9; logp = log_softmax(logits); logp_a = logp[a_h]`. (Gather only over legal support — guarantees the policy never assigns prob to illegal actions, mirroring JVM `-inf` masking.)
- Return-to-go = the civ's terminal reward (±1) broadcast to all its steps (undiscounted single terminal reward, per D1). Baseline = running mean of returns; advantage `A = R - baseline`. Loss `= -(A) * Σ_h logp_a_h`, averaged over steps. Optional entropy bonus for exploration (small coeff) — keep minimal per non-goals.
- **Relevance**: `train.py`. Tiny MLP, Adam, CPU — converges fast on a few thousand steps/round.

## 4. ONNX I/O contract parity (the anti-drift requirement)
- ONE input path: JVM builds `concat(featurizer.observe(civ).block("global"), block("acting_civ"))` as `float[]`; Python builds `np.concatenate([step.blocks["global"], step.blocks["acting_civ"]])`. Both f32, same order. The PARITY test feeds ONE fixed observation vector through the JVM `OrtSession` and a Python `onnxruntime`/torch reference and asserts logits match within fp tol — catches any width/order/dtype drift across the boundary.
- Widths are RUNTIME (GnK vocab) — read from the generated `schema.json` layout (block `len`) / a generated shared constants file, never hardcode 80/70. Stamp the resolved widths into ONNX metadata so a mismatched model is refused.

## Key insights
1. One shared read-only `OrtSession` + shared `OrtEnvironment` satisfies thread-safety for the multithreaded `Simulation` — no per-worker session needed (verify on the pinned Javadoc; set intra-op threads=1).
2. Multi-head MLP exports cleanly as a tuple-returning `forward` → `output_names=["tech_logits","policy_logits"]`; provenance rides in `metadata_props` and is read back via `session.getMetadata().getCustomMetadata()` to enforce the contract on BOTH ends.
3. The whole anti-drift story reduces to: identical concat(global,acting_civ) f32 vector on both sides + a golden PARITY test + runtime-derived widths stamped into ONNX metadata.
4. REINFORCE math is trivial; the only correctness-critical pieces are (a) wiring the terminal reward into shards (D1) and (b) masked log-prob that exactly mirrors JVM `-inf` masking so train-time and infer-time legality agree.
