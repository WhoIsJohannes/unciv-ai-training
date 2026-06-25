# v4 Structured Encoder — SHIPPED (local branch `onnx-selfplay-loop`)

**One-liner:** replaced the v3 rich-critic's permutation-invariant masked-pool spatial encoder with a
structured encoder (categorical embeddings + hex-GNN over true adjacency + hand-rolled self/cross-attention,
capacity ladder + throughput guard), fixing the v3 Medium regression.

**Verdict: SHIPPED — all 7 acceptance criteria met.**
- **AC1 (primary) PASS:** Medium structured GNN **23.0% vs v3 rich-pool 14.7%** (z=+2.20, p=0.014); Tiny
  non-regression met with the demand-appropriate medium rung (52.2% vs 57%, z=−1.35, not-significantly-below).
- AC2 rung sweep + demand-driven rule demonstrated (Tiny→medium, Medium→small); AC3 parity (logits +
  adjacency-fidelity); AC4 throughput (bench-onnx PASS, ~24 ms/dec Medium); AC5 provenance/legality;
  AC6 frozen core + no new heads; AC7 construction bug + test.
- **Honest limit:** Medium structured still < blind (28.9%) — the from-scratch-per-round *training* ceiling
  (pre-registered confound), NOT the encoder. **Next unlock: weight carryover** (see RESULTS.md + memory).

**Contract:** bumped v2→v3 in lockstep (Kotlin emit/bridge + Python train/export + reader). Frozen training
core (`_optimize_actor_critic`/`compute_gae`, terminal-reward+critic) untouched — only the nn.Module swapped.

**Key files:** `hexgraph.py` (adjacency builder), `model.py` (StructuredPolicyValueNet), `Featurizer.kt`
(v3 emit + spatial_coords + construction fix), `OnnxPolicy.kt` (v3 inference, live-engine neighbor tensors),
`SelfPlayRunner.kt` (parity adj + adjacency-dump + bench-onnx), `contract.py`/`features.py`/`dataset.py`/
`export_onnx.py`/`train.py`/`run_loop.py` (v3 + structured variant + ladder). Tests: hexgraph, structured
smoke/attn/train, contract fail-loud, ConstructionCodeTest, parity (v3 logits + adjacency-fidelity).

**Process:** /feature run — Phase 1 discovery, Phase 2 (4-cluster adversarial design workflow + plan +
2 plan-council rounds + approval gate), Phase 3 build (Phase A/B/C), Phase 4 ship-council (1 critical
NaN-grad fixed + regression test). ~22 commits. Worktree+branch were externally deleted mid-session and
recreated from master (no loss).

**Open / next:** weight-carryover (the data-backed unlock); medium/large rung sweep on Medium is OOM-gated
by the whole-round dense batch (micro-batching = noted future work). Cleanup debt in
`.feature-workflow/cleanup-opportunities.md` (model.py/SelfPlayRunner.kt size; HEX_OFFSETS duplication).

**NOT merged** — local fork research; merge decision deferred to the user.
