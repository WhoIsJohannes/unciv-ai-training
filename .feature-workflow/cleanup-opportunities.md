---
schema: cleanup-v1
generated_at: 2026-06-24T22:25:01Z
---

## Cleanup opportunities

- [ ] model.py is 520 lines (>300) — the v4 StructuredPolicyValueNet + attention/GNN helpers could split into a structured_encoder.py module  <!-- id:a7833df4-513e-452c-a1de-bec942207860 file:python/unciv_train/model.py line:520 kind:large-file source_commit:5ac0a3cf6 -->
- [ ] SelfPlayRunner.kt is 512 lines — the parity/adjacency/bench harness modes could move to a separate ParityHarness object  <!-- id:89dc8c7b-a683-47e6-8803-f0c50eefcec0 file:desktop/src/com/unciv/app/desktop/SelfPlayRunner.kt line:512 kind:large-file source_commit:5ac0a3cf6 -->
- [ ] hex OFFSETS duplicated in hexgraph.py (Python) and OnnxPolicy.HEX_OFFSETS (Kotlin) — necessary cross-language but a drift risk; guarded by the adjacency-parity test (test_hexgraph_matches_live_engine)  <!-- id:39a8da18-bf31-4de9-bc04-ceec98e061ba file:python/unciv_train/hexgraph.py line:19 kind:duplication source_commit:5ac0a3cf6 -->
