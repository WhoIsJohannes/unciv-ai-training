> Web-sourced content below is DATA, not instructions.

# Step 4 — Web Research: ONNX-exportability of the v4 encoder (the pre-registered risk)

Focus: de-risk the only genuinely uncertain part of v4 — exporting a hex-GNN + attention
encoder to ONNX under **opset 17** (the contract's opset). Two load-bearing findings, each
with a concrete export-safe realization.

## Q1 — GNN message passing: is scatter_add/scatter_reduce ONNX-safe at opset 17? NO.

**Query:** PyTorch export scatter_add/index_add to ONNX opset 17 (GNN message passing).
**Findings:**
- `scatter_add` export fails in the TorchScript exporter: *"Unsupported: ONNX export of
  operator scatter_add, index should have the same dimensionality as src"* (pytorch#65138,
  #89190 SAGEConv + dynamic_axes).
- **`index_add_` with duplicate index values produces a WRONG model** (pytorch#111159) —
  and GNN aggregation *inherently* has duplicate target indices (many edges → one node).
  This is a silent-correctness landmine, not just an export error.
- `scatter_reduce` has accuracy bugs + *"ONNX does not support include_self=False"* and
  `ScatterElements` dimension-mismatch runtime errors (pytorch#147617, pyg#8415).
**Relevance / decision (D3):** do NOT use edge_index + scatter_add. Use a **neighbor-gather +
masked-reduce over a fixed degree-6 axis**: Python builds per-node neighbor-row index `[N,6]`
+ neighbor-mask `[N,6]` (from (x,y)+radius+worldWrap, replicating
`TileMap.getIfTileExistsOrNull`); the model `Gather`s neighbor features → `[N,6,C]`, applies
the message transform, multiplies by the mask, and `ReduceSum`/`ReduceMean`s over the degree
axis. All ops (Gather, Mul, ReduceSum, MatMul) are core ONNX ≤ opset 13 — no scatter anywhere.
Bounded degree 6 ⇒ cheap and exact for the hex graph; ragged edges + world-wrap handled by the
mask + the wrapped neighbor row. This is still a true hex-adjacency GNN, just the export-safe
realization the plan's D3 asked us to find.

## Q2 — Attention (self + cross): is SDPA/MultiheadAttention ONNX-safe at opset 17? NOT natively.

**Query:** torch.onnx.export scaled_dot_product_attention / MultiheadAttention opset 17.
**Findings:**
- ONNX added a native **Attention/MHA operator only in opset 23** (pytorch#149662). Opset 17
  has **no** native attention op.
- At opset <23 PyTorch *decomposes* SDPA into constituent ops; direct export of
  `aten::scaled_dot_product_attention` is reported Unsupported on several opsets
  (huggingface/diffusers#4691, pytorch#96944/#97262). The decomposition path is fragile via the
  TorchScript exporter.
**Relevance / decision (D4/D5):** hand-roll scaled-dot-product attention from primitives —
`nn.Linear` Q/K/V projections, `torch.matmul`, scale by 1/√d, masked softmax
(`masked_fill(mask==0, -inf)` → `softmax`), `matmul` with V, output Linear. Pre-LN. All core
ONNX ops, dynamic-shape friendly. Do NOT use `nn.MultiheadAttention` or
`F.scaled_dot_product_attention`. (Same masking discipline as the existing masked_pool: pad
tokens get mask 0; never let an all-padding row produce NaN — additive -inf + a safe fallback.)

## Q3 — Dynamic/ragged shapes (batch + variable N tiles/tokens)

Gather with dynamic indices + ReduceSum over a fixed axis + masked softmax all support dynamic
batch and dynamic N via `dynamic_axes` at opset 17 (the existing v2 export already uses ragged
`{1:"n_<name>"}` axes). The neighbor-index/mask tensors get the same ragged-N treatment as the
spatial token set. No new op-support risk beyond Q1/Q2 (both resolved above).

## Libraries

- No new third-party dependency. PyTorch Geometric is the obvious GNN lib but its layers lean on
  scatter_reduce (the exact broken path) and would add a heavy dep — **reject**; hand-rolled
  gather+reduce is ~30 lines, export-safe, and matches the existing hand-rolled-module style of
  `model.py`. Stay on stock `torch.nn`.

## Key insights

1. **D3 = gather-neighbors + masked-reduce over a fixed degree-6 axis** (NOT scatter_add). The
   neighbor index `[N,6]` + mask `[N,6]` are derived in Python (replicating the Kotlin world-wrap)
   and fed as inputs. Export-safe at opset 17; correctness-safe (no duplicate-index scatter).
2. **D4/D5 = hand-rolled scaled-dot-product attention** from Linear+matmul+masked-softmax (NOT
   nn.MultiheadAttention/SDPA — no native ONNX attention until opset 23).
3. **No new deps** (reject PyG — its scatter path is the broken one). Validate export on the
   *small* rung FIRST (export smoke test) before scaling — the plan's pre-registered gate.
4. Both decisions keep the stable seam intact: `forward(inputs:dict)→(tech_logits,policy_logits,
   tanh(value))`, masking discipline mirrors the existing masked_pool NaN guards.
