"""The policy net: a small shared-trunk MLP with per-head linear outputs + a value critic.

Input (blind variant) = concat(global, acting_civ) (fixed width from schema).
Output = {tech_logits, policy_logits, value}. The value head is TRAINING-ONLY — `export_onnx`
wraps the net to emit only (tech_logits, policy_logits), so the play-time ONNX contract is
policy-only. Deliberately tiny — trains on CPU in seconds.

`RichPolicyValueNet` (Stage B) consumes the FULL observation as a multi-tensor input: a
permutation-invariant masked pool over the per-tile spatial token set and over each entity token
type, concatenated with global+acting_civ → shared trunk → the same three heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .contract import Dims


def _small_init_value_head(layer: nn.Linear) -> None:
    """Initialize the value head near zero so V≈0 at init — game features are unnormalized
    (gold, science, …), so a default-init linear head would otherwise emit huge values and the
    bounded discounted-terminal return target couldn't be fit in a few epochs. v1-reinforce ignores
    the value head, so this does not perturb the attributable baseline."""
    nn.init.uniform_(layer.weight, -1e-3, 1e-3)
    nn.init.zeros_(layer.bias)


class PolicyNet(nn.Module):
    """Blind variant: trunk over concat(global, acting_civ) → {tech, policy, value}."""

    def __init__(self, dims: Dims, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dims.input_w, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.tech_head = nn.Linear(hidden, dims.tech_w)
        self.policy_head = nn.Linear(hidden, dims.policy_w)
        self.value_head = nn.Linear(hidden, 1)  # training-only critic
        _small_init_value_head(self.value_head)  # V≈0 at init (game features are unnormalized)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        # tanh-bound the value: the true value = expected discounted terminal reward ∈ [-1,1], so
        # this is the correct range and it keeps V bounded despite unnormalized game features.
        return self.tech_head(h), self.policy_head(h), torch.tanh(self.value_head(h))


def masked_pool(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Permutation-invariant masked mean+max pool over a [B, N, F] token set.

    `mask` is [B, N] (1 = present, 0 = padding). NaN-guarded (council R3): mean divides by
    clamp(count, min=1); max over an empty set → 0 (not −inf). An all-padding row → zero vector.
    Returns [B, 2F] (mean ‖ max).
    """
    m = mask.unsqueeze(-1)                                  # [B, N, 1]
    safe_count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1] avoid /0
    mean = (tokens * m).sum(dim=1) / safe_count             # [B, F]; all-padding row → 0
    masked = tokens.masked_fill(m == 0, float("-inf"))      # true -inf so empty set is catchable
    mx = masked.max(dim=1).values                           # [B, F]; all-padding → -inf
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))  # empty set → 0 (NaN/inf guard)
    return torch.cat([mean, mx], dim=1)                     # [B, 2F]


class _TokenEncoder(nn.Module):
    """Per-token MLP → masked mean+max pool. Input [B,N,in_dim] + mask [B,N] → [B, 2*out_dim]."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),
                                 nn.Linear(out_dim, out_dim), nn.ReLU())
        self.out_w = 2 * out_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_pool(self.mlp(tokens), mask)


class RichPolicyValueNet(nn.Module):
    """Rich variant: masked-pool encoders over the spatial tile set + each entity token type,
    concatenated with global+acting_civ → trunk → {tech, policy, value}.

    `token_specs` maps an input tensor name → its per-token feature width, e.g.
    {"spatial": 13, "own_units": 8, "opp_units": 8, "own_cities": 16, "opp_cities": 16,
     "civ_tokens": 84}. `forward(inputs)` takes a dict of {name: tensor} plus {name+"_mask": mask}.
    """

    INPUT_GLOBAL = "global"
    INPUT_ACTING = "acting_civ"

    def __init__(self, dims: Dims, token_specs: dict[str, int], *, token_dim: int = 32,
                 hidden: int = 256):
        super().__init__()
        self.token_names = list(token_specs.keys())
        self.encoders = nn.ModuleDict(
            {name: _TokenEncoder(width, token_dim) for name, width in token_specs.items()}
        )
        trunk_in = dims.global_w + dims.acting_w + sum(e.out_w for e in self.encoders.values())
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.tech_head = nn.Linear(hidden, dims.tech_w)
        self.policy_head = nn.Linear(hidden, dims.policy_w)
        self.value_head = nn.Linear(hidden, 1)
        _small_init_value_head(self.value_head)

    def forward(self, inputs: dict[str, torch.Tensor]):
        parts = [inputs[self.INPUT_GLOBAL], inputs[self.INPUT_ACTING]]
        for name in self.token_names:
            parts.append(self.encoders[name](inputs[name], inputs[name + "_mask"]))
        h = self.trunk(torch.cat(parts, dim=1))
        return self.tech_head(h), self.policy_head(h), torch.tanh(self.value_head(h))


# --------------------------------------------------------------------------------------------------
# v4 STRUCTURED encoder (D2 embeddings + D3 hex-GNN). Phase A is GNN-only (attn_layers default 0);
# entity/cross attention is Phase B. The FROZEN seam is preserved exactly:
#   forward(inputs:dict) -> (tech_logits, policy_logits, torch.tanh(value)); INPUT_GLOBAL/INPUT_ACTING.
# All ops are opset-17 core: Gather/index, Mul, ReduceSum/ReduceMean, MatMul/Linear, LayerNorm,
# Softmax, tanh — NO scatter_add/index_add, NO nn.MultiheadAttention/F.scaled_dot_product_attention.
# --------------------------------------------------------------------------------------------------

# Spatial channel field plan for the FIXED 13-channel v3 spatial block (SSOT order is Kotlin
# Featurizer.buildSpatial). Each entry is either ("num",) for a raw scalar channel or ("emb", table)
# for a categorical channel looked up in the named shared embedding table. Channel indices:
#   0 visibility(num) 1 terrain_base(emb terrain) 2 terrain_feature(emb terrain SHARED)
#   3 resource(emb resource) 4 road(emb road) 5 river(num) 6 is_city_center(num)
#   7 owner_slot(emb slot) 8 improvement(emb improvement) 9 unit_present(num)
#   10 unit_owner_slot(emb slot SHARED) 11 unit_type_cat(emb unit_type) 12 unit_health_bucket(num)
_SPATIAL_FIELD_PLAN = (
    ("num",),               # 0 visibility (raw 0/1/2)
    ("emb", "terrain"),     # 1 terrain_base
    ("emb", "terrain"),     # 2 terrain_feature (SHARED terrain table)
    ("emb", "resource"),    # 3 resource
    ("emb", "road"),        # 4 road status (ordinal 0..2)
    ("num",),               # 5 river
    ("num",),               # 6 is_city_center
    ("emb", "slot"),        # 7 owner_slot (255 -> reindex)
    ("emb", "improvement"), # 8 improvement
    ("num",),               # 9 unit_present
    ("emb", "slot"),        # 10 unit_owner_slot (SHARED slot table; 255 -> reindex)
    ("emb", "unit_type"),   # 11 unit_type_cat (0..4)
    ("num",),               # 12 unit_health_bucket
)
SLOT_SENTINEL_RAW = 255            # Featurizer emits 255 for "self" in owner-slot fields
UNIT_TYPE_TABLE = 5                # unit_type_cat is 0..4; 0 == real "none" (no +1 sentinel)
ROAD_TABLE = 3                     # roadStatus ordinal 0..2 (None/Road/Railroad); no sentinel

RUNGS = {
    # small stays Phase-A GNN-only (attn_layers=0) — the GNN-only baseline must remain intact.
    "small":  dict(embed_dim=8,  gnn_layers=1, gnn_channels=32, attn_layers=0, attn_heads=2,
                   attn_dim=32,  trunk_w=128),
    # medium/large turn on Phase-B attention (D4 self-attn + entity↔node join + D5 cross-attn).
    "medium": dict(embed_dim=16, gnn_layers=2, gnn_channels=64, attn_layers=2, attn_heads=4,
                   attn_dim=64,  trunk_w=256),
    "large":  dict(embed_dim=24, gnn_layers=3, gnn_channels=96, attn_layers=3, attn_heads=8,
                   attn_dim=96,  trunk_w=384),
}

# D4 entity↔node join (FND-0036): each entity token carries a tile index naming its co-located GNN
# node. own_units/opp_units → field index 8 (currentTile.zeroBasedIndex), own_cities/opp_cities →
# field index 12 (centerTile.zeroBasedIndex). civ_tokens have NO tile index → no join. The tile index
# is a JOIN KEY (used to gather the post-GNN node embedding), NOT a learned scalar feature.
_ENTITY_TILE_FIELD = {
    "own_units": 8,
    "opp_units": 8,
    "own_cities": 12,
    "opp_cities": 12,
    # civ_tokens: absent → no entity↔node join (no tile index).
}


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean over [B,N,F] with mask [B,N] (1 present / 0 pad). NaN-guarded: divide by
    clamp(count,1) so an all-padding row → zero vector (mirrors masked_pool, model.py:58)."""
    m = mask.unsqueeze(-1)                                       # [B,N,1]
    safe = mask.sum(dim=1, keepdim=True).clamp(min=1.0)         # [B,1]
    return (tokens * m).sum(dim=1) / safe                       # [B,F]


class _SpatialEmbed(nn.Module):
    """Embed the 13-channel spatial token: categorical channels → shared embedding tables, numeric
    channels → raw scalars; concat → per-node feature [B,N,proj_w]. Counts are read from the schema
    `vocabCounts` (num_embeddings = count + 1 sentinel row), NEVER hardcoded (D2)."""

    def __init__(self, vocab_counts: dict, embed_dim: int, max_civ_tokens: int):
        super().__init__()
        E = embed_dim
        self.slot_table = max_civ_tokens + 2          # 0 none/unknown, 1..mct civ, mct+1 self
        self.slot_self = max_civ_tokens + 1           # reindex 255 -> this row
        self.tables = nn.ModuleDict({
            # terrain_base + terrain_feature SHARE this table (both index Vocab.TERRAINS).
            "terrain": nn.Embedding(vocab_counts["terrain"] + 1, E),
            "resource": nn.Embedding(vocab_counts["resource"] + 1, E),
            "improvement": nn.Embedding(vocab_counts["improvement"] + 1, E),
            # owner_slot + unit_owner_slot SHARE this slot table.
            "slot": nn.Embedding(self.slot_table, E),
            "unit_type": nn.Embedding(UNIT_TYPE_TABLE, E),    # 0..4, 0 real "none" (no +1)
            "road": nn.Embedding(ROAD_TABLE, E),              # 0..2 ordinal (no +1)
        })
        self.plan = _SPATIAL_FIELD_PLAN
        n_num = sum(1 for s in self.plan if s[0] == "num")
        n_emb = sum(1 for s in self.plan if s[0] == "emb")
        self.proj_w = n_num + n_emb * E

    def forward(self, spatial: torch.Tensor) -> torch.Tensor:  # spatial: [B,N,13] float
        cols = []
        for j, s in enumerate(self.plan):
            v = spatial[..., j]                                # [B,N]
            if s[0] == "num":
                cols.append(v.unsqueeze(-1))                   # [B,N,1]
            else:
                table = s[1]
                if table == "slot":
                    v = torch.where(v == float(SLOT_SENTINEL_RAW),
                                    torch.full_like(v, float(self.slot_self)), v)
                idx = v.clamp(0, self.tables[table].num_embeddings - 1).long()
                cols.append(self.tables[table](idx))          # [B,N,E]
        return torch.cat(cols, dim=-1)                         # [B,N,proj_w]


class _GatherGNNLayer(nn.Module):
    """One degree-6 gather-GNN layer (opset-17 safe: Gather + Mul + ReduceSum + Linear + LayerNorm).
    Node features carry a ZERO pad row at index N so a sentinel-index-N neighbor gathers zeros."""

    def __init__(self, channels: int):
        super().__init__()
        self.msg = nn.Linear(channels, channels)
        self.upd = nn.Linear(2 * channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, h: torch.Tensor, nbr_idx: torch.Tensor, nbr_mask: torch.Tensor) -> torch.Tensor:
        # h: [B,N,C]; nbr_idx: [B,N,6] int64 in [0,N]; nbr_mask: [B,N,6] f32.
        B, N, C = h.shape
        h_pad = torch.cat([h, torch.zeros(B, 1, C, dtype=h.dtype, device=h.device)], dim=1)  # [B,N+1,C]
        deg = nbr_idx.shape[2]
        flat = nbr_idx.reshape(B, N * deg)                                       # [B,N*6]
        gathered = torch.gather(h_pad, 1, flat.unsqueeze(-1).expand(-1, -1, C))  # [B,N*6,C]
        nbr = gathered.reshape(B, N, deg, C)                                     # [B,N,6,C]
        m = self.msg(nbr) * nbr_mask.unsqueeze(-1)                               # mask padding edges
        denom = nbr_mask.sum(dim=2, keepdim=True).clamp(min=1.0)                 # [B,N,1] NaN-guard
        agg = m.sum(dim=2) / denom                                              # masked mean -> [B,N,C]
        out = torch.relu(self.upd(torch.cat([h, agg], dim=-1)))                  # [B,N,C]
        return self.norm(h + out)                                               # residual + LN


class _EntityEncoder(nn.Module):
    """Phase-A entity token encoder: per-token Linear → masked-mean pool. Clean seam for the
    Phase-B upgrade to embeddings + self-attention (just swap the body, keep the masked pool)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
        self.out_w = out_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return _masked_mean(self.proj(tokens), mask)


# --------------------------------------------------------------------------------------------------
# Phase B (D4/D5) hand-rolled attention. ONLY Linear Q/K/V + matmul + scale(1/√dk) + masked softmax
# (masked_fill(-inf) → softmax) + matmul + out-proj, pre-LN, multi-head via reshape. NO
# nn.MultiheadAttention / F.scaled_dot_product_attention / scatter — every op is opset-17 core.
# --------------------------------------------------------------------------------------------------


def _masked_softmax_attend(scores: torch.Tensor, kv_mask: torch.Tensor,
                           v: torch.Tensor) -> torch.Tensor:
    """NaN-safe masked attention readout (FND-0025).

    `scores`: [B,H,Lq,Lk] raw QKᵀ/√dk. `kv_mask`: [B,Lk] (1 keep / 0 pad). `v`: [B,H,Lk,dk].
    Masks padded keys with −inf then softmax. A row whose keys are ALL masked yields NaN after
    softmax (−inf − (−inf)); we detect it and force the context to ZEROS (never NaN), mirroring the
    max-over-empty → 0 guard in masked_pool (model.py:62). Returns context [B,H,Lq,dk]."""
    m = kv_mask[:, None, None, :]                                   # [B,1,1,Lk] broadcast over H,Lq
    scores = scores.masked_fill(m == 0, float("-inf"))             # padded keys → −inf
    w = torch.softmax(scores, dim=-1)                              # all-masked row → NaN here
    # all-masked detector: a query row has NO live key iff every kv_mask entry is 0.
    any_key = (kv_mask.sum(dim=-1) > 0)                            # [B] True if ≥1 live key
    w = torch.where(any_key[:, None, None, None], w, torch.zeros_like(w))  # dead row → 0 weights
    ctx = torch.matmul(w, v)                                       # [B,H,Lq,dk]; dead row → 0 ctx
    return ctx


class _MHA(nn.Module):
    """Hand-rolled multi-head attention: Linear Q/K/V → reshape to heads → scaled QKᵀ → masked
    softmax (NaN-safe) → ·V → merge heads → out-proj. q and kv may differ (cross-attention)."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"_MHA: attn_dim {dim} not divisible by heads {heads}")
        self.h = heads
        self.dk = dim // heads
        self.scale = self.dk ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

    def _split(self, x: torch.Tensor) -> torch.Tensor:            # [B,L,dim] → [B,H,L,dk]
        B, L, _ = x.shape
        return x.reshape(B, L, self.h, self.dk).transpose(1, 2)

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor,
                kv_mask: torch.Tensor) -> torch.Tensor:
        q = self._split(self.q(q_in))                             # [B,H,Lq,dk]
        k = self._split(self.k(kv_in))                            # [B,H,Lk,dk]
        v = self._split(self.v(kv_in))                            # [B,H,Lk,dk]
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B,H,Lq,Lk]
        ctx = _masked_softmax_attend(scores, kv_mask, v)         # [B,H,Lq,dk]; NaN-safe
        B, Lq = q_in.shape[0], q_in.shape[1]
        ctx = ctx.transpose(1, 2).reshape(B, Lq, self.h * self.dk)  # merge heads [B,Lq,dim]
        return self.o(ctx)


class _AttnBlock(nn.Module):
    """Pre-LN attention residual block: x = x + MHA(LN(x_q), LN(x_kv), mask), then a small FFN
    residual. For self-attention q_in is kv_in; for cross-attention they differ (separate LN)."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.mha = _MHA(dim, heads)
        self.ln_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor,
                kv_mask: torch.Tensor) -> torch.Tensor:
        # When q and kv are the same tensor (self-attn) both LNs see it; for cross-attn they differ.
        q_n = self.ln_q(q_in)
        kv_n = self.ln_kv(kv_in)
        x = q_in + self.mha(q_n, kv_n, kv_mask)
        x = x + self.ff(self.ln_ff(x))
        return x


class StructuredPolicyValueNet(nn.Module):
    """v4 structured encoder: categorical embeddings + a hex-aware gather-GNN over the spatial tile
    graph. FROZEN seam: forward(inputs:dict) → (tech, policy, tanh(value)).

    Two paths, switched on `attn_layers`:
    - **attn_layers == 0 (Phase A, GNN-only):** entity sets are projected + masked-mean-pooled, the
      GNN nodes are masked-mean-pooled to a board vector, all concatenated with global+acting_civ →
      split trunk. This is the untouched Phase-A baseline (the `small` rung).
    - **attn_layers > 0 (Phase B):** D4 per-entity-set self-attention (with the FND-0036 entity↔node
      join: each entity token is fused with its co-located post-GNN node embedding via its tile
      index before self-attention) + D5 single-query cross-attention over [GNN nodes ⊕ refined
      entity tokens] → a fixed context vector that replaces the masked-mean pooling as the trunk
      input. The `medium`/`large` rungs use this path.

    The model never sees raw coords — 2D locality comes entirely from the GNN over the
    externally-built `neighbor_index`/`neighbor_mask` graph tensors. The tile index in each entity
    token is a JOIN KEY into the node axis, not a learned feature.
    """

    INPUT_GLOBAL = "global"
    INPUT_ACTING = "acting_civ"
    INPUT_NEIGHBOR_INDEX = "neighbor_index"
    INPUT_NEIGHBOR_MASK = "neighbor_mask"
    SPATIAL = "spatial"
    ENTITY_NAMES = ("own_units", "opp_units", "own_cities", "opp_cities", "civ_tokens")

    def __init__(self, dims: Dims, token_specs: dict[str, int], vocab_counts: dict, *,
                 embed_dim: int = 8, gnn_layers: int = 2, gnn_channels: int = 32,
                 attn_layers: int = 0, attn_heads: int = 2, attn_dim: int = 32,
                 trunk_w: int = 128, max_civ_tokens: int = 40):
        super().__init__()
        if int(token_specs.get("spatial", 0)) != len(_SPATIAL_FIELD_PLAN):
            raise ValueError(
                f"StructuredPolicyValueNet: spatial width {token_specs.get('spatial')} != field-plan "
                f"length {len(_SPATIAL_FIELD_PLAN)} (Phase-A targets the 13-channel v3 spatial block)"
            )
        self.attn_layers = attn_layers
        self.attn_dim = attn_dim
        self.gnn_channels = gnn_channels
        self.use_attn = attn_layers > 0

        # D2 embeddings + per-node input projection -> gnn_channels.
        self.spatial_embed = _SpatialEmbed(vocab_counts, embed_dim, max_civ_tokens)
        self.spatial_in = nn.Linear(self.spatial_embed.proj_w, gnn_channels)

        # D3 GNN stack.
        self.gnn = nn.ModuleList(_GatherGNNLayer(gnn_channels) for _ in range(gnn_layers))

        if not self.use_attn:
            # ---- Phase A: project + masked-mean pool entity sets; mean-pool the board. ----
            self.entity_enc = nn.ModuleDict({
                name: _EntityEncoder(int(token_specs[name]), gnn_channels)
                for name in self.ENTITY_NAMES
            })
            agg_w = gnn_channels + sum(e.out_w for e in self.entity_enc.values())
        else:
            # ---- Phase B: self-attention + entity↔node join + single-query cross-attention. ----
            # Per-entity-set token projection raw_width -> attn_dim.
            self.entity_in = nn.ModuleDict({
                name: nn.Linear(int(token_specs[name]), attn_dim) for name in self.ENTITY_NAMES
            })
            # Static 0/1 feature masks that zero each entity's tile-index column (join key, not a
            # feature). Registered as buffers (move with .to(device), constant-folded in ONNX).
            self._entity_feat_mask = {}
            for name, fld in _ENTITY_TILE_FIELD.items():
                mask = torch.ones(1, 1, int(token_specs[name]))
                mask[..., fld] = 0.0
                buf = f"_featmask_{name}"
                self.register_buffer(buf, mask)
                self._entity_feat_mask[name] = getattr(self, buf)
            # FND-0036 join: project the gathered post-GNN node embedding (C) -> attn_dim, then ADD
            # it into the entity token. Reused as the cross-attn node-key projection (same node space).
            self.node_to_attn = nn.Linear(gnn_channels, attn_dim)
            # D4 per-entity-set self-attention stacks (only entity sets WITH a join get the node fuse;
            # civ_tokens still get self-attention, just no join).
            self.self_attn = nn.ModuleDict({
                name: nn.ModuleList(_AttnBlock(attn_dim, attn_heads) for _ in range(attn_layers))
                for name in self.ENTITY_NAMES
            })
            # D5 single-query cross-attention: query from [global ⊕ acting] -> attn_dim.
            self.q_proj = nn.Linear(dims.global_w + dims.acting_w, attn_dim)
            self.cross = nn.ModuleList(_AttnBlock(attn_dim, attn_heads) for _ in range(attn_layers))
            agg_w = attn_dim                      # the cross-attn context vector replaces pooling

        # Aggregate -> split trunk (D6). (Phase A) board ⊕ entities OR (Phase B) cross-ctx, then
        # ⊕ global ⊕ acting_civ. Unchanged heads.
        trunk_in = agg_w + dims.global_w + dims.acting_w
        self.shared = nn.Sequential(nn.Linear(trunk_in, trunk_w), nn.ReLU(),
                                    nn.Linear(trunk_w, trunk_w), nn.ReLU())
        self.policy_body = nn.Sequential(nn.Linear(trunk_w, trunk_w), nn.ReLU())
        self.value_body = nn.Sequential(nn.Linear(trunk_w, trunk_w), nn.ReLU())
        self.tech_head = nn.Linear(trunk_w, dims.tech_w)
        self.policy_head = nn.Linear(trunk_w, dims.policy_w)
        self.value_head = nn.Linear(trunk_w, 1)
        _small_init_value_head(self.value_head)

    def _gather_node_by_tile(self, h_pad: torch.Tensor, tile_idx: torch.Tensor,
                             N: int) -> torch.Tensor:
        """Gather each entity's co-located GNN node embedding (FND-0036 entity↔node join).

        `h_pad`: [B,N+1,C] post-GNN node features with a ZERO pad row at index N (so an OOB/clamped
        index gathers zeros). `tile_idx`: [B,M] float tile indices (currentTile/centerTile
        zeroBasedIndex). Returns [B,M,C]. Indices are clamped to [0, N] — anything outside the real
        node range (including the −1/absent sentinel after clamp) lands on the zero pad row."""
        idx = tile_idx.clamp(0, N).long()                            # [B,M] in [0,N] (pad row safe)
        B, M = idx.shape
        C = h_pad.shape[2]
        gathered = torch.gather(h_pad, 1, idx.unsqueeze(-1).expand(B, M, C))  # [B,M,C]
        return gathered

    def forward(self, inputs: dict[str, torch.Tensor]):
        g = inputs[self.INPUT_GLOBAL]
        a = inputs[self.INPUT_ACTING]
        spatial = inputs[self.SPATIAL]                       # [B,N,13] float
        spatial_mask = inputs[self.SPATIAL + "_mask"]        # [B,N]
        nbr_idx = inputs[self.INPUT_NEIGHBOR_INDEX]          # [B,N,6] int64
        nbr_mask = inputs[self.INPUT_NEIGHBOR_MASK]          # [B,N,6] f32

        # D2 embeddings -> per-node projection -> D3 GNN.
        h = self.spatial_in(self.spatial_embed(spatial))     # [B,N,C]
        for layer in self.gnn:
            h = layer(h, nbr_idx, nbr_mask)                  # [B,N,C]

        if not self.use_attn:
            # ---- Phase A path (GNN-only) — unchanged. ----
            board = _masked_mean(h, spatial_mask)            # [B,C] board vector
            parts = [board]
            for name in self.ENTITY_NAMES:
                parts.append(self.entity_enc[name](inputs[name], inputs[name + "_mask"]))
            parts += [g, a]
            body = self.shared(torch.cat(parts, dim=1))
            ph = self.policy_body(body)
            vh = self.value_body(body)
            return self.tech_head(ph), self.policy_head(ph), torch.tanh(self.value_head(vh))

        # ---- Phase B path (attention). ----
        B, N, C = h.shape
        # Zero pad row at index N so a clamped/absent tile index gathers zeros (mirrors the GNN pad).
        h_pad = torch.cat([h, torch.zeros(B, 1, C, dtype=h.dtype, device=h.device)], dim=1)  # [B,N+1,C]
        gnn_kv = self.node_to_attn(h)                        # [B,N,attn_dim] cross-attn node keys

        # D4 per-entity-set self-attention (with the FND-0036 entity↔node join where a tile index
        # exists). Refined tokens + masks are kept for the cross-attention union.
        ent_tok, ent_mask = [], []
        for name in self.ENTITY_NAMES:
            tok = inputs[name]                               # [B,M,raw_width]
            m = inputs[name + "_mask"]                       # [B,M]
            tile_field = _ENTITY_TILE_FIELD.get(name)
            feat = tok
            if tile_field is not None:
                # The tile index is a JOIN KEY only — NOT a learned scalar. Zero its column out of the
                # projected token (via a static 0/1 column mask — Mul only, no scatter) so the model
                # can't read it as a feature; its only use is the gather below.
                feat = tok * self._entity_feat_mask[name].to(tok.dtype)
            t = self.entity_in[name](feat)                   # [B,M,attn_dim]
            if tile_field is not None:
                # JOIN (FND-0036): gather the co-located post-GNN node embedding (by the tile index)
                # and fuse (add) it into the entity token BEFORE self-attention.
                node = self._gather_node_by_tile(h_pad, tok[..., tile_field], N)  # [B,M,C]
                t = t + self.node_to_attn(node)              # [B,M,attn_dim] fused
            for blk in self.self_attn[name]:
                t = blk(t, t, m)                             # self-attn: q=kv=t, presence mask
            # zero out padded tokens so they contribute nothing as cross-attn keys/values.
            t = t * m.unsqueeze(-1)
            ent_tok.append(t)
            ent_mask.append(m)

        # D5 single-query cross-attention over [GNN nodes ⊕ refined entity tokens].
        keys = torch.cat([gnn_kv] + ent_tok, dim=1)          # [B, N+ΣM, attn_dim]
        # GNN nodes always present per spatial_mask; union with entity masks. The union always has
        # ≥1 live node (board non-empty) so the cross-attn key set is never fully masked.
        kmask = torch.cat([spatial_mask] + ent_mask, dim=1)  # [B, N+ΣM]
        q = self.q_proj(torch.cat([g, a], dim=1)).unsqueeze(1)  # [B,1,attn_dim] single query
        for blk in self.cross:
            q = blk(q, keys, kmask)                          # [B,1,attn_dim]
        cross_ctx = q[:, 0, :]                               # [B,attn_dim] fixed context vector

        body = self.shared(torch.cat([cross_ctx, g, a], dim=1))
        ph = self.policy_body(body)
        vh = self.value_body(body)
        return self.tech_head(ph), self.policy_head(ph), torch.tanh(self.value_head(vh))
