"""Assemble the rich/structured multi-tensor input from per-step observation blocks.

The spatial block (nTiles*n_channels, schema-driven) becomes a per-tile token SET; each entity block
(own_units/opp_units/own_cities/opp_cities/civ_tokens) is a present-only token set. All are padded
to the batch-max count with a presence mask (1 = real token, 0 = padding) so the model's masked
pool/GNN ignores padding. This is the SAME layout the JVM builds at decision time — the parity test
guards that the two agree.

For the v4 structured encoder this also derives the degree-6 hex-adjacency tensors the gather-GNN
needs — ``neighbor_index`` [B,N,6] int64 + ``neighbor_mask`` [B,N,6] f32 — from the shard-only
``spatial_coords`` block (per-tile (x,y)) plus the global head's map-dim scalars (effWrapRadius,
worldWrap, shape) via ``hexgraph.build_neighbor_graph``. These are NEW ONNX inputs (not token sets,
not folded into spatial). They are emitted only when coords are available (contract v3); on v2 shards
the neighbor tensors are simply absent and the rich (pool) net does not consume them.
"""
from __future__ import annotations

import numpy as np
import torch

from .contract import Dims, GLOBAL_MAPDIM_OFFSET, INPUT_NEIGHBOR_INDEX, INPUT_NEIGHBOR_MASK
from .hexgraph import HEX_DEGREE, build_neighbor_graph


def _pad_token_set(arrays: list[np.ndarray], width: int) -> tuple[np.ndarray, np.ndarray]:
    """Pad a list of [n_i, width] arrays to [B, maxN, width] + mask [B, maxN]."""
    b = len(arrays)
    counts = [int(a.shape[0]) for a in arrays]
    maxn = max(1, max(counts) if counts else 1)
    padded = np.zeros((b, maxn, width), dtype=np.float32)
    mask = np.zeros((b, maxn), dtype=np.float32)
    for i, a in enumerate(arrays):
        c = int(a.shape[0])
        if c:
            w = min(width, int(a.shape[1]))
            padded[i, :c, :w] = a[:, :w]
            mask[i, :c] = 1.0
    return padded, mask


def _read_mapdims(global_vec: np.ndarray) -> tuple[int, bool, int]:
    """Read (eff_wrap_radius, world_wrap, shape) from the global head's map-dim slots.

    Global head layout (positional, frozen with Kotlin buildGlobal):
        [turns, era, tileCount, knownMajors, aliveMajors, effWrapRadius, worldWrap, shape, ...demog]
    so the 3 map-dim scalars sit at slots GLOBAL_MAPDIM_OFFSET, +1, +2 (BEFORE the demographics agg).
    """
    g = np.asarray(global_vec, dtype=np.float32).reshape(-1)
    o = GLOBAL_MAPDIM_OFFSET
    eff_wrap_radius = int(round(float(g[o])))
    ww_raw = float(g[o + 1])
    ww_round = round(ww_raw)
    assert ww_round in (0, 1), (
        f"worldWrap global slot {o + 1} = {ww_raw!r} is not in {{0,1}} "
        "(global head map-dim offset drift? check GLOBAL_MAPDIM_OFFSET vs Kotlin buildGlobal)"
    )
    world_wrap = bool(ww_round)
    shape = int(round(float(g[o + 2])))
    return eff_wrap_radius, world_wrap, shape


def _neighbor_graph_for(coords: np.ndarray, global_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Degree-6 (idx [N,6] int64, mask [N,6] f32) for one step from its coords + global map-dims."""
    eff_wrap_radius, world_wrap, shape = _read_mapdims(global_vec)
    return build_neighbor_graph(coords, eff_wrap_radius=eff_wrap_radius,
                                world_wrap=world_wrap, shape=shape)


def build_rich_batch(trajectories, dims: Dims, token_specs: dict[str, int]) -> dict:
    """Flatten all trajectories' per-step rich blocks into one padded multi-tensor batch.
    Returns {name: tensor, name+"_mask": tensor, "global": tensor, "acting_civ": tensor} and, when
    per-step `spatial_coords` are present, the structured graph tensors neighbor_index/neighbor_mask
    (padded to the SAME max-N as the spatial token set — degree axis static 6).
    """
    steps = [blk for t in trajectories for blk in (t.rich or [])]
    n = len(steps)
    assert n > 0, "build_rich_batch called with no steps"
    out: dict[str, torch.Tensor] = {
        "global": torch.tensor(np.stack([s["global"] for s in steps]).astype(np.float32)),
        "acting_civ": torch.tensor(np.stack([s["acting_civ"] for s in steps]).astype(np.float32)),
    }
    # Build the padded spatial token set first so the neighbor tensors share its max-N exactly.
    spatial_arrays = [np.asarray(s["spatial"], np.float32) for s in steps]
    maxn = max(1, max(int(a.shape[0]) for a in spatial_arrays))

    have_coords = all(s.get("spatial_coords") is not None for s in steps)
    if have_coords:
        nbr_idx = np.zeros((n, maxn, HEX_DEGREE), dtype=np.int64)
        nbr_mask = np.zeros((n, maxn, HEX_DEGREE), dtype=np.float32)
        for i, s in enumerate(steps):
            coords = np.asarray(s["spatial_coords"], np.float32)
            ni = int(coords.shape[0])
            if ni:
                gi, gm = _neighbor_graph_for(coords, s["global"])
                # Reindex this step's sentinel (== ni, the per-step pad row) to the batch pad row
                # (== maxn): the model appends ONE zero pad row at the padded node count maxn, so a
                # missing neighbor must point at maxn, not at the per-step ni (a real padded row).
                gi = np.where(gi == ni, maxn, gi)
                nbr_idx[i, :ni] = gi
                nbr_mask[i, :ni] = gm
        out[INPUT_NEIGHBOR_INDEX] = torch.tensor(nbr_idx)
        out[INPUT_NEIGHBOR_MASK] = torch.tensor(nbr_mask)

    for name, width in token_specs.items():
        padded, mask = _pad_token_set([s[name] for s in steps], width)
        out[name] = torch.tensor(padded)
        out[name + "_mask"] = torch.tensor(mask)
    # Padding-consistency invariant: neighbor tensors share spatial's padded node axis exactly.
    assert out["spatial"].shape[1] == maxn
    if have_coords:
        assert out[INPUT_NEIGHBOR_INDEX].shape[1] == out["spatial"].shape[1]
    return out


def build_rich_single(step_blocks: dict, token_specs: dict[str, int]) -> dict:
    """Batch-of-1 multi-tensor input from ONE step's raw blocks (the parity / inference reference).

    `step_blocks` holds: global, acting_civ (1D), spatial (nTiles*n_channels flat OR [nTiles,n_ch]),
    each entity block as [count, perItem] (possibly empty), and optionally `spatial_coords`
    (nTiles*2 flat OR [nTiles,2]) — when present the degree-6 neighbor tensors are derived and added.
    Mirrors exactly what the JVM builds.
    """
    g = np.asarray(step_blocks["global"], dtype=np.float32).reshape(1, -1)
    a = np.asarray(step_blocks["acting_civ"], dtype=np.float32).reshape(1, -1)
    out: dict[str, torch.Tensor] = {"global": torch.tensor(g), "acting_civ": torch.tensor(a)}
    n_spatial = 0
    for name, width in token_specs.items():
        arr = np.asarray(step_blocks[name], dtype=np.float32)
        if name == "spatial":
            arr = arr.reshape(-1, width)
            n_spatial = int(arr.shape[0])
        elif arr.ndim == 1:
            arr = arr.reshape(0, width) if arr.size == 0 else arr.reshape(-1, width)
        padded, mask = _pad_token_set([arr], width)
        out[name] = torch.tensor(padded)
        out[name + "_mask"] = torch.tensor(mask)

    coords = step_blocks.get("spatial_coords")
    if coords is not None:
        coords = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
        gi, gm = _neighbor_graph_for(coords, g[0])
        # batch-of-1: padded node count == n_spatial, so the per-step sentinel (== n_spatial) already
        # equals the model's pad row index — no reindex needed.
        out[INPUT_NEIGHBOR_INDEX] = torch.tensor(gi[None, ...])     # [1,N,6] int64
        out[INPUT_NEIGHBOR_MASK] = torch.tensor(gm[None, ...])      # [1,N,6] f32
        assert out[INPUT_NEIGHBOR_INDEX].shape[1] == out["spatial"].shape[1] == n_spatial
    return out
