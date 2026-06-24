"""v4 hex adjacency builder (D3) — pure, mirrors the JVM engine exactly.

Turns per-tile (x,y) coords into a fixed degree-6 ``neighbor_index [N,6]`` + ``neighbor_mask [N,6]``
for the ONNX-safe gather-GNN (gather neighbors → masked-reduce over the degree axis; NO scatter).
A missing neighbor (off-map, no world-wrap) → **sentinel index N** (the model appends a zero pad row
at index N) + mask 0, so even a forgotten mask contributes zero (council FND-0035). All lookups are
exact integer dict hits; world-wrap replicates ``TileMap.getIfTileExistsOrNull`` branch C then D.

Used at TRAIN time (Python, from the shard's ``spatial_coords`` block). At INFERENCE the JVM builds
the same tensors from the live ``TileMap``; an engine-fidelity test + a Python↔Kotlin adjacency-parity
test guard that the two agree (FND-0036). This function and the Kotlin pure builder are the SSOT.
"""
from __future__ import annotations

import numpy as np

# FROZEN clock-direction order (dirs 12,2,4,6,8,10) — identical in Kotlin emit/inference/parity.
# HexMath.clockPositionToHexcoordMap: 12=(1,1) 2=(0,1) 4=(-1,0) 6=(-1,-1) 8=(0,-1) 10=(1,0).
OFFSETS: tuple[tuple[int, int], ...] = ((1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1), (1, 0))
HEX_DEGREE = len(OFFSETS)  # 6


def build_neighbor_graph(coords, eff_wrap_radius, world_wrap, shape=1):
    """Build the degree-6 hex adjacency for one observation.

    Args:
        coords: [N,2] per-tile (x,y). Must be integer-exact (they come from integer hex positions
            via the f32 ``spatial_coords`` block; a fractional value signals storage/decode drift).
        eff_wrap_radius: PRE-RESOLVED wrap radius emitted by Kotlin (rectangular = width/2, else
            mapSize.radius) — so Python never needs map width.
        world_wrap: bool — whether the map wraps.
        shape: map-shape ordinal (kept for signature symmetry; the wrap rule is fully captured by
            ``eff_wrap_radius``, identical for hex/rectangular given R).

    Returns:
        (neighbor_index [N,6] int64, neighbor_mask [N,6] float32). Missing slot → index N + mask 0.
    """
    coords = np.asarray(coords)
    n = int(coords.shape[0])
    if n == 0:
        return np.zeros((0, HEX_DEGREE), dtype=np.int64), np.zeros((0, HEX_DEGREE), dtype=np.float32)

    xs = np.rint(coords[:, 0]).astype(np.int64)
    ys = np.rint(coords[:, 1]).astype(np.int64)
    if not (np.array_equal(xs, coords[:, 0]) and np.array_equal(ys, coords[:, 1])):
        raise ValueError("hexgraph: tile coords are not integer-exact (storage/decode drift?)")

    pos_to_row: dict[tuple[int, int], int] = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(xs, ys))}
    wrap = bool(world_wrap)
    r = int(round(float(eff_wrap_radius)))

    sentinel = n  # the model pads node features with a zero row at index N
    idx = np.full((n, HEX_DEGREE), sentinel, dtype=np.int64)
    mask = np.zeros((n, HEX_DEGREE), dtype=np.float32)

    for i in range(n):
        x, y = int(xs[i]), int(ys[i])
        for d, (dx, dy) in enumerate(OFFSETS):
            nx, ny = x + dx, y + dy
            row = pos_to_row.get((nx, ny))
            if row is None and wrap:                       # mirror getIfTileExistsOrNull C then D
                row = pos_to_row.get((nx + r, ny - r))
                if row is None:
                    row = pos_to_row.get((nx - r, ny + r))
            if row is not None:
                idx[i, d] = row
                mask[i, d] = 1.0

    # bounds: every index in [0, N]; the N sentinel is the pad row — no OOB Gather (FND-0025)
    assert int(idx.min()) >= 0 and int(idx.max()) <= n, "neighbor index out of bounds"
    return idx, mask
