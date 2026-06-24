"""RED-first test for the v4 hex adjacency builder (D3).

Captures the load-bearing new behavior: build_neighbor_graph turns per-tile (x,y) coords into a
fixed degree-6 neighbor_index [N,6] + neighbor_mask [N,6], using the SAME 6 clock-direction
offsets and world-wrap rule as the JVM engine (HexMath.clockPositionToHexcoordMap +
TileMap.getIfTileExistsOrNull). Until python/unciv_train/hexgraph.py exists this is RED (import error).

Clock direction order (FROZEN — must match Kotlin emit + inference + parity):
    [ (+1,+1), (0,+1), (-1,0), (-1,-1), (0,-1), (+1,0) ]   # dirs 12,2,4,6,8,10
Missing neighbor (off-map, no world-wrap) -> index 0 (sentinel) AND mask 0.
"""
import numpy as np

# RED-first: direct import so pytest reports a hard failure (collection error) until
# python/unciv_train/hexgraph.py is implemented by D3 — NOT a skip.
from unciv_train import hexgraph

OFFSETS = [(1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1), (1, 0)]
SHAPE_HEX = 1  # MapShape ordinal: 0 rectangular, 1 hexagonal, 2 flatEarth


def _center_patch():
    # row 0 = center (0,0); rows 1..6 = its 6 neighbors in clock order.
    coords = [(0, 0)] + [(0 + dx, 0 + dy) for (dx, dy) in OFFSETS]
    return np.asarray(coords, dtype=np.float32)


def test_returns_correct_shapes_and_dtype():
    coords = _center_patch()
    idx, mask = hexgraph.build_neighbor_graph(coords, eff_wrap_radius=0, world_wrap=False, shape=SHAPE_HEX)
    n = coords.shape[0]
    assert idx.shape == (n, 6)
    assert mask.shape == (n, 6)
    assert np.issubdtype(np.asarray(idx).dtype, np.integer)
    # every index in [0, N]: N is the sentinel/pad row the model appends; no OOB Gather (FND-0025/0035)
    assert idx.min() >= 0 and idx.max() <= n


def test_interior_tile_has_all_six_neighbors_in_clock_order():
    coords = _center_patch()
    idx, mask = hexgraph.build_neighbor_graph(coords, eff_wrap_radius=0, world_wrap=False, shape=SHAPE_HEX)
    # center is row 0; rows 1..6 were placed in exactly clock order.
    assert mask[0].tolist() == [1, 1, 1, 1, 1, 1]
    assert idx[0].tolist() == [1, 2, 3, 4, 5, 6]


def test_edge_tile_masks_absent_neighbors():
    # row 1 is (1,1); within the patch only 3 of its 6 clock-neighbors exist:
    #   dir4 (-1,0)->(0,1)=row2, dir6 (-1,-1)->(0,0)=row0, dir8 (0,-1)->(1,0)=row6.
    coords = _center_patch()
    idx, mask = hexgraph.build_neighbor_graph(coords, eff_wrap_radius=0, world_wrap=False, shape=SHAPE_HEX)
    assert mask[1].tolist() == [0, 0, 1, 1, 1, 0]
    present = [(i, v) for i, (m, v) in enumerate(zip(mask[1].tolist(), idx[1].tolist())) if m == 1]
    assert present == [(2, 2), (3, 0), (4, 6)]


def test_world_wrap_resolves_off_map_neighbor():
    # row0=(2,2); its dir10 (+1,0) neighbor (3,2) is off-map but wrap-retry D (nx-R, ny+R)=(0,5)=row1.
    # (getIfTileExistsOrNull: direct miss -> (nx+R,ny-R)=(6,-1) miss -> (nx-R,ny+R)=(0,5) hit.)
    R = 3
    coords = np.asarray([(2, 2), (0, 5)], dtype=np.float32)
    DIR10 = 5  # OFFSETS index of (+1,0)
    idx, mask = hexgraph.build_neighbor_graph(coords, eff_wrap_radius=R, world_wrap=True, shape=SHAPE_HEX)
    assert mask[0, DIR10] == 1 and idx[0, DIR10] == 1, "wrap must find row1 as row0's +1,0 neighbor"
    idx_nw, mask_nw = hexgraph.build_neighbor_graph(coords, eff_wrap_radius=R, world_wrap=False, shape=SHAPE_HEX)
    assert mask_nw[0, DIR10] == 0, "without world wrap that neighbor is off-map (absent)"
    assert mask.sum() > mask_nw.sum(), "world wrap must find strictly more neighbors here"
