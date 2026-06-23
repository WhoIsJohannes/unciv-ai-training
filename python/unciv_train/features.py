"""Assemble the rich multi-tensor input from per-step observation blocks.

The spatial block (FIXED nTiles*13) becomes a per-tile token SET; each entity block
(own_units/opp_units/own_cities/opp_cities/civ_tokens) is a present-only token set. All are padded
to the batch-max count with a presence mask (1 = real token, 0 = padding) so the model's masked
pool ignores padding. This is the SAME layout the JVM builds at decision time — the parity test
guards that the two agree.
"""
from __future__ import annotations

import numpy as np
import torch

from .contract import Dims


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


def build_rich_batch(trajectories, dims: Dims, token_specs: dict[str, int]) -> dict:
    """Flatten all trajectories' per-step rich blocks into one padded multi-tensor batch.
    Returns {name: tensor, name+"_mask": tensor, "global": tensor, "acting_civ": tensor}.
    """
    steps = [blk for t in trajectories for blk in (t.rich or [])]
    n = len(steps)
    out: dict[str, torch.Tensor] = {
        "global": torch.tensor(np.stack([s["global"] for s in steps]).astype(np.float32)),
        "acting_civ": torch.tensor(np.stack([s["acting_civ"] for s in steps]).astype(np.float32)),
    }
    assert n > 0, "build_rich_batch called with no steps"
    for name, width in token_specs.items():
        padded, mask = _pad_token_set([s[name] for s in steps], width)
        out[name] = torch.tensor(padded)
        out[name + "_mask"] = torch.tensor(mask)
    return out


def build_rich_single(step_blocks: dict, token_specs: dict[str, int]) -> dict:
    """Batch-of-1 multi-tensor input from ONE step's raw blocks (the parity / inference reference).

    `step_blocks` holds: global, acting_civ (1D), spatial (nTiles*13 flat OR [nTiles,13]), and each
    entity block as [count, perItem] (possibly empty). Mirrors exactly what the JVM builds.
    """
    g = np.asarray(step_blocks["global"], dtype=np.float32).reshape(1, -1)
    a = np.asarray(step_blocks["acting_civ"], dtype=np.float32).reshape(1, -1)
    out: dict[str, torch.Tensor] = {"global": torch.tensor(g), "acting_civ": torch.tensor(a)}
    for name, width in token_specs.items():
        arr = np.asarray(step_blocks[name], dtype=np.float32)
        if name == "spatial":
            arr = arr.reshape(-1, width)
        elif arr.ndim == 1:
            arr = arr.reshape(0, width) if arr.size == 0 else arr.reshape(-1, width)
        padded, mask = _pad_token_set([arr], width)
        out[name] = torch.tensor(padded)
        out[name + "_mask"] = torch.tensor(mask)
    return out
