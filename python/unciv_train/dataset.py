"""Load trajectory shards → (obs, chosen action, legal mask, return-to-go) for the learner civ.

Provenance is STRICT (criterion 6): a shard whose schema VERSION mismatches is refused by the
reader (`reader.load` raises); a shard whose ruleset fingerprint differs from the expected one is
refused here. The learner's steps are found per-shard via the header's `majorCivSlots` slot↔civId
map (turn-order shuffle makes civ_slot vary per game). The terminal record's reward is broadcast as
the undiscounted return-to-go for that civ's non-terminal steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from unciv_dataplane import reader

from .contract import LEARNER_CIV_ID


class ProvenanceError(Exception):
    """Raised when a shard's schema VERSION or ruleset fingerprint differs from the running engine."""


@dataclass
class TrainStep:
    obs: np.ndarray        # concat(global, acting_civ), float32
    a_tech: int            # chosen tech index, or -1 (no decision this turn)
    a_policy: int          # chosen policy index, or -1
    mask_tech: np.ndarray  # legal tech mask (float32 0/1)
    mask_policy: np.ndarray
    ret: float             # return-to-go = the civ's terminal reward


# Per-type entity/spatial token blocks consumed by the rich variant (perItem widths come from the
# generated schema; spatial is nTiles*n_channels (schema-driven) and reshaped to [nTiles, n_channels]).
RICH_TOKEN_BLOCKS = ("spatial", "own_units", "opp_units", "own_cities", "opp_cities", "civ_tokens")
# Contract-v3 (structured) shard-only block: per-tile (x,y) coords as f32 [nTiles*2]. Consumed by
# the Python hex-adjacency builder (hexgraph) to derive neighbor_index/neighbor_mask — it is NOT an
# ONNX model input and is absent on contract-v2 shards (handled gracefully: returns None).
SPATIAL_COORDS_BLOCK = "spatial_coords"


@dataclass
class TrainTrajectory:
    """One game's ordered learner-step sequence (GAE needs the temporal sequence).

    ALL non-terminal learner steps are kept in emission order — no-action steps are NOT dropped
    (dropping them would break GAE's temporal contiguity and risk losing the reward-bearing step).
    The terminal ±1 is placed at the last step's reward slot; reward is 0 everywhere else.
    """
    obs: np.ndarray          # [T, input_w] blind concat(global, acting_civ), float32
    a_tech: np.ndarray       # [T] int (−1 = head did not act)
    a_policy: np.ndarray     # [T] int
    mask_tech: np.ndarray    # [T, tech_w] float32
    mask_policy: np.ndarray  # [T, policy_w] float32
    rewards: np.ndarray      # [T] float32 — 0 except the terminal ±1 at the last step
    rich: list | None = None  # per-step dict{name -> np.ndarray} for the rich variant (else None)
    # v6: per-step behavior-policy log π_b(a|s) recorded at sampling time, [T] f32, 0 where the head
    # did not act. None ONLY for synthetic (non-shard) trajectories that never exercise off-policy
    # replay; real shards (SCHEMA_VERSION≥4) always populate both. _stack_traj treats None as zeros.
    b_logp_tech: np.ndarray | None = None
    b_logp_policy: np.ndarray | None = None
    # v7.2: per-step log-stabilized economy potential Φ(s), [T] f32. None for synthetic trajectories
    # (no shaping). The trainer forms the policy-invariant shaping reward F = γ·Φ(s')−Φ(s) from it.
    phi: np.ndarray | None = None


def _learner_slot(header: dict, learner_civ_id: str) -> int | None:
    for entry in header.get("majorCivSlots", []):
        if entry.get("civId") == learner_civ_id:
            return int(entry["slot"])
    return None


def load_training_steps(
    paths,
    learner_civ_id: str = LEARNER_CIV_ID,
    *,
    expected_version: int,
    expected_fingerprint: str,
) -> list[TrainStep]:
    out: list[TrainStep] = []
    # Sorted for deterministic ordering (multithreaded GENERATE writes shards in any order).
    for p in sorted(str(x) for x in paths):
        shard = reader.load(p)  # raises ShardError on a VERSION mismatch (perishable datasets)
        prov = shard.provenance
        if prov.schema_version != expected_version:
            raise ProvenanceError(f"{p}: schema_version {prov.schema_version} != {expected_version}")
        if prov.ruleset_fingerprint != expected_fingerprint:
            raise ProvenanceError(
                f"{p}: ruleset_fingerprint {prov.ruleset_fingerprint!r} != {expected_fingerprint!r} "
                "(shard was generated against different ruleset content — regenerate)"
            )
        slot = _learner_slot(shard.header, learner_civ_id)
        if slot is None:
            continue  # this shard's game did not contain the learner civ

        learner_steps = [s for s in shard.steps if s.civ_slot == slot and not s.is_terminal]
        terminal = next((s for s in shard.steps if s.civ_slot == slot and s.is_terminal), None)
        ret = float(terminal.reward) if terminal is not None else 0.0

        for s in learner_steps:
            actions = s.blocks["actions"]
            a_tech, a_policy = int(actions[0]), int(actions[1])
            if a_tech < 0 and a_policy < 0:
                continue  # no modeled (tech/policy) decision on this turn — nothing to learn from
            obs = np.concatenate([s.blocks["global"], s.blocks["acting_civ"]]).astype(np.float32)
            out.append(TrainStep(
                obs=obs, a_tech=a_tech, a_policy=a_policy,
                mask_tech=s.blocks["mask_tech"].astype(np.float32),
                mask_policy=s.blocks["mask_policy"].astype(np.float32),
                ret=ret,
            ))
    return out


def _rich_step_blocks(blocks: dict, n_channels: int) -> dict:
    """Extract the per-step rich token blocks. spatial (nTiles*n_channels, schema-driven) →
    [nTiles, n_channels]; entity blocks (VARIABLE) are already [count, perItem]; an absent/empty
    block → [0, perItem]. The contract-v3 `spatial_coords` block (nTiles*2 f32), when present, is
    reshaped to [nTiles, 2] and returned for the Python adjacency builder (None on v2 shards).

    `n_channels` is the schema's spatial channel count (FAIL-LOUD god-constant, no hardcoded 13)."""
    out = {
        "global": np.asarray(blocks["global"], dtype=np.float32),
        "acting_civ": np.asarray(blocks["acting_civ"], dtype=np.float32),
    }
    spatial = np.asarray(blocks["spatial"], dtype=np.float32)
    if spatial.size % n_channels != 0:
        raise ValueError(
            f"spatial block size {spatial.size} not divisible by n_channels {n_channels} "
            "(schema/shard channel-count drift)"
        )
    out["spatial"] = spatial.reshape(-1, n_channels)
    coords = blocks.get(SPATIAL_COORDS_BLOCK)
    if coords is not None:
        coords = np.asarray(coords, dtype=np.float32)
        if coords.size % 2 != 0:
            raise ValueError(f"spatial_coords block size {coords.size} not divisible by 2")
        out[SPATIAL_COORDS_BLOCK] = coords.reshape(-1, 2)
    else:
        out[SPATIAL_COORDS_BLOCK] = None
    for name in ("own_units", "opp_units", "own_cities", "opp_cities", "civ_tokens"):
        b = blocks.get(name)
        arr = np.asarray(b, dtype=np.float32) if b is not None else np.zeros((0, 0), np.float32)
        if arr.ndim == 1:                      # empty VARIABLE block decoded as 1D
            arr = arr.reshape(0, arr.shape[0] if arr.size else 0)
        out[name] = arr
    # v7: per-city construction — the legal mask [ncities, constr_w] + the recorded action / behavior
    # logp [ncities] (−1 / 0 where the city did not decide), aligned to own_cities (orderedOwnCities).
    mc = blocks.get("mask_construction")
    mc = np.asarray(mc, dtype=np.float32) if mc is not None else np.zeros((0, 0), np.float32)
    if mc.ndim == 1:
        mc = mc.reshape(0, mc.shape[0] if mc.size else 0)
    out["mask_construction"] = mc
    ca = blocks.get("construction_action")
    out["construction_action"] = (np.asarray(ca, dtype=np.float32).reshape(-1)
                                  if ca is not None else np.zeros(0, np.float32))
    cl = blocks.get("construction_logp")
    out["construction_logp"] = (np.asarray(cl, dtype=np.float32).reshape(-1)
                                if cl is not None else np.zeros(0, np.float32))
    # v7.3: per-city raw log-economy (aligned to construction), for the per-city value/advantage.
    ec = blocks.get("econ_city")
    out["econ_city"] = (np.asarray(ec, dtype=np.float32).reshape(-1)
                        if ec is not None else np.zeros(0, np.float32))
    # v7.4: per-city current construction mask idx (heuristic pick when gen'd with control off) — BC target.
    cc = blocks.get("construction_current")
    out["construction_current"] = (np.asarray(cc, dtype=np.float32).reshape(-1)
                                   if cc is not None else np.zeros(0, np.float32))
    return out


def load_trajectories(
    paths,
    learner_civ_id: str = LEARNER_CIV_ID,
    *,
    expected_version: int,
    expected_fingerprint: str,
    rich: bool = False,
    expected_spatial_channels: int | None = None,
) -> list[TrainTrajectory]:
    """Ordered per-game learner-step trajectories for actor-critic + GAE. Same provenance gates as
    `load_training_steps`. `rich=True` also attaches per-step token blocks for the rich/structured
    variant; `expected_spatial_channels` is the schema-driven spatial channel count (REQUIRED when
    rich — no hardcoded 13, FND-0007).
    """
    if rich and expected_spatial_channels is None:
        raise ValueError("load_trajectories(rich=True) requires expected_spatial_channels "
                         "(schema-driven spatial channel count; SSOT is Kotlin SampleSchema)")
    out: list[TrainTrajectory] = []
    for p in sorted(str(x) for x in paths):
        shard = reader.load(p)
        prov = shard.provenance
        if prov.schema_version != expected_version:
            raise ProvenanceError(f"{p}: schema_version {prov.schema_version} != {expected_version}")
        if prov.ruleset_fingerprint != expected_fingerprint:
            raise ProvenanceError(
                f"{p}: ruleset_fingerprint {prov.ruleset_fingerprint!r} != {expected_fingerprint!r} "
                "(shard was generated against different ruleset content — regenerate)"
            )
        slot = _learner_slot(shard.header, learner_civ_id)
        if slot is None:
            continue
        # ALL non-terminal learner steps in emission (chronological) order — keep no-action steps.
        steps = [s for s in shard.steps if s.civ_slot == slot and not s.is_terminal]
        terminal = next((s for s in shard.steps if s.civ_slot == slot and s.is_terminal), None)
        term_r = float(terminal.reward) if terminal is not None else 0.0
        if not steps:
            continue
        t = len(steps)
        obs = np.stack([np.concatenate([s.blocks["global"], s.blocks["acting_civ"]]) for s in steps]).astype(np.float32)
        a_tech = np.array([int(s.blocks["actions"][0]) for s in steps], dtype=np.int64)
        a_policy = np.array([int(s.blocks["actions"][1]) for s in steps], dtype=np.int64)
        mask_tech = np.stack([s.blocks["mask_tech"] for s in steps]).astype(np.float32)
        mask_policy = np.stack([s.blocks["mask_policy"] for s in steps]).astype(np.float32)
        rewards = np.zeros(t, dtype=np.float32)
        rewards[-1] = term_r                              # terminal-only ±1
        # v6: behavior-policy log π_b per head (MASK_HEADS order {tech, policy}). No .get() fallback —
        # SCHEMA_VERSION=4 guarantees the block (fail-loud, matching the perishable-dataset discipline).
        b_logp_tech = np.array([float(s.blocks["behavior_logp"][0]) for s in steps], dtype=np.float32)
        b_logp_policy = np.array([float(s.blocks["behavior_logp"][1]) for s in steps], dtype=np.float32)
        # v7.2: per-step economy potential Φ(s) (SCHEMA_VERSION≥6). Fail-loud — no .get() fallback.
        phi = np.array([float(s.blocks["phi"][0]) for s in steps], dtype=np.float32)
        rich_blocks = ([_rich_step_blocks(s.blocks, expected_spatial_channels) for s in steps]
                       if rich else None)
        out.append(TrainTrajectory(obs, a_tech, a_policy, mask_tech, mask_policy, rewards, rich_blocks,
                                   b_logp_tech, b_logp_policy, phi=phi))
    return out
