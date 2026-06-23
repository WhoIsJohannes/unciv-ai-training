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
