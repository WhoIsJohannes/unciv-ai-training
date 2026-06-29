"""v6 (plan council 🔴) — the --resume deque refill must EXCLUDE round 0.

The in-process replay deque excludes round 0 (RandomPolicy data, maximally off the current net). On
--resume the deque is empty and must be refilled from disk over the SAME window — i.e. the recent
NON-zero rounds `max(1, start-(K-1)) .. start-1`, never round 0. A naive `[start-K .. start-1]` glob
would re-admit round 0 when start ≤ K (e.g. resume at round 2 with K=4 → would load round 0).

RED until v6 adds `run_loop._replay_refill_rounds(start_round, replay_window)`.
"""
from __future__ import annotations

from unciv_train import run_loop


def test_refill_excludes_round_zero():
    # resume at round 2, K=4: window would be [-2..1]; must clamp to [1] (round 0 excluded).
    assert run_loop._replay_refill_rounds(2, 4) == [1]


def test_refill_recent_window_when_far_in():
    # resume at round 10, K=4: the K-1=3 most recent non-zero rounds before 10.
    assert run_loop._replay_refill_rounds(10, 4) == [7, 8, 9]


def test_refill_k1_is_empty():
    # K=1 ⇒ no replay ⇒ nothing to refill.
    assert run_loop._replay_refill_rounds(5, 1) == []


def test_refill_at_round_one_is_empty():
    # resuming at round 1: only round 0 precedes it, which is excluded ⇒ empty.
    assert run_loop._replay_refill_rounds(1, 4) == []
