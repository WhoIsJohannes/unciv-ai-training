"""v6 replay-buffer analysis: the two AC2 framings + AC1 no-op check, per rung.

Consumes the 4 arm dirs produced by run_v6.sh (each a run_loop --out with curve.csv + the per-arm
acceptance_v6.json written by analyze_v5). Reports, per rung (small target 40.7%, medium 46.6%):

  AC1 (no-op): the K=1 arm's ceiling vs v5's reference (40.7% small / 46.6% medium) — must be
               statistically indistinguishable (the window-gated path is the literal v5 recompute).
  AC2 framing 1 (sample efficiency): rounds-to-target — the first generation round whose 80-game
               eval winrate reaches the target. Fewer rounds for K=4 ⇒ replay WINS on sample efficiency.
  AC2 framing 2 (ceiling at equal rounds): the 200-game ceiling (seed 4242424) of K=4 vs K=1 at round
               16, two-proportion z-test. Higher K=4 ceiling with p<0.05 ⇒ replay WINS on ceiling.

Reuses the FROZEN primitive analyze._two_proportion_z. Pure reporting — runs no gradle.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .analyze import _two_proportion_z

# v5 references (the bar each rung must reach / reproduce) + the blind baseline (AC2 headline).
REF = {"small": 0.407, "medium": 0.466}
BLIND = (58, 200)  # 28.9%


def _curve(root: Path) -> list[tuple[int, float]]:
    f = root / "curve.csv"
    if not f.is_file():
        return []
    out = []
    with open(f, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                out.append((int(row["round"]), float(row["winrate"])))
            except (ValueError, KeyError):
                continue
    return out


def _rounds_to_target(curve: list[tuple[int, float]], target: float) -> int | None:
    for rnd, wr in curve:
        if wr >= target:
            return rnd
    return None


def _ceiling(root: Path) -> tuple[int, int] | None:
    """(wins, games) from the per-arm acceptance_v6.json (written by analyze_v5), else None."""
    f = root / "acceptance_v6.json"
    if not f.is_file():
        f = root / "acceptance_v5.json"
    if not f.is_file():
        return None
    try:
        c = json.loads(f.read_text()).get("ceiling")
        if c and c.get("games"):
            return int(c["wins"]), int(c["games"])
    except (ValueError, KeyError, TypeError):
        pass
    return None


def _arm(root: Path, rung: str, k: int) -> dict:
    d = root / f"structured-{rung}-K{k}"
    curve = _curve(d)
    ceil = _ceiling(d)
    return {
        "dir": d.name,
        "exists": d.is_dir(),
        "rounds_to_target": _rounds_to_target(curve, REF[rung]),
        "final_round": curve[-1][0] if curve else None,
        "final_winrate_80g": curve[-1][1] if curve else None,
        "ceiling": {"wins": ceil[0], "games": ceil[1], "winrate": ceil[0] / ceil[1]} if ceil else None,
    }


def analyze(root: Path) -> dict:
    report = {"reference": REF, "blind_baseline": "58/200=28.9%", "rungs": {}}
    for rung in ("small", "medium"):
        k1, k4 = _arm(root, rung, 1), _arm(root, rung, 4)
        rep: dict = {"k1": k1, "k4": k4}
        # AC1 no-op: K=1 ceiling vs blind (should clear it like v5 did) — and report its winrate vs REF.
        if k1["ceiling"]:
            w, n = k1["ceiling"]["wins"], k1["ceiling"]["games"]
            z, p = _two_proportion_z(w, n, *BLIND)
            rep["ac1_k1_vs_blind"] = {"k1": f"{w}/{n}={w/n:.3f}", "z": z, "p_one_sided": p,
                                      "ref": REF[rung], "clears_blind": p < 0.05 and w / n > BLIND[0] / BLIND[1]}
        # AC2 framing 1 — sample efficiency (fewer generation rounds to the target).
        rep["ac2_sample_efficiency"] = {
            "k1_rounds_to_target": k1["rounds_to_target"], "k4_rounds_to_target": k4["rounds_to_target"],
            "replay_faster": (k4["rounds_to_target"] is not None and
                              (k1["rounds_to_target"] is None or k4["rounds_to_target"] < k1["rounds_to_target"])),
        }
        # AC2 framing 2 — ceiling at equal rounds (K=4 vs K=1, two-proportion z).
        if k1["ceiling"] and k4["ceiling"]:
            w1, n1 = k1["ceiling"]["wins"], k1["ceiling"]["games"]
            w4, n4 = k4["ceiling"]["wins"], k4["ceiling"]["games"]
            z, p = _two_proportion_z(w4, n4, w1, n1)
            rep["ac2_ceiling"] = {"k1": f"{w1}/{n1}={w1/n1:.3f}", "k4": f"{w4}/{n4}={w4/n4:.3f}",
                                  "z": z, "p_one_sided": p, "replay_higher": w4 / n4 > w1 / n1 and p < 0.05}
        # Overall per-rung verdict.
        won_eff = rep["ac2_sample_efficiency"]["replay_faster"]
        won_ceil = rep.get("ac2_ceiling", {}).get("replay_higher", False)
        rep["verdict"] = "REPLAY WINS" if (won_eff or won_ceil) else "NULL (no replay benefit at this rung)"
        report["rungs"][rung] = rep
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="v6 replay analysis — AC1 no-op + AC2 both framings, per rung")
    ap.add_argument("--root", required=True, help="v6 root containing the 4 arm dirs structured-{small,medium}-K{1,4}")
    args = ap.parse_args(argv)
    report = analyze(Path(args.root))
    print(json.dumps(report, indent=2))
    (Path(args.root) / "acceptance_v6_compare.json").write_text(json.dumps(report, indent=2))
    for rung, rep in report["rungs"].items():
        print(f"\n[{rung}] verdict: {rep['verdict']}")
        eff = rep["ac2_sample_efficiency"]
        print(f"  sample-efficiency: K1 reaches {REF[rung]*100:.1f}% at round {eff['k1_rounds_to_target']}, "
              f"K4 at round {eff['k4_rounds_to_target']} → replay_faster={eff['replay_faster']}")
        if "ac2_ceiling" in rep:
            c = rep["ac2_ceiling"]
            print(f"  ceiling@16: K1={c['k1']} K4={c['k4']} z={c['z']:.3f} p={c['p_one_sided']:.4g} "
                  f"→ replay_higher={c['replay_higher']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
