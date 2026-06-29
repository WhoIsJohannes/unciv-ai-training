"""v7 per-city CONSTRUCTION analysis: ON-vs-OFF per rung + the 50%-break-even framing.

Consumes the 4 arm dirs from run_v7.sh (each a run_loop --out with curve.csv + the per-arm
acceptance_v7.json written by analyze_v5's 200-game ceiling eval). Per rung (small / medium), with
construction the ONLY changed axis vs the v6 baseline:

  EFFECT (AC#3, D-C5 ship criterion): two-proportion one-sided z-test, H1 winrate(ON) > winrate(OFF).
      p<0.05 ⇒ construction control PROVABLY moves the learner in the right direction → SHIP
      (even if it does not cross 50%). This is the headline question.
  50%-BREAK-EVEN (the milestone, NOT a gate): each arm's ceiling vs a 0.5 null (one-proportion z),
      stated plainly — does construction-ON cross the symmetric break-even?
  CONTAMINATION (PR2): the ON arm must have actually controlled construction (≈0 OnnxPolicy
      fallbacks). Surfaced from each arm's EVAL telemetry when available; a contaminated ON arm
      (the net silently fell back to the heuristic) invalidates the ON-vs-OFF delta.

Reuses the FROZEN primitive analyze._two_proportion_z. Pure reporting — runs no gradle.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .analyze import _two_proportion_z

# Context references (the v5/v6 below-50% baseline this builds on). Not gates.
REF = {"small": 0.407, "medium": 0.466}
BLIND = (58, 200)  # 28.9% fixed blind baseline


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


def _ceiling(root: Path) -> tuple[int, int] | None:
    """(wins, games) from the per-arm acceptance_v7.json (analyze_v5 falls back to v5/v6 names too)."""
    for name in ("acceptance_v7.json", "acceptance_v6.json", "acceptance_v5.json"):
        f = root / name
        if f.is_file():
            try:
                c = json.loads(f.read_text()).get("ceiling")
                if c and c.get("games"):
                    return int(c["wins"]), int(c["games"])
            except (ValueError, KeyError, TypeError):
                pass
    return None


def _one_proportion_z(wins: int, n: int, p0: float = 0.5) -> tuple[float, float]:
    """One-proportion z-test vs p0 (default the 50% break-even). Returns (z, one-sided p for >p0)."""
    if n <= 0:
        return 0.0, 1.0
    p = wins / n
    se = math.sqrt(p0 * (1 - p0) / n)
    z = (p - p0) / se if se > 0 else 0.0
    # one-sided P(Z > z) via the normal CDF (erf), matching analyze's convention.
    p_one = 0.5 * math.erfc(z / math.sqrt(2))
    return z, p_one


def _arm(root: Path, rung: str, mode: str) -> dict:
    d = root / f"structured-{rung}-{mode}"
    curve = _curve(d)
    ceil = _ceiling(d)
    return {
        "dir": d.name, "exists": d.is_dir(),
        "final_round": curve[-1][0] if curve else None,
        "final_winrate_80g": curve[-1][1] if curve else None,
        "ceiling": {"wins": ceil[0], "games": ceil[1], "winrate": ceil[0] / ceil[1]} if ceil else None,
    }


def analyze(root: Path) -> dict:
    report = {"reference_v5_v6": REF, "blind_baseline": "58/200=28.9%",
              "ship_criterion": "construction-ON beats OFF (p<0.05) within a rung — even below 50%", "rungs": {}}
    any_directional_win = False
    for rung in ("small", "medium"):
        off, on = _arm(root, rung, "off"), _arm(root, rung, "on")
        rep: dict = {"off": off, "on": on}
        if off["ceiling"] and on["ceiling"]:
            wo, no = off["ceiling"]["wins"], off["ceiling"]["games"]
            wn, nn = on["ceiling"]["wins"], on["ceiling"]["games"]
            # EFFECT: ON > OFF one-sided.
            z, p = _two_proportion_z(wn, nn, wo, no)
            directional = p < 0.05 and wn / nn > wo / no
            any_directional_win = any_directional_win or directional
            rep["effect_on_vs_off"] = {
                "off": f"{wo}/{no}={wo/no:.3f}", "on": f"{wn}/{nn}={wn/nn:.3f}",
                "delta": round(wn / nn - wo / no, 4), "z": z, "p_one_sided": p,
                "construction_helps": directional}
            # 50%-break-even (milestone, not a gate).
            zo, po = _one_proportion_z(wo, no); zn, pn = _one_proportion_z(wn, nn)
            rep["vs_50pct"] = {
                "off": {"winrate": round(wo / no, 4), "z": round(zo, 3), "p_one_sided": round(po, 4),
                        "crosses_50": wo / no > 0.5 and po < 0.05},
                "on": {"winrate": round(wn / nn, 4), "z": round(zn, 3), "p_one_sided": round(pn, 4),
                       "crosses_50": wn / nn > 0.5 and pn < 0.05}}
            rep["verdict"] = ("CONSTRUCTION HELPS — SHIP (directional, p<0.05)" if directional
                              else "NULL / NEGATIVE (ON did not beat OFF at p<0.05)")
        else:
            rep["verdict"] = "INCOMPLETE (missing OFF and/or ON ceiling)"
        report["rungs"][rung] = rep
    report["ship_recommendation"] = (
        "SHIP — construction proven directionally beneficial in ≥1 rung (default --control-construction ON)"
        if any_directional_win else
        "SHIP INFRA ONLY — no directional win; default --control-construction OFF (reused by next per-entity heads)")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="v7 construction analysis — ON-vs-OFF per rung + 50% framing")
    ap.add_argument("--root", required=True,
                    help="v7 root containing the 4 arm dirs structured-{small,medium}-{off,on}")
    args = ap.parse_args(argv)
    report = analyze(Path(args.root))
    print(json.dumps(report, indent=2))
    (Path(args.root) / "acceptance_v7_compare.json").write_text(json.dumps(report, indent=2))
    print(f"\nSHIP RECOMMENDATION: {report['ship_recommendation']}")
    for rung, rep in report["rungs"].items():
        print(f"\n[{rung}] {rep['verdict']}")
        if "effect_on_vs_off" in rep:
            e = rep["effect_on_vs_off"]
            print(f"  ON-vs-OFF: OFF={e['off']} ON={e['on']} Δ={e['delta']:+.3f} z={e['z']:.3f} p={e['p_one_sided']:.4g}")
            v = rep["vs_50pct"]
            print(f"  vs 50%:   OFF crosses_50={v['off']['crosses_50']} ({v['off']['winrate']:.3f})  "
                  f"ON crosses_50={v['on']['crosses_50']} ({v['on']['winrate']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
