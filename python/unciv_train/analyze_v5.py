"""v5 continual-training acceptance analysis.

For one arm (a run_loop --out dir), run the 200-game Medium ceiling eval at the fixed v4 seed and
z-test the result against the FIXED recorded v4 baselines (one-sided, p<0.05):
  - v4 structured-from-scratch (small rung): 47/204 = 23.0%   (AC1: continual must BEAT this)
  - v2 blind-critic baseline:                58/200 = 28.9%   (AC2: the bar to clear, report either way)

Reuses the FROZEN primitives: run_loop.evaluate (gradle eval) + analyze._two_proportion_z. Also pulls
the round-8 winrate from curve.csv to disentangle regime-vs-round-count (council FND-0029): v5 at 8
rounds (matched to v4's 8-round run) vs v5 final (16 rounds).

  python -m unciv_train.analyze_v5 --root <arm-dir> --label structured-small-Medium
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

from . import run_loop
from .analyze import _two_proportion_z

# Fixed v4/v2 baselines (RESULTS.md of the v4 run; do NOT recompute — apples-to-apples reference).
V4_STRUCTURED = (47, 204)   # 23.0% — from-scratch small-rung structured (AC1 target to beat)
V2_BLIND = (58, 200)        # 28.9% — blind-critic baseline (AC2 headline bar)


def _last_onnx(root: Path) -> Path | None:
    cands = glob.glob(str(root / "policy_round_*.onnx"))
    if not cands:
        return None
    return Path(max(cands, key=lambda p: int(re.search(r"policy_round_(\d+)\.onnx", p).group(1))))


def _curve_winrates(root: Path) -> list[tuple[int, float]]:
    csv_path = root / "curve.csv"
    if not csv_path.is_file():
        return []
    out = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                out.append((int(row["round"]), float(row["winrate"])))
            except (ValueError, KeyError):
                continue
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="v5 continual ceiling eval + z-tests vs fixed v4 baselines")
    ap.add_argument("--root", required=True, help="a run_loop --out arm dir (curve.csv + policy_round_*.onnx)")
    ap.add_argument("--label", default="", help="human label for the arm (e.g. structured-small-Medium)")
    ap.add_argument("--ceiling-games", type=int, default=200)
    ap.add_argument("--turn-cap", type=int, default=250)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--eval-seed", type=int, default=4242424)
    ap.add_argument("--gradle-timeout", type=float, default=3600.0)
    ap.add_argument("--control-construction", choices=["on", "off"], default="off",
                    help="v7: run the ceiling eval WITH per-city construction control (REQUIRED for an "
                         "ON arm — else the ceiling measures tech/policy only, not the construction lever)")
    ap.add_argument("--skip-ceiling-eval", action="store_true",
                    help="report from the last curve.csv round instead of a fresh high-N eval")
    args = ap.parse_args(argv)

    root = Path(args.root)
    label = args.label or root.name
    curve = _curve_winrates(root)

    result: dict = {"label": label, "root": str(root), "eval_seed": args.eval_seed,
                    "n_rounds": len(curve)}
    # Regime-vs-rounds disentangling (council FND-0029): v5 @ 8 rounds (matched to v4) vs v5 final.
    wr_by_round = dict(curve)
    result["winrate_round7_80g"] = wr_by_round.get(7)   # 8th round (0-indexed) — matched to v4's 8 rounds
    result["winrate_final_80g"] = curve[-1][1] if curve else None

    onnx = _last_onnx(root)
    if args.skip_ceiling_eval or onnx is None:
        result["note"] = "ceiling eval skipped or no onnx; winrates are the 80-game per-round evals"
        result["ceiling"] = None
    else:
        ev = run_loop.evaluate(onnx, args.ceiling_games, args.turn_cap, args.threads,
                               args.eval_seed, args.gradle_timeout, "Medium",
                               control_construction=(args.control_construction == "on"))
        w, n = int(ev["wins"]), int(ev["games"])
        z_v4, p_v4 = _two_proportion_z(w, n, *V4_STRUCTURED)    # AC1: continual vs v4-from-scratch
        z_bl, p_bl = _two_proportion_z(w, n, *V2_BLIND)         # AC2: vs blind baseline
        result["ceiling"] = {
            "onnx": onnx.name, "wins": w, "games": n, "winrate": w / n if n else 0.0,
            "vs_v4_structured_23pct": {"baseline": "47/204", "z": z_v4, "p_one_sided": p_v4,
                                       "beats_p05": (w / n > V4_STRUCTURED[0] / V4_STRUCTURED[1]) and p_v4 < 0.05},
            "vs_blind_28_9pct": {"baseline": "58/200", "z": z_bl, "p_one_sided": p_bl,
                                 "clears_p05": (w / n > V2_BLIND[0] / V2_BLIND[1]) and p_bl < 0.05},
        }

    out_json = root / "acceptance_v5.json"
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\n[analyze_v5] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
