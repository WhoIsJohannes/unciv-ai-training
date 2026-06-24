"""Acceptance analysis: read the per-variant curve.csv files, compute the operationalized criteria,
and write a report + overlaid comparison plots.

AC1  attributable curves        — overlay v1-reinforce / blind-critic / rich-critic (Tiny).
AC2  convergence (the critic)   — last-K=4 win-rate stddev: blind-critic vs v1-reinforce (+ mean late
                                  win-rate, since "steadier" is only meaningful with the level).
AC3  ceiling (the board)        — rich-critic vs blind-critic on Medium: a final high-N eval +
                                  two-proportion z-test (one-sided rich>blind). Reported either way.

No scipy: the binomial/normal tail uses math.erfc. Run AFTER run_acceptance.sh finishes the curves.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from . import run_loop  # reuse evaluate() (gradle eval) for the final ceiling eval  # noqa: E402

LAST_K = 4


def _read_curve(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with open(path, newline="") as f:
        for d in csv.DictReader(f):
            try:
                rows.append({"round": int(d["round"]), "winrate": float(d["winrate"]),
                             "games": int(d["games"]), "pval": float(d["pval"])})
            except (ValueError, KeyError):
                continue
    return rows


def _stddev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _two_proportion_z(w1: int, n1: int, w2: int, n2: int) -> tuple[float, float]:
    """One-sided two-proportion z-test for p1 > p2. Returns (z, p_one_sided)."""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1, p2 = w1 / n1, w2 / n2
    p = (w1 + w2) / (n1 + n2)
    se = math.sqrt(max(p * (1 - p) * (1 / n1 + 1 / n2), 1e-12))
    z = (p1 - p2) / se
    p_one = 0.5 * math.erfc(z / math.sqrt(2))  # P(Z > z)
    return z, p_one


def _overlay(curves: dict[str, list[dict]], png: Path, title: str) -> None:
    try:  # best-effort: overlay PNG is nice-to-have, not required
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze] matplotlib not installed — skipping overlay PNG")
        return
    plt.figure(figsize=(8, 4.5))
    plt.axhline(50, color="gray", ls="--", lw=1, label="RandomPolicy (50%)")
    for i, (name, rows) in enumerate(curves.items()):
        if rows:
            plt.plot([r["round"] for r in rows], [r["winrate"] * 100 for r in rows],
                     "-o", color=f"C{i}", label=name)
    plt.xlabel("round"); plt.ylabel("win-rate vs RandomPolicy (%)")
    plt.title(title); plt.ylim(0, 100); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(png, dpi=120); plt.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Acceptance analysis over the v2 training runs")
    ap.add_argument("--root", required=True, help="dir containing <variant>-<size>/curve.csv subdirs")
    ap.add_argument("--ceiling-games", type=int, default=200, help="final Medium eval game count (AC3)")
    ap.add_argument("--turn-cap", type=int, default=250)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--eval-seed", type=int, default=4242424)
    ap.add_argument("--gradle-timeout", type=float, default=3600.0)
    ap.add_argument("--skip-ceiling-eval", action="store_true", help="use last Medium round instead of a fresh high-N eval")
    args = ap.parse_args(argv)

    root = Path(args.root)
    tiny = {v: _read_curve(root / f"{v}-tiny" / "curve.csv")
            for v in ["v1-reinforce", "blind-critic", "rich-critic"]}
    medium = {v: _read_curve(root / f"{v}-medium" / "curve.csv")
              for v in ["blind-critic", "rich-critic"]}

    lines: list[str] = ["# v2 Acceptance Analysis\n"]

    # AC1 — curves
    lines.append("## AC1 — attributable curves\n")
    for v, rows in tiny.items():
        lines.append(f"- Tiny `{v}`: {len(rows)} rounds, final win-rate "
                     f"{rows[-1]['winrate']*100:.1f}% (n={rows[-1]['games']})" if rows else f"- Tiny `{v}`: (no curve)")
    _overlay(tiny, root / "curve_tiny_overlay.png", "Tiny — v1-reinforce vs blind-critic vs rich-critic")
    lines.append(f"\nOverlay plot: `{root / 'curve_tiny_overlay.png'}`\n")

    # AC2 — convergence (last-K stddev: blind-critic vs v1-reinforce)
    lines.append("## AC2 — convergence (does the critic steady the curve?)\n")
    def lastk(rows): return [r["winrate"] for r in rows[-LAST_K:]]
    v1, bc = tiny["v1-reinforce"], tiny["blind-critic"]
    ac2 = {}
    if len(v1) >= LAST_K and len(bc) >= LAST_K:
        sd_v1, sd_bc = _stddev(lastk(v1)), _stddev(lastk(bc))
        mean_v1 = sum(lastk(v1)) / LAST_K
        mean_bc = sum(lastk(bc)) / LAST_K
        steadier = sd_bc < sd_v1
        ac2 = {"last_k": LAST_K, "stddev_v1_reinforce": sd_v1, "stddev_blind_critic": sd_bc,
               "mean_winrate_v1_reinforce": mean_v1, "mean_winrate_blind_critic": mean_bc,
               "blind_critic_steadier": steadier}
        lines.append(f"- last-{LAST_K} win-rate stddev: v1-reinforce={sd_v1*100:.2f}pp, "
                     f"blind-critic={sd_bc*100:.2f}pp → blind-critic {'STEADIER ✓' if steadier else 'NOT steadier'}")
        lines.append(f"- last-{LAST_K} mean win-rate: v1-reinforce={mean_v1*100:.1f}%, blind-critic={mean_bc*100:.1f}%")
    else:
        lines.append(f"- insufficient rounds for last-{LAST_K} stddev (v1={len(v1)}, blind={len(bc)})")

    # AC3 — ceiling (rich vs blind on Medium)
    lines.append("\n## AC3 — ceiling (does seeing the board help on Medium?)\n")
    ac3 = {}
    bm = root / "blind-critic-medium" / f"policy_round_{len(medium['blind-critic'])-1}.onnx" if medium["blind-critic"] else None
    rm = root / "rich-critic-medium" / f"policy_round_{len(medium['rich-critic'])-1}.onnx" if medium["rich-critic"] else None
    if bm and rm and bm.is_file() and rm.is_file() and not args.skip_ceiling_eval:
        be = run_loop.evaluate(bm, args.ceiling_games, args.turn_cap, args.threads, args.eval_seed, args.gradle_timeout, "Medium")
        re_ = run_loop.evaluate(rm, args.ceiling_games, args.turn_cap, args.threads, args.eval_seed, args.gradle_timeout, "Medium")
        z, p = _two_proportion_z(int(re_["wins"]), int(re_["games"]), int(be["wins"]), int(be["games"]))
        ac3 = {"blind": be, "rich": re_, "z": z, "p_one_sided_rich_gt_blind": p, "significant_p05": p < 0.05}
        lines.append(f"- Medium final eval (n={args.ceiling_games}): blind-critic={be['winrate']*100:.1f}%, "
                     f"rich-critic={re_['winrate']*100:.1f}%")
        lines.append(f"- two-proportion z={z:.2f}, one-sided p(rich>blind)={p:.4g} → "
                     f"{'SIGNIFICANT at p<0.05 ✓' if p < 0.05 else 'NOT significant at p<0.05 (reported plainly)'}")
    else:
        for v, rows in medium.items():
            lines.append(f"- Medium `{v}`: {len(rows)} rounds" + (f", final {rows[-1]['winrate']*100:.1f}%" if rows else " (no curve)"))
        lines.append("- (ceiling eval skipped or models missing — comparison from last Medium rounds above)")
    _overlay(medium, root / "curve_medium_overlay.png", "Medium — blind-critic vs rich-critic")
    lines.append(f"\nMedium overlay plot: `{root / 'curve_medium_overlay.png'}`\n")

    report = root / "acceptance-report.md"
    report.write_text("\n".join(lines) + "\n")
    (root / "acceptance.json").write_text(json.dumps({"ac2": ac2, "ac3": ac3}, indent=2))
    print(f"wrote {report}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
