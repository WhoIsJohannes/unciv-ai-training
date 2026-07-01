"""v7.3 replicated-experiment analysis: per-arm mean ceiling ± SE + PAIRED per-seed differences.

The self-play baseline is high-variance, so single-seed ceilings are unreliable. This aggregates the
replicates: per-arm mean/SE across seeds, and the paired differences (same seed → same gen-luck) that
give a far tighter effect estimate than comparing raw means. Verdict: does per-city credit (on-pcc) beat
the OFF baseline and the shared-adv construction arm (on-shared)?
"""
import json
import math
import sys
from pathlib import Path

ARMS = ["off", "on-shared", "on-pcc"]


def _ceiling(root: Path, arm: str, seed: str):
    f = root / f"{arm}_s{seed}" / "acceptance_v5.json"
    if not f.is_file():
        return None
    c = (json.loads(f.read_text()).get("ceiling") or {})
    w, n = c.get("wins"), c.get("games")
    return (w / n) if (w is not None and n) else None


def _mean_se(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, None, len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var / len(xs)), len(xs)


def _paired(root, a, b, seeds):
    """Paired differences a-b over seeds where BOTH arms completed. Returns (diffs, mean, se, n)."""
    diffs = []
    for s in seeds:
        ca, cb = _ceiling(root, a, s), _ceiling(root, b, s)
        if ca is not None and cb is not None:
            diffs.append(ca - cb)
    if not diffs:
        return [], None, None, 0
    m = sum(diffs) / len(diffs)
    if len(diffs) < 2:
        return diffs, m, None, len(diffs)
    var = sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1)
    return diffs, m, math.sqrt(var / len(diffs)), len(diffs)


def main():
    root = Path(sys.argv[1])
    seeds = (sys.argv[2].split() if len(sys.argv) > 2 else ["1000", "2000", "3000", "4000"])
    print(f"\n=== v7.3 REPLICATED EFFICACY  (root={root}, seeds={seeds}) ===\n")
    print(f"{'arm':<12} " + " ".join(f"s{s:<6}" for s in seeds) + "   mean±SE (n)")
    per_arm = {}
    for arm in ARMS:
        vals = [_ceiling(root, arm, s) for s in seeds]
        per_arm[arm] = vals
        cells = " ".join((f"{v*100:5.1f} " if v is not None else "  --  ") for v in vals)
        m, se, n = _mean_se(vals)
        ms = f"{m*100:.1f}%±{se*100:.1f} (n={n})" if (m is not None and se is not None) else \
             (f"{m*100:.1f}% (n={n})" if m is not None else "no data")
        print(f"{arm:<12} {cells}   {ms}")

    print("\n--- PAIRED per-seed differences (same seed ⇒ controls gen-variance) ---")
    for a, b in (("on-pcc", "off"), ("on-shared", "off"), ("on-pcc", "on-shared")):
        diffs, m, se, n = _paired(root, a, b, seeds)
        if m is None:
            print(f"  {a} − {b}: no paired data"); continue
        dd = " ".join(f"{d*100:+.1f}" for d in diffs)
        if se and n >= 2:
            t = m / se if se > 0 else float("inf")
            print(f"  {a:<10} − {b:<10}: mean Δ={m*100:+.1f}pp ±{se*100:.1f} (n={n})  per-seed[{dd}]  t≈{t:+.2f}")
        else:
            print(f"  {a:<10} − {b:<10}: Δ={m*100:+.1f}pp (n={n})  per-seed[{dd}]")

    # Verdict
    moff = _mean_se(per_arm["off"])[0]; mpcc = _mean_se(per_arm["on-pcc"])[0]; msh = _mean_se(per_arm["on-shared"])[0]
    print("\n--- VERDICT ---")
    if None in (moff, mpcc, msh):
        print("  incomplete — some arms have no completed runs yet")
    else:
        print(f"  mean ceilings: off={moff*100:.1f}%  on-shared={msh*100:.1f}%  on-pcc={mpcc*100:.1f}%")
        _, dpo, spo, npo = _paired(root, "on-pcc", "off", seeds)
        if dpo is not None:
            moves_right = dpo >= 0
            sig = (spo and npo >= 2 and abs(dpo) > 2 * spo)
            print(f"  per-city credit vs OFF: Δ={dpo*100:+.1f}pp{' (>2·SE)' if sig else ''} — "
                  f"{'MOVES THE RIGHT WAY (construction no longer hurts)' if moves_right else 'still below OFF'}")
        _, dps, sps, nps = _paired(root, "on-pcc", "on-shared", seeds)
        if dps is not None:
            print(f"  per-city credit vs shared-adv: Δ={dps*100:+.1f}pp — "
                  f"{'the credit MECHANISM helps' if dps > 0 else 'no mechanism gain over shared-adv'}")


if __name__ == "__main__":
    main()
