#!/usr/bin/env python3
"""v8 unit-intent PAIRED efficacy analysis (clone of analyze_v73rep.py + a real p-value).

Reads each arm's 200-game ceiling win-rate from `<arm>_s<seed>/acceptance_v5.json`, prints the arm×seed
table with mean±SE, and the PAIRED per-seed differences (same seed ⇒ controls the gen-variance the
baseline-variance memory warns about) with a one-sample paired t-stat + two-sided p-value (df = n−1). Also
reports each arm's absolute win-rate vs the 50% break-even (net-intent vs random-intent).

    python3 analyze_v8.py <root> "<seeds>" "off,on" "on-off"

Arm names + pairs are argv-overridable (default: off/on, paired on−off), so the same script serves any
arm set. The v8 base has construction control ON in BOTH arms; the paired variable is unit-intent off vs on.
"""
import json
import math
import sys
from pathlib import Path

ARMS = ["off", "on"]
PAIRS = [("on", "off")]


def _t_sf(t: float, df: int) -> float:
    """Two-sided p-value for Student-t |t| with `df` dof. Uses scipy when available (exact), else the
    regularized incomplete beta (also exact) — no scipy dependency required."""
    try:
        from scipy import stats  # type: ignore
        return float(2.0 * stats.t.sf(abs(t), df))
    except Exception:
        # p = I_{df/(df+t^2)}(df/2, 1/2)  (two-sided), via a continued-fraction betainc.
        x = df / (df + t * t)
        a, b = df / 2.0, 0.5

        def _betacf(x, a, b):
            MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
            qab, qap, qam = a + b, a + 1.0, a - 1.0
            c = 1.0
            d = 1.0 - qab * x / qap
            d = FPMIN if abs(d) < FPMIN else d
            d = 1.0 / d
            h = d
            for m in range(1, MAXIT + 1):
                m2 = 2 * m
                aa = m * (b - m) * x / ((qam + m2) * (a + m2))
                d = 1.0 + aa * d; d = FPMIN if abs(d) < FPMIN else d
                c = 1.0 + aa / c; c = FPMIN if abs(c) < FPMIN else c
                d = 1.0 / d; h *= d * c
                aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
                d = 1.0 + aa * d; d = FPMIN if abs(d) < FPMIN else d
                c = 1.0 + aa / c; c = FPMIN if abs(c) < FPMIN else c
                d = 1.0 / d; de = d * c; h *= de
                if abs(de - 1.0) < EPS:
                    break
            return h

        lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        if x < (a + 1.0) / (a + b + 2.0):
            ib = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) * _betacf(x, a, b) / a
        else:
            ib = 1.0 - math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) * _betacf(1.0 - x, b, a) / b
        return float(min(1.0, max(0.0, ib)))


def _one_prop_z(wins: int, n: int, p0: float = 0.5):
    """One-proportion z vs p0 (the 50% break-even) + one-sided p for >p0."""
    if n <= 0:
        return 0.0, 1.0
    p = wins / n
    se = math.sqrt(p0 * (1 - p0) / n)
    z = (p - p0) / se if se > 0 else 0.0
    return z, 0.5 * math.erfc(z / math.sqrt(2))


def _ceiling(root: Path, arm: str, seed: str):
    f = root / f"{arm}_s{seed}" / "acceptance_v5.json"
    if not f.is_file():
        return None
    c = (json.loads(f.read_text()).get("ceiling") or {})
    w, n = c.get("wins"), c.get("games")
    return (w, n) if (w is not None and n) else None


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
    diffs = []
    for s in seeds:
        ca, cb = _ceiling(root, a, s), _ceiling(root, b, s)
        if ca is not None and cb is not None:
            diffs.append(ca[0] / ca[1] - cb[0] / cb[1])
    if not diffs:
        return [], None, None, 0
    m = sum(diffs) / len(diffs)
    if len(diffs) < 2:
        return diffs, m, None, len(diffs)
    var = sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1)
    return diffs, m, math.sqrt(var / len(diffs)), len(diffs)


def main(argv):
    global ARMS, PAIRS
    root = Path(argv[1])
    seeds = (argv[2].split() if len(argv) > 2 else ["1000", "2000", "3000", "4000"])
    if len(argv) > 3 and argv[3]:
        ARMS = argv[3].split(",")
    if len(argv) > 4 and argv[4]:
        PAIRS = [tuple(p.split("-", 1)) for p in argv[4].split(";")]

    print(f"\n=== v8 UNIT-INTENT PAIRED EFFICACY  (root={root}, seeds={seeds}) ===\n")
    print(f"{'arm':<12} " + " ".join(f"s{s:<6}" for s in seeds) + "   mean±SE (n)   vs-50%")
    for arm in ARMS:
        cells, rates, tot_w, tot_n = [], [], 0, 0
        for s in seeds:
            c = _ceiling(root, arm, s)
            if c is None:
                cells.append("  --  "); rates.append(None)
            else:
                cells.append(f"{c[0]/c[1]*100:5.1f} "); rates.append(c[0] / c[1]); tot_w += c[0]; tot_n += c[1]
        m, se, n = _mean_se(rates)
        ms = f"{m*100:.1f}%±{se*100:.1f} (n={n})" if (m is not None and se is not None) else \
             (f"{m*100:.1f}% (n={n})" if m is not None else "no data")
        z, pz = _one_prop_z(tot_w, tot_n)
        vs50 = f"z={z:+.2f} p={pz:.3f}" if tot_n else ""
        print(f"{arm:<12} {' '.join(cells)}   {ms}   {vs50}")

    print("\n--- PAIRED per-seed differences (same seed ⇒ controls gen-variance) ---")
    for a, b in PAIRS:
        diffs, m, se, n = _paired(root, a, b, seeds)
        if m is None:
            print(f"  {a} − {b}: no paired data"); continue
        dd = " ".join(f"{d*100:+.1f}" for d in diffs)
        npos = sum(1 for d in diffs if d > 0)
        if se and n >= 2 and se > 0:
            t = m / se
            p = _t_sf(t, n - 1)
            verdict = "SIGNIFICANT (p<0.05)" if p < 0.05 else "within noise"
            print(f"  {a:<8} − {b:<8}: Δ={m*100:+.1f}pp ±{se*100:.1f} (n={n})  {npos}/{n} +  "
                  f"per-seed[{dd}]  t={t:+.2f} p={p:.3f}  [{verdict}]")
        else:
            print(f"  {a:<8} − {b:<8}: Δ={m*100:+.1f}pp (n={n})  per-seed[{dd}]  (need n≥2 for t)")
    print()


if __name__ == "__main__":
    main(sys.argv)
