"""Round-based self-play driver: gen → train → export → eval, K rounds → curve + verdict.

Round 0 bootstraps with RandomPolicy generation; each round r≥1 generates with the current
`policy.onnx`. Writes `curve.csv` (round, games, winrate, pval) + `curve.png`, and prints an explicit
GO / PLATEAU / INCONCLUSIVE verdict. The JVM is invoked via `./gradlew selfPlay` as an argument LIST
(no shell) with a per-call timeout; a failed/timed-out round aborts the loop loudly.

Run from anywhere:  python -m unciv_train.run_loop --rounds 10 --gen-games 24 --eval-games 100
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import contract, dataset as ds, export_onnx as ex, train as tr  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
GRADLEW = REPO / "gradlew"
DEFAULT_JAVA_HOME = "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
EVAL_RE = re.compile(r"EVAL_RESULT\s+(\{.*\})")
GEN_RE = re.compile(r"SELFPLAY_GEN_DONE\s+games=(\d+)")


def gradle_selfplay(args: list[str], timeout: float) -> str:
    """Invoke the selfPlay task with an argument LIST (no shell). Returns combined stdout/stderr.
    Raises on a non-zero gradle build or a timeout."""
    env = dict(os.environ)
    env.setdefault("JAVA_HOME", env.get("JAVA_HOME") or DEFAULT_JAVA_HOME)
    cmd = [str(GRADLEW), "selfPlay", "--console=plain", "--args=" + " ".join(args)]
    p = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=timeout)
    out = p.stdout + p.stderr
    if p.returncode != 0:
        sys.stderr.write(out[-4000:])
        raise RuntimeError(f"gradle selfPlay failed (rc={p.returncode}) for args={args}")
    return out


def generate(model: str, out_dir: Path, n: int, max_turns: int, threads: int, seed: int, timeout: float) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = gradle_selfplay(["gen", model, str(out_dir), str(n), str(max_turns), str(threads), str(seed)], timeout)
    m = GEN_RE.search(out)
    if not m:
        sys.stderr.write(out[-4000:])
        raise RuntimeError("gen produced no SELFPLAY_GEN_DONE marker")
    return int(m.group(1))


def evaluate(model: Path, m_games: int, max_turns: int, threads: int, seed: int, timeout: float) -> dict:
    out = gradle_selfplay(["eval", str(model), str(m_games), str(max_turns), str(threads), str(seed)], timeout)
    m = EVAL_RE.search(out)
    if not m:
        sys.stderr.write(out[-4000:])
        raise RuntimeError("eval produced no EVAL_RESULT line")
    return json.loads(m.group(1))


def verdict(rows: list[dict]) -> str:
    if not rows:
        return "INCONCLUSIVE"
    last = rows[-1]
    if last["winrate"] >= 0.60 and last["pval"] < 0.05:
        return "GO"
    tail = rows[-3:]
    if len(tail) >= 3 and all(0.45 <= r["winrate"] <= 0.55 for r in tail) and tail[-1]["winrate"] <= tail[0]["winrate"] + 0.02:
        return "PLATEAU"
    return "INCONCLUSIVE"


def plot(rows: list[dict], png: Path) -> None:
    rounds = [r["round"] for r in rows]
    wr = [r["winrate"] * 100 for r in rows]
    plt.figure(figsize=(7, 4))
    plt.axhline(50, color="gray", ls="--", lw=1, label="RandomPolicy baseline (50%)")
    plt.axhline(60, color="green", ls=":", lw=1, label="target (60%)")
    plt.plot(rounds, wr, "-o", color="C0", label="OnnxPolicy win-rate")
    plt.xlabel("round"); plt.ylabel("win-rate vs RandomPolicy (%)")
    plt.title("Self-play learning curve (Tiny GnK, learner vs RandomPolicy)")
    plt.ylim(0, 100); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(png, dpi=110); plt.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Self-play round loop → learning curve")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--gen-games", type=int, default=24)
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--turn-cap", type=int, default=1000)  # high: games play to a real victory; score-leader is the rare fallback
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--gen-seed", type=int, default=1000)
    ap.add_argument("--eval-seed", type=int, default=999000)
    ap.add_argument("--out", type=str, default=str(REPO / "training-runs" / "run"))
    ap.add_argument("--keep-shards", type=int, default=2, help="retain only the last N rounds' shard dirs")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gradle-timeout", type=float, default=1800.0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    curve_csv, curve_png = out / "curve.csv", out / "curve.png"
    rows: list[dict] = []
    model_path: Path | None = None  # None ⇒ round-0 RandomPolicy generation

    with open(curve_csv, "w", newline="") as f:
        csv.writer(f).writerow(["round", "games", "winrate", "pval", "n_steps", "loss", "ret_pos", "onnx_decisions"])

    for r in range(args.rounds):
        t0 = time.time()
        round_dir = out / f"round_{r}"
        gen_model = str(model_path) if model_path is not None else "random"
        n_games = generate(gen_model, round_dir, args.gen_games, args.turn_cap, args.threads,
                           args.gen_seed + r * 1000, args.gradle_timeout)

        schema = round_dir / "schema.json"
        dims = contract.dims_from_schema(schema)
        fp = contract.fingerprint_from_schema(schema)
        ver = contract.schema_version_from_schema(schema)
        steps = ds.load_training_steps(glob.glob(str(round_dir / "*.bin")),
                                       expected_version=ver, expected_fingerprint=fp)
        net, stats = tr.train(steps, dims, epochs=args.epochs, lr=args.lr, seed=r)

        model_path = out / f"policy_round_{r}.onnx"
        ex.export(net, dims, model_path, schema_version=ver, ruleset_fingerprint=fp)

        ev = evaluate(model_path, args.eval_games, args.turn_cap, args.threads, args.eval_seed, args.gradle_timeout)
        row = {"round": r, "games": ev["games"], "winrate": ev["winrate"], "pval": ev["pval"],
               "n_steps": stats["n"], "loss": stats["loss"], "ret_pos": stats.get("ret_pos", 0),
               "onnx_decisions": ev["onnx_decisions"]}
        rows.append(row)
        with open(curve_csv, "a", newline="") as f:
            csv.writer(f).writerow([row["round"], row["games"], f"{row['winrate']:.4f}",
                                    f"{row['pval']:.4g}", row["n_steps"], f"{row['loss']:.4f}",
                                    row["ret_pos"], row["onnx_decisions"]])
        plot(rows, curve_png)
        print(f"[round {r}] gen_games={n_games} train_steps={stats['n']} "
              f"winrate={row['winrate']*100:.1f}% pval={row['pval']:.3g} "
              f"onnx_decisions={row['onnx_decisions']} ({time.time()-t0:.0f}s)", flush=True)

        # shard retention: drop round dirs older than the keep window
        if args.keep_shards >= 0:
            old = out / f"round_{r - args.keep_shards}"
            if old.is_dir():
                shutil.rmtree(old, ignore_errors=True)

    v = verdict(rows)
    print(f"\nVERDICT: {v}  (curve: {curve_csv} | plot: {curve_png})")
    (out / "verdict.txt").write_text(f"{v}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
