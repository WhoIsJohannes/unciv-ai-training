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


def generate(model: str, out_dir: Path, n: int, max_turns: int, threads: int, seed: int,
             timeout: float, map_size: str = "Tiny") -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = gradle_selfplay(["gen", model, str(out_dir), str(n), str(max_turns), str(threads),
                           str(seed), map_size], timeout)
    m = GEN_RE.search(out)
    if not m:
        sys.stderr.write(out[-4000:])
        raise RuntimeError("gen produced no SELFPLAY_GEN_DONE marker")
    return int(m.group(1))


def evaluate(model: Path, m_games: int, max_turns: int, threads: int, seed: int,
             timeout: float, map_size: str = "Tiny") -> dict:
    out = gradle_selfplay(["eval", str(model), str(m_games), str(max_turns), str(threads),
                           str(seed), map_size], timeout)
    m = EVAL_RE.search(out)
    if not m:
        sys.stderr.write(out[-4000:])
        raise RuntimeError("eval produced no EVAL_RESULT line")
    return json.loads(m.group(1))


def train_round(variant: str, trajectories_or_steps, dims, schema_path, args, seed: int):
    """Dispatch to the right trainer by variant. Returns (net, stats, exporter_callable)."""
    from . import contract
    if variant == "v1-reinforce":
        net, stats = tr.train_reinforce(trajectories_or_steps, dims, epochs=args.epochs,
                                        lr=args.lr, seed=seed, entropy_coef=args.entropy_coef)
        return net, stats, "blind"
    if variant == "blind-critic":
        net, stats = tr.train_actor_critic_blind(
            trajectories_or_steps, dims, epochs=args.epochs, lr=args.lr, seed=seed,
            gamma=args.gamma, lam=args.lam, value_coef=args.value_coef,
            entropy_coef=args.entropy_coef, clip_eps=args.clip_eps)
        return net, stats, "blind"
    if variant == "rich-critic":
        token_specs = contract.token_specs_from_schema(schema_path)
        net, stats = tr.train_actor_critic_rich(
            trajectories_or_steps, dims, token_specs, epochs=args.epochs, lr=args.lr, seed=seed,
            gamma=args.gamma, lam=args.lam, value_coef=args.value_coef,
            entropy_coef=args.entropy_coef, clip_eps=args.clip_eps)
        return net, stats, ("rich", token_specs)
    raise ValueError(f"unknown variant {variant!r}")


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


CURVE_COLS = ["round", "games", "winrate", "pval", "n_steps", "loss", "value_loss",
              "entropy", "mean_value", "grad_norm", "diverged", "ret_pos", "onnx_decisions"]


def plot(rows: list[dict], png: Path, variant: str, map_size: str) -> None:
    rounds = [r["round"] for r in rows]
    wr = [r["winrate"] * 100 for r in rows]
    plt.figure(figsize=(7, 4))
    plt.axhline(50, color="gray", ls="--", lw=1, label="RandomPolicy baseline (50%)")
    plt.axhline(60, color="green", ls=":", lw=1, label="target (60%)")
    plt.plot(rounds, wr, "-o", color="C0", label=f"{variant} win-rate")
    plt.xlabel("round"); plt.ylabel("win-rate vs RandomPolicy (%)")
    plt.title(f"Self-play learning curve ({map_size} GnK · {variant} · vs RandomPolicy)")
    plt.ylim(0, 100); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(png, dpi=110); plt.close()


def _read_existing_rows(curve_csv: Path) -> list[dict]:
    if not curve_csv.is_file():
        return []
    rows = []
    with open(curve_csv, newline="") as f:
        for d in csv.DictReader(f):
            try:
                rows.append({"round": int(d["round"]), "games": int(d["games"]),
                             "winrate": float(d["winrate"]), "pval": float(d["pval"])})
            except (ValueError, KeyError):
                continue
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Self-play round loop → learning curve")
    ap.add_argument("--variant", choices=["v1-reinforce", "blind-critic", "rich-critic"],
                    default="blind-critic", help="algorithm + representation (attributable axis)")
    ap.add_argument("--map-size", default="Tiny", help="Tiny (comparability) or Medium (ceiling test)")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--gen-games", type=int, default=24)
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--turn-cap", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--gen-seed", type=int, default=1000)
    ap.add_argument("--eval-seed", type=int, default=999000)
    ap.add_argument("--out", type=str, default=str(REPO / "training-runs" / "run"))
    ap.add_argument("--keep-shards", type=int, default=2, help="retain only the last N rounds' shard dirs")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--clip-eps", type=float, default=0.2,
                    help="PPO clip epsilon (default 0.2; required because K epochs reuse fixed "
                         "advantages — pass 0 for single-epoch plain A2C)")
    ap.add_argument("--resume", action="store_true", help="skip rounds already in curve.csv")
    ap.add_argument("--gradle-timeout", type=float, default=1800.0)
    args = ap.parse_args(argv)

    import torch  # local import: keep CLI help fast

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    curve_csv, curve_png = out / "curve.csv", out / "curve.png"
    metrics_jsonl = out / "metrics.jsonl"

    rows: list[dict] = []
    start_round = 0
    model_path: Path | None = None
    if args.resume:
        rows = _read_existing_rows(curve_csv)
        start_round = len(rows)
        if start_round > 0:
            cand = out / f"policy_round_{start_round - 1}.onnx"
            model_path = cand if cand.is_file() else None
            print(f"[resume] continuing from round {start_round} ({len(rows)} rows in curve.csv)")
    if start_round == 0:
        with open(curve_csv, "w", newline="") as f:
            csv.writer(f).writerow(CURVE_COLS)

    for r in range(start_round, args.rounds):
        t0 = time.time()
        round_dir = out / f"round_{r}"
        gen_model = str(model_path) if model_path is not None else "random"
        n_games = generate(gen_model, round_dir, args.gen_games, args.turn_cap, args.threads,
                           args.gen_seed + r * 1000, args.gradle_timeout, args.map_size)

        schema = round_dir / "schema.json"
        dims = contract.dims_from_schema(schema)
        fp = contract.fingerprint_from_schema(schema)
        ver = contract.schema_version_from_schema(schema)
        shards = glob.glob(str(round_dir / "*.bin"))

        if args.variant == "v1-reinforce":
            data = ds.load_training_steps(shards, expected_version=ver, expected_fingerprint=fp)
        else:
            data = ds.load_trajectories(shards, expected_version=ver, expected_fingerprint=fp,
                                        rich=(args.variant == "rich-critic"))
        net, stats, mode = train_round(args.variant, data, dims, schema, args, seed=r)

        # checkpoint (state_dict only; loaded with weights_only=True on resume — R7)
        torch.save(net.state_dict(), out / f"ckpt_round_{r}.pt")

        model_path = out / f"policy_round_{r}.onnx"
        if mode == "blind":
            ex.export(net, dims, model_path, schema_version=ver, ruleset_fingerprint=fp)
        else:  # ("rich", token_specs)
            _, token_specs = mode
            from .features import build_rich_batch
            sample = None
            if data:
                sample = {k: v[:1] for k, v in build_rich_batch(data[:1], dims, token_specs).items()}
                sample = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in sample.items()}
            ex.export_rich(net, dims, token_specs, model_path, schema_version=ver,
                           ruleset_fingerprint=fp, sample_inputs=sample)

        ev = evaluate(model_path, args.eval_games, args.turn_cap, args.threads, args.eval_seed,
                      args.gradle_timeout, args.map_size)
        row = {"round": r, "games": ev["games"], "winrate": ev["winrate"], "pval": ev["pval"],
               "n_steps": stats.get("n", 0), "loss": stats.get("loss", 0.0),
               "value_loss": stats.get("value_loss", 0.0), "entropy": stats.get("entropy", 0.0),
               "mean_value": stats.get("mean_value", 0.0), "grad_norm": stats.get("grad_norm", 0.0),
               "diverged": int(bool(stats.get("diverged", False))),
               "ret_pos": stats.get("ret_pos", 0), "onnx_decisions": ev["onnx_decisions"]}
        rows.append(row)
        with open(curve_csv, "a", newline="") as f:
            csv.writer(f).writerow([row[c] if not isinstance(row[c], float) else f"{row[c]:.5g}"
                                    for c in CURVE_COLS])
        with open(metrics_jsonl, "a") as f:
            f.write(json.dumps({**row, "variant": args.variant, "map_size": args.map_size,
                                "mean_adv": stats.get("mean_adv", 0.0)}) + "\n")
        plot(rows, curve_png, args.variant, args.map_size)
        print(f"[{args.variant} r{r}] gen={n_games} steps={row['n_steps']} "
              f"winrate={row['winrate']*100:.1f}% pval={row['pval']:.3g} "
              f"vloss={row['value_loss']:.3f} ent={row['entropy']:.2f} "
              f"onnx_dec={row['onnx_decisions']} diverged={row['diverged']} "
              f"({time.time()-t0:.0f}s)", flush=True)

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
