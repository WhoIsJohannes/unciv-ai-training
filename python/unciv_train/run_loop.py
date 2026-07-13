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

# perf: idle OpenMP pool threads must SLEEP, not spin — 4 concurrent trainers' spin-wait was measured
# starving the sim JVMs (verified bit-identical training numerics). Must be set BEFORE torch/libgomp
# initialize, hence before the `train` import below; setdefault so an explicit env wins.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

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


class JvmWorker:
    """perf: persistent selfPlay JVM (`serve` mode) — ONE gradle+JVM spawn and ONE JIT warmup per job
    instead of one per gen/eval phase (2×rounds spawns; the C2 tax alone was ~4% of each phase's CPU).
    Commands are seed-scoped exactly like one-shot invocations (sessions opened/closed per command,
    hooks uninstalled per Simulation teardown), so results are identical to spawn mode. Any error,
    timeout, or unexpected death kills the worker; the next command respawns a clean JVM."""

    _MARKERS = ("SELFPLAY_GEN_DONE", "EVAL_RESULT", "SERVE_ERROR")

    def __init__(self):
        self._p: subprocess.Popen | None = None

    def _spawn(self, timeout: float) -> None:
        import collections
        env = dict(os.environ)
        env.setdefault("JAVA_HOME", env.get("JAVA_HOME") or DEFAULT_JAVA_HOME)
        cmd = [str(GRADLEW), "selfPlay", "--console=plain", "--args=serve"]
        self._tail = collections.deque(maxlen=400)
        # start_new_session ⇒ own process group, so kill() reaps the gradle→JVM tree, not just gradle
        self._p = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, start_new_session=True)
        self._read_until(("SERVE_READY",), timeout)   # engine bootstrap (rulesets etc.)

    def _read_until(self, prefixes: tuple[str, ...], timeout: float) -> list[str]:
        import select
        deadline = time.time() + timeout
        lines: list[str] = []
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                self.kill()
                raise RuntimeError(f"serve worker timed out after {timeout:.0f}s waiting for {prefixes} "
                                   f"(tail: {''.join(list(self._tail)[-20:])[-2000:]})")
            ready, _, _ = select.select([self._p.stdout], [], [], min(remaining, 5.0))
            if not ready:
                continue
            line = self._p.stdout.readline()
            if line == "":   # EOF — worker died
                self.kill()
                raise RuntimeError(f"serve worker died (tail: {''.join(list(self._tail)[-20:])[-2000:]})")
            self._tail.append(line)
            lines.append(line)
            if any(line.startswith(p) for p in prefixes):
                return lines

    def run(self, args: list[str], timeout: float) -> str:
        """Run one gen/eval command; returns the output up to its marker line (same contract as
        gradle_selfplay: the caller regex-scans for SELFPLAY_GEN_DONE / EVAL_RESULT)."""
        if self._p is None or self._p.poll() is not None:
            self._spawn(timeout)
        try:
            self._p.stdin.write(" ".join(args) + "\n")
            self._p.stdin.flush()
            lines = self._read_until(self._MARKERS, timeout)
            if lines[-1].startswith("SERVE_ERROR"):
                self.kill()   # the worker exits(1) after SERVE_ERROR by design — never reuse it
                raise RuntimeError(f"serve command failed for args={args}: {lines[-1].strip()}")
            self._read_until(("SERVE_READY",), 120.0)   # drain to idle so the next command starts clean
            return "".join(lines)
        except Exception:
            self.kill()
            raise

    def kill(self) -> None:
        import signal
        p, self._p = self._p, None
        if p is None or p.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            p.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass

    def close(self) -> None:
        if self._p is not None and self._p.poll() is None:
            try:
                self._p.stdin.write("quit\n")
                self._p.stdin.flush()
                self._p.wait(timeout=30)
            except Exception:
                pass
        self.kill()


def _selfplay(args: list[str], timeout: float, worker: "JvmWorker | None") -> str:
    """Route one gen/eval command through the persistent worker when given, else one-shot gradle."""
    return worker.run(args, timeout) if worker is not None else gradle_selfplay(args, timeout)


def generate(model: str, out_dir: Path, n: int, max_turns: int, threads: int, seed: int,
             timeout: float, map_size: str = "Tiny", control_construction: bool = False,
             control_unit_intent: bool = False, worker: "JvmWorker | None" = None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = _selfplay(["gen", model, str(out_dir), str(n), str(max_turns), str(threads),
                     str(seed), map_size, str(control_construction).lower(),
                     str(control_unit_intent).lower()], timeout, worker)
    m = GEN_RE.search(out)
    if not m:
        sys.stderr.write(out[-4000:])
        raise RuntimeError("gen produced no SELFPLAY_GEN_DONE marker")
    return int(m.group(1))


def evaluate(model: Path, m_games: int, max_turns: int, threads: int, seed: int,
             timeout: float, map_size: str = "Tiny", control_construction: bool = False,
             control_unit_intent: bool = False, worker: "JvmWorker | None" = None) -> dict:
    out = _selfplay(["eval", str(model), str(m_games), str(max_turns), str(threads),
                     str(seed), map_size, str(control_construction).lower(),
                     str(control_unit_intent).lower()], timeout, worker)
    m = EVAL_RE.search(out)
    if not m:
        sys.stderr.write(out[-4000:])
        raise RuntimeError("eval produced no EVAL_RESULT line")
    return json.loads(m.group(1))


def _atomic_torch_save(obj, path):
    """v5: crash-safe checkpoint write (mirrors export_onnx's atomic pattern) — tmp sibling → os.replace.
    A crash mid-write never leaves a half-written ckpt/opt that --resume would then load."""
    import torch
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _load_warm(variant: str, dims, schema_path, args, out: Path, start: int):
    """v5 --resume (continual): rebuild the net arch for `variant`, load ckpt_round_{start-1}.pt +
    opt_round_{start-1}.pt (weights_only=True). Requires BOTH sidecars (fail-fast); load_state_dict
    errors loudly on a rung/dims mismatch — no silent fresh optimizer."""
    import torch
    from . import contract
    ckpt = out / f"ckpt_round_{start - 1}.pt"
    optf = out / f"opt_round_{start - 1}.pt"
    if not ckpt.is_file() or not optf.is_file():
        raise FileNotFoundError(
            f"[resume] continual restart needs BOTH {ckpt.name} and {optf.name} in {out} "
            f"(found ckpt={ckpt.is_file()} opt={optf.is_file()}). A pre-v5 run has no opt sidecar — "
            f"run with --no-continual or start fresh from round 0.")
    if variant == "blind-critic":
        from .model import PolicyNet
        net = PolicyNet(dims)
    elif variant == "rich-critic":
        from .model import RichPolicyValueNet
        net = RichPolicyValueNet(dims, contract.token_specs_from_schema(schema_path))
    elif variant == "structured":
        from .model import RUNGS, StructuredPolicyValueNet
        net = StructuredPolicyValueNet(dims, contract.token_specs_from_schema(schema_path),
                                       contract.vocab_counts_from_schema(schema_path), **RUNGS[args.rung])
    else:
        raise ValueError(f"--continual --resume unsupported for variant {variant!r}")
    net.load_state_dict(torch.load(ckpt, weights_only=True))   # raises on shape/rung mismatch
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    opt.load_state_dict(torch.load(optf, weights_only=True))
    print(f"[resume] warm-loaded net+opt from {ckpt.name} + {optf.name}")
    return net, opt


def _replay_refill_rounds(start_round: int, replay_window: int) -> list[int]:
    """v6 (plan council 🔴) — the round indices whose shards refill the replay deque on --resume.

    Mirrors the in-process window EXACTLY: the recent NON-zero rounds `max(1, start-(K-1)) .. start-1`.
    Round 0 (RandomPolicy, maximally off-policy) is EXCLUDED — a naive `[start-K .. start-1]` glob would
    re-admit it when start ≤ K. K≤1 ⇒ no replay ⇒ [].
    """
    if replay_window <= 1:
        return []
    lo = max(1, start_round - (replay_window - 1))
    return list(range(lo, start_round))


def train_round(variant: str, trajectories_or_steps, dims, schema_path, args, seed: int,
                *, net=None, optimizer=None, replay_active: bool | None = None, bc_ref=None,
                device: str = "cpu"):
    """Dispatch to the right trainer by variant. Returns (net, stats, exporter_callable, optimizer).
    v5: `net`/`optimizer` are the warm continual pair (None ⇒ fresh round); the trainer returns the
    optimizer so run_loop can carry it forward. micro_batch_steps is threaded from args.
    v6: `replay_active` (per-round; None ⇒ derive from --replay-window) gates the stored-logp source.
    perf: `device` is threaded to the trainers; nets/checkpoints/export remain CPU-side."""
    from . import contract
    mb = getattr(args, "micro_batch_steps", 0) or None
    # v6: window-gated source switch — replay active ⇒ use the STORED behavior logp as old_logp for
    # the (off-policy) replayed steps; else behavior_logp=False ⇒ literal v5 recompute path.
    bl = replay_active if replay_active is not None else (getattr(args, "replay_window", 1) > 1)
    rsc = getattr(args, "reward_shaping_coef", 0.0)   # v7.2: PBRS coefficient (0 ⇒ terminal-only)
    ccc = getattr(args, "construction_credit_coef", 0.0)   # v7.3: per-city economy-advantage weight (0 ⇒ shared-adv)
    if variant == "v1-reinforce":   # v1 is not continual — ignores warm net/opt
        net, stats = tr.train_reinforce(trajectories_or_steps, dims, epochs=args.epochs,
                                        lr=args.lr, seed=seed, entropy_coef=args.entropy_coef)
        return net, stats, "blind", None
    if variant == "blind-critic":
        net, stats, optimizer = tr.train_actor_critic_blind(
            trajectories_or_steps, dims, epochs=args.epochs, lr=args.lr, seed=seed,
            gamma=args.gamma, lam=args.lam, value_coef=args.value_coef,
            entropy_coef=args.entropy_coef, clip_eps=args.clip_eps,
            net=net, optimizer=optimizer, micro_batch_steps=mb, behavior_logp=bl, reward_shaping_coef=rsc,
            device=device)
        return net, stats, "blind", optimizer
    if variant == "rich-critic":
        token_specs = contract.token_specs_from_schema(schema_path)
        net, stats, optimizer = tr.train_actor_critic_rich(
            trajectories_or_steps, dims, token_specs, epochs=args.epochs, lr=args.lr, seed=seed,
            gamma=args.gamma, lam=args.lam, value_coef=args.value_coef,
            entropy_coef=args.entropy_coef, clip_eps=args.clip_eps,
            net=net, optimizer=optimizer, micro_batch_steps=mb, behavior_logp=bl, reward_shaping_coef=rsc,
            device=device)
        return net, stats, ("rich", token_specs), optimizer
    if variant == "structured":
        from .model import RUNGS
        token_specs = contract.token_specs_from_schema(schema_path)
        vocab_counts = contract.vocab_counts_from_schema(schema_path)
        rung = RUNGS[args.rung]
        net, stats, optimizer = tr.train_actor_critic_structured(
            trajectories_or_steps, dims, token_specs, vocab_counts, rung, epochs=args.epochs,
            lr=args.lr, seed=seed, gamma=args.gamma, lam=args.lam, value_coef=args.value_coef,
            entropy_coef=args.entropy_coef, clip_eps=args.clip_eps,
            net=net, optimizer=optimizer, micro_batch_steps=mb, behavior_logp=bl, reward_shaping_coef=rsc,
            construction=(getattr(args, "control_construction", "off") == "on"),   # v7: train the per-city head only when ON
            construction_credit_coef=ccc,   # v7.3: >0 ⇒ per-city credit (separate per-city construction PG term)
            bc_ref=bc_ref, construction_kl_coef=getattr(args, "construction_kl_coef", 0.0),  # v7.4 KL-to-clone leash
            unit_intent=(getattr(args, "control_unit_intent", "off") == "on"),   # v8: train the per-unit intent head only when ON
            unit_kl_coef=getattr(args, "unit_kl_coef", 0.0),  # v8 KL-to-clone leash for the unit-intent head
            device=device)
        return net, stats, ("structured", token_specs), optimizer
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
    try:  # best-effort: a curve PNG is nice-to-have, not required for the run
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed — skipping curve PNG (run unaffected)")
        return
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
    ap.add_argument("--rung", choices=["small", "medium", "large"], default="small",
                    help="v4 structured-encoder capacity rung (D7 ladder)")
    ap.add_argument("--variant", choices=["v1-reinforce", "blind-critic", "rich-critic", "structured", "rich-v2"],
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
    ap.add_argument("--continual", action=argparse.BooleanOptionalAction, default=True,
                    help="v5: carry (net, optimizer) across rounds (warm-start). --no-continual ⇒ "
                         "fresh net+opt every round (v4 from-scratch; for rollback / regime A/B)")
    ap.add_argument("--micro-batch-steps", type=int, default=0,
                    help="v5: chunk the dense forward/backward into K-step sub-batches (0 ⇒ whole-batch "
                         "no-op). Set on the medium rung on Medium to avoid OOM. Changes ONLY traversal.")
    ap.add_argument("--replay-window", type=int, default=4,
                    help="v6: recent-round replay window K. Each update trains on the current round ∪ the "
                         "last K-1 rounds' trajectories (K=4 ≈ 64 games vs the v5 ~16 — a sample-efficiency "
                         "multiplier; the PPO clip + small window keep the importance-ratio variance bounded "
                         "as the warm-started policy drifts only a few hundred grad-steps over K rounds). "
                         "K=1 ⇒ NO replay (bit-identical to v5: window-gated old_logp recompute, no behavior "
                         "logp). Round 0 (RandomPolicy) is always excluded from the window.")
    ap.add_argument("--control-construction", choices=["on", "off"], default="off",
                    help="v7: when 'on', the policy DRIVES each deciding city's production (per-city "
                         "construction head) in gen+eval AND the trainer sums the construction logp into "
                         "the joint PPO ratio. 'off' ⇒ construction stays heuristic (the v6 / no-op path). "
                         "Only the STRUCTURED variant carries the head.")
    ap.add_argument("--control-unit-intent", choices=["on", "off"], default="off",
                    help="v8: when 'on', the policy DRIVES each controlled land-military unit's INTENT (which "
                         "existing UnitAutomation behaviour to run; pathfinding stays heuristic) in gen+eval "
                         "AND the trainer sums the unit-intent logp into the joint PPO ratio (shared advantage). "
                         "'off' ⇒ units stay fully heuristic (the v7 / no-op path). Only STRUCTURED carries the head.")
    ap.add_argument("--reward-shaping-coef", type=float, default=0.0,
                    help="v7.2: potential-based reward shaping coefficient. Adds F = coef·(γ·Φ(s')−Φ(s)) "
                         "to each step, where Φ is the recorded log-stabilized economy potential. PBRS is "
                         "policy-invariant (Ng-Harada) — it shortens the credit horizon for long-payoff "
                         "decisions (construction) WITHOUT changing the optimal 'win the game' policy. "
                         "0 ⇒ terminal-only (unchanged). Try ~0.1.")
    ap.add_argument("--construction-credit-coef", type=float, default=0.0,
                    help="v7.3: per-city credit weight. When >0, construction is pulled OUT of the joint "
                         "PPO ratio and trained by a SEPARATE per-city policy-gradient term whose advantage "
                         "is shared_adv + coef·A_city, where A_city is a per-city GAE over the recorded "
                         "per-city log-economy (baseline = a per-city value head). This credits each city's "
                         "construction by ITS OWN economy return (the attribution fix). 0 ⇒ legacy shared-adv "
                         "(construction in the joint ratio). Requires --control-construction on. Use "
                         "--replay-window 1 (the per-city term is on-policy). Try ~0.5.")
    ap.add_argument("--bc-pretrain-dir", type=str, default=None,
                    help="v7.4: directory of HEURISTIC-gen shards (control-construction OFF, schema 8 with "
                         "construction_current). Before RL round 0, supervised behavior-clone the construction "
                         "head to the heuristic's per-city picks, so it starts ~heuristic instead of collapsing "
                         "below random. Requires --continual --control-construction on. Raise --entropy-coef to "
                         "leash the finetune against re-collapse.")
    ap.add_argument("--bc-epochs", type=int, default=8, help="v7.4: behavior-cloning epochs.")
    ap.add_argument("--construction-kl-coef", type=float, default=0.0,
                    help="v7.4: KL-to-clone leash weight. When >0 (with --bc-pretrain-dir), penalize "
                         "KL(current construction || frozen BC clone) each RL epoch, anchoring the head "
                         "near the clone so finetune can't drift it back into the collapse basin. Try ~0.5.")
    ap.add_argument("--unit-kl-coef", type=float, default=0.0,
                    help="v8: KL-to-clone leash weight for the per-unit intent head. When >0 (with "
                         "--bc-pretrain-dir), penalize KL(current unit-intent || frozen BC clone) each RL "
                         "epoch — the direct analog of --construction-kl-coef for units. Try ~0.5.")
    ap.add_argument("--gradle-timeout", type=float, default=1800.0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="perf: training device. 'auto' (default) uses CUDA when available — verified "
                         "~145x per epoch on this box's RTX 5090 vs CPU whole-batch — else CPU. "
                         "Checkpoints/ONNX export are always CPU-side. A CUDA OOM (e.g. several "
                         "concurrent whole-batch trainers) falls back to CPU for that call, loudly.")
    ap.add_argument("--torch-threads", type=int, default=int(os.environ.get("TORCH_THREADS", "8")),
                    help="perf: torch intra-op thread cap (default $TORCH_THREADS or 8). Uncapped "
                         "torch (32 threads/trainer) was measured starving the sim JVMs to ~2-3 of "
                         "their 6 requested cores when several experiment arms run concurrently.")
    ap.add_argument("--jvm-worker", choices=["serve", "spawn"], default="serve",
                    help="perf: 'serve' (default) keeps ONE persistent selfPlay JVM for all gen/eval "
                         "phases (no per-phase gradle+JVM spawn or cold JIT; identical seed-scoped "
                         "results). 'spawn' restores the legacy one-JVM-per-phase invocation.")
    args = ap.parse_args(argv)
    # perf: cap trainer parallelism BEFORE any tensor work. interop must be set before the first
    # parallel op and raises if torch already initialized it — guard, don't crash a run over it.
    import torch
    torch.set_num_threads(max(1, args.torch_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if args.variant == "rich-v2":   # alias for the v4 structured encoder
        args.variant = "structured"
    # v6: keep enough recent round_*/ shard dirs alive that the last K-1 rounds survive the prune for
    # the replay window (the in-process deque is lost on --resume and refilled from these dirs).
    args.keep_shards = max(args.keep_shards, args.replay_window - 1)

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

    # perf: persistent selfPlay JVM for every gen/eval phase of this job (closed at process exit —
    # atexit also fires on the loud abort paths, so no gradle/JVM orphans survive a failed run).
    worker = None
    if args.jvm_worker == "serve":
        import atexit
        worker = JvmWorker()
        atexit.register(worker.close)

    warm_net = warm_opt = None       # v5: the persistent continual (net, optimizer) pair
    bc_ref = None                    # v7.4: frozen BC-clone KL-leash reference (set in the BC block)
    # v7.4 BEHAVIOR-CLONING WARM-START: before RL, supervised-pretrain the construction head to mimic the
    # heuristic's picks (recorded in the --bc-pretrain-dir shards, gen'd with construction OFF). Gives the
    # head a non-collapsed ~heuristic start so RL has a positive-advantage foothold instead of collapsing
    # below random. Only on a fresh run (start_round==0), structured variant. The BC net becomes round 0's
    # warm net; RL then finetunes it (raise --entropy-coef to leash against re-collapse).
    bc_dir = getattr(args, "bc_pretrain_dir", None)
    if bc_dir and start_round == 0 and args.variant == "structured":
        import torch
        from .model import RUNGS, StructuredPolicyValueNet
        bc_dir = Path(bc_dir)
        bc_schema = bc_dir / "schema.json"
        bc_dims = contract.dims_from_schema(bc_schema)
        bc_ts = contract.token_specs_from_schema(bc_schema)
        bc_vc = contract.vocab_counts_from_schema(bc_schema)
        bc_ver = contract.schema_version_from_schema(bc_schema)
        bc_fp = contract.fingerprint_from_schema(bc_schema)
        bc_ch = contract.spatial_channels_from_schema(bc_schema)
        bc_shards = glob.glob(str(bc_dir / "*.bin"))
        bc_trajs = ds.load_trajectories(bc_shards, expected_version=bc_ver, expected_fingerprint=bc_fp,
                                        rich=True, expected_spatial_channels=bc_ch)
        torch.manual_seed(0)
        bc_net = StructuredPolicyValueNet(bc_dims, bc_ts, bc_vc, **RUNGS[args.rung])
        # v8: bc_pretrain_construction clones BOTH the construction AND unit-intent heads in one pass (when the
        # net carries the unit-intent head) — one warm-start serves both heads + both KL leashes.
        try:
            bc_net, bc_stats = tr.bc_pretrain_construction(
                bc_net, bc_trajs, bc_dims, bc_ts, epochs=getattr(args, "bc_epochs", 8), lr=args.lr,
                micro_batch_steps=(args.micro_batch_steps or 512), device=args.device)
        except torch.cuda.OutOfMemoryError:
            sys.stderr.write("[bc] CUDA OOM — falling back to CPU for the BC pretrain\n")
            torch.cuda.empty_cache()
            bc_net = bc_net.cpu()
            bc_net, bc_stats = tr.bc_pretrain_construction(
                bc_net, bc_trajs, bc_dims, bc_ts, epochs=getattr(args, "bc_epochs", 8), lr=args.lr,
                micro_batch_steps=(args.micro_batch_steps or 512), device="cpu")
        _uacc = bc_stats.get("bc_unit_acc")
        print(f"[bc] pretrained per-entity heads on {len(bc_trajs)} heuristic trajectories "
              f"({bc_stats['n']} steps): bc_loss={bc_stats['bc_loss']:.4f} bc_acc={bc_stats['bc_acc']:.3f}"
              + (f" bc_unit_acc={_uacc:.3f}" if _uacc is not None else ""))
        warm_net = bc_net                # round 0 RL starts from the BC-warm net
        # v7.4/v8 KL leash: freeze a copy of the clone as the KL reference (anti-drift during RL finetune).
        if getattr(args, "construction_kl_coef", 0.0) > 0 or getattr(args, "unit_kl_coef", 0.0) > 0:
            import copy
            bc_ref = copy.deepcopy(bc_net).eval()
            for p in bc_ref.parameters():
                p.requires_grad_(False)
            print(f"[bc] KL-to-clone leash active (construction_kl={getattr(args, 'construction_kl_coef', 0.0)} "
                  f"unit_kl={getattr(args, 'unit_kl_coef', 0.0)})")
    # v6: recent-round replay window. The deque's maxlen gives the sliding window for free (appending the
    # (K+1)-th round evicts the oldest). K=1 ⇒ maxlen 1 ⇒ degenerates to the single-round batch (== v5).
    from collections import deque
    replay: deque = deque(maxlen=max(1, args.replay_window))
    if args.resume and args.variant != "v1-reinforce" and args.replay_window > 1:
        # The in-process deque is empty on restart — refill it from disk over the SAME non-zero window.
        for rr in _replay_refill_rounds(start_round, args.replay_window):
            rdir = out / f"round_{rr}"
            rshards = glob.glob(str(rdir / "*.bin"))
            if not rshards:
                print(f"[resume] replay refill: round_{rr} shards missing — skipping (window under-filled)")
                continue
            rschema = rdir / "schema.json"
            is_rich_r = args.variant in ("rich-critic", "structured")
            try:   # ship-council 🔴: a corrupt/unreadable kept shard must NOT kill the multi-hour resume
                replay.append(ds.load_trajectories(
                    rshards, expected_version=contract.schema_version_from_schema(rschema),
                    expected_fingerprint=contract.fingerprint_from_schema(rschema), rich=is_rich_r,
                    expected_spatial_channels=(contract.spatial_channels_from_schema(rschema) if is_rich_r else None)))
            except Exception as e:
                print(f"[resume] replay refill: round_{rr} failed to load ({e!r}) — skipping (window under-filled)")
        print(f"[resume] replay refilled {len(replay)} round(s) for window K={args.replay_window}")
    for r in range(start_round, args.rounds):
        t0 = time.time()
        round_dir = out / f"round_{r}"
        gen_model = str(model_path) if model_path is not None else "random"
        n_games = generate(gen_model, round_dir, args.gen_games, args.turn_cap, args.threads,
                           args.gen_seed + r * 1000, args.gradle_timeout, args.map_size,
                           control_construction=(args.control_construction == "on"),
                           control_unit_intent=(args.control_unit_intent == "on"), worker=worker)

        schema = round_dir / "schema.json"
        dims = contract.dims_from_schema(schema)
        fp = contract.fingerprint_from_schema(schema)
        ver = contract.schema_version_from_schema(schema)
        shards = glob.glob(str(round_dir / "*.bin"))

        if args.variant == "v1-reinforce":
            data = ds.load_training_steps(shards, expected_version=ver, expected_fingerprint=fp)
        else:
            is_rich = args.variant in ("rich-critic", "structured")
            data = ds.load_trajectories(
                shards, expected_version=ver, expected_fingerprint=fp, rich=is_rich,
                expected_spatial_channels=(contract.spatial_channels_from_schema(schema)
                                           if is_rich else None))
        # v6: assemble the replay batch. Round 0 (RandomPolicy) is EXCLUDED from the window — its policy
        # is maximally off the current net (high-variance ratios) and contributes little; it trains on its
        # own data directly. For r≥1 (and the trajectory variants), append this round and flatten the
        # deque (current ∪ last K-1 rounds) into one list — each trajectory carries its own behavior_logp
        # so stored_old_logp is correct per-step regardless of source round. v1-reinforce is single-round.
        frac_replayed = 0.0
        if args.variant != "v1-reinforce" and r > 0:
            replay.append(data)
            train_data = [t for round_trajs in replay for t in round_trajs]
            if train_data:
                frac_replayed = 1.0 - (len(data) / len(train_data))
        else:
            train_data = data
        if args.continual and warm_net is None and r > 0:   # --resume restart: load warm net+opt from disk
            warm_net, warm_opt = _load_warm(args.variant, dims, schema, args, out, r)
        # v6 (ship-council): round 0 (RandomPolicy bootstrap) ALWAYS trains on-policy (recompute) — its
        # stored logp is the uniform RandomPolicy logp, and forcing recompute keeps round 0 IDENTICAL across
        # the K=1 and K=4 arms (the v5 bootstrap), isolating the replay effect to rounds ≥1.
        replay_active = (args.variant != "v1-reinforce" and args.replay_window > 1 and r > 0)
        try:
            net, stats, mode, opt = train_round(
                args.variant, train_data, dims, schema, args, seed=r,
                net=(warm_net if args.continual else None),
                optimizer=(warm_opt if args.continual else None),
                replay_active=replay_active, bc_ref=bc_ref, device=args.device)
        except torch.cuda.OutOfMemoryError:
            # perf: several concurrent whole-batch trainers can exceed VRAM — retry THIS round on CPU
            # (weights may have taken some GPU epochs' steps already; the pipeline is not bit-repro
            # across devices anyway, and a loud fallback beats a dead multi-hour run).
            sys.stderr.write(f"[r{r}] CUDA OOM — falling back to CPU for this round's training\n")
            torch.cuda.empty_cache()
            if warm_net is not None:
                warm_net = warm_net.cpu()
            net, stats, mode, opt = train_round(
                args.variant, train_data, dims, schema, args, seed=r,
                net=(warm_net if args.continual else None),
                optimizer=(warm_opt if args.continual else None),
                replay_active=replay_active, bc_ref=bc_ref, device="cpu")
        if args.continual:
            warm_net, warm_opt = net, opt                   # carry the persistent pair to next round

        # checkpoint net + optimizer (atomic; weights_only=True on resume — R7). opt sidecar = v5.
        _atomic_torch_save(net.state_dict(), out / f"ckpt_round_{r}.pt")
        if opt is not None:
            _atomic_torch_save(opt.state_dict(), out / f"opt_round_{r}.pt")

        model_path = out / f"policy_round_{r}.onnx"
        if mode == "blind":
            ex.export(net, dims, model_path, schema_version=ver, ruleset_fingerprint=fp)
        else:  # ("rich"|"structured", token_specs)
            kind, token_specs = mode
            from .features import build_rich_batch
            sample = None
            if data:
                sample = {k: v[:1] for k, v in build_rich_batch(data[:1], dims, token_specs).items()}
                sample = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in sample.items()}
            if kind == "structured":   # v3: emit the int64 neighbor inputs + stamp contract_version=3
                ex.export_rich(net, dims, token_specs, model_path, schema_version=ver,
                               ruleset_fingerprint=fp, sample_inputs=sample, neighbors=True,
                               contract_version=contract.CONTRACT_VERSION_STRUCTURED)
            else:
                ex.export_rich(net, dims, token_specs, model_path, schema_version=ver,
                               ruleset_fingerprint=fp, sample_inputs=sample)

        ev = evaluate(model_path, args.eval_games, args.turn_cap, args.threads, args.eval_seed,
                      args.gradle_timeout, args.map_size,
                      control_construction=(args.control_construction == "on"),
                      control_unit_intent=(args.control_unit_intent == "on"), worker=worker)
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
                                "rung": args.rung,
                                "continual": bool(args.continual),
                                "warm_start": bool(args.continual and r > 0),
                                "micro_batch_steps": int(args.micro_batch_steps),
                                "turns_per_sec": ev.get("turns_per_sec"),
                                "ms_per_decision": ev.get("ms_per_decision"),
                                "mean_adv": stats.get("mean_adv", 0.0),
                                # v6 off-policy-health diagnostics (read-only; let the multi-hour run be watched)
                                "replay_window": int(args.replay_window),
                                "frac_replayed": float(frac_replayed),
                                "mean_ratio": stats.get("mean_ratio", 1.0),
                                "clip_frac": stats.get("clip_frac", 0.0)}) + "\n")
        plot(rows, curve_png, args.variant, args.map_size)
        print(f"[{args.variant} r{r}] gen={n_games} steps={row['n_steps']} "
              f"winrate={row['winrate']*100:.1f}% pval={row['pval']:.3g} "
              f"vloss={row['value_loss']:.3f} ent={row['entropy']:.2f} "
              f"onnx_dec={row['onnx_decisions']} diverged={row['diverged']} "
              f"({time.time()-t0:.0f}s)", flush=True)
        # v6 (ship-council): surface growing off-policy variance loudly, not just on disk. mean_ratio≈1 is
        # healthy near-on-policy; a high mean ratio (or heavy clipping) means the replay window is stale.
        mr, cf = stats.get("mean_ratio", 1.0), stats.get("clip_frac", 0.0)
        if replay_active and (mr > 1.5 or cf > 0.5):
            sys.stderr.write(f"[replay-health WARN] r{r}: mean_ratio={mr:.3f} clip_frac={cf:.3f} — "
                             f"off-policy variance high; consider lowering --replay-window or --clip-eps.\n")

        if args.keep_shards >= 0:
            old = out / f"round_{r - args.keep_shards}"
            if old.is_dir():
                shutil.rmtree(old, ignore_errors=True)
            # v5: prune stale ckpt/opt sidecars, keeping the last 3 (resume needs round r-1). Best-effort
            # — a prune failure must NEVER crash a multi-hour run (ship-council FND-0012).
            for stale in (out / f"ckpt_round_{r - 3}.pt", out / f"opt_round_{r - 3}.pt"):
                try:
                    stale.unlink(missing_ok=True)
                except OSError:
                    pass

        # v5: drop this round's big batch tensors before the next gen (memory insurance over 16 rounds;
        # the persistent net+opt are small and intentionally retained) — ship-council FND-0013/0033.
        import gc
        gc.collect()

    v = verdict(rows)
    print(f"\nVERDICT: {v}  (curve: {curve_csv} | plot: {curve_png})")
    (out / "verdict.txt").write_text(f"{v}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
