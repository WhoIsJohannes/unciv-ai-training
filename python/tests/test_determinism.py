"""AC3 DETERMINISM — a fixed seed reproduces byte-identical trajectory shards.

Generation is seeded end-to-end (the policy RNG is the engine's state-based RNG; the map seed is
`seedBase + per-iteration`), so two GENERATE runs with the same seed produce identical shards →
identical SHA-256. (A fixed `policy.onnx` behaves identically: its sample/argmax RNG is seeded the
same way; `random` is used here to avoid coupling the test to a ruleset fingerprint.) Skips cleanly
if the JVM toolchain is unavailable.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GRADLEW = REPO / "gradlew"
JAVA_HOME = os.environ.get("JAVA_HOME") or "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"


def _toolchain_ok() -> bool:
    return GRADLEW.is_file() and Path(JAVA_HOME, "bin", "java").is_file()


def _gen(out_dir: Path, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, JAVA_HOME=JAVA_HOME)
    cmd = [str(GRADLEW), "selfPlay", "--console=plain",
           "--args=" + f"gen random {out_dir} 2 30 1 {seed}"]
    p = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=600)
    assert "SELFPLAY_GEN_DONE" in (p.stdout + p.stderr), (p.stdout + p.stderr)[-2000:]


def _digests(d: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("*.bin"))}


@pytest.mark.skipif(not _toolchain_ok(), reason="JVM toolchain (gradlew + JDK) not available")
def test_same_seed_byte_identical_shards(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _gen(a, 12345)
    _gen(b, 12345)
    da, db = _digests(a), _digests(b)
    assert da, "no shards produced"
    assert da == db, f"determinism broken: {da} != {db}"
