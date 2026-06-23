"""AC4 PARITY — the anti-drift test: a fixed model + fixed observation must produce the SAME logits
through the JVM onnxruntime session (`selfPlay parity-run`) and a Python onnxruntime reference.

Synthetic small dims are sufficient — parity is about cross-boundary NUMERICAL agreement on one
shared (model, obs), not the specific GnK widths. Skips cleanly if the JVM toolchain is unavailable.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from unciv_train import contract  # noqa: E402
from unciv_train.contract import Dims  # noqa: E402
from unciv_train.export_onnx import export  # noqa: E402
from unciv_train.model import PolicyNet  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
GRADLEW = REPO / "gradlew"
JAVA_HOME = os.environ.get("JAVA_HOME") or "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
ATOL = 1e-4


def _toolchain_ok() -> bool:
    return GRADLEW.is_file() and Path(JAVA_HOME, "bin", "java").is_file()


@pytest.mark.skipif(not _toolchain_ok(), reason="JVM toolchain (gradlew + JDK) not available")
def test_jvm_python_logits_match(tmp_path):
    dims = Dims(global_w=4, acting_w=4, tech_w=5, policy_w=4)
    torch.manual_seed(123)
    net = PolicyNet(dims, hidden=16)
    model = tmp_path / "parity.onnx"
    export(net, dims, model, schema_version=2, ruleset_fingerprint="paritytest")

    rng = np.random.default_rng(0)
    obs = rng.standard_normal(dims.input_w).astype(np.float32)
    obs_file = tmp_path / "obs.csv"
    obs_file.write_text(",".join(repr(float(x)) for x in obs))

    # Python reference
    sess = ort.InferenceSession(str(model))
    py = sess.run([contract.OUTPUT_TECH, contract.OUTPUT_POLICY], {contract.INPUT_NAME: obs[None, :]})
    py_tech, py_policy = py[0][0], py[1][0]

    # JVM side
    out_file = tmp_path / "jvm.json"
    env = dict(os.environ, JAVA_HOME=JAVA_HOME)
    cmd = [str(GRADLEW), "selfPlay", "--console=plain",
           "--args=" + f"parity-run {model} {obs_file} {out_file}"]
    p = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert out_file.is_file(), "JVM parity-run produced no logits file"
    jvm = json.loads(out_file.read_text())
    jvm_tech, jvm_policy = np.array(jvm["tech"], np.float32), np.array(jvm["policy"], np.float32)

    assert np.allclose(py_tech, jvm_tech, atol=ATOL), f"tech logits drift > {ATOL}: {py_tech} vs {jvm_tech}"
    assert np.allclose(py_policy, jvm_policy, atol=ATOL), f"policy logits drift > {ATOL}: {py_policy} vs {jvm_policy}"
