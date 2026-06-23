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


def _fixture_line_vec(name: str, v: np.ndarray) -> str:
    return f"{name} vec " + " ".join(repr(float(x)) for x in v)


def _fixture_line_set(name: str, arr: np.ndarray, width: int) -> str:
    return f"{name} set {arr.shape[0]} {width} " + " ".join(repr(float(x)) for x in arr.reshape(-1))


@pytest.mark.skipif(not _toolchain_ok(), reason="JVM toolchain (gradlew + JDK) not available")
def test_jvm_python_rich_logits_match(tmp_path):
    """AC4 over the contract-v2 MULTI-TENSOR input: JVM-built tensors + logits must match the Python
    reference for a fixed rich observation (incl. an empty entity set → the masked-pool NaN guard)."""
    from unciv_train.export_onnx import export_rich
    from unciv_train.features import build_rich_single
    from unciv_train.model import RichPolicyValueNet

    dims = Dims(global_w=4, acting_w=4, tech_w=5, policy_w=4)
    token_specs = {"spatial": 3, "own_units": 2, "opp_units": 2,
                   "own_cities": 2, "opp_cities": 2, "civ_tokens": 3}
    torch.manual_seed(7)
    net = RichPolicyValueNet(dims, token_specs, token_dim=8, hidden=16)
    model = tmp_path / "parity_rich.onnx"
    export_rich(net, dims, token_specs, model, schema_version=2, ruleset_fingerprint="paritytest")

    rng = np.random.default_rng(0)
    blocks = {
        "global": rng.standard_normal(4).astype(np.float32),
        "acting_civ": rng.standard_normal(4).astype(np.float32),
        "spatial": rng.standard_normal((7, 3)).astype(np.float32),
        "own_units": rng.standard_normal((2, 2)).astype(np.float32),
        "opp_units": np.zeros((0, 2), np.float32),        # empty set → NaN-guard path
        "own_cities": rng.standard_normal((1, 2)).astype(np.float32),
        "opp_cities": np.zeros((0, 2), np.float32),
        "civ_tokens": rng.standard_normal((2, 3)).astype(np.float32),
    }

    # Python reference: build the same multi-tensor input and run ORT.
    feed = {k: (v.numpy() if hasattr(v, "numpy") else np.asarray(v))
            for k, v in build_rich_single(blocks, token_specs).items()}
    sess = ort.InferenceSession(str(model))
    py = sess.run([contract.OUTPUT_TECH, contract.OUTPUT_POLICY], feed)
    py_tech, py_policy = py[0][0], py[1][0]

    # JVM side: write the shared fixture, run parity-run-rich.
    obs_file = tmp_path / "obs_rich.txt"
    lines = [_fixture_line_vec("global", blocks["global"]),
             _fixture_line_vec("acting_civ", blocks["acting_civ"])]
    for name in ["spatial", "own_units", "opp_units", "own_cities", "opp_cities", "civ_tokens"]:
        lines.append(_fixture_line_set(name, blocks[name], token_specs[name]))
    obs_file.write_text("\n".join(lines) + "\n")

    out_file = tmp_path / "jvm_rich.json"
    env = dict(os.environ, JAVA_HOME=JAVA_HOME)
    cmd = [str(GRADLEW), "selfPlay", "--console=plain",
           "--args=" + f"parity-run-rich {model} {obs_file} {out_file}"]
    p = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert out_file.is_file(), "JVM parity-run-rich produced no logits file"
    jvm = json.loads(out_file.read_text())
    jvm_tech, jvm_policy = np.array(jvm["tech"], np.float32), np.array(jvm["policy"], np.float32)

    assert np.allclose(py_tech, jvm_tech, atol=ATOL), f"rich tech drift > {ATOL}: {py_tech} vs {jvm_tech}"
    assert np.allclose(py_policy, jvm_policy, atol=ATOL), f"rich policy drift > {ATOL}: {py_policy} vs {jvm_policy}"
