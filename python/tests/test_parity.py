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


def _fixture_line_adj(name: str, arr: np.ndarray) -> str:
    """<name> adj <N> <deg> <N*deg values> — v3 hex-GNN neighbor_index/neighbor_mask."""
    return f"{name} adj {arr.shape[0]} {arr.shape[1]} " + " ".join(repr(float(x)) for x in arr.reshape(-1))


@pytest.mark.skipif(not _toolchain_ok(), reason="JVM toolchain (gradlew + JDK) not available")
def test_jvm_python_structured_logits_match(tmp_path):
    """AC4 over contract v3 (structured): JVM-built tensors + the hex-GNN neighbor inputs must match
    the Python reference (atol 1e-4). Exercises the wider entity tokens (9/17), the int64
    neighbor_index path, an empty entity set (NaN-guard), and the 255 slot reindex."""
    from unciv_train.export_onnx import export_rich
    from unciv_train.features import build_rich_single
    from unciv_train.model import RUNGS, StructuredPolicyValueNet, _SPATIAL_FIELD_PLAN

    spatial_w = len(_SPATIAL_FIELD_PLAN)  # 13
    dims = Dims(global_w=8, acting_w=6, tech_w=5, policy_w=4)
    token_specs = {"spatial": spatial_w, "own_units": 9, "opp_units": 9,
                   "own_cities": 17, "opp_cities": 17, "civ_tokens": 84}
    vocab_counts = {"terrain": 6, "resource": 5, "improvement": 4, "religion": 3,
                    "era": 4, "building": 7, "unit": 8, "nation": 2, "promotion": 3}
    torch.manual_seed(7)
    net = StructuredPolicyValueNet(dims, token_specs, vocab_counts, **RUNGS["small"]).eval()
    model = tmp_path / "parity_structured.onnx"
    export_rich(net, dims, token_specs, model, schema_version=3, ruleset_fingerprint="paritytest",
                neighbors=True, contract_version=contract.CONTRACT_VERSION_STRUCTURED)

    # A small consistent hex patch: row 0 center (0,0) + its 6 clock-neighbors; no world wrap.
    coords = np.array([(0, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1), (1, 0)], dtype=np.float32)
    n = coords.shape[0]
    rng = np.random.default_rng(0)
    spatial = rng.integers(0, 4, size=(n, spatial_w)).astype(np.float32)
    spatial[:, 7] = 255.0    # owner_slot self sentinel → exercises the 255 reindex in parity too
    spatial[:, 10] = 255.0   # unit_owner_slot self sentinel
    g = rng.standard_normal(8).astype(np.float32)
    g[contract.GLOBAL_MAPDIM_OFFSET] = 0.0       # effWrapRadius
    g[contract.GLOBAL_MAPDIM_OFFSET + 1] = 0.0   # worldWrap
    g[contract.GLOBAL_MAPDIM_OFFSET + 2] = 1.0   # shape (hex)
    blocks = {
        "global": g,
        "acting_civ": rng.standard_normal(6).astype(np.float32),
        "spatial": spatial,
        "spatial_coords": coords,
        "own_units": rng.standard_normal((2, 9)).astype(np.float32),
        "opp_units": np.zeros((0, 9), np.float32),    # empty set → masked-pool / NaN-guard path
        "own_cities": rng.standard_normal((1, 17)).astype(np.float32),
        "opp_cities": np.zeros((0, 17), np.float32),
        "civ_tokens": rng.standard_normal((2, 84)).astype(np.float32),
    }

    # Python reference: same multi-tensor input (incl. derived neighbor tensors) → ORT.
    built = {k: (v.numpy() if hasattr(v, "numpy") else np.asarray(v))
             for k, v in build_rich_single(blocks, token_specs).items()}
    sess = ort.InferenceSession(str(model))
    feed = {i.name: (built[i.name].astype(np.int64) if "int64" in i.type else built[i.name].astype(np.float32))
            for i in sess.get_inputs()}
    py = sess.run([contract.OUTPUT_TECH, contract.OUTPUT_POLICY], feed)
    py_tech, py_policy = py[0][0], py[1][0]

    # JVM side: vec global/acting, set the 6 token sets, adj the SAME neighbor tensors Python derived.
    lines = [_fixture_line_vec("global", blocks["global"]),
             _fixture_line_vec("acting_civ", blocks["acting_civ"])]
    for name in ["spatial", "own_units", "opp_units", "own_cities", "opp_cities", "civ_tokens"]:
        arr = np.asarray(blocks[name], np.float32).reshape(-1, token_specs[name])
        lines.append(_fixture_line_set(name, arr, token_specs[name]))
    lines.append(_fixture_line_adj("neighbor_index", feed["neighbor_index"][0]))
    lines.append(_fixture_line_adj("neighbor_mask", feed["neighbor_mask"][0]))
    obs_file = tmp_path / "obs_structured.txt"
    obs_file.write_text("\n".join(lines) + "\n")

    out_file = tmp_path / "jvm_structured.json"
    env = dict(os.environ, JAVA_HOME=JAVA_HOME)
    cmd = [str(GRADLEW), "selfPlay", "--console=plain",
           "--args=" + f"parity-run-rich {model} {obs_file} {out_file}"]
    p = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert out_file.is_file(), "JVM parity-run-rich produced no logits file"
    jvm = json.loads(out_file.read_text())
    jvm_tech, jvm_policy = np.array(jvm["tech"], np.float32), np.array(jvm["policy"], np.float32)

    assert np.allclose(py_tech, jvm_tech, atol=ATOL), f"structured tech drift > {ATOL}: {py_tech} vs {jvm_tech}"
    assert np.allclose(py_policy, jvm_policy, atol=ATOL), f"structured policy drift > {ATOL}: {py_policy} vs {jvm_policy}"
