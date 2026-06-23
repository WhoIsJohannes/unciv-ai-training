"""RED-first spec for v2 Stage A: actor-critic value head + GAE + terminal-only reward.

Pins the contracts the build must satisfy:
  1. PolicyNet has a value head; forward() returns (tech_logits, policy_logits, value).
  2. compute_gae() implements episodic GAE with V(terminal)=0 bootstrap, reward 0
     except the terminal step (terminal-only ±1 — no shaped intermediate reward).
  3. The EXPORTED play ONNX drops the value head (policy-only contract).

These FAIL on v1 (forward is a 2-tuple; no value_head; no compute_gae) and go GREEN
after Stage A.
"""
import numpy as np
import pytest

from unciv_train.contract import Dims
from unciv_train.model import PolicyNet

torch = pytest.importorskip("torch")


def _dims() -> Dims:
    # synthetic small contract (same shape family as the GnK contract)
    return Dims(global_w=4, acting_w=4, tech_w=5, policy_w=4)


def test_model_forward_returns_value():
    net = PolicyNet(_dims())
    assert hasattr(net, "value_head"), "PolicyNet must have a value_head (training-only critic)"
    obs = torch.zeros(3, _dims().input_w)
    out = net(obs)
    assert len(out) == 3, "forward() must return (tech_logits, policy_logits, value)"
    tech, policy, value = out
    assert tech.shape == (3, 5)
    assert policy.shape == (3, 4)
    # value is a per-state scalar
    assert value.reshape(3, -1).shape[1] == 1


def test_compute_gae_terminal_only_reward():
    """Hand-computed episodic GAE, V(terminal)=0, reward 0 except terminal +1.

    T=2, V=[0.5,0.2], r=[0,1], gamma=0.99, lam=0.95:
      d1 = 1 + 0.99*0 - 0.2  = 0.8
      d0 = 0 + 0.99*0.2 - 0.5 = -0.302
      A1 = d1                 = 0.8
      A0 = d0 + 0.99*0.95*A1  = 0.4504
      R  = A + V              = [0.9504, 1.0]
    """
    from unciv_train.train import compute_gae

    values = np.array([0.5, 0.2], dtype=np.float32)
    rewards = np.array([0.0, 1.0], dtype=np.float32)  # terminal-only
    adv, ret = compute_gae(rewards, values, gamma=0.99, lam=0.95)
    np.testing.assert_allclose(adv, [0.4504, 0.8], atol=1e-4)
    np.testing.assert_allclose(ret, [0.9504, 1.0], atol=1e-4)


def test_gae_loss_reward_is_terminal_only():
    """Guard AC7: the trajectory's reward vector is zero everywhere except the last step."""
    from unciv_train.train import compute_gae

    values = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    rewards = np.zeros(4, dtype=np.float32)
    rewards[-1] = -1.0  # a loss; still terminal-only
    adv, ret = compute_gae(rewards, values, gamma=0.99, lam=0.95)
    # final-step advantage = r_T + gamma*0 - V_T = -1 - 0.1 = -1.1
    np.testing.assert_allclose(adv[-1], -1.1, atol=1e-4)


def test_export_drops_value_head(tmp_path):
    """Exported play ONNX is policy-only — value head provably dropped (council 🟡)."""
    import onnx
    from unciv_train import export_onnx

    dims = _dims()
    net = PolicyNet(dims)
    out = tmp_path / "m.onnx"
    export_onnx.export(net, dims, out, schema_version=2, ruleset_fingerprint="deadbeef")
    model = onnx.load(str(out))
    names = {o.name for o in model.graph.output}
    assert names == {"tech_logits", "policy_logits"}, f"value head leaked into export: {names}"
