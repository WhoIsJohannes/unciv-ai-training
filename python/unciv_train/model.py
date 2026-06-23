"""The policy net: a small shared-trunk MLP with per-head linear outputs + a value critic.

Input (blind variant) = concat(global, acting_civ) (fixed width from schema).
Output = {tech_logits, policy_logits, value}. The value head is TRAINING-ONLY — `export_onnx`
wraps the net to emit only (tech_logits, policy_logits), so the play-time ONNX contract is
policy-only. Deliberately tiny — trains on CPU in seconds.

`RichPolicyValueNet` (Stage B) consumes the FULL observation as a multi-tensor input: a
permutation-invariant masked pool over the per-tile spatial token set and over each entity token
type, concatenated with global+acting_civ → shared trunk → the same three heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .contract import Dims


def _small_init_value_head(layer: nn.Linear) -> None:
    """Initialize the value head near zero so V≈0 at init — game features are unnormalized
    (gold, science, …), so a default-init linear head would otherwise emit huge values and the
    bounded discounted-terminal return target couldn't be fit in a few epochs. v1-reinforce ignores
    the value head, so this does not perturb the attributable baseline."""
    nn.init.uniform_(layer.weight, -1e-3, 1e-3)
    nn.init.zeros_(layer.bias)


class PolicyNet(nn.Module):
    """Blind variant: trunk over concat(global, acting_civ) → {tech, policy, value}."""

    def __init__(self, dims: Dims, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dims.input_w, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.tech_head = nn.Linear(hidden, dims.tech_w)
        self.policy_head = nn.Linear(hidden, dims.policy_w)
        self.value_head = nn.Linear(hidden, 1)  # training-only critic
        _small_init_value_head(self.value_head)  # V≈0 at init (game features are unnormalized)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        # tanh-bound the value: the true value = expected discounted terminal reward ∈ [-1,1], so
        # this is the correct range and it keeps V bounded despite unnormalized game features.
        return self.tech_head(h), self.policy_head(h), torch.tanh(self.value_head(h))


def masked_pool(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Permutation-invariant masked mean+max pool over a [B, N, F] token set.

    `mask` is [B, N] (1 = present, 0 = padding). NaN-guarded (council R3): mean divides by
    clamp(count, min=1); max over an empty set → 0 (not −inf). An all-padding row → zero vector.
    Returns [B, 2F] (mean ‖ max).
    """
    m = mask.unsqueeze(-1)                                  # [B, N, 1]
    safe_count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1] avoid /0
    mean = (tokens * m).sum(dim=1) / safe_count             # [B, F]; all-padding row → 0
    masked = tokens.masked_fill(m == 0, float("-inf"))      # true -inf so empty set is catchable
    mx = masked.max(dim=1).values                           # [B, F]; all-padding → -inf
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))  # empty set → 0 (NaN/inf guard)
    return torch.cat([mean, mx], dim=1)                     # [B, 2F]


class _TokenEncoder(nn.Module):
    """Per-token MLP → masked mean+max pool. Input [B,N,in_dim] + mask [B,N] → [B, 2*out_dim]."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),
                                 nn.Linear(out_dim, out_dim), nn.ReLU())
        self.out_w = 2 * out_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_pool(self.mlp(tokens), mask)


class RichPolicyValueNet(nn.Module):
    """Rich variant: masked-pool encoders over the spatial tile set + each entity token type,
    concatenated with global+acting_civ → trunk → {tech, policy, value}.

    `token_specs` maps an input tensor name → its per-token feature width, e.g.
    {"spatial": 13, "own_units": 8, "opp_units": 8, "own_cities": 16, "opp_cities": 16,
     "civ_tokens": 84}. `forward(inputs)` takes a dict of {name: tensor} plus {name+"_mask": mask}.
    """

    INPUT_GLOBAL = "global"
    INPUT_ACTING = "acting_civ"

    def __init__(self, dims: Dims, token_specs: dict[str, int], *, token_dim: int = 32,
                 hidden: int = 256):
        super().__init__()
        self.token_names = list(token_specs.keys())
        self.encoders = nn.ModuleDict(
            {name: _TokenEncoder(width, token_dim) for name, width in token_specs.items()}
        )
        trunk_in = dims.global_w + dims.acting_w + sum(e.out_w for e in self.encoders.values())
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.tech_head = nn.Linear(hidden, dims.tech_w)
        self.policy_head = nn.Linear(hidden, dims.policy_w)
        self.value_head = nn.Linear(hidden, 1)
        _small_init_value_head(self.value_head)

    def forward(self, inputs: dict[str, torch.Tensor]):
        parts = [inputs[self.INPUT_GLOBAL], inputs[self.INPUT_ACTING]]
        for name in self.token_names:
            parts.append(self.encoders[name](inputs[name], inputs[name + "_mask"]))
        h = self.trunk(torch.cat(parts, dim=1))
        return self.tech_head(h), self.policy_head(h), torch.tanh(self.value_head(h))
