"""The policy net: a small shared-trunk MLP with per-head linear outputs.

Input = concat(global, acting_civ) (fixed width from schema). Output = {tech_logits, policy_logits}.
Deliberately tiny — trains on CPU in seconds. `forward` returns a TUPLE so the ONNX export has two
named outputs (`tech_logits`, `policy_logits`).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .contract import Dims


class PolicyNet(nn.Module):
    def __init__(self, dims: Dims, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dims.input_w, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.tech_head = nn.Linear(hidden, dims.tech_w)
        self.policy_head = nn.Linear(hidden, dims.policy_w)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        return self.tech_head(h), self.policy_head(h)
