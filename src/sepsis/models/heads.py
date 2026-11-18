"""Task heads shared by every multi-task model.

Output dict convention used everywhere in the codebase:
    "vitals": (B, H, V)  forecast of normalised vitals at t+1..t+H
    "risk":   (B,)       logit of SepsisLabel at hour t
    "onset":  (B, K)     logits over time-to-onset buckets
A model may omit keys it does not predict (e.g. persistence only forecasts).
"""

import torch
from torch import nn


class MultiTaskHeads(nn.Module):
    def __init__(self, in_dim: int, horizon: int, n_vitals: int, n_onset: int,
                 tasks=("vitals", "risk", "onset"), hidden: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.tasks = tuple(tasks)
        self.horizon, self.n_vitals = horizon, n_vitals
        hidden = hidden or in_dim

        def mlp(out):
            return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out))

        self.heads = nn.ModuleDict()
        if "vitals" in self.tasks:
            self.heads["vitals"] = mlp(horizon * n_vitals)
        if "risk" in self.tasks:
            self.heads["risk"] = mlp(1)
        if "onset" in self.tasks:
            self.heads["onset"] = mlp(n_onset)

    def forward(self, z: torch.Tensor) -> dict:
        out = {}
        if "vitals" in self.heads:
            out["vitals"] = self.heads["vitals"](z).view(-1, self.horizon, self.n_vitals)
        if "risk" in self.heads:
            out["risk"] = self.heads["risk"](z).squeeze(-1)
        if "onset" in self.heads:
            out["onset"] = self.heads["onset"](z)
        return out


def window_features(batch: dict) -> torch.Tensor:
    """Per-hour input features: values, mask, log time-since-measured and a
    padding flag -> (B, L, 3C + 1). Every model gets the same inputs."""
    return torch.cat([batch["x"], batch["m"], batch["d"], batch["pad"].unsqueeze(-1)], dim=-1)
