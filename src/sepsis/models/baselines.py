"""Simple reference models: persistence, linear regression, multi-task MLP."""

import torch
from torch import nn

from ..data.targets import VITAL_IDX
from .heads import MultiTaskHeads, window_features


class Persistence(nn.Module):
    """Forecast = last (forward-filled) value of each vital. No training."""

    trainable = False

    def __init__(self, horizon: int, **_):
        super().__init__()
        self.horizon = horizon
        self.register_buffer("vital_idx", torch.tensor(VITAL_IDX))

    def forward(self, batch):
        last = batch["x"][:, -1, :].index_select(-1, self.vital_idx)
        return {"vitals": last.unsqueeze(1).expand(-1, self.horizon, -1)}


class LinearForecaster(nn.Module):
    """Linear regression from the flattened window to future vitals."""

    def __init__(self, n_features: int, lookback: int, horizon: int, n_vitals: int, **_):
        super().__init__()
        self.horizon, self.n_vitals = horizon, n_vitals
        self.lin = nn.Linear(n_features * lookback, horizon * n_vitals)

    def forward(self, batch):
        f = window_features(batch).flatten(1)
        return {"vitals": self.lin(f).view(-1, self.horizon, self.n_vitals)}


class MLPMultiTask(nn.Module):
    """Simplest model producing all three outputs."""

    def __init__(self, n_features: int, n_static: int, lookback: int, horizon: int, n_vitals: int,
                 n_onset: int, hidden_dim: int = 128, dropout: float = 0.1, tasks=("vitals", "risk", "onset"), **_):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_features * lookback + n_static, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.heads = MultiTaskHeads(hidden_dim, horizon, n_vitals, n_onset, tasks, dropout=dropout)

    def forward(self, batch):
        z = self.body(torch.cat([window_features(batch).flatten(1), batch["s"]], dim=-1))
        return self.heads(z)
