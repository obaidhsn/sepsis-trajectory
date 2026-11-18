"""GRU-D (Che et al., 2018): GRU with trainable decay for missing values.

Inputs are already forward-filled and normalised (train mean = 0), so the
GRU-D imputation  x_hat = m*x + (1-m)*(g*x_last + (1-g)*mean)  reduces to
x_hat = m*x + (1-m)*g*x, with g = exp(-relu(W_g * delta)).
"""

import torch
from torch import nn

from .heads import MultiTaskHeads


class GRUD(nn.Module):
    def __init__(self, n_vars: int, n_static: int, horizon: int, n_vitals: int, n_onset: int,
                 hidden_dim: int = 64, dropout: float = 0.1, tasks=("vitals", "risk", "onset"), **_):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gamma_x = nn.Linear(n_vars, n_vars)           # per-variable input decay
        self.gamma_h = nn.Linear(n_vars, hidden_dim)       # hidden-state decay
        self.cell = nn.GRUCell(2 * n_vars, hidden_dim)     # [x_hat, mask]
        self.static = nn.Linear(n_static, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.heads = MultiTaskHeads(2 * hidden_dim, horizon, n_vitals, n_onset, tasks, dropout=dropout)

    def forward(self, batch):
        x, m, d, pad = batch["x"], batch["m"], torch.expm1(batch["d"]), batch["pad"]
        B, L, _ = x.shape
        h = x.new_zeros(B, self.hidden_dim)
        for t in range(L):
            g_x = torch.exp(-torch.relu(self.gamma_x(d[:, t])))
            g_h = torch.exp(-torch.relu(self.gamma_h(d[:, t])))
            x_hat = m[:, t] * x[:, t] + (1 - m[:, t]) * g_x * x[:, t]
            h_new = self.cell(torch.cat([x_hat, m[:, t]], -1), g_h * h)
            keep = pad[:, t].unsqueeze(-1)                 # padded steps leave h unchanged
            h = keep * h + (1 - keep) * h_new
        z = torch.cat([self.drop(h), torch.relu(self.static(batch["s"]))], -1)
        return self.heads(z)
