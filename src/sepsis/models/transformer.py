"""Temporal Transformer: causal self-attention across hours (primary baseline)."""

import torch
from torch import nn

from .heads import MultiTaskHeads, window_features


class TemporalEncoder(nn.Module):
    """(B, L, F) -> (B, D): representation of the last hour.

    Padding is passed as an input flag rather than a key-padding mask, which
    keeps fully padded causal rows NaN-free.
    """

    def __init__(self, n_features: int, lookback: int, d_model: int = 64, n_layers: int = 2,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, lookback, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        causal = torch.triu(torch.full((lookback, lookback), float("-inf")), diagonal=1)
        self.register_buffer("causal", causal)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        L = feats.shape[1]
        h = self.inp(feats) + self.pos[:, :L]
        h = self.enc(h, mask=self.causal[:L, :L])
        return self.norm(h[:, -1])


class TemporalTransformer(nn.Module):
    def __init__(self, n_features: int, n_static: int, lookback: int, horizon: int, n_vitals: int,
                 n_onset: int, hidden_dim: int = 64, n_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.1, tasks=("vitals", "risk", "onset"), **_):
        super().__init__()
        self.temporal = TemporalEncoder(n_features, lookback, hidden_dim, n_layers, n_heads, dropout)
        self.static = nn.Linear(n_static, hidden_dim)
        self.fuse = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.heads = MultiTaskHeads(hidden_dim, horizon, n_vitals, n_onset, tasks, dropout=dropout)

    def forward(self, batch):
        z_t = self.temporal(window_features(batch))
        z = self.fuse(torch.cat([z_t, self.static(batch["s"])], -1))
        return self.heads(z)
