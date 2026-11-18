"""Proposed model: Temporal Transformer + GAT over clinical variables.

Identical to `TemporalTransformer` except for the GAT branch, so that any
difference can be attributed to modelling inter-variable relationships.

GAT branch: node c gets the window of (value, mask, delta) for variable c,
projected to `hidden_dim` and summed with a learned node-identity embedding.
"""

import torch
from torch import nn

from .gat import DenseGATLayer
from .heads import MultiTaskHeads, window_features
from .transformer import TemporalEncoder


class GATTransformer(nn.Module):
    def __init__(self, n_vars: int, n_features: int, n_static: int, lookback: int, horizon: int,
                 n_vitals: int, n_onset: int, adjacency, hidden_dim: int = 64, n_layers: int = 2,
                 n_heads: int = 4, gat_layers: int = 2, gat_heads: int = 4, dropout: float = 0.1,
                 tasks=("vitals", "risk", "onset"), **_):
        super().__init__()
        self.temporal = TemporalEncoder(n_features, lookback, hidden_dim, n_layers, n_heads, dropout)
        self.node_in = nn.Linear(3 * lookback, hidden_dim)
        self.node_emb = nn.Parameter(torch.randn(n_vars, hidden_dim) * 0.02)
        self.gat = nn.ModuleList(
            DenseGATLayer(hidden_dim, hidden_dim, gat_heads, dropout) for _ in range(gat_layers)
        )
        self.act = nn.ELU()
        self.register_buffer("adj", torch.as_tensor(adjacency, dtype=torch.float32))
        self.static = nn.Linear(n_static, hidden_dim)
        self.fuse = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.heads = MultiTaskHeads(hidden_dim, horizon, n_vitals, n_onset, tasks, dropout=dropout)

    def graph_branch(self, batch) -> torch.Tensor:
        # (B, L, C) x3 -> (B, C, 3L)
        nodes = torch.cat([batch["x"], batch["m"], batch["d"]], dim=1).transpose(1, 2)
        h = self.node_in(nodes) + self.node_emb
        for layer in self.gat:
            h = self.act(layer(h, self.adj))
        return h.mean(1)

    def forward(self, batch):
        z_t = self.temporal(window_features(batch))
        z_g = self.graph_branch(batch)
        z = self.fuse(torch.cat([z_t, z_g, self.static(batch["s"])], -1))
        return self.heads(z)

    def attention_maps(self) -> list:
        """Per-layer (B, heads, C, C) attention from the last forward pass."""
        return [layer.last_attention for layer in self.gat]
