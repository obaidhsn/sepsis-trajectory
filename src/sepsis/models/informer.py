"""Informer encoder (Zhou et al., AAAI 2021) adapted to the multi-task setting.

Keeps the two ideas of the Informer encoder:
  * ProbSparse self-attention: only the u = c·ln(L) "active" queries (largest
    max-minus-mean score on a random sample of keys) get full attention; the
    rest get the mean of V.
  * Self-attention distilling: Conv1d + BatchNorm + ELU + max-pool (stride 2)
    between attention layers, halving the sequence each time.
The representation of the last remaining position feeds the shared heads.
The generative decoder is not used: vitals come from the same forecast head
as every other model.
"""

import math

import torch
from torch import nn

from .heads import MultiTaskHeads, window_features


class ProbSparseAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, factor: int = 5, dropout: float = 0.1):
        super().__init__()
        self.h, self.dk, self.factor = n_heads, d_model // n_heads, factor
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)  # (B, H, L, dk) each
        u = min(L, int(self.factor * math.ceil(math.log(L + 1))))
        # sparsity measure on a random sample of keys
        idx = torch.randint(L, (u,), device=x.device)
        s_sample = q @ k[:, :, idx].transpose(-2, -1) / math.sqrt(self.dk)          # (B, H, L, u)
        m = s_sample.max(-1).values - s_sample.mean(-1)
        top = m.topk(u, dim=-1).indices                                              # (B, H, u)
        # lazy queries: mean of V; active queries: full attention
        ctx = v.mean(2, keepdim=True).expand(B, self.h, L, self.dk).clone()
        q_top = torch.gather(q, 2, top.unsqueeze(-1).expand(-1, -1, -1, self.dk))
        attn = torch.softmax(q_top @ k.transpose(-2, -1) / math.sqrt(self.dk), -1)
        ctx.scatter_(2, top.unsqueeze(-1).expand(-1, -1, -1, self.dk), self.drop(attn) @ v)
        return self.out(ctx.transpose(1, 2).reshape(B, L, D))


class InformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, factor: int, dropout: float):
        super().__init__()
        self.attn = ProbSparseAttention(d_model, n_heads, factor, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(4 * d_model, d_model))
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.n1(x + self.drop(self.attn(x)))
        return self.n2(x + self.drop(self.ff(x)))


class DistilLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, 3, padding=1, padding_mode="circular")
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.ELU()
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)

    def forward(self, x):  # (B, L, D) -> (B, ceil(L/2), D)
        return self.pool(self.act(self.bn(self.conv(x.transpose(1, 2))))).transpose(1, 2)


class Informer(nn.Module):
    def __init__(self, n_features: int, n_static: int, lookback: int, horizon: int, n_vitals: int,
                 n_onset: int, hidden_dim: int = 64, n_layers: int = 2, n_heads: int = 4,
                 factor: int = 5, dropout: float = 0.1, tasks=("vitals", "risk", "onset"), **_):
        super().__init__()
        self.inp = nn.Linear(n_features, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, lookback, hidden_dim))
        nn.init.normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList(InformerLayer(hidden_dim, n_heads, factor, dropout) for _ in range(n_layers))
        self.distil = nn.ModuleList(DistilLayer(hidden_dim) for _ in range(n_layers - 1))
        self.norm = nn.LayerNorm(hidden_dim)
        self.static = nn.Linear(n_static, hidden_dim)
        self.fuse = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.heads = MultiTaskHeads(hidden_dim, horizon, n_vitals, n_onset, tasks, dropout=dropout)

    def forward(self, batch):
        feats = window_features(batch)
        h = self.inp(feats) + self.pos[:, :feats.shape[1]]
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.distil):
                h = self.distil[i](h)
        z_t = self.norm(h[:, -1])
        z = self.fuse(torch.cat([z_t, self.static(batch["s"])], -1))
        return self.heads(z)
