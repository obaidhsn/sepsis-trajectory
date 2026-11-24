"""Autoformer encoder (Wu et al., NeurIPS 2021) adapted to the multi-task setting.

Keeps the two ideas of the Autoformer encoder:
  * Series decomposition blocks (moving average) that split every hidden
    sequence into trend and seasonal parts; layers work on the seasonal part.
  * Auto-correlation instead of self-attention: period-based dependencies are
    found with FFT, and values are aggregated over the top-k time delays
    (k = c·ln L) weighted by their softmaxed correlation.
The trend of the embedded input is kept and fused with the encoder output.
The decoder is not used: vitals come from the same forecast head as every
other model.
"""

import math

import torch
from torch import nn

from .heads import MultiTaskHeads, window_features


class SeriesDecomp(nn.Module):
    def __init__(self, kernel: int):
        super().__init__()
        self.pad = (kernel - 1) // 2
        self.avg = nn.AvgPool1d(kernel, stride=1)

    def forward(self, x):  # (B, L, D) -> seasonal, trend
        front = x[:, :1].expand(-1, self.pad, -1)
        end = x[:, -1:].expand(-1, self.pad, -1)
        trend = self.avg(torch.cat([front, x, end], 1).transpose(1, 2)).transpose(1, 2)
        return x - trend, trend


class AutoCorrelation(nn.Module):
    def __init__(self, d_model: int, n_heads: int, factor: int = 1, dropout: float = 0.1):
        super().__init__()
        self.h, self.dk, self.factor = n_heads, d_model // n_heads, factor
        self.q, self.k, self.v = (nn.Linear(d_model, d_model) for _ in range(3))
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q = self.q(x).view(B, L, self.h, self.dk)
        k = self.k(x).view(B, L, self.h, self.dk)
        v = self.v(x).view(B, L, self.h, self.dk)
        # period-based dependencies via FFT: R(tau) for every lag tau
        corr = torch.fft.irfft(torch.fft.rfft(q, dim=1) * torch.conj(torch.fft.rfft(k, dim=1)), n=L, dim=1)
        score = corr.mean(-1)                                           # (B, L, H)
        top_k = max(1, int(self.factor * math.log(L)))
        w, lags = score.topk(top_k, dim=1)                              # (B, k, H)
        w = self.drop(torch.softmax(w, dim=1))
        # time-delay aggregation: roll V by each selected lag
        base = torch.arange(L, device=x.device).view(1, L, 1, 1)
        out = torch.zeros_like(v)
        for i in range(top_k):
            idx = (base + lags[:, i].view(B, 1, self.h, 1)) % L          # (B, L, H, 1)
            out = out + torch.gather(v, 1, idx.expand(-1, -1, -1, self.dk)) * w[:, i].view(B, 1, self.h, 1)
        return self.out(out.reshape(B, L, D))


class AutoformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, factor: int, kernel: int, dropout: float):
        super().__init__()
        self.attn = AutoCorrelation(d_model, n_heads, factor, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(4 * d_model, d_model))
        self.d1, self.d2 = SeriesDecomp(kernel), SeriesDecomp(kernel)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.d1(x + self.drop(self.attn(x)))
        x, _ = self.d2(x + self.drop(self.ff(x)))
        return x


class SeasonalNorm(nn.Module):
    """LayerNorm followed by removing the mean over time (Autoformer's `my_Layernorm`)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm(x)
        return x - x.mean(1, keepdim=True)


class Autoformer(nn.Module):
    def __init__(self, n_features: int, n_static: int, lookback: int, horizon: int, n_vitals: int,
                 n_onset: int, hidden_dim: int = 64, n_layers: int = 2, n_heads: int = 4,
                 factor: int = 1, moving_avg: int = 7, dropout: float = 0.1,
                 tasks=("vitals", "risk", "onset"), **_):
        super().__init__()
        self.inp = nn.Linear(n_features, hidden_dim)
        self.decomp = SeriesDecomp(moving_avg)
        self.layers = nn.ModuleList(AutoformerLayer(hidden_dim, n_heads, factor, moving_avg, dropout)
                                    for _ in range(n_layers))
        self.norm = SeasonalNorm(hidden_dim)
        self.static = nn.Linear(n_static, hidden_dim)
        self.fuse = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.heads = MultiTaskHeads(hidden_dim, horizon, n_vitals, n_onset, tasks, dropout=dropout)

    def forward(self, batch):
        h = self.inp(window_features(batch))
        seasonal, trend = self.decomp(h)
        for layer in self.layers:
            seasonal = layer(seasonal)
        z_s = self.norm(seasonal)[:, -1]
        z = self.fuse(torch.cat([z_s, trend[:, -1], self.static(batch["s"])], -1))
        return self.heads(z)
