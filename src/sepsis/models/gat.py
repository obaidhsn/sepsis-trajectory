"""Dense multi-head Graph Attention (Velickovic et al., 2018) over variables.

With only C = 33 nodes a dense implementation is simpler and as fast as
PyTorch Geometric, and makes the attention matrix easy to inspect. The
adjacency restricts which variable pairs may attend to each other.
"""

import torch
from torch import nn
import torch.nn.functional as F


class DenseGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.n_heads, self.concat = n_heads, concat
        self.f = out_dim // n_heads if concat else out_dim
        self.W = nn.Linear(in_dim, n_heads * self.f, bias=False)
        self.a_src = nn.Parameter(torch.empty(n_heads, self.f))
        self.a_dst = nn.Parameter(torch.empty(n_heads, self.f))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Linear(in_dim, n_heads * self.f if concat else out_dim, bias=False)
        self.last_attention = None  # (B, heads, N, N), kept for analysis

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, _ = h.shape
        z = self.W(h).view(B, N, self.n_heads, self.f)                       # B N H F
        e_src = (z * self.a_src).sum(-1)                                      # B N H
        e_dst = (z * self.a_dst).sum(-1)
        e = F.leaky_relu(e_src.unsqueeze(2) + e_dst.unsqueeze(1), 0.2)        # B i j H
        e = e.permute(0, 3, 1, 2)                                             # B H i j
        e = e.masked_fill(adj.view(1, 1, N, N) == 0, float("-inf"))
        att = torch.softmax(e, dim=-1)
        self.last_attention = att.detach()
        out = torch.einsum("bhij,bjhf->bihf", self.drop(att), z)             # B N H F
        out = out.reshape(B, N, -1) if self.concat else out.mean(2)
        return out + self.res(h)
