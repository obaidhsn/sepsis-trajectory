"""Variable-to-variable graphs over the C = 33 clinical variables.

All builders return a binary (C, C) adjacency with self-loops. The GAT
attends only over edges present in the adjacency, so:

  * "learned"        - fully connected; attention decides which edges matter
  * "clinical_prior" - edges between variables in the same organ system
  * "correlation"    - edges where |corr| on TRAINING data exceeds a threshold
  * "none"           - identity only (ablation: GAT branch sees no neighbours)
"""

from __future__ import annotations

import numpy as np

from ..constants import DYNAMIC, ORGAN_GROUPS


def full_graph(n: int = len(DYNAMIC)) -> np.ndarray:
    return np.ones((n, n), dtype=np.float32)


def identity_graph(n: int = len(DYNAMIC)) -> np.ndarray:
    return np.eye(n, dtype=np.float32)


def clinical_prior_graph() -> np.ndarray:
    n = len(DYNAMIC)
    adj = np.eye(n, dtype=np.float32)
    for members in ORGAN_GROUPS.values():
        idx = [DYNAMIC.index(v) for v in members]
        for i in idx:
            for j in idx:
                adj[i, j] = 1.0
    return adj


def correlation_graph(records: list[dict], threshold: float = 0.2, min_pairs: int = 200) -> np.ndarray:
    """Pairwise Pearson correlation on co-observed (raw, normalised) values
    from training patients only."""
    y = np.concatenate([r["y_raw"] for r in records])
    n = y.shape[1]
    adj = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            both = ~np.isnan(y[:, i]) & ~np.isnan(y[:, j])
            if both.sum() < min_pairs:
                continue
            c = np.corrcoef(y[both, i], y[both, j])[0, 1]
            if np.isfinite(c) and abs(c) >= threshold:
                adj[i, j] = adj[j, i] = 1.0
    return adj


def build_adjacency(kind: str, train_records: list[dict] | None = None, **kw) -> np.ndarray:
    if kind == "learned":
        return full_graph()
    if kind == "clinical_prior":
        return clinical_prior_graph()
    if kind == "correlation":
        if train_records is None:
            raise ValueError("correlation graph needs training records")
        return correlation_graph(train_records, **kw)
    if kind == "none":
        return identity_graph()
    raise ValueError(f"unknown graph type: {kind}")
