"""Targets for the three tasks at prediction hour t."""

from __future__ import annotations

import numpy as np

from ..constants import DYNAMIC, VITALS

VITAL_IDX = [DYNAMIC.index(v) for v in VITALS]

# Time-to-onset buckets (hours until t_sepsis). Default edges give
#   0: <=6 h (includes hours at/after onset, see docs/methodology.md "Open decisions")
#   1: 6-12 h   2: 12-24 h   3: >24 h or never septic
DEFAULT_ONSET_EDGES = (6, 12, 24)


def onset_bucket(t: int, t_sepsis: int, edges=DEFAULT_ONSET_EDGES) -> int:
    if t_sepsis < 0:
        return len(edges)
    hours_left = t_sepsis - t
    for k, e in enumerate(edges):
        if hours_left <= e:
            return k
    return len(edges)


def vital_targets(y_raw: np.ndarray, t: int, horizon: int):
    """Future vitals at t+1..t+horizon and a mask of which were measured."""
    T = y_raw.shape[0]
    fut = np.full((horizon, len(VITAL_IDX)), np.nan, dtype=np.float32)
    end = min(T, t + 1 + horizon)
    if end > t + 1:
        fut[: end - t - 1] = y_raw[t + 1 : end][:, VITAL_IDX]
    mask = (~np.isnan(fut)).astype(np.float32)
    return np.nan_to_num(fut), mask
