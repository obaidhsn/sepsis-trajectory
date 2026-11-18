"""PyTorch datasets.

`SepsisWindowDataset`  - training: samples a fixed number of windows per
                         patient each epoch (septic patients oversampled) so
                         long stays do not dominate.
`PatientSequenceDataset` - evaluation: every hour of every patient, in order,
                         which is what the utility score and alarm metrics need.

A window ending at hour t contains hours [t-L+1, t] (left-padded with zeros
and pad=1 when the stay is shorter than L).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .targets import DEFAULT_ONSET_EDGES, onset_bucket, vital_targets


def make_window(rec: dict, t: int, lookback: int, horizon: int, onset_edges=DEFAULT_ONSET_EDGES) -> dict:
    start = max(0, t - lookback + 1)
    n = t - start + 1
    pad = lookback - n

    def win(a):
        w = a[start : t + 1]
        if pad:
            w = np.concatenate([np.zeros((pad,) + a.shape[1:], dtype=a.dtype), w])
        return w

    vit, vit_mask = vital_targets(rec["y_raw"], t, horizon)
    pad_mask = np.zeros(lookback, dtype=np.float32)
    pad_mask[:pad] = 1.0
    return {
        "x": win(rec["x"]),
        "m": win(rec["mask"]),
        "d": win(rec["delta_n"]),
        "pad": pad_mask,
        "s": rec["s"][t],
        "y_vitals": vit,
        "y_vitals_mask": vit_mask,
        "y_risk": np.float32(rec["label"][t]),
        "y_onset": np.int64(onset_bucket(t, rec["t_sepsis"], onset_edges)),
        "hour": np.int64(t),
    }


class SepsisWindowDataset(Dataset):
    def __init__(self, patients: dict, ids: list[str], lookback=24, horizon=6,
                 windows_per_patient=8, septic_oversample=4, seed=0,
                 onset_edges=DEFAULT_ONSET_EDGES):
        self.patients = patients
        self.ids = list(ids)
        self.lookback, self.horizon = lookback, horizon
        self.wpp, self.oversample = windows_per_patient, septic_oversample
        self.onset_edges = tuple(onset_edges)
        self.rng = np.random.default_rng(seed)
        self.resample()

    def resample(self):
        """Call once per epoch to draw a fresh set of (patient, hour) windows."""
        index = []
        for pid in self.ids:
            rec = self.patients[pid]
            T = len(rec["label"])
            k = self.wpp * (self.oversample if rec["t_sepsis"] >= 0 else 1)
            for t in self.rng.integers(0, T, size=k):
                index.append((pid, int(t)))
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        pid, t = self.index[i]
        return make_window(self.patients[pid], t, self.lookback, self.horizon, self.onset_edges)


class PatientSequenceDataset(Dataset):
    """All hours of the given patients; item also carries the patient index."""

    def __init__(self, patients: dict, ids: list[str], lookback=24, horizon=6,
                 onset_edges=DEFAULT_ONSET_EDGES):
        self.patients, self.ids = patients, list(ids)
        self.lookback, self.horizon = lookback, horizon
        self.onset_edges = tuple(onset_edges)
        self.index = [(k, pid, t) for k, pid in enumerate(self.ids)
                      for t in range(len(patients[pid]["label"]))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        k, pid, t = self.index[i]
        item = make_window(self.patients[pid], t, self.lookback, self.horizon, self.onset_edges)
        item["patient"] = np.int64(k)
        return item


def collate_windows(batch: list[dict]) -> dict:
    return {k: torch.as_tensor(np.stack([b[k] for b in batch])) for k in batch[0]}
