"""Strictly causal preprocessing.

Rules (see docs/data.md):
  * Missing values are forward-filled from EARLIER hours only.
  * The observation mask M is kept as an input (informative missingness).
  * delta = hours since the variable was last measured (GRU-D style).
  * Normalisation statistics are computed on TRAINING patients only.
  * Values never observed so far are set to 0 after normalisation (= train mean).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import DYNAMIC, LABEL, LABEL_SHIFT, STATIC


def compute_delta(mask: np.ndarray) -> np.ndarray:
    """Hours since last observation for each (t, c). Before the first
    observation the counter grows from 1 at t=0."""
    T, C = mask.shape
    delta = np.zeros((T, C), dtype=np.float32)
    last = np.full(C, -1.0)
    for t in range(T):
        delta[t] = t - last
        last = np.where(mask[t] > 0, t, last)
    # delta[t] counts hours since the observation *before* t; if observed at t
    # itself the model also sees mask=1, which is the standard GRU-D encoding.
    return delta


def forward_fill(values: np.ndarray) -> np.ndarray:
    return pd.DataFrame(values).ffill().to_numpy(dtype=np.float32)


def build_patient_record(df: pd.DataFrame, hospital: str) -> dict:
    """Turn one raw patient table into un-normalised arrays."""
    raw = df[DYNAMIC].to_numpy(dtype=np.float32)
    mask = (~np.isnan(raw)).astype(np.float32)
    labels = df[LABEL].to_numpy(dtype=np.int64)
    positive = np.flatnonzero(labels)
    t_sepsis = int(positive[0] + LABEL_SHIFT) if len(positive) else -1
    return {
        "hospital": hospital,
        "raw": raw,                                   # (T, C) with NaN, used for forecast targets
        "values": forward_fill(raw),                  # (T, C) causal ffill, NaN before first obs
        "mask": mask,                                 # (T, C)
        "delta": compute_delta(mask),                 # (T, C)
        "static": df[STATIC].to_numpy(dtype=np.float32),  # (T, S)
        "label": labels,                              # (T,)
        "t_sepsis": t_sepsis,                         # onset hour index, -1 if never septic
    }


# Arrays kept in patients.pkl (everything the datasets, models and graphs read).
KEEP_KEYS = ("hospital", "label", "t_sepsis", "x", "mask", "delta_n", "s", "y_raw")


@dataclass
class Normalizer:
    dyn_mean: np.ndarray = field(default=None)
    dyn_std: np.ndarray = field(default=None)
    sta_mean: np.ndarray = field(default=None)
    sta_std: np.ndarray = field(default=None)

    def fit(self, records: list[dict]) -> "Normalizer":
        raw = np.concatenate([r["raw"] for r in records])
        sta = np.concatenate([r["static"] for r in records])
        self.dyn_mean = np.nanmean(raw, axis=0)
        self.dyn_std = np.nanstd(raw, axis=0) + 1e-6
        self.sta_mean = np.nanmean(sta, axis=0)
        self.sta_std = np.nanstd(sta, axis=0) + 1e-6
        return self

    def transform(self, rec: dict) -> dict:
        out = dict(rec)
        out["x"] = np.nan_to_num((rec["values"] - self.dyn_mean) / self.dyn_std).astype(np.float32)
        out["y_raw"] = ((rec["raw"] - self.dyn_mean) / self.dyn_std).astype(np.float32)  # NaN kept
        out["s"] = np.nan_to_num((rec["static"] - self.sta_mean) / self.sta_std).astype(np.float32)
        out["delta_n"] = np.log1p(rec["delta"]).astype(np.float32)
        return out

    def to_dict(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("dyn_mean", "dyn_std", "sta_mean", "sta_std")}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(**{k: np.asarray(v, dtype=np.float32) for k, v in d.items()})


def preprocess_all(raw_patients: dict, splits: dict, out_dir: str | Path) -> Path:
    """Build records, fit the normaliser on train ids, write data/processed/."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = {pid: build_patient_record(df, h) for pid, (h, df) in raw_patients.items()}
    norm = Normalizer().fit([records[p] for p in splits["train"]])
    # Only the model-ready arrays are stored; the intermediate raw/ffill/delta
    # arrays are dropped, which halves the file and the RAM needed for training.
    processed = {pid: {k: v for k, v in norm.transform(r).items() if k in KEEP_KEYS}
                 for pid, r in records.items()}
    with open(out_dir / "patients.pkl", "wb") as f:
        pickle.dump(processed, f, protocol=pickle.HIGHEST_PROTOCOL)
    from ..utils import save_json
    save_json(norm.to_dict(), out_dir / "normalizer.json")
    save_json(splits, out_dir / "splits.json")
    save_json(dataset_profile(records, splits), out_dir / "profile.json")
    return out_dir


def dataset_profile(records: dict, splits: dict) -> dict:
    """Cohort statistics per hospital and per split (for the report)."""
    def stats(ids):
        ids = list(ids)
        lengths = np.array([len(records[p]["label"]) for p in ids])
        pos = int(sum(records[p]["label"].sum() for p in ids))
        septic = sum(records[p]["t_sepsis"] >= 0 for p in ids)
        return {
            "patients": len(ids),
            "hours": int(lengths.sum()),
            "septic_patients": int(septic),
            "septic_rate": septic / max(len(ids), 1),
            "positive_hours": pos,
            "positive_hour_rate": pos / max(int(lengths.sum()), 1),
            "median_stay_hours": float(np.median(lengths)) if len(ids) else float("nan"),
        }

    out = {f"hospital_{h}": stats(p for p, r in records.items() if r["hospital"] == h)
           for h in sorted({r["hospital"] for r in records.values()})}
    out["all"] = stats(records)
    out["splits"] = {k: stats(v) for k, v in splits.items() if isinstance(v, list)}
    return out


def load_processed(processed_dir: str | Path) -> tuple[dict, dict]:
    processed_dir = Path(processed_dir)
    with open(processed_dir / "patients.pkl", "rb") as f:
        patients = pickle.load(f)
    from ..utils import load_json
    return patients, load_json(processed_dir / "splits.json")
