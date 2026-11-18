"""Patient-level splits: 70/15/15 on Hospital A, Hospital B held out entirely."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def make_splits(raw_patients: dict, seed: int = 0, train=0.70, val=0.15) -> dict:
    """`raw_patients` is {pid: (hospital, df)}. Stratified by ever-septic."""
    a_ids = sorted(p for p, (h, _) in raw_patients.items() if h == "A")
    b_ids = sorted(p for p, (h, _) in raw_patients.items() if h == "B")
    septic = np.array([int(raw_patients[p][1]["SepsisLabel"].max()) for p in a_ids])

    train_ids, rest = train_test_split(a_ids, train_size=train, stratify=septic, random_state=seed)
    rest_septic = np.array([int(raw_patients[p][1]["SepsisLabel"].max()) for p in rest])
    val_frac = val / (1 - train)
    val_ids, test_ids = train_test_split(rest, train_size=val_frac, stratify=rest_septic, random_state=seed)
    return {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test_A": sorted(test_ids),
        "test_B": b_ids,
        "seed": seed,
    }
