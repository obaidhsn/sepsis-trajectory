"""Write small fake .psv files with the real column layout, so the whole
pipeline can be tested before (or without) downloading PhysioNet data.

    python scripts/make_synthetic_data.py --out data/synthetic --n 60
"""
import argparse
from pathlib import Path

import _bootstrap_path  # noqa: F401
import numpy as np
import pandas as pd

from sepsis.constants import DROPPED, DYNAMIC, LABEL, LABEL_SHIFT, STATIC, VITALS

COLUMNS = VITALS[:7] + DROPPED + [c for c in DYNAMIC if c not in VITALS] + ["Age", "Gender", "Unit1", "Unit2",
                                                                              "HospAdmTime", "ICULOS", LABEL]


def fake_patient(rng, septic: bool) -> pd.DataFrame:
    T = int(rng.integers(20, 70))
    data = {}
    for c in VITALS + DROPPED + [c for c in DYNAMIC if c not in VITALS]:
        base = rng.normal(80 if c == "HR" else 1.0, 5)
        series = base + np.cumsum(rng.normal(0, 1, T))
        p_obs = 0.85 if c in VITALS else 0.05
        series[rng.random(T) > p_obs] = np.nan
        data[c] = series
    data.update(Age=np.full(T, rng.integers(20, 90)), Gender=np.full(T, rng.integers(0, 2)),
                Unit1=np.full(T, 1.0), Unit2=np.full(T, 0.0),
                HospAdmTime=np.full(T, -rng.uniform(0, 50)), ICULOS=np.arange(1, T + 1))
    label = np.zeros(T, dtype=int)
    if septic:
        onset = int(rng.integers(LABEL_SHIFT + 2, T))
        label[onset - LABEL_SHIFT:] = 1
        data["HR"][onset - LABEL_SHIFT:] += 25  # make it learnable
    data[LABEL] = label
    return pd.DataFrame(data)[COLUMNS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=60, help="patients per hospital")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for h, start in (("training_setA", 0), ("training_setB", 100000)):
        d = Path(a.out) / h
        d.mkdir(parents=True, exist_ok=True)
        for i in range(a.n):
            fake_patient(rng, septic=rng.random() < 0.3).to_csv(d / f"p{start + i:06d}.psv", sep="|", index=False)
    print(f"wrote {2 * a.n} synthetic patients to {a.out}")


if __name__ == "__main__":
    main()
