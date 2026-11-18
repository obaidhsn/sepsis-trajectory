"""Read the raw Challenge 2019 .psv files.

Expected layout after scripts/download_data.sh:

    data/raw/training_setA/p000001.psv ...   (Hospital A, 20,336 patients)
    data/raw/training_setB/p100001.psv ...   (Hospital B, 20,000 patients)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

HOSPITAL_DIRS = {"A": "training_setA", "B": "training_setB"}


def read_psv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="|")


def load_raw_patients(raw_dir: str | Path, hospitals=("A", "B"), limit: int | None = None):
    """Return {patient_id: (hospital, DataFrame)}.

    `limit` caps the number of files per hospital (useful for quick debugging).
    """
    raw_dir = Path(raw_dir)
    patients = {}
    for h in hospitals:
        files = sorted((raw_dir / HOSPITAL_DIRS[h]).glob("*.psv"))
        if not files:
            raise FileNotFoundError(
                f"No .psv files in {raw_dir / HOSPITAL_DIRS[h]}. Run scripts/download_data.sh first."
            )
        if limit:
            files = files[:limit]
        skipped = []
        for f in tqdm(files, desc=f"Hospital {h}"):
            try:
                df = read_psv(f)
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                skipped.append(f.name)
                warnings.warn(f"Skipping unreadable file {f}: {e}")
                continue
            if df.empty:
                skipped.append(f.name)
                warnings.warn(f"Skipping file with no rows: {f}")
                continue
            patients[f.stem] = (h, df)
        if skipped:
            print(f"Hospital {h}: skipped {len(skipped)} of {len(files)} files: {', '.join(skipped[:10])}"
                  + (" ..." if len(skipped) > 10 else ""))
    return patients
