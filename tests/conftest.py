import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session")
def synthetic_raw():
    from make_synthetic_data import fake_patient
    rng = np.random.default_rng(0)
    patients = {}
    for i in range(40):
        h = "A" if i < 30 else "B"
        patients[f"p{i:06d}"] = (h, fake_patient(rng, septic=(i % 3 == 0)))
    return patients


@pytest.fixture(scope="session")
def processed(synthetic_raw, tmp_path_factory):
    from sepsis.data import make_splits, preprocess_all
    from sepsis.data.preprocess import load_processed
    splits = make_splits(synthetic_raw, seed=0)
    out = preprocess_all(synthetic_raw, splits, tmp_path_factory.mktemp("proc"))
    return load_processed(out)
