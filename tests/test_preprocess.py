import numpy as np
import pandas as pd

from sepsis.constants import DYNAMIC
from sepsis.data.preprocess import build_patient_record, compute_delta, forward_fill


def test_forward_fill_is_causal():
    v = np.array([[np.nan], [1.0], [np.nan], [3.0]], dtype=np.float32)
    out = forward_fill(v)
    assert np.isnan(out[0, 0])          # nothing before the first observation
    assert out[2, 0] == 1.0             # filled from the past, never from t=3


def test_delta_counts_hours_since_last_observation():
    m = np.array([[1], [0], [0], [1], [0]], dtype=np.float32)
    assert compute_delta(m)[:, 0].tolist() == [1, 1, 2, 3, 1]


def test_onset_from_label_shift(synthetic_raw):
    for _, (h, df) in synthetic_raw.items():
        rec = build_patient_record(df, h)
        if df["SepsisLabel"].any():
            assert rec["t_sepsis"] == int(np.argmax(df["SepsisLabel"].to_numpy())) + 6
        else:
            assert rec["t_sepsis"] == -1
        assert rec["raw"].shape[1] == len(DYNAMIC) == 33


def test_splits_are_patient_level_and_disjoint(processed):
    _, splits = processed
    sets = [set(splits[k]) for k in ("train", "val", "test_A", "test_B")]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not sets[i] & sets[j]


def test_config_override_parses_scientific_notation(tmp_path):
    from sepsis.config import load_config
    f = tmp_path / "c.yaml"
    f.write_text("train:\n  lr: 1.0e-3\n")
    cfg = load_config(f, ["train.lr=3e-4", "train.tasks=[risk]", "model.graph=none"])
    assert cfg["train"]["lr"] == 3e-4
    assert cfg["train"]["tasks"] == ["risk"]
    assert cfg["model"]["graph"] == "none"
