import numpy as np
import pytest
import torch

from sepsis.config import load_config
from sepsis.constants import DYNAMIC
from sepsis.data.dataset import SepsisWindowDataset, collate_windows
from sepsis.models import build_model

EXPERIMENTS = ["persistence", "linear", "mlp", "grud", "transformer", "gat_transformer"]


@pytest.mark.parametrize("exp", EXPERIMENTS)
def test_forward_shapes(exp, processed):
    patients, splits = processed
    cfg = load_config(f"configs/experiments/{exp}.yaml")
    ds = SepsisWindowDataset(patients, splits["train"], cfg["data"]["lookback"], cfg["data"]["horizon"], 1, 1)
    batch = collate_windows([ds[i] for i in range(8)])
    model = build_model(cfg, np.ones((len(DYNAMIC), len(DYNAMIC)), dtype=np.float32))
    out = model(batch)
    assert out["vitals"].shape == (8, 6, 7)
    if "risk" in cfg["train"]["tasks"]:
        assert out["risk"].shape == (8,)
        assert out["onset"].shape == (8, 4)
    for v in out.values():
        assert torch.isfinite(v).all()


def test_gat_respects_adjacency(processed):
    patients, splits = processed
    cfg = load_config("configs/experiments/gat_transformer.yaml")
    adj = np.eye(len(DYNAMIC), dtype=np.float32)
    model = build_model(cfg, adj).eval()
    ds = SepsisWindowDataset(patients, splits["train"], 24, 6, 1, 1)
    model(collate_windows([ds[i] for i in range(4)]))
    att = model.attention_maps()[0]
    off_diag = att * (1 - torch.eye(len(DYNAMIC)))
    assert off_diag.abs().max() == 0
