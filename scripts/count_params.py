"""Print parameter counts so the plain Transformer can be matched to the
GAT + Transformer (proposal: equal parameter budget).

    python scripts/count_params.py
"""
import _bootstrap_path  # noqa: F401
import numpy as np

from sepsis.config import load_config
from sepsis.constants import DYNAMIC
from sepsis.models import build_model, count_parameters

CONFIGS = ["mlp", "grud", "transformer", "gat_transformer"]


def main():
    adj = np.ones((len(DYNAMIC), len(DYNAMIC)), dtype=np.float32)
    counts = {}
    for c in CONFIGS:
        cfg = load_config(f"configs/experiments/{c}.yaml")
        counts[c] = count_parameters(build_model(cfg, adj))
        print(f"{c:18s} {counts[c]:>10,}")
    target = counts["gat_transformer"]
    print(f"\nTransformer hidden_dim that best matches gat_transformer ({target:,}):")
    cfg = load_config("configs/experiments/transformer.yaml")
    best = min(range(48, 129, 4), key=lambda h: abs(count_parameters(
        build_model({**cfg, "model": {**cfg["model"], "hidden_dim": h}})) - target))
    n = count_parameters(build_model({**cfg, "model": {**cfg["model"], "hidden_dim": best}}))
    print(f"  hidden_dim={best} -> {n:,} params (set it in configs/experiments/transformer.yaml)")


if __name__ == "__main__":
    main()
