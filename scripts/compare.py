"""Compare two trained runs on the same split with patient-level bootstrap CIs.

    python scripts/compare.py results/gat_transformer/seed0 results/transformer/seed0 --split test_A
"""
import argparse
from pathlib import Path

import _bootstrap_path  # noqa: F401
import numpy as np

from sepsis.evaluation import paired_bootstrap, risk_metrics, vitals_metrics
from sepsis.evaluation.early_warning import early_warning_metrics
from sepsis.utils import load_json

METRICS = {
    "auprc": lambda p: risk_metrics(p["y_risk"], p["pred_risk"])["auprc"],
    "auroc": lambda p: risk_metrics(p["y_risk"], p["pred_risk"])["auroc"],
    "vitals_mae": lambda p: vitals_metrics(p["y_vitals"], p["pred_vitals"], p["y_vitals_mask"])["mae"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--split", default="test_A")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    pa = dict(np.load(Path(a.run_a) / f"predictions_{a.split}.npz"))
    pb = dict(np.load(Path(a.run_b) / f"predictions_{a.split}.npz"))
    metrics = dict(METRICS)
    thr_a = load_json(Path(a.run_a) / "thresholds.json").get("max_utility")
    thr_b = load_json(Path(a.run_b) / "thresholds.json").get("max_utility")
    if thr_a is not None and thr_b is not None:
        # each model keeps its own validation-chosen threshold
        def utility_diff_fn(p, thr):
            return early_warning_metrics(p["y_risk"], p["pred_risk"], p["patient"], p["hour"], thr)["utility"]
        print("utility A/B:", utility_diff_fn(pa, thr_a), utility_diff_fn(pb, thr_b))
    for name, fn in metrics.items():
        try:
            r = paired_bootstrap(fn, pa, pb, n_boot=a.n_boot)
        except KeyError:
            continue
        print(f"{name:12s} A-B = {r['diff']:+.4f}  95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")


if __name__ == "__main__":
    main()
