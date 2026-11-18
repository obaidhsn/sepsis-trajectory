"""Check src/sepsis/evaluation/utility.py against the official Challenge 2019
scorer (data/raw/evaluate_sepsis_score.py, fetched by download_data.sh).

Uses real SepsisLabel sequences and several kinds of alarm sequences
(random at different rates, label-shifted, all-on, all-off) and compares the
normalised utility from both implementations.

    python scripts/verify_utility.py --n 3000
"""
import argparse
import importlib.util
from pathlib import Path

import _bootstrap_path  # noqa: F401
import numpy as np
import pandas as pd

from sepsis.evaluation.utility import compute_utility_score
from sepsis.utils import save_json


def load_official(path: Path):
    spec = importlib.util.spec_from_file_location("official_scorer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def official_score(mod, labels, preds) -> float:
    """Same aggregation as evaluate_sepsis_score(), without the file I/O."""
    obs = best = inaction = 0.0
    for y, p in zip(labels, preds):
        n = len(y)
        b = np.zeros(n)
        if np.any(y):
            t_sepsis = np.argmax(y) - (-6)
            b[max(0, t_sepsis - 12): min(t_sepsis + 3 + 1, n)] = 1
        obs += mod.compute_prediction_utility(y, p)
        best += mod.compute_prediction_utility(y, b)
        inaction += mod.compute_prediction_utility(y, np.zeros(n))
    return (obs - inaction) / (best - inaction)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--n", type=int, default=3000, help="patients per hospital")
    ap.add_argument("--out", default="results/verification/utility_check.json")
    a = ap.parse_args()
    raw = Path(a.raw)
    mod = load_official(raw / "evaluate_sepsis_score.py")

    files = sorted((raw / "training_setA").glob("*.psv"))[: a.n] + sorted((raw / "training_setB").glob("*.psv"))[: a.n]
    labels = [pd.read_csv(f, sep="|", usecols=["SepsisLabel"])["SepsisLabel"].to_numpy() for f in files]
    rng = np.random.default_rng(0)

    cases = {"all_off": [np.zeros_like(y) for y in labels], "all_on": [np.ones_like(y) for y in labels],
             "labels_as_alarms": labels,
             "labels_shifted_-6h": [np.roll(y, -6) * (np.arange(len(y)) < len(y) - 6) for y in labels]}
    for rate in (0.01, 0.05, 0.2, 0.5):
        cases[f"random_{rate}"] = [(rng.random(len(y)) < rate).astype(int) for y in labels]

    rows, max_diff = {}, 0.0
    for name, preds in cases.items():
        ours = compute_utility_score(labels, preds)
        ref = official_score(mod, labels, preds)
        rows[name] = {"ours": ours, "official": ref, "abs_diff": abs(ours - ref)}
        max_diff = max(max_diff, abs(ours - ref))
        print(f"{name:20s} ours={ours:+.6f} official={ref:+.6f} diff={abs(ours - ref):.2e}")
    result = {"n_patients": len(labels), "cases": rows, "max_abs_diff": max_diff, "pass": max_diff < 1e-9}
    save_json(result, a.out)
    print(("PASS" if result["pass"] else "FAIL") + f": max |diff| = {max_diff:.2e} -> {a.out}")


if __name__ == "__main__":
    main()
