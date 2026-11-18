"""Patient-level paired bootstrap for model comparisons (95% CI)."""

from __future__ import annotations

import numpy as np


def paired_bootstrap(metric_fn, pred_a: dict, pred_b: dict, n_boot: int = 1000, seed: int = 0,
                     alpha: float = 0.05) -> dict:
    """CI for metric(A) - metric(B), resampling PATIENTS with replacement.

    `pred_a` / `pred_b` are outputs of `training.predict` on the same dataset
    (same row order). `metric_fn(pred_dict) -> float`.
    """
    rng = np.random.default_rng(seed)
    patient = pred_a["patient"]
    ids = np.unique(patient)
    rows = {i: np.flatnonzero(patient == i) for i in ids}

    def subset(pred, idx):
        return {k: v[idx] for k, v in pred.items()}

    point = metric_fn(pred_a) - metric_fn(pred_b)
    diffs = []
    for b in range(n_boot):
        sample = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([rows[i] for i in sample])
        # re-label patients so duplicated patients count as distinct
        new_pid = np.concatenate([np.full(len(rows[i]), k) for k, i in enumerate(sample)])
        sa, sb = subset(pred_a, idx), subset(pred_b, idx)
        sa["patient"] = sb["patient"] = new_pid
        diffs.append(metric_fn(sa) - metric_fn(sb))
    lo, hi = np.nanpercentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": float(point), "ci_low": float(lo), "ci_high": float(hi)}
