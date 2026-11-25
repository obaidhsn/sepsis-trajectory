"""Per-hour metrics for the three tasks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, roc_auc_score)

from ..constants import VITALS


def expected_calibration_error(y, p, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(y[sel].mean() - p[sel].mean())
    return float(ece)


def risk_metrics(y, p) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p)
    if y.min() == y.max():
        return {"auroc": float("nan"), "auprc": float("nan"), "brier": float("nan"), "ece": float("nan")}
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
    }


def vitals_metrics(y, pred, mask, std=None) -> dict:
    """MAE/RMSE over measured future values, overall, per vital, per horizon.

    Values are in normalised units unless `std` (per-vital train std) is given,
    in which case errors are converted back to clinical units.
    """
    err = pred - y
    if std is not None:
        err = err * np.asarray(std).reshape(1, 1, -1)
    m = mask.astype(bool)
    out = {
        "mae": float(np.abs(err[m]).mean()) if m.any() else float("nan"),
        "rmse": float(np.sqrt((err[m] ** 2).mean())) if m.any() else float("nan"),
        "mae_per_vital": {},
        "mae_per_horizon": [],
    }
    for j, v in enumerate(VITALS):
        mj = m[:, :, j]
        out["mae_per_vital"][v] = float(np.abs(err[:, :, j][mj]).mean()) if mj.any() else float("nan")
    for h in range(err.shape[1]):
        mh = m[:, h]
        out["mae_per_horizon"].append(float(np.abs(err[:, h][mh]).mean()) if mh.any() else float("nan"))
    return out


# Representative hours-to-onset for each bucket, used for an "absolute error
# in hours" summary from bucket predictions.
BUCKET_MIDPOINTS = (3.0, 9.0, 18.0, 36.0)


def onset_metrics(y, prob) -> dict:
    y = np.asarray(y)
    pred = prob.argmax(-1)
    labels = list(range(prob.shape[1]))
    out = {"macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
           "f1_per_bucket": [float(x) for x in f1_score(y, pred, labels=labels, average=None, zero_division=0)],
           "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else float("nan"),
           "accuracy": float((pred == y).mean())}
    mids = np.asarray(BUCKET_MIDPOINTS[: prob.shape[1]])
    septic = y < prob.shape[1] - 1
    if septic.any():
        exp_hours = prob[septic] @ mids
        out["abs_error_hours_septic"] = float(np.abs(exp_hours - mids[y[septic]]).mean())
    return out
