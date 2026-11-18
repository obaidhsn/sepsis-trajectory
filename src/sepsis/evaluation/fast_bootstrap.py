"""Fast patient-level paired bootstrap over several seeds.

Resampling patients with replacement is the same as giving every row of
patient i the weight k_i (how often i was drawn). So each resample only
needs re-weighted cumulative sums over rows sorted once per model, instead
of re-sorting or re-scoring ~10^6 hours. The metric of a model in a resample
is the mean over its seeds; the reported quantity is metric(A) - metric(B).

`weighted_auprc` / `weighted_auroc` match sklearn's average_precision_score
and roc_auc_score with sample_weight (tests/test_metrics.py).
"""

from __future__ import annotations

import numpy as np

from .early_warning import group_by_patient
from .utility import UtilityTable


class RankedScores:
    """Rows sorted by descending score, with the end index of every tie group."""

    def __init__(self, y: np.ndarray, p: np.ndarray, patient: np.ndarray):
        order = np.argsort(-p, kind="mergesort")
        self.y = y[order].astype(np.float64)
        self.patient = patient[order]
        ps = p[order]
        self.ends = np.r_[np.flatnonzero(np.diff(ps)), len(ps) - 1]

    def _curves(self, w: np.ndarray):
        wr = w[self.patient]
        tp = np.cumsum(wr * self.y)[self.ends]
        fp = np.cumsum(wr * (1.0 - self.y))[self.ends]
        return tp, fp

    def auprc(self, w: np.ndarray) -> float:
        tp, fp = self._curves(w)
        if tp[-1] <= 0:
            return float("nan")
        precision = tp / np.maximum(tp + fp, 1e-12)
        recall = tp / tp[-1]
        return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

    def auroc(self, w: np.ndarray) -> float:
        tp, fp = self._curves(w)
        if tp[-1] <= 0 or fp[-1] <= 0:
            return float("nan")
        tpr = np.r_[0.0, tp / tp[-1]]
        fpr = np.r_[0.0, fp / fp[-1]]
        return float(np.trapezoid(tpr, fpr))


def weighted_auprc(y, p, w_rows) -> float:
    r = RankedScores(np.asarray(y), np.asarray(p), np.arange(len(y)))
    return r.auprc(np.asarray(w_rows, dtype=np.float64))


def weighted_auroc(y, p, w_rows) -> float:
    r = RankedScores(np.asarray(y), np.asarray(p), np.arange(len(y)))
    return r.auroc(np.asarray(w_rows, dtype=np.float64))


class PatientSums:
    """Per-patient additive statistics, so ratio metrics are weighted sums."""

    def __init__(self, num: np.ndarray, den: np.ndarray):
        self.num, self.den = num, den

    def value(self, w: np.ndarray) -> float:
        d = float(w @ self.den)
        return float(w @ self.num) / d if d else float("nan")


def utility_sums(pred: dict, threshold: float, n_patients: int) -> PatientSums:
    """Normalised utility = sum_i(obs_i - inaction_i) / sum_i(best_i - inaction_i)."""
    ys, ps = group_by_patient(pred["patient"], pred["hour"], pred["y_risk"], pred["pred_risk"])
    num, den = np.zeros(n_patients), np.zeros(n_patients)
    pids = np.unique(pred["patient"])
    for k, y, p in zip(pids, ys, ps):
        t = UtilityTable([y])
        alarms = p >= threshold
        obs = float(np.where(alarms, t.u_alarm, t.u_quiet).sum())
        num[k], den[k] = obs - t.inaction, t.best - t.inaction
    return PatientSums(num, den)


def mae_sums(pred: dict, vital_std: np.ndarray, n_patients: int) -> PatientSums:
    err = np.abs(pred["pred_vitals"] - pred["y_vitals"]) * vital_std.reshape(1, 1, -1)
    err = (err * pred["y_vitals_mask"]).sum((1, 2))
    cnt = pred["y_vitals_mask"].sum((1, 2))
    return PatientSums(np.bincount(pred["patient"], err, n_patients),
                       np.bincount(pred["patient"], cnt, n_patients))


def multi_seed_bootstrap(stats_a: list, stats_b: list, n_patients: int, n_boot: int = 1000,
                         seed: int = 0, alpha: float = 0.05) -> dict:
    """`stats_a/b`: per-seed callables w -> metric (w = per-patient weights)."""
    def value(stats, w):
        return float(np.mean([s(w) for s in stats]))

    ones = np.ones(n_patients)
    point_a, point_b = value(stats_a, ones), value(stats_b, ones)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n_patients, n_patients), minlength=n_patients).astype(np.float64)
        diffs[b] = value(stats_a, w) - value(stats_b, w)
    lo, hi = np.nanpercentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # two-sided bootstrap p-value for "no difference"
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return {"a": point_a, "b": point_b, "diff": point_a - point_b, "ci_low": float(lo),
            "ci_high": float(hi), "p_value": float(min(p, 1.0)), "n_boot": n_boot}
