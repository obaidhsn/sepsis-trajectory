"""Alarm-level metrics from hourly risk scores.

Thresholds are chosen on VALIDATION predictions and then frozen for test.
"""

from __future__ import annotations

import numpy as np

from ..constants import LABEL_SHIFT
from .utility import UtilityTable, compute_utility_score


def group_by_patient(patient: np.ndarray, hour: np.ndarray, *arrays):
    """Split flat per-hour arrays into per-patient sequences ordered by hour."""
    order = np.lexsort((hour, patient))
    patient = patient[order]
    cuts = np.flatnonzero(np.diff(patient)) + 1
    return [np.split(a[order], cuts) for a in arrays]


def early_warning_metrics(labels, probs, patient, hour, threshold: float,
                          min_prelabel_hours: int = 6) -> dict:
    ys, ps = group_by_patient(patient, hour, labels, probs)
    alarms = [p >= threshold for p in ps]
    utility = compute_utility_score(ys, alarms)

    warn_hours, detected, eligible = [], 0, 0
    fa_hours, fa_episodes, neg_hours = 0, 0, 0
    for y, a in zip(ys, alarms):
        if y.any():
            first_pos = int(np.argmax(y))
            if first_pos < min_prelabel_hours:
                continue  # already (nearly) positive at admission
            eligible += 1
            t_sepsis = first_pos + LABEL_SHIFT
            hits = np.flatnonzero(a[: t_sepsis + 1])
            if len(hits):
                detected += 1
                warn_hours.append(t_sepsis - hits[0])
        else:
            neg_hours += len(y)
            fa_hours += int(a.sum())
            fa_episodes += int(np.sum(np.diff(np.concatenate([[0], a.astype(int)])) == 1))
    days = max(neg_hours / 24.0, 1e-9)
    return {
        "threshold": float(threshold),
        "utility": float(utility),
        "detection_rate": detected / max(eligible, 1),
        "n_eligible_septic": eligible,
        "median_warning_hours": float(np.median(warn_hours)) if warn_hours else float("nan"),
        "mean_warning_hours": float(np.mean(warn_hours)) if warn_hours else float("nan"),
        "false_alarm_hours_per_patient_day": fa_hours / days,
        "false_alarm_episodes_per_patient_day": fa_episodes / days,
    }


def select_threshold_utility(labels, probs, patient, hour, grid=None) -> float:
    grid = np.linspace(0.01, 0.99, 99) if grid is None else grid
    ys, ps = group_by_patient(patient, hour, labels, probs)
    table = UtilityTable(ys)
    p = np.concatenate(ps)
    scores = [table.score(p >= t) for t in grid]
    return float(grid[int(np.argmax(scores))])


def select_threshold_fixed_fa(labels, probs, patient, hour, target_fa_per_day: float = 1.0) -> float:
    """Lowest threshold whose false-alarm hours per non-septic patient-day <= target.

    Alarms are `p >= threshold`, so the threshold is placed just above the
    (allowed+1)-th highest negative score: at most `allowed` hours fire.
    """
    ys, ps = group_by_patient(patient, hour, labels, probs)
    neg = np.concatenate([p for y, p in zip(ys, ps) if not y.any()])
    days = len(neg) / 24.0
    allowed = int(target_fa_per_day * days)
    if allowed >= len(neg):
        return 0.0
    cut = np.sort(neg)[::-1][allowed]
    return float(np.nextafter(np.float32(cut), np.float32(np.inf)))
