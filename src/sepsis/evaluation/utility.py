"""PhysioNet/CinC Challenge 2019 normalised utility score.

Re-implementation of the official `compute_utility` logic:
  * Septic patient, onset t_sepsis = first positive label + 6.
  * Reward for alarms between t_sepsis-12 and t_sepsis+3, peaking at t_sepsis-6.
  * Small penalty (-0.05) for every alarm hour in non-septic patients.
  * Penalty for missing the window after t_sepsis.
Normalised so that "always predict 0" = 0 and the best possible = 1.

Checked against the official evaluate_sepsis_score.py with
scripts/verify_utility.py (scripts/download_data.sh fetches it into data/raw/).

`_patient_utility` is a line-by-line port of the official loop. The score
itself is computed from per-hour utility tables (`hour_utilities`), which
gives identical numbers but makes threshold sweeps over ~10^6 hours cheap.
"""

from __future__ import annotations

import numpy as np

DT_EARLY, DT_OPTIMAL, DT_LATE = -12, -6, 3
MAX_U_TP, MIN_U_FN, U_FP, U_TN = 1.0, -2.0, -0.05, 0.0


def _patient_utility(labels: np.ndarray, preds: np.ndarray) -> float:
    n = len(labels)
    septic = labels.any()
    t_sepsis = int(np.argmax(labels)) - DT_OPTIMAL if septic else np.inf

    m1 = MAX_U_TP / (DT_OPTIMAL - DT_EARLY)
    b1 = -m1 * DT_EARLY
    m2 = -MAX_U_TP / (DT_LATE - DT_OPTIMAL)
    b2 = -m2 * DT_LATE
    m3 = MIN_U_FN / (DT_LATE - DT_OPTIMAL)
    b3 = -m3 * DT_OPTIMAL

    u = np.zeros(n)
    for t in range(n):
        if t > t_sepsis + DT_LATE:
            continue
        rel = t - t_sepsis
        if septic and preds[t]:
            u[t] = max(m1 * rel + b1, U_FP) if t <= t_sepsis + DT_OPTIMAL else m2 * rel + b2
        elif not septic and preds[t]:
            u[t] = U_FP
        elif septic and not preds[t]:
            u[t] = 0.0 if t <= t_sepsis + DT_OPTIMAL else m3 * rel + b3
        else:
            u[t] = U_TN
    return float(u.sum())


def _best_predictions(labels: np.ndarray) -> np.ndarray:
    n = len(labels)
    best = np.zeros(n, dtype=int)
    if labels.any():
        t_sepsis = int(np.argmax(labels)) - DT_OPTIMAL
        best[max(0, t_sepsis + DT_EARLY): min(t_sepsis + DT_LATE + 1, n)] = 1
    return best


def hour_utilities(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Utility of each hour if an alarm is raised (u_alarm) or not (u_quiet).

    Vectorised form of `_patient_utility`: for any binary predictions p,
    _patient_utility(labels, p) == sum(where(p, u_alarm, u_quiet)).
    """
    labels = np.asarray(labels)
    n = len(labels)
    if not labels.any():
        return np.full(n, U_FP), np.full(n, U_TN)
    t = np.arange(n)
    t_sepsis = int(np.argmax(labels)) - DT_OPTIMAL
    rel = t - t_sepsis
    m1 = MAX_U_TP / (DT_OPTIMAL - DT_EARLY)
    b1 = -m1 * DT_EARLY
    m2 = -MAX_U_TP / (DT_LATE - DT_OPTIMAL)
    b2 = -m2 * DT_LATE
    m3 = MIN_U_FN / (DT_LATE - DT_OPTIMAL)
    b3 = -m3 * DT_OPTIMAL
    early = t <= t_sepsis + DT_OPTIMAL
    u_alarm = np.where(early, np.maximum(m1 * rel + b1, U_FP), m2 * rel + b2)
    u_quiet = np.where(early, 0.0, m3 * rel + b3)
    after = t > t_sepsis + DT_LATE
    u_alarm[after] = 0.0
    u_quiet[after] = 0.0
    return u_alarm, u_quiet


class UtilityTable:
    """Pre-computed utilities for a fixed set of patients, so the normalised
    score of many alarm sequences (e.g. a threshold sweep) is a dot product."""

    def __init__(self, labels_by_patient: list[np.ndarray]):
        parts = [hour_utilities(y) for y in labels_by_patient]
        self.u_alarm = np.concatenate([a for a, _ in parts]) if parts else np.zeros(0)
        self.u_quiet = np.concatenate([q for _, q in parts]) if parts else np.zeros(0)
        best = np.concatenate([_best_predictions(y) for y in labels_by_patient]).astype(bool) \
            if parts else np.zeros(0, dtype=bool)
        self.inaction = float(self.u_quiet.sum())
        self.best = float(np.where(best, self.u_alarm, self.u_quiet).sum())

    def score(self, alarms: np.ndarray) -> float:
        """`alarms`: flat boolean array in the same patient/hour order."""
        observed = float(np.where(alarms, self.u_alarm, self.u_quiet).sum())
        return (observed - self.inaction) / (self.best - self.inaction)


def compute_utility_score(labels_by_patient: list[np.ndarray], preds_by_patient: list[np.ndarray]) -> float:
    table = UtilityTable([np.asarray(y).astype(int) for y in labels_by_patient])
    alarms = np.concatenate([np.asarray(p).astype(bool) for p in preds_by_patient])
    return table.score(alarms)
