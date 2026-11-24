"""One call that turns predictions into the full metric table."""

from __future__ import annotations

import numpy as np

from .early_warning import early_warning_metrics
from .metrics import onset_metrics, risk_metrics, vitals_metrics


def evaluate_predictions(pred: dict, thresholds: dict | None = None, vital_std=None,
                         t_sepsis_by_patient: np.ndarray | None = None) -> dict:
    out = {}
    if "pred_vitals" in pred:
        out["vitals"] = vitals_metrics(pred["y_vitals"], pred["pred_vitals"], pred["y_vitals_mask"], vital_std)
        if t_sepsis_by_patient is not None:
            # trajectory error in the 12 h before onset, septic patients only
            ts = t_sepsis_by_patient[pred["patient"]]
            sel = (ts >= 0) & (pred["hour"] >= ts - 12) & (pred["hour"] < ts)
            if sel.any():
                out["vitals_pre_onset"] = vitals_metrics(pred["y_vitals"][sel], pred["pred_vitals"][sel],
                                                         pred["y_vitals_mask"][sel], vital_std)
    if "pred_risk" in pred:
        out["risk"] = risk_metrics(pred["y_risk"], pred["pred_risk"])
        for name, thr in (thresholds or {}).items():
            out[f"early_warning@{name}"] = early_warning_metrics(
                pred["y_risk"], pred["pred_risk"], pred["patient"], pred["hour"], thr)
    if "pred_onset" in pred:
        out["onset"] = onset_metrics(pred["y_onset"], pred["pred_onset"])
        if t_sepsis_by_patient is not None:
            # time-to-alert quality where it matters: hours of patients who do become septic
            sel = t_sepsis_by_patient[pred["patient"]] >= 0
            if sel.any():
                out["onset_septic"] = onset_metrics(pred["y_onset"][sel], pred["pred_onset"][sel])
    return out
