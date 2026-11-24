"""L = lambda_v * L_vitals + lambda_r * L_risk + lambda_t * L_time  (proposal eq. 1)

L_vitals: MAE only on future values that were actually measured.
L_risk:   class-weighted BCE (positive hours are ~1.8%).
L_time:   cross-entropy over onset buckets, class-weighted. With
          `onset_class_weights: auto` the weights are inverse bucket frequencies
          in the training windows (see `onset_class_weights_from_counts`);
          without them the head collapses to the majority ">24 h / never" bucket.
"""

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


def onset_class_weights_from_counts(counts) -> list[float]:
    """Balanced weights n / (K * n_k), as in scikit-learn's class_weight='balanced'.

    The weighted share of every bucket in the loss is then equal.
    """
    counts = np.asarray(counts, dtype=np.float64)
    w = counts.sum() / (len(counts) * np.maximum(counts, 1.0))
    return [float(x) for x in w]


class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_vitals=1.0, lambda_risk=1.0, lambda_onset=1.0,
                 risk_pos_weight=10.0, onset_class_weights=None):
        super().__init__()
        self.lv, self.lr, self.lt = lambda_vitals, lambda_risk, lambda_onset
        self.register_buffer("pos_weight", torch.tensor(float(risk_pos_weight)))
        w = None if onset_class_weights is None else torch.tensor(onset_class_weights, dtype=torch.float32)
        self.register_buffer("onset_w", w)

    def forward(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        parts = {}
        if "vitals" in out:
            mask = batch["y_vitals_mask"]
            err = (out["vitals"] - batch["y_vitals"]).abs() * mask
            parts["vitals"] = err.sum() / mask.sum().clamp(min=1.0)
        if "risk" in out:
            parts["risk"] = F.binary_cross_entropy_with_logits(
                out["risk"], batch["y_risk"], pos_weight=self.pos_weight)
        if "onset" in out:
            parts["onset"] = F.cross_entropy(out["onset"], batch["y_onset"], weight=self.onset_w)
        weights = {"vitals": self.lv, "risk": self.lr, "onset": self.lt}
        total = sum(weights[k] * v for k, v in parts.items())
        return total, {k: float(v.detach()) for k, v in parts.items()}
