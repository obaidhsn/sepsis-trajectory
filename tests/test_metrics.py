import numpy as np

from sepsis.evaluation.utility import compute_utility_score


def _patients():
    septic = np.array([0] * 10 + [1] * 10)
    normal = np.zeros(30, dtype=int)
    return [septic, normal]


def test_utility_perfect_and_inaction():
    from sepsis.evaluation.utility import _best_predictions
    ys = _patients()
    assert np.isclose(compute_utility_score(ys, [_best_predictions(y) for y in ys]), 1.0)
    assert np.isclose(compute_utility_score(ys, [np.zeros_like(y) for y in ys]), 0.0)


def test_utility_penalises_false_alarms():
    ys = _patients()
    always = compute_utility_score(ys, [np.ones_like(y) for y in ys])
    assert always < 1.0


def test_weighted_ranking_metrics_match_sklearn():
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sepsis.evaluation.fast_bootstrap import weighted_auprc, weighted_auroc
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    p = np.round(rng.random(2000) * 0.5 + 0.4 * y, 2)          # rounding creates ties
    w = rng.integers(0, 3, 2000).astype(float)
    assert np.isclose(weighted_auprc(y, p, w), average_precision_score(y, p, sample_weight=w))
    assert np.isclose(weighted_auroc(y, p, w), roc_auc_score(y, p, sample_weight=w))


def test_threshold_fixed_fa_respects_budget():
    from sepsis.evaluation.early_warning import select_threshold_fixed_fa
    rng = np.random.default_rng(1)
    patient = np.repeat(np.arange(50), 48)
    hour = np.tile(np.arange(48), 50)
    labels = np.zeros(len(patient), dtype=int)
    probs = rng.random(len(patient)).astype(np.float32)
    thr = select_threshold_fixed_fa(labels, probs, patient, hour, 1.0)
    assert (probs >= thr).sum() <= int(len(probs) / 24.0)
