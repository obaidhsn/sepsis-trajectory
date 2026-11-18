# Evaluation

All metrics are in `src/sepsis/evaluation/`. They are computed per split (val, test_A, test_B) by `scripts/train.py`.

## Vital forecasting (`metrics.vitals_metrics`)

- MAE and RMSE over **measured** future values only, in clinical units (de-normalised with train std)
- per vital and per forecast hour (trajectory error across the horizon)
- separately for the 12 h before onset in septic patients (`vitals_pre_onset`)

## Sepsis risk (`metrics.risk_metrics`)

- AUROC, **AUPRC (main metric, given the 1.8% positive rate)**
- calibration: Brier score and expected calibration error (15 bins)

## Time to onset (`metrics.onset_metrics`)

- macro-F1 and accuracy over buckets
- absolute error in hours for septic hours, using bucket midpoints (3, 9, 18, 36 h) as the expected value

## Early warning (`early_warning.py`, `utility.py`)

- **Challenge 2019 normalised utility score**: re-implemented in `utility.py` and checked against the
  official `evaluate_sepsis_score.py` with `scripts/verify_utility.py` (6,000 real label sequences, eight
  kinds of alarm sequences, max |difference| 1.5e-14; `results/verification/utility_check.json`).
- **Detection rate and warning hours**: computed on septic patients with ≥6 h of history before the first
  positive label (2,199 patients per the proposal's profiling). Warning time is `t_sepsis − first alarm`,
  for alarms at or before onset.
- **False alarms per patient-day**: alarm hours and alarm episodes (rising edges) in non-septic patients.

## Threshold protocol

Two thresholds are chosen on **validation** and then frozen:

- `max_utility`: maximises the utility score over a grid
- `fixed_fa`: the lowest threshold with ≤ `eval.fixed_false_alarm_rate` alarm-hours per non-septic patient-day

## Statistical comparison

`scripts/bootstrap_compare.py` runs a paired, patient-level bootstrap (1,000 resamples) of
metric(A) − metric(B) over all seeds the two runs share (each model's metric is its mean over seeds) and
reports a 95% CI and a two-sided bootstrap p-value, for AUPRC, AUROC, utility (each model at its own
validation threshold) and vitals MAE. Output: `results/tables/comparisons.{md,csv,json}`.
`scripts/compare.py` does the same for a single pair of runs. Report every comparison against the plain
Transformer this way.
Across seeds, `scripts/summarize.py` reports mean ± std.

## Generalisation

Report the change in each metric from test_A to test_B (Hospital B is never seen during training or tuning).
