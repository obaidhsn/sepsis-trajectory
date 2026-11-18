# Changelog

All notable changes to the code, in reverse chronological order.

## 2026-11-18: first full run on the real data

### Fixed

- **CLI overrides in scientific notation** (`config.py`): PyYAML follows YAML 1.1 and parses `3e-4`
  (no decimal point) as the *string* `"3e-4"`, so `--override train.lr=3e-4` crashed AdamW. Override values
  that parse as strings are now tried as floats first. Covered by a new test.
- **Official scorer download** (`scripts/download_data.sh`): the PhysioNet URL for
  `evaluate_sepsis_score.py` returns 404. The script now fetches it from the challenge's GitHub repository
  (`physionetchallenges/evaluation-2019`) with `curl -f`, so a failed download is reported instead of
  silently leaving no file.
- **Fixed-false-alarm threshold off by one** (`evaluation/early_warning.py::select_threshold_fixed_fa`):
  the threshold was set *equal* to the (allowed+1)-th highest negative score, and alarms fire at
  `p >= threshold`, so one more alarm hour than the budget (plus any ties) fired. The threshold is now
  the next float above that score, so at most `allowed` alarm-hours fire. Covered by a new test.
- **JSON export of NumPy booleans** (`utils/io.py`): `np.bool_` values raised `TypeError` in `save_json`.
- **Memory use of `patients.pkl`** (`data/preprocess.py`): the file also stored the intermediate arrays
  (`raw`, `values`, `delta`, `static`) that nothing reads after preprocessing. Only the model-ready arrays
  are kept now (`KEEP_KEYS`), which shrinks the file from 1.53 GB to 0.88 GB and the RAM needed for
  training by about the same. On a 16 GB laptop this matters.

### Changed

- **Utility score is vectorised** (`evaluation/utility.py`): `hour_utilities` gives, per hour, the utility
  of raising or not raising an alarm, and `UtilityTable` turns any alarm sequence into a dot product.
  The line-by-line port of the official loop (`_patient_utility`) is kept as the reference.
  The threshold sweep over the validation set went from tens of seconds to well under a second, so the
  grid was refined from 49 to 99 thresholds (0.01 … 0.99).
- **Deterministic cuDNN** (`utils/seed.py`): `cudnn.benchmark = False`, `cudnn.deterministic = True`.
- **Training history** (`training/trainer.py`): every epoch now records the per-task training losses,
  the monitored validation metric under its own name (`val_auprc` or `neg_val_vitals_mae`), and timings.
  `fit()` returns `best_epoch` and `epochs_run`. The progress bar is only shown in a terminal, so log files
  stay readable.
- **Run metadata** (`scripts/train.py`): each run writes `run_info.json` (command, overrides, seed, device,
  GPU, library versions, parameter count, best epoch, wall-clock times).
- **`scripts/run_all.sh`**: runs all nine experiments (five baselines, GAT model, three graph ablations)
  seed by seed, logs each run to `results/logs/<experiment>_seed<k>.log`, skips runs that already have a
  `metrics.json` (resumable), and finishes with `summarize.py`. `PY`, `SEEDS`, `EXPS`, `SUFFIX` and `OVERRIDES` can be set
  from the environment.
- **`scripts/summarize.py`**: writes `results/tables/runs.csv` (every scalar metric of every run and split),
  `summary_<split>.{csv,md}` (mean ± std over seeds) and `vitals_<split>.md` (per-vital and per-horizon MAE).

### Added

- `data.use_static` config flag (default `true`): `false` zeroes the static/timeline features, as an input
  ablation (`scripts/train.py`).
- `data/processed/profile.json`: cohort statistics per hospital and per split, written by preprocessing.
- `scripts/verify_utility.py`: compares our utility score with the official scorer on real label
  sequences and eight kinds of alarm sequences; writes `results/verification/utility_check.json`.
- `evaluation/fast_bootstrap.py` and `scripts/bootstrap_compare.py`: paired patient-level bootstrap
  that uses all shared seeds (metric = mean over seeds). Resampling patients is done with per-patient
  weights on rows sorted once, so 1,000 resamples on Hospital B (762k hours) take seconds.
  Weighted AUPRC/AUROC are tested against scikit-learn.
- `CHANGELOG.md` and `docs/results.md`.
