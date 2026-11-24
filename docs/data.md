# Data

## Source

PhysioNet/Computing in Cardiology Challenge 2019, *Early Prediction of Sepsis from Clinical Data*
(open access, https://physionet.org/content/challenge-2019/1.0.0/).

| | Hospital A | Hospital B | Total |
|---|---:|---:|---:|
| Patients | 20,336 | 20,000 | 40,336 |
| Hourly rows | 790,215 | 761,995 | 1,552,210 |
| Septic patients | 1,790 | 1,142 | 2,932 (7.3%) |
| Positive hours | 2.17% | 1.41% | 1.8% |

Median stay is 38 h (39 h at A, 38 h at B).

## Getting the data

**Option 1, PhysioNet directly (no account):** `SOURCE=physionet bash scripts/download_data.sh`. This lists
the files on physionet.org and fetches the 40,336 files in parallel from PhysioNet's open S3 bucket (313 MB,
a few minutes). Re-running it only fetches files that are missing.

**Option 2, Kaggle mirror:** we originally downloaded from the Kaggle mirror
[farjanayesmin/the-physionet-challenge-2019-dataset](https://www.kaggle.com/datasets/farjanayesmin/the-physionet-challenge-2019-dataset),
which is faster than PhysioNet's per-file server.

1. `pip install kaggle`
2. On kaggle.com go to Settings → API → *Create New Token*. This downloads `kaggle.json`.
3. `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
   (or set `KAGGLE_USERNAME` and `KAGGLE_KEY` instead).
4. `bash scripts/download_data.sh`

The script unzips into `data/raw/_kaggle/`, then moves every `p0*.psv` to `data/raw/training_setA/` and
every `p1*.psv` to `data/raw/training_setB/`, whatever folder nesting the archive uses. It ends by printing
the file counts (expect 20,336 and 20,000) and how many files are empty. The loader also searches subfolders
and skips empty or unreadable files with a warning.

## Variables

Defined in `src/sepsis/constants.py`.

- **Vitals (7, forecast targets):** HR, O2Sat, Temp, SBP, MAP, DBP, Resp
- **Labs (26):** BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride,
  Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total,
  TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets
- **Dropped:** EtCO2 (entirely missing at Hospital A)
- **Static / timeline:** Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS
- **Target:** SepsisLabel

The 33 time-varying variables (7 vitals + 26 labs) are the graph nodes.

## Missingness

HR, O2Sat, SBP, MAP and Resp are missing in 10–15% of hours, DBP in 31%, Temp in 66%, and labs in 83–99.8%.
Whether a lab was ordered is itself informative, so the binary mask is always a model input.

## Labels and onset time

`SepsisLabel` is set to 1 from **6 h before** clinical onset, and stays 1 afterwards. Therefore

```
t_sepsis = (index of first positive hour) + 6
```

Septic records end about 10 h after the first positive label (roughly 4 h after onset).

## Leakage rules (do not break these)

1. **No future values.** Forward-fill only. Values never observed so far become 0 after
   normalisation, i.e. the training mean.
2. **Train-only statistics.** Normalisation means/stds and the correlation graph use training patients only.
3. **No stay length.** Septic records are truncated, so total length of stay leaks the label.
   `ICULOS` (hours elapsed *so far*) is causal and allowed.
4. **Patient-level splits.** A patient's hours never appear in two splits.
5. **Hospital B is never used for training, model selection or threshold tuning.**
6. **Thresholds** are chosen on validation and frozen before touching test data.

## Preprocessing pipeline

`scripts/preprocess.py` → `src/sepsis/data/preprocess.py`

For each patient, with T hours and C = 33 variables:

| Array | Shape | Meaning |
|---|---|---|
| `raw` | T×C | raw values with NaN |
| `values` | T×C | causal forward-fill |
| `mask` | T×C | 1 if measured at that hour |
| `delta` | T×C | hours since last measurement (GRU-D convention) |
| `x` | T×C | normalised `values`, NaN → 0 |
| `y_raw` | T×C | normalised `raw`, NaN kept (forecast targets, only measured values count) |
| `delta_n` | T×C | log(1 + delta) |
| `s` | T×S | normalised static features |
| `label` | T | SepsisLabel |
| `t_sepsis` | scalar | onset hour, −1 if never septic |

`patients.pkl` stores only what training and evaluation read: `hospital`, `label`, `t_sepsis`, `x`, `mask`,
`delta_n`, `s` and `y_raw` (`KEEP_KEYS` in `preprocess.py`). `raw`, `values`, `delta` and `static` are
intermediate and dropped (0.88 GB instead of 1.53 GB). Preprocessing also writes `profile.json` with
cohort statistics per hospital and per split.

### Cohort (from `data/processed/profile.json`)

| | Hospital A | Hospital B | Total |
|---|---:|---:|---:|
| Patients | 20,336 | 20,000 | 40,336 |
| Hourly rows | 790,215 | 761,995 | 1,552,210 |
| Septic patients | 1,790 (8.8%) | 1,142 (5.7%) | 2,932 (7.3%) |
| Positive hours | 2.17% | 1.41% | 1.80% |
| Median stay (h) | 39 | 38 | 38 |

| Split | Patients | Hours | Septic patients | Positive hours |
|---|---:|---:|---:|---:|
| train (A) | 14,235 | 553,951 | 1,253 | 2.17% |
| val (A) | 3,050 | 116,624 | 268 | 2.19% |
| test_A | 3,051 | 119,640 | 269 | 2.16% |
| test_B | 20,000 | 761,995 | 1,142 | 1.41% |

## Splits

`src/sepsis/data/splits.py`: Hospital A is split 70/15/15 into train/val/test_A, stratified by ever-septic,
using `data.split_seed` (fixed at 0, so all model seeds share the same split). Hospital B becomes `test_B`.

## Windows

`src/sepsis/data/dataset.py`

- A sample is the window of the last `lookback` hours ending at hour *t* (left-padded, with a pad flag).
- **Training:** each epoch draws `windows_per_patient` random hours per patient, ×`septic_oversample`
  for septic patients, so long stays don't dominate.
- **Evaluation:** every hour of every patient, in order. The utility score and alarm metrics need this.
