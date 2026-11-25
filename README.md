# Predicting Sepsis Deterioration Trajectories

**A multi-task temporal and graph-based approach to sepsis early warning.**

Course project for AI651 (LUMS, Fall 2026). We build models that, from the last 24 hours of routinely
recorded ICU data, forecast a patient's vital signs, estimate their current sepsis risk and estimate how
far away onset is, all at once. We then test whether modelling the relationships between clinical variables
with a graph attention network adds anything over a temporal Transformer of the same size. This README
describes the project and ends with the results of the mid-project evaluation.

Team: Syed Muhammad Raza Abbas Jafferi · Taha Razzaq · Syed Obaidullah Hassan Chishti

## The project

At every ICU hour *t*, a single model jointly predicts:

1. **Future vitals:** HR, O2Sat, Temp, SBP, MAP, DBP and Resp at *t+1 … t+6 h* (trajectory forecasting)
2. **Sepsis risk:** probability that `SepsisLabel = 1` at hour *t* (onset within 6 h, or already septic)
3. **Time to onset:** which bucket the onset falls in (≤6, 6–12, 12–24, >24 h / never)

**Research question:** does explicitly modelling the relationships between the 33 clinical variables with a
Graph Attention Network (GAT), combined with a temporal Transformer, improve these tasks over a plain
temporal Transformer of the same size? A well-tested negative result is a valid outcome.

**Scope:** early warning only. The models flag rising sepsis risk and forecast vital-sign trajectories; they
make no treatment or drug recommendations. The data contain vitals, labs and labels but almost no
treatment history, so treatment questions are out of scope.

## Data

PhysioNet/CinC Challenge 2019, *Early Prediction of Sepsis from Clinical Data* (open access): 40,336 ICU
patients from two hospitals, 1.55 M hourly rows of vitals, labs and demographics, 7.3% of patients septic
(1.8% of hours positive). `SepsisLabel` turns on 6 h before clinical onset.

- Hospital A is split 70/15/15 by patient into train / validation / test_A (stratified on sepsis).
- **Hospital B is held out entirely** as an external test set (test_B), never used for training or tuning.
- Preprocessing is strictly causal: values are forward-filled only, normalised with training-set
  statistics, and never-observed values become 0 (the training mean). Total length of stay is never a
  feature, because septic records are truncated and it would leak the label.

Every model gets the same inputs for a 24 h lookback window: the 33 normalised variables (7 vitals and 26
labs), a missingness mask, log time since each variable was last measured, a padding flag, and static
features (age, gender, ICU unit, time since hospital admission, hours in ICU so far). Inputs are clipped to the
range seen in the training patients, which guards against data-entry errors at Hospital B. During training each
patient contributes 8 random windows per epoch and septic patients 4× as many. Details: [docs/data.md](docs/data.md).

## Models

| Model | Config | Predicts | Notes |
|---|---|---|---|
| Persistence | `persistence.yaml` | vitals | last observed value |
| Linear regression | `linear.yaml` | vitals | flattened window → vitals |
| MLP multi-task | `mlp.yaml` | all | simplest all-task model |
| GRU-D | `grud.yaml` | all | recurrent, learned missingness decay |
| Informer (encoder) | `informer.yaml` | all | ProbSparse self-attention + distilling |
| Autoformer (encoder) | `autoformer.yaml` | all | series decomposition + auto-correlation |
| Temporal Transformer | `transformer.yaml` | all | **primary baseline**, parameter-matched to the GAT model |
| GAT + Temporal Transformer | `gat_transformer.yaml` | all | **proposed**, learned graph over variables |
| GAT + Temporal Transformer | `gat_clinical_prior.yaml` | all | same model, graph from clinical organ systems |
| GAT + Temporal Transformer | `gat_correlation.yaml` | all | same model, graph from training-set correlations |

All multi-task models share the same three prediction heads and loss, so differences come from the encoder.
The onset loss is class-weighted (balanced inverse bucket frequencies), because most hours fall in the
">24 h / never" bucket.

**Model selection (since 2026-11-25).** Each head is taken from the epoch with its own best validation
score (risk AUPRC, vitals MAE, onset macro-F1). Training stops when no head has improved for 5 epochs.
Previously one checkpoint was chosen on AUPRC alone, which stopped every head after 1–6 epochs.
`--override train.selection=risk_auprc` restores the old protocol.
Informer and Autoformer use their encoders only (vitals come from the shared forecast head, not their
decoders). Details: [docs/methodology.md](docs/methodology.md).

## Evaluation

| Project brief | Reported as |
|---|---|
| AUROC, AUPRC | hourly sepsis risk on test_A and test_B |
| Early-warning utility | official Challenge 2019 normalised utility; detection rate; utility at 1 false-alarm hour per patient-day |
| Trajectory MAE | vitals forecast MAE over 7 vitals × 6 hours, clinical units, measured values only |
| (time to alert) | onset-bucket macro-F1 |

Alarm thresholds are chosen on the validation set and then frozen. Onset is also reported per bucket, as
balanced accuracy, and on septic patients only. Details: [docs/evaluation.md](docs/evaluation.md).

## Repository layout

```
sepsis-trajectory/
├── configs/
│   ├── default.yaml           shared settings (data, model, training, evaluation)
│   └── experiments/           one file per model (inherits default.yaml)
├── data/                      raw/ and processed/ (not committed)
├── docs/                      data, methodology, evaluation, experiment plan
├── scripts/
│   ├── download_data.sh       fetch the Challenge 2019 data (Kaggle) and the official scorer
│   ├── preprocess.py          raw → processed (causal, train-only normalisation)
│   ├── train.py               train and evaluate one model / seed
│   ├── run_mid_eval.sh        all models, seed 0, then summary tables
│   └── summarize.py           results tables
├── src/sepsis/                data, graphs, models, training, evaluation
└── results/                   mid-project results (see below)
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
SOURCE=physionet bash scripts/download_data.sh   # straight from PhysioNet, no account needed
# or: bash scripts/download_data.sh              # Kaggle mirror, needs a Kaggle API token (docs/data.md)
python scripts/preprocess.py      # once; writes data/processed/ (~1 min, ~2 GB RAM)
OUT=results/per_task bash scripts/run_mid_eval.sh   # 10 models, seed 0 (~2.5 h on an M2 MacBook)
```

Hardware: the device is picked automatically (CUDA, then Apple-silicon MPS, then CPU). An M2 MacBook with
16 GB runs one model in 1–31 min (all ten in about 2.5 h), so no cloud GPU is needed for single-seed runs.
A 3-seed grid (about 7.5 h on the M2) is quicker on a CUDA GPU or Kaggle.

One model on its own: `python scripts/train.py --config configs/experiments/gat_transformer.yaml --seed 0`.
Each run writes `config.yaml`, `run_info.json`, `history.json`, `thresholds.json` and `metrics.json` to
`results/<model>/seed0/`, plus one weight snapshot per head (`best_<task>.pt`, not committed).
`run_mid_eval.sh` skips runs that are already done. `--eval-only` re-evaluates a finished run from its
saved snapshots without retraining.

## Results so far

### Latest: all models with per-task selection and onset fix (2026-11-25)

Seed 0, Apple M2 (MPS). Full write-up, Hospital B table and comparison with the old protocol:
[results/per_task/RESULTS.md](results/per_task/RESULTS.md).

Hospital A test set (test_A); in brackets, Hospital B (test_B):

| Model | Risk from epoch | AUROC | AUPRC | ECE | Utility | Utility@1FA | Onset macro-F1 | Onset error (h) | Vitals MAE |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Persistence | – | – | – | – | – | – | – | – | 6.437 (6.961) |
| Linear | – | – | – | – | – | – | – | – | 6.000 (6.618) |
| MLP | 1 | 0.766 (0.741) | 0.061 (0.037) | 0.375 | 0.337 (0.141) | 0.164 (0.048) | 0.250 | 12.3 | 6.450 (7.208) |
| GRU-D | 8 | 0.729 (0.694) | 0.054 (0.034) | 0.095 | 0.245 (0.139) | 0.181 (0.130) | 0.261 | 12.8 | **5.803 (6.487)** |
| Informer | 25 | 0.739 (0.687) | 0.064 (0.040) | 0.068 | 0.268 (0.160) | 0.213 (0.141) | **0.293** | 16.0 | 6.028 (6.832) |
| Autoformer | 3 | 0.767 (0.765) | 0.057 (0.045) | 0.313 | **0.350** (0.222) | 0.076 (0.077) | 0.239 | **9.3** | 5.938 (6.598) |
| Transformer | 24 | 0.737 (0.700) | 0.064 (0.042) | **0.064** | 0.243 (0.155) | 0.199 (0.148) | 0.283 | 16.5 | 5.919 (6.638) |
| GAT + Transformer (learned) | 1 | **0.769 (0.785)** | **0.064 (0.061)** | 0.291 | 0.345 (**0.242**) | 0.180 (0.154) | 0.282 | 14.5 | 5.926 (6.675) |
| GAT + Transformer (clinical prior) | 11 | 0.756 (0.731) | 0.062 (0.044) | 0.083 | 0.286 (0.120) | **0.240 (0.195)** | 0.282 | 15.6 | 5.870 (6.545) |
| GAT + Transformer (correlation) | 1 | 0.768 (0.783) | 0.063 (0.058) | 0.290 | 0.324 (0.235) | 0.177 (0.155) | 0.281 | 15.2 | 5.894 (6.612) |

- **Vitals:** the models that the old protocol stopped at epoch 1 gained most. The fixed-graph GATs went
  from 6.46/6.42 to 5.87/5.89. GRU-D is still the best forecaster (5.80).
- **Onset:** every model now predicts all four buckets. Error for septic hours fell from 17.5–19.3 h to
  9.3–17.4 h, Autoformer lowest. No model separates neighbouring buckets well (best macro-F1 0.29).
- **Risk:** results split by the epoch the risk head came from.
  - Epoch 1–3 snapshots (MLP, Autoformer, learned and correlation GAT) rank best, with AUROC up to 0.785
    on Hospital B, but are poorly calibrated.
  - Later snapshots (GRU-D, Informer, Transformer, clinical-prior GAT) are well calibrated and mostly
    best at one false alarm per day.
  - Validation AUPRC is too noisy for the epoch choice to mean much.
- **Graph question:** the fixed-graph GATs are slightly ahead of the matched Transformer on vitals MAE
  and on utility at one false alarm per day. This is one seed, within seed noise, and the earlier 3-seed
  run found that a GAT without edges does as well. Still no evidence for the graph.

### Mid-project evaluation (old protocol, 2026-11-24)

**These are single-seed results (seed 0)**, selected on validation AUPRC alone, without onset class
weights or input clipping. Superseded by the run above; kept for reference. Differences of a few thousandths are within run-to-run noise.
Full tables (including validation, per-vital and per-horizon errors) are in
[results/RESULTS.md](results/RESULTS.md) and `results/tables/`.

Run on 2026-11-24, one run per model (seed 0), on one NVIDIA RTX 4060 laptop GPU.

### Hospital A test set (test_A)

| Model | Params | AUROC | AUPRC | Utility | Detection | Utility@1FA | Onset macro-F1 | Vitals MAE |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Persistence | 0 | – | – | – | – | – | – | 6.437 |
| Linear | 100,842 | – | – | – | – | – | – | 5.965 |
| MLP | 380,207 | 0.759 | 0.061 | 0.345 | 0.947 | 0.117 | 0.254 | 6.458 |
| GRU-D | 84,689 | 0.757 | 0.056 | 0.318 | 0.942 | 0.067 | 0.256 | **5.803** |
| Informer | 162,567 | 0.746 | 0.057 | 0.295 | 0.855 | 0.135 | 0.257 | 6.100 |
| Autoformer | 168,455 | 0.765 | 0.060 | 0.330 | 0.947 | 0.115 | 0.252 | 5.927 |
| Transformer | 165,575 | 0.755 | **0.064** | 0.306 | **0.966** | **0.207** | **0.267** | 6.045 |
| GAT + Transformer (learned) | 159,855 | 0.760 | 0.060 | 0.323 | 0.957 | 0.167 | 0.251 | 6.130 |
| GAT + Transformer (clinical prior) | 159,855 | **0.777** | 0.062 | 0.355 | 0.942 | 0.130 | 0.246 | 6.455 |
| GAT + Transformer (correlation) | 159,855 | 0.776 | 0.063 | **0.356** | 0.928 | 0.136 | 0.247 | 6.424 |

### Hospital B, external test set (test_B)

| Model | AUROC | AUPRC | Utility | Detection | Utility@1FA | Onset macro-F1 | Vitals MAE |
|---|--:|--:|--:|--:|--:|--:|--:|
| Persistence | – | – | – | – | – | – | 6.961 |
| Linear | – | – | – | – | – | – | 6.747 |
| MLP | 0.713 | 0.032 | 0.120 | 0.791 | 0.029 | 0.252 | 7.200 |
| GRU-D | 0.742 | 0.039 | 0.170 | 0.800 | 0.029 | 0.251 | **6.489** |
| Informer | 0.696 | 0.032 | 0.161 | 0.890 | 0.090 | 0.254 | 6.851 |
| Autoformer | 0.753 | 0.047 | 0.197 | 0.891 | 0.085 | 0.250 | 6.662 |
| Transformer | 0.730 | 0.044 | 0.095 | **0.980** | **0.192** | **0.266** | 6.699 |
| GAT + Transformer (learned) | 0.730 | 0.043 | 0.134 | 0.953 | 0.162 | 0.250 | 7.051 |
| GAT + Transformer (clinical prior) | **0.772** | **0.054** | **0.232** | 0.844 | 0.093 | 0.246 | 7.312 |
| GAT + Transformer (correlation) | 0.771 | 0.053 | 0.227 | 0.815 | 0.101 | 0.246 | 7.252 |

Utility = official Challenge utility at the validation-chosen best threshold; Utility@1FA = utility at a
threshold allowing 1 false-alarm hour per non-septic patient-day; Vitals MAE in clinical units.

### What we see so far

- **Risk prediction is close across models.** On Hospital A, AUROC is 0.746–0.777 and AUPRC 0.056–0.064.
  The GAT models with a fixed graph (clinical prior, correlation) have the highest AUROC and utility, but
  with one seed the gap to the parameter-matched Transformer is not yet evidence of a real improvement.
- **Generalisation to a new hospital is the main gap.** Every model drops on Hospital B (AUPRC
  0.032–0.054). The fixed-graph GAT models and Autoformer hold up best in this run.
- **The alarm threshold changes the ranking.** At a fixed false-alarm budget the Transformer has the best
  utility on both hospitals, while at the utility-optimal threshold the fixed-graph GAT models lead.
- **Vital-sign trajectories:** GRU-D and Autoformer forecast best; the attention models are close to the
  linear baseline.
- **Training stops early.** Several models reach their best validation AUPRC after 1–3 epochs, so the
  forecast heads are undertrained. Fixed by per-task selection (see the latest results above).
