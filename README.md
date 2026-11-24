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
features (age, gender, ICU unit, time since hospital admission, hours in ICU so far). During training each
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
Informer and Autoformer use their encoders only (vitals come from the shared forecast head, not their
decoders). Details: [docs/methodology.md](docs/methodology.md).

## Evaluation

| Project brief | Reported as |
|---|---|
| AUROC, AUPRC | hourly sepsis risk on test_A and test_B |
| Early-warning utility | official Challenge 2019 normalised utility; detection rate; utility at 1 false-alarm hour per patient-day |
| Trajectory MAE | vitals forecast MAE over 7 vitals × 6 hours, clinical units, measured values only |
| (time to alert) | onset-bucket macro-F1 |

Alarm thresholds are chosen on the validation set and then frozen. Early stopping monitors validation
AUPRC. Details: [docs/evaluation.md](docs/evaluation.md).

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
bash scripts/download_data.sh     # PhysioNet 2019 from Kaggle into data/raw/ (see docs/data.md)
python scripts/preprocess.py      # once; writes data/processed/
bash scripts/run_mid_eval.sh      # 10 models, seed 0 (~50 min on an RTX 4060 laptop GPU)
```

One model on its own: `python scripts/train.py --config configs/experiments/gat_transformer.yaml --seed 0`.
Each run writes `config.yaml`, `run_info.json`, `history.json`, `thresholds.json` and `metrics.json` to
`results/<model>/seed0/`; `run_mid_eval.sh` skips runs that are already done.

## Results so far (mid-project evaluation)

**These are single-seed results (seed 0).** Differences of a few thousandths are within run-to-run noise.
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
  forecast heads are undertrained.
