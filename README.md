# Predicting Sepsis Deterioration Trajectories

**a multi-task temporal and graph-based approach to sepsis early warning**.

At every ICU hour *t* the model jointly predicts:

1. **Future vitals**: HR, O2Sat, Temp, SBP, MAP, DBP, Resp at *t+1 … t+6 h*
2. **Sepsis risk**: probability that `SepsisLabel = 1` at hour *t* (onset within 6 h, or already septic)
3. **Time to onset**: bucket of hours until onset (≤6, 6–12, 12–24, >24 h / never)

**Research question:** does explicitly modelling relationships between physiological variables with a
Graph Attention Network (GAT) improve these tasks over a plain temporal Transformer of the same size?
A well-tested negative result is a valid outcome. The scope is early warning only, with no treatment claims.

Team: Syed Muhammad Raza Abbas Jafferi · Taha Razzaq · Syed Obaidullah Hassan Chishti

---

## Repository layout

```
sepsis-trajectory/
├── README.md                  you are here
├── requirements.txt           pip dependencies
├── pyproject.toml             makes `src/sepsis` installable (pip install -e .)
├── configs/
│   ├── default.yaml           shared hyper-parameters (data, model, training, eval)
│   └── experiments/           one file per model / ablation (inherit default.yaml)
├── data/
│   ├── raw/                   PhysioNet .psv files (not committed)
│   └── processed/             patients.pkl, normalizer.json, splits.json (not committed)
├── docs/
│   ├── data.md                dataset facts, leakage rules, preprocessing
│   ├── methodology.md         models, losses, graphs, design decisions
│   ├── evaluation.md          metrics, threshold protocol, bootstrap
│   ├── experiments.md         experiment plan, priority order, results log
│   └── project_plan.md        milestones, owners, open decisions
├── notebooks/                 exploration and figures (not the source of truth)
├── results/                   one folder per experiment/seed (not committed)
├── scripts/
│   ├── download_data.sh       fetch Challenge 2019 data (Kaggle) + official scorer
│   ├── make_synthetic_data.py fake data with the real schema, for testing
│   ├── preprocess.py          raw → processed (causal, train-only normalisation)
│   ├── train.py               train + evaluate one config/seed
│   ├── run_all.sh             full experiment grid over 3 seeds
│   ├── summarize.py           mean ± std table across seeds
│   ├── compare.py             paired patient-level bootstrap between two runs
│   └── count_params.py        parameter budgets (match Transformer to GAT model)
├── src/sepsis/
│   ├── constants.py           column lists, organ-system groups, label shift
│   ├── config.py              YAML loader with inheritance and CLI overrides
│   ├── data/                  loading, preprocessing, splits, targets, datasets
│   ├── graphs/                learned / clinical-prior / correlation / no graph
│   ├── models/                persistence, linear, MLP, GRU-D, Transformer, GAT+Transformer
│   ├── training/              multi-task loss, trainer with early stopping
│   ├── evaluation/            metrics, Challenge utility, alarms, bootstrap
│   └── utils/                 seeding, JSON I/O
└── tests/                     pytest suite on synthetic data
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .           # optional; scripts also work without it
pytest                     # 16 tests, runs on synthetic data in a few seconds
```

Python ≥ 3.10 and PyTorch ≥ 2.1. A GPU helps but is not required for the baselines.

## Quick start

```bash
# 0. Check the whole pipeline on fake data (no download needed, ~1 min on CPU)
python scripts/make_synthetic_data.py --out data/synthetic --n 60
python scripts/preprocess.py --raw data/synthetic --out data/processed_synthetic
python scripts/train.py --config configs/experiments/debug.yaml --processed data/processed_synthetic

# 1. Real data
pip install kaggle                      # once; needs ~/.kaggle/kaggle.json, see docs/data.md
bash scripts/download_data.sh           # ~40k .psv files into data/raw/ (from Kaggle)
python scripts/preprocess.py            # add --limit 500 for a fast debug subset

# 2. Train and evaluate one model
python scripts/train.py --config configs/experiments/transformer.yaml --seed 0
python scripts/train.py --config configs/experiments/gat_transformer.yaml --seed 0

# 3. Compare (patient-level bootstrap, 95% CI)
python scripts/compare.py results/gat_transformer/seed0 results/transformer/seed0 --split test_A

# 4. Everything, 3 seeds, then summary tables and bootstrap comparisons over all seeds
python scripts/verify_utility.py                # utility score vs the official scorer
bash scripts/run_all.sh                         # logs in results/logs/, tables in results/tables/
python scripts/summarize.py --split test_B      # print one split (all are written to results/tables/)
python scripts/bootstrap_compare.py             # every model vs the Transformer, test_A and test_B
```

Override any config value from the command line, e.g. lookback ablation or single-task runs:

```bash
python scripts/train.py --config configs/experiments/gat_transformer.yaml \
    --name gat_lookback48 --override data.lookback=48
python scripts/train.py --config configs/experiments/transformer.yaml \
    --name transformer_risk_only --override "train.tasks=[risk]"
```

Each run writes to `results/<name>/seed<k>/`: the resolved `config.yaml`, `run_info.json`, `best.pt`, `history.json`,
`thresholds.json` (chosen on validation), `metrics.json` (val, test_A, test_B) and `predictions_*.npz`.

## Models

| Model | Config | Vitals | Risk | Onset | Notes |
|---|---|:-:|:-:|:-:|---|
| Persistence | `persistence.yaml` | ✓ | – | – | last observed value |
| Linear regression | `linear.yaml` | ✓ | – | – | flattened window → vitals |
| MLP multi-task | `mlp.yaml` | ✓ | ✓ | ✓ | simplest all-task model |
| GRU-D | `grud.yaml` | ✓ | ✓ | ✓ | recurrent, learned missingness decay |
| Temporal Transformer | `transformer.yaml` | ✓ | ✓ | ✓ | **primary baseline**, parameter-matched |
| GAT + Temporal Transformer | `gat_transformer.yaml` | ✓ | ✓ | ✓ | **proposed** |
| Graph ablations | `gat_clinical_prior.yaml`, `gat_correlation.yaml`, `gat_none.yaml` | ✓ | ✓ | ✓ | same model, different graph |

All models get identical inputs: values, missingness mask, time since last measurement, and static features.
See [docs/methodology.md](docs/methodology.md).

## Data and leakage rules

PhysioNet/CinC Challenge 2019 (open access): 40,336 ICU patients from hospitals A and B.
We train/validate/test on Hospital A (70/15/15, patient-level, stratified) and keep **Hospital B
entirely held out** as an external test set. Preprocessing is strictly causal (forward-fill only),
normalisation statistics come from training patients only, and length of stay is never a feature.
Details: [docs/data.md](docs/data.md).

## Evaluation

AUROC/AUPRC/Brier/ECE for risk, MAE/RMSE per vital and per horizon for forecasts, macro-F1 for onset buckets,
the official Challenge utility score, warning hours, detection rate and false alarms per patient-day.
Alarm thresholds are chosen on validation and frozen. Comparisons use a patient-level bootstrap.
Details: [docs/evaluation.md](docs/evaluation.md).
