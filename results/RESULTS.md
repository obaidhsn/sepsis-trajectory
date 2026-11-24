# Mid-project evaluation results (seed 0)

**Single-seed numbers.** Every model below was trained once (seed 0) with the same data, splits, inputs,
training loop, early stopping and threshold selection. Differences of a few thousandths in AUROC/AUPRC are
within run-to-run noise and should not be read as one model being better than another.

**Scope.** This work is early warning only: it predicts sepsis risk, time to onset and vital-sign
trajectories from routinely recorded ICU data. It makes no treatment or drug recommendations (the
PhysioNet 2019 data contain almost no treatment history).

Data: PhysioNet/CinC Challenge 2019, 40,336 ICU patients. Hospital A is split 70/15/15 by patient
(train/val/test_A); Hospital B is held out completely as an external test set (test_B).

## Metrics (mapping to the project brief)

| Brief | Column | Meaning |
|---|---|---|
| AUROC | AUROC | hourly sepsis risk (`SepsisLabel` at hour t) |
| AUPRC | AUPRC | same; the positive class is rare, so this is the harder number |
| Early-warning utility | Utility | official PhysioNet/CinC 2019 normalised utility, threshold chosen on val to maximise utility |
| Early-warning utility | Detection | share of septic patients with at least one alarm in the utility window, same threshold |
| Early-warning utility | Utility@1FA | utility with the threshold set on val for 1 false-alarm hour per non-septic patient-day |
| (time to alert) | Onset macro-F1 | time-to-onset buckets: ≤6, 6–12, 12–24, >24 h / never |
| Trajectory MAE | Vitals MAE | forecast of HR, O2Sat, Temp, SBP, MAP, DBP, Resp at t+1…t+6 h, clinical units, measured values only |

Persistence and linear regression only forecast vitals, so their risk and onset cells are empty.
Run time is the whole run (data loading, training, evaluation) on one RTX 4060 laptop GPU.

## Models

- Baselines: persistence (last observed value), linear regression, MLP multi-task, GRU-D.
- Informer (encoder): ProbSparse self-attention (u = 5·ln L active queries, the rest get the mean of V) and
  self-attention distilling (Conv1d k=3 + BatchNorm + ELU + max-pool stride 2) between its two layers;
  representation of the last remaining position, fused with the static features. hidden_dim 68.
- Autoformer (encoder): series decomposition with a moving-average kernel of 7 (the paper's 25 is longer
  than the 24 h window) and Auto-Correlation (FFT, top-k = ln L delays, time-delay aggregation by roll) on
  the seasonal part; representation = last step of the seasonal output plus the last step of the input
  trend, fused with the static features. hidden_dim 72.
- Temporal Transformer: primary baseline, parameter-matched to the GAT model.
- GAT + Temporal Transformer (proposed), with three graphs over the clinical variables: learned
  (`gat_transformer`), clinical prior (`gat_clinical_prior`) and training-set correlation (`gat_correlation`).

Informer and Autoformer use only their encoders, with the same inputs and the same multi-task heads as
every other model, so the comparison is about the encoder alone.

## Results

### Hospital A test set (test_A, in-distribution)

| Model | Params | AUROC | AUPRC | Utility | Detection | Utility@1FA | Onset macro-F1 | Vitals MAE | Best epoch | Run time (min) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| persistence | 0 | – | – | – | – | – | – | 6.437 | – | 1.7 |
| linear | 100,842 | – | – | – | – | – | – | 5.965 | 10 | 6.5 |
| mlp | 380,207 | 0.759 | 0.061 | 0.345 | 0.947 | 0.117 | 0.254 | 6.458 | 2 | 3.2 |
| grud | 84,689 | 0.757 | 0.056 | 0.318 | 0.942 | 0.067 | 0.256 | 5.803 | 2 | 4.8 |
| informer | 162,567 | 0.746 | 0.057 | 0.295 | 0.855 | 0.135 | 0.257 | 6.100 | 7 | 7.8 |
| autoformer | 168,455 | 0.765 | 0.060 | 0.330 | 0.947 | 0.115 | 0.252 | 5.927 | 3 | 6.0 |
| transformer | 165,575 | 0.755 | 0.064 | 0.306 | 0.966 | 0.207 | 0.267 | 6.045 | 6 | 5.0 |
| gat_transformer | 159,855 | 0.760 | 0.060 | 0.323 | 0.957 | 0.167 | 0.251 | 6.130 | 4 | 4.6 |
| gat_clinical_prior | 159,855 | 0.777 | 0.062 | 0.355 | 0.942 | 0.130 | 0.246 | 6.455 | 1 | 3.7 |
| gat_correlation | 159,855 | 0.776 | 0.063 | 0.356 | 0.928 | 0.136 | 0.247 | 6.424 | 1 | 3.7 |

### Hospital B (test_B, external hospital, never used for training or tuning)

| Model | Params | AUROC | AUPRC | Utility | Detection | Utility@1FA | Onset macro-F1 | Vitals MAE | Best epoch | Run time (min) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| persistence | 0 | – | – | – | – | – | – | 6.961 | – | 1.7 |
| linear | 100,842 | – | – | – | – | – | – | 6.747 | 10 | 6.5 |
| mlp | 380,207 | 0.713 | 0.032 | 0.120 | 0.791 | 0.029 | 0.252 | 7.200 | 2 | 3.2 |
| grud | 84,689 | 0.742 | 0.039 | 0.170 | 0.800 | 0.029 | 0.251 | 6.489 | 2 | 4.8 |
| informer | 162,567 | 0.696 | 0.032 | 0.161 | 0.890 | 0.090 | 0.254 | 6.851 | 7 | 7.8 |
| autoformer | 168,455 | 0.753 | 0.047 | 0.197 | 0.891 | 0.085 | 0.250 | 6.662 | 3 | 6.0 |
| transformer | 165,575 | 0.730 | 0.044 | 0.095 | 0.980 | 0.192 | 0.266 | 6.699 | 6 | 5.0 |
| gat_transformer | 159,855 | 0.730 | 0.043 | 0.134 | 0.953 | 0.162 | 0.250 | 7.051 | 4 | 4.6 |
| gat_clinical_prior | 159,855 | 0.772 | 0.054 | 0.232 | 0.844 | 0.093 | 0.246 | 7.312 | 1 | 3.7 |
| gat_correlation | 159,855 | 0.771 | 0.053 | 0.227 | 0.815 | 0.101 | 0.246 | 7.252 | 1 | 3.7 |

### Hospital A validation set (val, used for early stopping and thresholds)

| Model | Params | AUROC | AUPRC | Utility | Detection | Utility@1FA | Onset macro-F1 | Vitals MAE | Best epoch | Run time (min) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| persistence | 0 | – | – | – | – | – | – | 6.374 | – | 1.7 |
| linear | 100,842 | – | – | – | – | – | – | 5.944 | 10 | 6.5 |
| mlp | 380,207 | 0.793 | 0.071 | 0.401 | 0.943 | 0.135 | 0.259 | 6.435 | 2 | 3.2 |
| grud | 84,689 | 0.791 | 0.060 | 0.417 | 0.948 | 0.085 | 0.254 | 5.778 | 2 | 4.8 |
| informer | 162,567 | 0.771 | 0.066 | 0.344 | 0.853 | 0.134 | 0.261 | 6.094 | 7 | 7.8 |
| autoformer | 168,455 | 0.807 | 0.071 | 0.414 | 0.972 | 0.142 | 0.252 | 5.901 | 3 | 6.0 |
| transformer | 165,575 | 0.777 | 0.065 | 0.333 | 0.962 | 0.196 | 0.264 | 6.027 | 6 | 5.0 |
| gat_transformer | 159,855 | 0.797 | 0.070 | 0.394 | 0.943 | 0.201 | 0.251 | 6.112 | 4 | 4.6 |
| gat_clinical_prior | 159,855 | 0.804 | 0.069 | 0.412 | 0.948 | 0.144 | 0.245 | 6.452 | 1 | 3.7 |
| gat_correlation | 159,855 | 0.804 | 0.068 | 0.409 | 0.938 | 0.157 | 0.245 | 6.418 | 1 | 3.7 |

## Observations

- On Hospital A the risk models are close: AUROC 0.746–0.777 and AUPRC 0.056–0.064. The two GAT models
  with a fixed graph (clinical prior, correlation) have the highest AUROC and utility, but with one seed
  these gaps are within run-to-run noise.
- On the external Hospital B every model drops (AUPRC 0.032–0.054, utility 0.095–0.232), which is the main
  generalisation gap to address. The fixed-graph GAT models and Autoformer hold up best there in this run.
- The Transformer has the best utility at a fixed false-alarm budget on both test sets (0.207 / 0.192), so
  the ranking depends on how the alarm threshold is set; both settings are reported.
- For vital-sign trajectories, GRU-D (5.80) and Autoformer (5.93) have the lowest MAE on Hospital A,
  ahead of the linear baseline (5.97); persistence is 6.44.
- Several models reach their best validation AUPRC after only 1–3 epochs, so early stopping ends training
  before the forecast heads are fully trained.

## Files

- `tables/summary_<split>.{md,csv}`, `tables/vitals_<split>.md` (MAE per vital and per horizon), `tables/runs.csv`
- `<model>/seed0/`: `config.yaml`, `run_info.json`, `history.json`, `thresholds.json`, `metrics.json`
- `logs/<model>_seed0.log`
