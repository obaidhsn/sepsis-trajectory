# Results: first full run (2026-11-18)

All nine experiments in `configs/experiments/` were trained on the real PhysioNet 2019 data with seeds 0, 1
and 2, using the configs exactly as committed (`configs/default.yaml`). Numbers are mean ± std over the three
seeds. Hospital A is split 70/15/15 (train/val/test_A); Hospital B (`test_B`) is never used for training,
early stopping or threshold selection.

Generated files (not committed, see `.gitignore`):

| What | Where |
|---|---|
| Per-run config, metadata, history, thresholds, metrics, predictions | `results/<experiment>/seed<k>/` |
| Per-run training logs | `results/logs/<experiment>_seed<k>.log` |
| Every metric of every run, one row per run and split | `results/tables/runs.csv` |
| Mean ± std per experiment | `results/tables/summary_{val,test_A,test_B}.{md,csv}` |
| Forecast MAE per vital and per horizon | `results/tables/vitals_{val,test_A,test_B}.md` |
| Paired patient-level bootstrap vs the Transformer | `results/tables/comparisons.{md,csv,json}` |
| Utility score check against the official scorer | `results/verification/utility_check.json` |

Reproduce with:

```bash
python scripts/preprocess.py
python scripts/verify_utility.py
bash scripts/run_all.sh              # 9 experiments × 3 seeds, about 70 min on an RTX 4060 laptop GPU
python scripts/bootstrap_compare.py  # about 35 min on CPU, most of it Hospital B
```

## Hardware and run size

One NVIDIA RTX 4060 Laptop GPU (8 GB), 16 CPU threads, 16 GB RAM. With the default settings an epoch draws
about 150k training windows and takes 8 to 16 s, plus about 4 s for validation over every hour of the
3,050 validation patients. A full run (training plus evaluation on 997k test hours) takes 0.5 to 5 min, and the
whole 27-run grid took 70 min. Data loading is single-process (`num_workers: 0`) because one training process
already holds about 3 GB of RAM, and the machine had about 4 GB free. The configs were not scaled down:
the runs are small enough for this machine as they are.

## Test A (held-out patients, same hospital)

| Model | Params | AUROC | AUPRC | Utility | Detection | Utility @ 1 FA/day | Onset macro-F1 | Vitals MAE |
|---|---:|---|---|---|---|---|---|---|
| persistence | 0 | – | – | – | – | – | – | 6.437 |
| linear | 100,842 | – | – | – | – | – | – | 5.961 ± 0.008 |
| mlp | 380,207 | 0.750 ± 0.007 | 0.058 ± 0.002 | 0.317 ± 0.027 | 0.966 | 0.106 ± 0.008 | 0.258 | 6.389 ± 0.058 |
| grud | 84,689 | 0.739 ± 0.021 | 0.055 ± 0.002 | 0.284 ± 0.030 | 0.937 | 0.104 ± 0.070 | 0.259 | **5.819 ± 0.027** |
| transformer | 165,575 | 0.757 ± 0.005 | **0.063 ± 0.002** | 0.310 ± 0.003 | 0.916 | **0.179 ± 0.028** | 0.258 | 6.135 ± 0.071 |
| gat_transformer (learned) | 159,855 | **0.763 ± 0.006** | 0.058 ± 0.002 | 0.333 ± 0.008 | 0.945 | 0.113 ± 0.039 | 0.251 | 6.244 ± 0.091 |
| gat_clinical_prior | 159,855 | **0.763 ± 0.012** | 0.062 ± 0.004 | 0.324 ± 0.022 | 0.957 | 0.165 ± 0.057 | 0.257 | 6.306 ± 0.207 |
| gat_correlation | 159,855 | **0.763 ± 0.012** | 0.062 ± 0.003 | **0.335 ± 0.022** | 0.923 | 0.159 ± 0.063 | 0.256 | 6.220 ± 0.175 |
| gat_none | 159,855 | 0.759 ± 0.013 | **0.063 ± 0.003** | 0.323 ± 0.025 | 0.897 | **0.214 ± 0.058** | 0.256 | 6.188 ± 0.197 |

## Test B (external hospital)

| Model | AUROC | AUPRC | Utility | Detection | Utility @ 1 FA/day | Vitals MAE |
|---|---|---|---|---|---|---|
| persistence | – | – | – | – | – | 6.961 |
| linear | – | – | – | – | – | 6.736 ± 0.008 |
| mlp | 0.708 ± 0.004 | 0.033 ± 0.001 | 0.100 ± 0.018 | 0.897 | 0.032 ± 0.006 | 7.161 ± 0.051 |
| grud | 0.735 ± 0.030 | 0.040 ± 0.003 | 0.166 ± 0.033 | 0.843 | 0.061 ± 0.043 | **6.518 ± 0.062** |
| transformer | 0.738 ± 0.017 | 0.044 ± 0.003 | 0.161 ± 0.053 | 0.882 | **0.169 ± 0.022** | 6.890 ± 0.143 |
| gat_transformer (learned) | 0.743 ± 0.011 | 0.040 ± 0.003 | 0.155 ± 0.015 | 0.895 | 0.081 ± 0.058 | 7.174 ± 0.107 |
| gat_clinical_prior | 0.744 ± 0.026 | **0.046 ± 0.006** | 0.171 ± 0.045 | 0.927 | 0.112 ± 0.054 | 7.146 ± 0.200 |
| gat_correlation | **0.747 ± 0.021** | 0.045 ± 0.006 | **0.184 ± 0.031** | 0.885 | 0.110 ± 0.050 | 7.027 ± 0.181 |
| gat_none | 0.735 ± 0.026 | **0.046 ± 0.007** | 0.183 ± 0.037 | 0.885 | 0.154 ± 0.053 | 6.924 ± 0.244 |

Column meanings: AUROC/AUPRC are hourly, for `SepsisLabel`. Utility is the official Challenge 2019 normalised
utility at the threshold that maximises it on validation. Detection is the share of eligible septic patients
(≥ 6 h before the first positive label) with an alarm at or before onset, at that threshold.
"Utility @ 1 FA/day" uses the threshold that allows one false-alarm hour per non-septic patient-day on
validation. Vitals MAE is in clinical units over all seven vitals and six forecast hours, measured values only.
Full columns (Brier, ECE, warning hours, false alarms per patient-day, RMSE, pre-onset MAE) are in
`results/tables/summary_*.md`.

## Paired bootstrap against the Transformer (3 seeds, 1,000 patient resamples)

Difference GAT variant − Transformer, 95% CI. Bold = CI excludes 0.

| Model | Split | AUROC | AUPRC | Utility | Vitals MAE (lower is better) |
|---|---|---|---|---|---|
| gat_transformer | test_A | +0.006 [−0.004, +0.016] | **−0.005 [−0.010, −0.001]** | +0.023 [−0.001, +0.044] | **+0.109 [+0.097, +0.121]** |
| gat_clinical_prior | test_A | +0.005 [−0.005, +0.017] | −0.000 [−0.006, +0.005] | +0.014 [−0.011, +0.039] | **+0.171 [+0.159, +0.184]** |
| gat_correlation | test_A | +0.006 [−0.004, +0.016] | −0.000 [−0.005, +0.005] | +0.024 [−0.000, +0.049] | **+0.086 [+0.075, +0.097]** |
| gat_none | test_A | +0.002 [−0.006, +0.011] | +0.000 [−0.004, +0.004] | +0.013 [−0.007, +0.032] | **+0.053 [+0.040, +0.066]** |
| gat_transformer | test_B | **+0.006 [+0.001, +0.010]** | **−0.004 [−0.006, −0.002]** | −0.006 [−0.017, +0.003] | **+0.284 [+0.227, +0.370]** |
| gat_clinical_prior | test_B | **+0.007 [+0.001, +0.013]** | +0.002 [−0.000, +0.004] | +0.010 [−0.002, +0.022] | **+0.256 [+0.216, +0.323]** |
| gat_correlation | test_B | **+0.009 [+0.004, +0.014]** | +0.001 [−0.001, +0.002] | **+0.023 [+0.013, +0.032]** | **+0.138 [+0.111, +0.185]** |
| gat_none | test_B | −0.003 [−0.008, +0.002] | **+0.002 [+0.000, +0.004]** | **+0.022 [+0.012, +0.031]** | **+0.034 [+0.002, +0.084]** |

The bootstrap resamples *patients* with the three seeds' predictions held fixed, so it captures test-set
sampling noise but not seed-to-seed training noise. Seed noise is of the same size or larger here (for example
the AUROC std across seeds is 0.005 to 0.030), so a CI that excludes 0 on Hospital B is weaker evidence
than it looks. Treat the mean ± std tables and the CIs together.

## Key findings

1. **The GAT branch does not clearly help.** On test_A no GAT variant beats the parameter-matched Transformer on
   AUROC, AUPRC or utility once uncertainty is taken into account. The learned-graph model is slightly *worse*
   on AUPRC on both test sets. Small AUROC gains on Hospital B (+0.006 to +0.009) are within one seed std.
2. **Graph structure does not matter.** `gat_none` (self-loops only, so the GAT branch is just a per-variable
   encoder) performs as well as the learned, clinical-prior and correlation graphs on every risk metric.
   Whatever the graph branch contributes is not coming from message passing between variables.
3. **Multi-task models are poor forecasters under this protocol.** GRU-D gives the best vital-sign forecasts
   (MAE 5.82 test_A, 6.52 test_B), and a plain linear model (5.96) beats the Transformer (6.14) and every GAT
   variant (6.19 to 6.31). See finding 5 for why.
4. **External validation drops sharply.** From Hospital A to B, AUPRC roughly halves (0.06 → 0.04, partly because
   positives are rarer at B: 1.41% vs 2.17% of hours) and utility drops from about 0.32 to 0.16. Rankings between
   models are not stable across hospitals.
5. **Training stops almost immediately.** Validation AUPRC peaks in epoch 1 to 6 for 17 of the 21 trained
   risk models (median best epoch 2) while the training loss keeps falling (risk BCE 1.08 → 0.35 over 11
   epochs for the Transformer). Early stopping on AUPRC therefore also stops the vitals head after one or two
   epochs, which explains finding 3. Two seed-0 pilots did not change this: a regularised protocol
   (lr 3e-4, dropout 0.3, weight decay 0.01, patience 8) and removing static features
   (`data.use_static=false`, to rule out memorising patients from Age/HospAdmTime). Both peaked early at the
   same validation AUPRC (`results/*_reg`, `results/transformer_nostatic`).
6. **Calibration is poor by design.** With `risk_pos_weight: 10` predicted probabilities are inflated
   (mean ECE 0.15 to 0.24 on test_A). The max-utility thresholds sit low and fire 3 to 5 false-alarm hours per non-septic
   patient-day while detecting about 90 to 95% of septic patients with a median of about 44 h of warning.
   At one false alarm per day, detection falls to 60 to 73% on test_A.

For context, our validation utility (0.37 to 0.41) is in the range reported by the Challenge's top teams
on their own cross-validation, but hourly AUROC (about 0.76 on test_A) is below the 0.80 to 0.85 typically
reported for tuned gradient-boosting models on this dataset.

## Suggested next steps

- Decouple early stopping from AUPRC alone, for example monitor the validation total loss, or train the vitals
  head to convergence and select the risk checkpoint separately. Then re-run the forecasting comparison.
- Run the remaining ablations in `docs/experiments.md` (single-task vs multi-task, lookback) only after the
  protocol above is settled, since the current comparison is dominated by epoch-1 noise.
- Add a gradient-boosting baseline on the same windows as a sanity check on the achievable AUROC.
- Report the bootstrap and the seed std together, or add more seeds (runs are cheap: about 3 min each).
