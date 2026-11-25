# Results

Three rounds of experiments so far, newest first. All use PhysioNet 2019 with the same split: Hospital A
70/15/15 by patient (train / val / test_A), and Hospital B (test_B) never used for training, early stopping
or threshold selection.

| Run | Date | Protocol | Seeds | Models | Details |
|---|---|---|---|---|---|
| 3 | 2026-11-25 | per-task selection, weighted onset loss, input clipping | 0 | all 10 | [results/per_task/RESULTS.md](../results/per_task/RESULTS.md) |
| 2 | 2026-11-24 | one checkpoint on val AUPRC | 0 | all 10 incl. Informer, Autoformer | [results/RESULTS.md](../results/RESULTS.md) |
| 1 | 2026-11-18 | one checkpoint on val AUPRC | 0, 1, 2 | 9 incl. `gat_none`, with bootstrap | below |

## Run 3 (2026-11-25): all models with per-task selection and onset fix

What changed (details in `docs/methodology.md`):

1. **Model selection.** Each head (risk, vitals, onset) is taken from the epoch with its own best validation
   score (AUPRC, MAE, macro-F1). Training stops only when no head has improved by more than 0.001 for 5
   epochs (maximum 30). Before, one checkpoint was chosen on AUPRC alone, which froze every head at
   epoch 1–7.
2. **Onset loss** is class-weighted with balanced inverse bucket frequencies, computed from the training
   windows (weights 2.35 / 6.61 / 6.08 / 0.31).
3. **Input clipping** to the training range. Hospital B has data-entry errors (an FiO2 value 21,554 standard
   deviations from the mean). These had inflated the Hospital B vitals RMSE of Linear (26.0) and the GAT
   models (13.5–15.3).

Seed 0, Apple M2 (MPS). Utility@1FA = utility at one false-alarm hour per non-septic patient-day. Onset
error = expected hours-to-onset error on septic hours. Vitals MAE in clinical units. The Hospital B table and
more columns are in [results/per_task/RESULTS.md](../results/per_task/RESULTS.md).

### Hospital A test set (test_A)

| Model | Risk from epoch | AUROC | AUPRC | ECE | Utility | Utility@1FA | Onset macro-F1 | Onset error (h) | Vitals MAE |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Persistence | – | – | – | – | – | – | – | – | 6.437 |
| Linear | – | – | – | – | – | – | – | – | 6.000 |
| MLP | 1 | 0.766 | 0.061 | 0.375 | 0.337 | 0.164 | 0.250 | 12.3 | 6.450 |
| GRU-D | 8 | 0.729 | 0.054 | 0.095 | 0.245 | 0.181 | 0.261 | 12.8 | **5.803** |
| Informer | 25 | 0.739 | 0.064 | 0.068 | 0.268 | 0.213 | **0.293** | 16.0 | 6.028 |
| Autoformer | 3 | 0.767 | 0.057 | 0.313 | **0.350** | 0.076 | 0.239 | **9.3** | 5.938 |
| Transformer | 24 | 0.737 | 0.064 | **0.064** | 0.243 | 0.199 | 0.283 | 16.5 | 5.919 |
| GAT + Transformer (learned) | 1 | **0.769** | 0.064 | 0.291 | 0.345 | 0.180 | 0.282 | 14.5 | 5.926 |
| GAT + Transformer (clinical prior) | 11 | 0.756 | 0.062 | 0.083 | 0.286 | **0.240** | 0.282 | 15.6 | 5.870 |
| GAT + Transformer (correlation) | 1 | 0.768 | 0.063 | 0.290 | 0.324 | 0.177 | 0.281 | 15.2 | 5.894 |

### Did the fixes improve the results?

Same seed, old protocol (run 2) → new protocol (run 3), test_A unless noted. The Hospital B column is
vitals MAE.

| Model | Vitals MAE | Vitals MAE test_B | Onset error (h) | Onset macro-F1 | AUROC | ECE | Utility@1FA |
|---|---|---|---|---|---|---|---|
| Linear | 5.965 → 6.000 | 6.747 → **6.618** | – | – | – | – | – |
| MLP | 6.458 → 6.450 | 7.200 → 7.208 | 18.5 → **12.3** | 0.254 → 0.250 | 0.759 → 0.766 | 0.267 → 0.375 | 0.117 → **0.164** |
| GRU-D | 5.803 → 5.803 | 6.489 → 6.487 | 17.5 → **12.8** | 0.256 → 0.261 | 0.757 → 0.729 | 0.264 → **0.095** | 0.067 → **0.181** |
| Informer | 6.100 → **6.028** | 6.851 → 6.832 | 19.3 → **16.0** | 0.257 → **0.293** | 0.746 → 0.739 | 0.104 → **0.068** | 0.135 → **0.213** |
| Autoformer | 5.927 → 5.938 | 6.662 → **6.598** | 18.0 → **9.3** | 0.252 → 0.239 | 0.765 → 0.767 | 0.294 → 0.313 | 0.115 → 0.076 |
| Transformer | 6.045 → **5.919** | 6.699 → **6.638** | 19.0 → **16.5** | 0.267 → **0.283** | 0.755 → 0.737 | 0.119 → **0.064** | 0.207 → 0.199 |
| GAT (learned) | 6.130 → **5.926** | 7.051 → **6.675** | 19.2 → **14.5** | 0.251 → **0.282** | 0.760 → 0.769 | 0.142 → 0.291 | 0.167 → 0.180 |
| GAT (clinical prior) | 6.455 → **5.870** | 7.312 → **6.545** | 19.2 → **15.6** | 0.246 → **0.282** | 0.777 → 0.756 | 0.280 → **0.083** | 0.130 → **0.240** |
| GAT (correlation) | 6.424 → **5.894** | 7.252 → **6.612** | 19.2 → **15.2** | 0.247 → **0.281** | 0.776 → 0.768 | 0.286 → 0.290 | 0.136 → **0.177** |

**Summary.**

- **Vitals forecasting: better, by a lot where heads had been cut off early.**
  - The three GATs improved by 0.20–0.59 on Hospital A and 0.38–0.77 on Hospital B. The Transformer
    and Informer improved by about 0.1.
  - Models that already trained several epochs for vitals (GRU-D, Autoformer, MLP) are unchanged.
  - GRU-D is still the best forecaster (5.80). The best attention model is now the clinical-prior GAT
    (5.87).
  - Linear is slightly worse on A (5.965 → 6.000), because the 0.001 improvement threshold stops it at
    epoch 3.
- **Time to onset: better for every model, still weak.**
  - Expected error for septic hours fell from 17.5–19.3 h to 9.3–17.4 h; Autoformer is lowest (9.3 h).
  - Macro-F1 rose to 0.28–0.29 for Informer and the Transformer-family models. MLP and Autoformer are
    flat or slightly lower.
  - No model separates neighbouring buckets well.
- **Sepsis risk: not better overall, but the trade-off is now visible.** The new protocol lets each risk
  head land on any epoch:
  - **Epoch 1–3** (MLP, Autoformer, learned and correlation GAT): highest AUROC (0.766–0.769 on A, up to
    0.785 on B), but poorly calibrated (ECE 0.29–0.38) with many false alarms.
  - **Epoch 8–25** (GRU-D, Informer, Transformer, clinical-prior GAT): lower AUROC (0.729–0.756) but well
    calibrated (ECE 0.06–0.10). They are also mostly better at a fixed budget of one false alarm per day;
    utility@1FA improved for 6 of the 8 risk models (not Autoformer or the Transformer), and the clinical-prior GAT reaches 0.240.
  - Validation AUPRC moves by about ±0.005 between epochs, so which group a model lands in is close to
    chance. That explains much of the spread in risk metrics between models, in this run and in runs 1–2.
- **Graph question: unchanged.**
  - The fixed-graph GATs are slightly ahead of the matched Transformer on vitals MAE (5.87 and 5.89 vs
    5.92) and on utility@1FA (clinical prior 0.240 vs 0.199). The learned-graph GAT is level on vitals.
  - These are single-seed gaps of the size of seed noise, and run 1 found that `gat_none` does as well
    as any graph.

**Next.**

- 3 seeds for all models under this protocol, with `gat_none` and the paired bootstrap (restore from commit
  6320dbd).
- `train.epochs: 50`, since four models hit the 30-epoch cap. Use a relative `min_delta` so Linear is not
  stopped early.
- A less noisy risk selection rule (validation risk loss, or AUPRC averaged over neighbouring epochs).
- An ordinal or regression target for time to onset.

## Run 2 (2026-11-24): mid-project evaluation

All ten models, seed 0, old protocol. Tables and observations are in
[results/RESULTS.md](../results/RESULTS.md). It added Informer and Autoformer. Autoformer and GRU-D were the
best forecasters (MAE 5.93 / 5.80 on test_A); the fixed-graph GATs had the highest AUROC (0.777 / 0.776).
All its rows are superseded by run 3.

## Run 1 (2026-11-18): first full run, 3 seeds

> Old protocol (one checkpoint on validation AUPRC). The scripts referenced below (`run_all.sh`,
> `verify_utility.py`, `bootstrap_compare.py`), the `gat_none` config and these result files were removed in
> commit 9cc0b62 and are only in the git history (commit 6320dbd). The "Suggested next steps" at the end
> are kept as written; the first one (decouple early stopping from AUPRC) is done in run 3.

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

### Hardware and run size

One NVIDIA RTX 4060 Laptop GPU (8 GB), 16 CPU threads, 16 GB RAM. With the default settings an epoch draws
about 150k training windows and takes 8 to 16 s, plus about 4 s for validation over every hour of the
3,050 validation patients. A full run (training plus evaluation on 997k test hours) takes 0.5 to 5 min, and the
whole 27-run grid took 70 min. Data loading is single-process (`num_workers: 0`) because one training process
already holds about 3 GB of RAM, and the machine had about 4 GB free. The configs were not scaled down:
the runs are small enough for this machine as they are.

### Test A (held-out patients, same hospital)

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

### Test B (external hospital)

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

### Paired bootstrap against the Transformer (3 seeds, 1,000 patient resamples)

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

### Key findings

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

### Suggested next steps

- Decouple early stopping from AUPRC alone, for example monitor the validation total loss, or train the vitals
  head to convergence and select the risk checkpoint separately. Then re-run the forecasting comparison.
- Run the remaining ablations in `docs/experiments.md` (single-task vs multi-task, lookback) only after the
  protocol above is settled, since the current comparison is dominated by epoch-1 noise.
- Add a gradient-boosting baseline on the same windows as a sanity check on the achievable AUROC.
- Report the bootstrap and the seed std together, or add more seeds (runs are cheap: about 3 min each).
