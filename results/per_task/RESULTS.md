# All models under per-task selection, weighted onset loss and input clipping (seed 0)

Run on 2026-11-25 on an Apple M2 MacBook (16 GB, PyTorch MPS), one run per model (seed 0). Same data,
splits, inputs, architectures and parameter counts as the mid-project evaluation (`results/RESULTS.md`,
"old" below). Three things changed (details in `docs/methodology.md`):

1. **Model selection** (`train.selection: per_task`). Every epoch, validation is scored on each task:
   risk AUPRC, vitals MAE and onset macro-F1. Each head is taken from its own best epoch, and training
   stops only when no head has improved by more than 0.001 for 5 epochs (maximum 30). Before, one
   checkpoint was chosen on AUPRC alone, which froze every head at epoch 1–7.
2. **Onset head** (`onset_class_weights: auto`). The cross-entropy is class-weighted with balanced inverse
   bucket frequencies of the training windows (weights 2.35 / 6.61 / 6.08 / 0.31 for ≤6 h / 6–12 h /
   12–24 h / >24 h or never). Before, the head only ever predicted ">24 h / never".
3. **Input clipping** (`data.clip_inputs: true`). Normalised inputs are clipped to the range seen in the
   training patients. Hospital B has data-entry errors (an FiO2 value 21,554 standard deviations from the
   mean; the training maximum is 51). These inflated some models' Hospital B vitals RMSE: Linear 26.0,
   GAT models 13.5–15.3, and 35.9 for the new GAT + Transformer run before clipping. Clipping leaves
   training data unchanged and affects 0.4% of val/test values.

The Transformer and GAT + Transformer were trained first and re-evaluated with clipping
(`scripts/train.py --eval-only`). The other eight were trained with all three changes in place
(`OUT=results/per_task bash scripts/run_mid_eval.sh`).

**Single seed. Differences of a few thousandths are within run-to-run noise.** The old runs were on an
RTX 4060 (CUDA), these on MPS, so the same seed does not give the same weights.

## Results

Utility = official Challenge 2019 normalised utility at the validation-chosen max-utility threshold.
FA h/day = false-alarm hours per non-septic patient-day at that threshold. Utility@1FA and Detect@1FA =
utility and share of septic patients detected at the validation threshold allowing one false-alarm hour per
non-septic patient-day. Onset error = expected hours-to-onset error on hours of septic patients. Vitals MAE
in clinical units. "Risk ep." = epoch the risk head was taken from.

### Hospital A test set (test_A)

| Model | Params | Risk ep. | AUROC | AUPRC | ECE | Utility | FA h/day | Utility@1FA | Detect@1FA | Onset macro-F1 | Onset error (h) | Vitals MAE |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Persistence | 0 | – | – | – | – | – | – | – | – | – | – | 6.437 |
| Linear | 100,842 | – | – | – | – | – | – | – | – | – | – | 6.000 |
| MLP | 380,207 | 1 | 0.766 | 0.061 | 0.375 | 0.337 | 4.87 | 0.164 | 0.681 | 0.250 | 12.3 | 6.450 |
| GRU-D | 84,689 | 8 | 0.729 | 0.054 | 0.095 | 0.245 | 1.99 | 0.181 | 0.657 | 0.261 | 12.8 | **5.803** |
| Informer | 162,567 | 25 | 0.739 | **0.064** | 0.068 | 0.268 | 2.38 | 0.213 | **0.783** | **0.293** | 16.0 | 6.028 |
| Autoformer | 168,455 | 3 | 0.767 | 0.057 | 0.313 | **0.350** | 3.84 | 0.076 | 0.696 | 0.239 | **9.3** | 5.938 |
| Transformer | 165,575 | 24 | 0.737 | **0.064** | **0.064** | 0.243 | **1.88** | 0.199 | 0.773 | 0.283 | 16.5 | 5.919 |
| GAT + Transformer (learned) | 159,855 | 1 | **0.769** | **0.064** | 0.291 | 0.345 | 4.33 | 0.180 | 0.725 | 0.282 | 14.5 | 5.926 |
| GAT + Transformer (clinical prior) | 159,855 | 11 | 0.756 | 0.062 | 0.083 | 0.286 | 3.91 | **0.240** | 0.744 | 0.282 | 15.6 | 5.870 |
| GAT + Transformer (correlation) | 159,855 | 1 | 0.768 | 0.063 | 0.290 | 0.324 | 4.39 | 0.177 | 0.729 | 0.281 | 15.2 | 5.894 |

### Hospital B, external test set (test_B)

| Model | Risk ep. | AUROC | AUPRC | ECE | Utility | FA h/day | Utility@1FA | Detect@1FA | Onset macro-F1 | Onset error (h) | Vitals MAE |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Persistence | – | – | – | – | – | – | – | – | – | – | 6.961 |
| Linear | – | – | – | – | – | – | – | – | – | – | 6.618 |
| MLP | 1 | 0.741 | 0.037 | 0.298 | 0.141 | 3.41 | 0.048 | 0.464 | 0.244 | 15.6 | 7.208 |
| GRU-D | 8 | 0.694 | 0.034 | 0.089 | 0.139 | 2.35 | 0.130 | 0.653 | 0.254 | 14.8 | **6.487** |
| Informer | 25 | 0.687 | 0.040 | **0.056** | 0.160 | 2.38 | 0.141 | 0.823 | **0.276** | 17.4 | 6.832 |
| Autoformer | 3 | 0.765 | 0.045 | 0.262 | 0.222 | 3.14 | 0.077 | 0.596 | 0.237 | **10.7** | 6.598 |
| Transformer | 24 | 0.700 | 0.042 | 0.058 | 0.155 | 2.32 | 0.148 | **0.839** | 0.271 | 17.2 | 6.638 |
| GAT + Transformer (learned) | 1 | **0.785** | **0.061** | 0.221 | **0.242** | 2.33 | 0.154 | 0.507 | 0.268 | 15.4 | 6.675 |
| GAT + Transformer (clinical prior) | 11 | 0.731 | 0.044 | 0.074 | 0.120 | 4.23 | **0.195** | 0.771 | 0.268 | 16.8 | 6.545 |
| GAT + Transformer (correlation) | 1 | 0.783 | 0.058 | 0.224 | 0.235 | 2.69 | 0.155 | 0.535 | 0.274 | 16.7 | 6.612 |

### Change from the old protocol (test_A, same seed)

| Model | Vitals MAE old → new | Onset error (h) old → new | Onset macro-F1 old → new | AUROC old → new | Risk epoch old → new |
|---|---|---|---|---|---|
| Linear | 5.965 → 6.000 | – | – | – | 10 → 3 (vitals) |
| MLP | 6.458 → 6.450 | 18.5 → 12.3 | 0.254 → 0.250 | 0.759 → 0.766 | 2 → 1 |
| GRU-D | 5.803 → 5.803 | 17.5 → 12.8 | 0.256 → 0.261 | 0.757 → 0.729 | 2 → 8 |
| Informer | 6.100 → 6.028 | 19.3 → 16.0 | 0.257 → 0.293 | 0.746 → 0.739 | 7 → 25 |
| Autoformer | 5.927 → 5.938 | 18.0 → 9.3 | 0.252 → 0.239 | 0.765 → 0.767 | 3 → 3 |
| Transformer | 6.045 → 5.919 | 19.0 → 16.5 | 0.267 → 0.283 | 0.755 → 0.737 | 6 → 24 |
| GAT (learned) | 6.130 → 5.926 | 19.2 → 14.5 | 0.251 → 0.282 | 0.760 → 0.769 | 4 → 1 |
| GAT (clinical prior) | 6.455 → **5.870** | 19.2 → 15.6 | 0.246 → 0.282 | 0.777 → 0.756 | 1 → 11 |
| GAT (correlation) | 6.424 → **5.894** | 19.2 → 15.2 | 0.247 → 0.281 | 0.776 → 0.768 | 1 → 1 |

Full tables (all columns, val split, per-vital and per-horizon errors): `tables/`. Epoch-by-epoch validation
scores: `<model>/seed0/history.json`.

## What this shows

1. **Vitals forecasting improved where heads had been cut off early.**
   - The biggest gains are for the models that the old protocol stopped at epoch 1: the fixed-graph GATs
     (6.46 → 5.87 and 6.42 → 5.89) and the learned-graph GAT (6.13 → 5.93). The Transformer and Informer
     improved by about 0.1.
   - GRU-D is still the best forecaster (5.80 on A, 6.49 on B), and its best vitals epoch is epoch 1.
   - Every model except MLP beats persistence (6.44). Linear (6.00) is beaten by GRU-D, Autoformer,
     the Transformer and all three GATs, but not by Informer (6.03) or MLP (6.45).
   - Linear is slightly worse than before (5.965 → 6.000). With the new 0.001 threshold it stops at
     epoch 3 instead of 10, so `min_delta` is too coarse for it.
2. **The onset head now works for every model, but only roughly.**
   - Expected error for septic hours fell from 17.5–19.3 h to 9.3–17.4 h for every model. Autoformer is
     lowest (9.3 h on A, 10.7 h on B).
   - Macro-F1 rose for most models: 0.25 → 0.28–0.29 for the Transformer-family models and Informer.
   - No model separates neighbouring buckets well. The best macro-F1 is 0.293.
3. **The epoch the risk head comes from sets the risk results.** Two groups:
   - Risk heads from epoch 1–3 (MLP, Autoformer, learned-graph GAT, correlation GAT): higher AUROC
     (0.766–0.769 on A, 0.741–0.785 on B), poor calibration (ECE 0.22–0.38) and many false alarms.
   - Risk heads from epoch 8–25 (GRU-D, Informer, Transformer, clinical-prior GAT): lower AUROC
     (0.729–0.756 on A, 0.687–0.731 on B), good calibration (ECE 0.06–0.10) and mostly the best
     utility at one false alarm per day (0.181–0.240 on A).

   Validation AUPRC moves by about ±0.005 from epoch to epoch, so which group a model lands in is close
   to chance. That explains most of the risk differences between models in both the old and the new
   tables.
4. **The graph question.** The two fixed-graph GATs forecast vitals slightly better than the
   parameter-matched Transformer (5.87 and 5.89 vs 5.92); the learned-graph GAT is level (5.93). The
   clinical-prior GAT has the best utility at
   one false alarm per day (0.240 vs 0.199 on A; 0.195 vs 0.148 on B). These are single-seed differences
   and of the same size as seed noise in run 1 (std 0.07–0.21 for vitals MAE). Run 1 also found that a
   GAT with no edges (`gat_none`) does as well as any graph. So there is still no evidence that the
   graph structure helps.

## Next

- 3 seeds for all models under this protocol, plus `gat_none` and the paired bootstrap (restore from
  commit 6320dbd), before any model ranking.
- A less noisy risk selection rule: validation risk loss, or AUPRC averaged over neighbouring epochs.
  Alternatively report risk at a fixed epoch budget for all models.
- `train.epochs: 50`. Four models hit the 30-epoch cap with a head still improving (Informer,
  Transformer, both fixed-graph GATs). Consider a relative `min_delta` so Linear is not stopped early.
- Time to onset: an ordinal or regression target instead of 4-way classification.

## Reproduce

```bash
SOURCE=physionet bash scripts/download_data.sh   # or the Kaggle mirror, see docs/data.md
python scripts/preprocess.py
OUT=results/per_task bash scripts/run_mid_eval.sh   # all 10 models, seed 0, skips finished runs
```

About 2.5 h for all ten models on an M2 MacBook: 0.5 min (Persistence) to 31 min (clinical-prior
GAT). Informer is the slowest per epoch.
