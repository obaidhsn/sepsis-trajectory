# Per-task model selection and onset-head fix: first GAT + Transformer run (seed 0)

Run on 2026-11-25 on an Apple M2 MacBook (16 GB, PyTorch MPS backend), one run per model (seed 0).
Same data, splits, inputs, architectures and parameter counts as the mid-project evaluation
(`results/RESULTS.md`). Three things changed:

1. **Model selection** (`train.selection: per_task`). Every epoch, validation is scored on each task:
   risk AUPRC, vitals MAE and onset macro-F1. Each head is taken from its own best epoch, and training
   stops only when no head has improved by more than 0.001 for 5 epochs. Before, one checkpoint was
   chosen on AUPRC alone, which froze every head at epoch 1–6.
2. **Onset head** (`onset_class_weights: auto`). The cross-entropy is class-weighted with balanced inverse
   frequencies of the training windows. Counts were [15,327, 5,448, 5,920, 117,257], giving weights
   [2.35, 6.61, 6.08, 0.31]. Without weights the head only ever predicted ">24 h / never".
3. **Input clipping** (`data.clip_inputs: true`). Normalised inputs are clipped to the range seen in the
   training patients. Hospital B contains data-entry errors (for example an FiO2 value 21,554 standard
   deviations above the mean, where the training maximum is 51). Without clipping, the GAT branch turned
   these into forecasts of about 25,000 mmHg for two patients and a Hospital B vitals RMSE of 35.9.
   Clipping leaves training data unchanged and affects 0.4% of val/test values. Both runs below were
   re-evaluated with clipping (`scripts/train.py --eval-only`).

**Single seed. Differences of a few thousandths are within run-to-run noise.**

## Results

Utility = official Challenge 2019 normalised utility at the validation-chosen max-utility threshold.
Utility@1FA = utility at the validation threshold allowing one false-alarm hour per non-septic patient-day.
FA h/pt-day = false-alarm hours per non-septic patient-day at the max-utility threshold. Vitals MAE is in
clinical units. "Septic pts" restricts the onset metric to hours of patients who become septic.

### Hospital A test set (test_A)

| Model | Params | Heads from epoch (risk / vitals / onset) | AUROC | AUPRC | ECE | Utility | FA h/pt-day | Utility@1FA | Onset macro-F1 | Onset macro-F1 (septic pts) | Onset error (h, septic) | Vitals MAE |
|---|--:|:-:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Transformer | 165,575 | 24 / 29 / 27 of 30 | 0.737 | 0.064 | 0.064 | 0.243 | 1.88 | **0.199** | 0.283 | 0.223 | 16.5 | **5.919** |
| GAT + Transformer (learned) | 159,855 | 1 / 17 / 17 of 22 | **0.769** | 0.064 | 0.291 | **0.345** | 4.33 | 0.180 | 0.282 | 0.233 | **14.5** | 5.926 |

### Hospital B, external test set (test_B)

| Model | AUROC | AUPRC | ECE | Utility | FA h/pt-day | Utility@1FA | Onset macro-F1 | Onset macro-F1 (septic pts) | Onset error (h, septic) | Vitals MAE |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Transformer | 0.700 | 0.042 | 0.059 | 0.155 | 2.32 | 0.148 | 0.271 | 0.240 | 17.2 | **6.638** |
| GAT + Transformer (learned) | **0.785** | **0.061** | 0.221 | **0.242** | 2.33 | 0.154 | 0.268 | 0.235 | **15.4** | 6.675 |

### Compared with the mid-project evaluation (old protocol, same seed)

| | Transformer old → new | GAT + Transformer old → new |
|---|---|---|
| Vitals MAE test_A | 6.045 → **5.919** | 6.130 → **5.926** |
| Vitals MAE test_B | 6.699 → **6.638** | 7.051 → **6.675** |
| Onset macro-F1 test_A | 0.267 → **0.283** | 0.251 → **0.282** |
| Onset error, septic hours (h), test_A | 19.0 → **16.5** | 19.2 → **14.5** |
| AUROC / AUPRC test_A | 0.755 / 0.064 → 0.737 / 0.064 | 0.760 / 0.060 → 0.769 / 0.064 |
| AUROC / AUPRC test_B | 0.730 / 0.044 → 0.700 / 0.042 | 0.730 / 0.043 → 0.785 / 0.061 |
| Epochs trained | 11 → 30 (hit the cap) | 9 → 22 |

The old runs were on an RTX 4060 (CUDA) and these on MPS, so the same seed does not reproduce the same
weights. Part of each difference is ordinary run-to-run noise.

Full tables: `tables/summary_{val,test_A,test_B}.md`, per-vital and per-horizon errors in
`tables/vitals_*.md`, epoch-by-epoch validation scores in `<model>/seed0/history.json`.

## What this shows

- **The vitals head was undertrained before.** With its own selection it trains for 17–29 epochs.
  Hospital A MAE drops to 5.92 for both models, now better than the linear baseline (5.97). GRU-D under
  the old protocol is still lower (5.80) and should be re-run under the new one. The two models now
  forecast equally well (5.919 vs 5.926), so the GAT branch adds nothing to trajectory forecasting in
  this run.
- **The onset head now learns, but only partly.** It no longer predicts a single bucket. Its one-vs-rest
  AUROC per bucket is 0.75–0.77 on test_A (GAT model), about as good as the risk head, and the expected
  error for septic hours fell from about 19 h to 14.5–16.5 h. But it separates neighbouring buckets
  poorly: among septic patients, balanced accuracy is about 0.26, close to chance (0.25), and the
  balanced weights make it over-predict the 6–12 h and 12–24 h buckets. Possible next steps: ordinal
  regression or a direct hours-to-onset target instead of 4-way classification, softer weights
  (square-root inverse frequency), or treating hours after onset separately (open decision in
  `docs/methodology.md`).
- **The risk comparison depends on which epoch each risk head comes from.** The GAT model's best
  validation AUPRC was at epoch 1; the Transformer's at epoch 24.
  - The early GAT snapshot has the higher AUROC (+0.03 on A, +0.09 on B), AUPRC (B), and max-utility
    score.
  - It is also badly calibrated (ECE 0.29 vs 0.06) and raises more than twice as many false-alarm hours
    on Hospital A.
  - At a fixed budget of one false-alarm hour per day the Transformer is better on Hospital A (0.199 vs
    0.180), and the two are level on Hospital B.
  - A one-epoch-old snapshot generalising better to a new hospital is plausible (less fitted to Hospital
    A), but with one seed it is not evidence that the graph helps. This needs 3+ seeds and a
    `gat_none` ablation before any claim.
- **The Transformer stopped at the 30-epoch cap**, with its vitals head still improving at epoch 29.
  Raise `train.epochs` (e.g. 50) for the final runs.

## Reproduce

```bash
SOURCE=physionet bash scripts/download_data.sh   # or the Kaggle mirror, see docs/data.md
python scripts/preprocess.py
for e in gat_transformer transformer; do
  python scripts/train.py --config configs/experiments/$e.yaml --seed 0 --name $e \
    --override paths.results_dir=results/per_task
done
python scripts/summarize.py --results results/per_task
```

About 16 min (GAT + Transformer) and 23 min (Transformer) on an M2 MacBook, at 34–40 s per epoch.
