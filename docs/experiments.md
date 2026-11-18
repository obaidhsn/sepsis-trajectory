# Experiments

## Priority order

Time is short, so run the experiments in this order and stop wherever the deadline hits:

1. **Core result:** `transformer` vs `gat_transformer`, 3 seeds, test_A and test_B
2. **Baselines:** persistence, linear, MLP, GRU-D
3. **Graph ablation:** `gat_clinical_prior`, `gat_correlation`, `gat_none`
4. **Multi-task vs single-task**
5. **Lookback:** 12 / 24 / 48 h
6. **Mask/delta input ablation**
7. Autoformer / Informer forecasting baselines (only if time permits)

## Commands

```bash
bash scripts/run_all.sh                       # steps 1–3
SEEDS="0" bash scripts/run_all.sh             # one seed, for the mid-project update
python scripts/summarize.py --split test_A
python scripts/compare.py results/gat_transformer/seed0 results/transformer/seed0
```

## Results log

Add a row for every run you want to keep. Paste numbers from `results/<name>/seed<k>/metrics.json`.

| Date | Who | Experiment | Seeds | Test A AUPRC | Test A utility | Test B AUPRC | Vitals MAE | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-11-18 | Obaidullah | persistence | – | – | – | – | 6.437 (A) / 6.961 (B) | no training |
| 2026-11-18 | Obaidullah | linear | 0,1,2 | – | – | – | 5.961 / 6.736 | |
| 2026-11-18 | Obaidullah | mlp | 0,1,2 | 0.058 | 0.317 | 0.033 | 6.389 / 7.161 | |
| 2026-11-18 | Obaidullah | grud | 0,1,2 | 0.055 | 0.284 | 0.040 | 5.819 / 6.518 | best forecaster |
| 2026-11-18 | Obaidullah | transformer | 0,1,2 | 0.063 | 0.310 | 0.044 | 6.135 / 6.890 | primary baseline |
| 2026-11-18 | Obaidullah | gat_transformer | 0,1,2 | 0.058 | 0.333 | 0.040 | 6.244 / 7.174 | learned graph |
| 2026-11-18 | Obaidullah | gat_clinical_prior | 0,1,2 | 0.062 | 0.324 | 0.046 | 6.306 / 7.146 | |
| 2026-11-18 | Obaidullah | gat_correlation | 0,1,2 | 0.062 | 0.335 | 0.045 | 6.220 / 7.027 | |
| 2026-11-18 | Obaidullah | gat_none | 0,1,2 | 0.063 | 0.323 | 0.046 | 6.188 / 6.924 | |
| 2026-11-18 | Obaidullah | transformer_reg, gat_transformer_reg | 0 | 0.050 / 0.054 | 0.304 / 0.349 | 0.038 / 0.046 | 6.42 / 6.48 (A) | lr 3e-4, dropout 0.3, wd 0.01, patience 8: no gain |
| 2026-11-18 | Obaidullah | transformer_nostatic | 0 | 0.059 | 0.271 | 0.035 | 5.94 (A) | `data.use_static=false` |

Means over seeds. Full tables, std and bootstrap CIs: [results.md](results.md).
