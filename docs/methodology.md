# Methodology

## Inputs

Every model receives the same per-hour features for the window [t−L+1, t]:
normalised values `x`, mask `m`, log time-since-measured `d`, and a padding flag
(3·33 + 1 = 100 features per hour), plus the static vector `s` at hour *t*.
Default lookback L = 24 h, horizon H = 6 h.

## Outputs (`src/sepsis/models/heads.py`)

| Key | Shape | Task |
|---|---|---|
| `vitals` | B×H×7 | normalised vitals at t+1…t+6 |
| `risk` | B | logit of SepsisLabel at hour *t* |
| `onset` | B×4 | logits over onset buckets |

## Models (`src/sepsis/models/`)

- **Persistence** (`baselines.py`): repeat the last forward-filled value.
- **Linear** (`baselines.py`): one linear layer from the flattened window to future vitals.
- **MLP** (`baselines.py`): flattened window + static → 2 hidden layers → three heads.
- **GRU-D** (`grud.py`): GRU with learned decay of inputs toward the mean and of the hidden state,
  driven by time since last measurement (Che et al., 2018).
- **Temporal Transformer** (`transformer.py`): linear input projection + learned positional embedding
  → causal Transformer encoder (pre-norm) → last-hour representation, fused with static features.
  **Primary baseline.**
- **GAT + Temporal Transformer** (`gat_transformer.py`): the same temporal encoder, plus a GAT branch.
  Each of the 33 variables is a node whose features are its own window of (value, mask, delta),
  projected and summed with a learned node-identity embedding. Two dense multi-head GAT layers
  (`gat.py`) with residuals, then mean-pooling over nodes. Temporal, graph and static
  representations are concatenated into a shared patient state that feeds the heads. **Proposed model.**

The GAT is implemented densely in plain PyTorch rather than with PyTorch Geometric. With 33 nodes this is
as fast, avoids a heavy dependency, and makes attention matrices easy to extract
(`GATTransformer.attention_maps()`). This is a change from the proposal's wording and should be mentioned in the report.

### Fair comparison

The Transformer baseline differs from the proposed model **only** by the GAT branch:
same inputs, heads, splits, loss and training. Its `hidden_dim` is set so the parameter counts match.
Run `python scripts/count_params.py` after any architecture change. At the moment that gives 165,575 for
the Transformer (hidden 72) vs. 159,855 for the GAT model (hidden 64).

## Graphs (`src/sepsis/graphs/builders.py`)

| `model.graph` | Edges |
|---|---|
| `learned` | fully connected; attention learns weights |
| `clinical_prior` | variables in the same organ system (`constants.ORGAN_GROUPS`) |
| `correlation` | \|Pearson r\| ≥ 0.2 on co-observed training values |
| `none` | self-loops only (graph branch without neighbours) |

The organ-system grouping is a first draft. Review it as a team and cite a source in the report.

## Loss (`src/sepsis/training/losses.py`)

L = λ_v·L_vitals + λ_r·L_risk + λ_t·L_time

- L_vitals: MAE over future values that were actually measured
- L_risk: BCE with `risk_pos_weight` (default 10)
- L_time: cross-entropy over onset buckets

The λ's and the positive weight are tuned on validation.

## Training (`src/sepsis/training/trainer.py`)

AdamW, gradient clipping, early stopping on validation AUPRC (on vitals MAE for forecast-only models),
best checkpoint restored. Results are averaged over seeds 0, 1 and 2.

## Ablations

1. Graph type: learned / clinical prior / correlation / none
2. Inputs: with vs. without mask and delta features (to add as a config flag)
3. Single-task vs. multi-task: `--override "train.tasks=[risk]"`
4. Lookback: `--override data.lookback=12|24|48`

## Open decisions

- **Hours after onset.** `SepsisLabel` stays 1 after onset, so the risk target means
  "onset within 6 h, *or already septic*". The ≤6 h onset bucket currently also contains hours at
  or after onset (`data/targets.py`). Options are to keep this, to add a separate "post-onset" bucket,
  or to drop post-onset hours from the onset loss. Decide and state it in the report.
- **Input ablation flag.** Inputs without mask and delta are not implemented yet. Static features can be switched off with
  `--override data.use_static=false`.
