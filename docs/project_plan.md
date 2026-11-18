# Project plan

## Deadlines

| # | Milestone | Deadline | What is needed |
|---|---|---|---|
| 1 | Groups | Nov 4 | ✓ |
| 2 | Proposal (≤2 pages) | Nov 13 | ✓ submitted |
| 3 | Mid-project update | **Mon Nov 30** | data preparation, baselines, preliminary experiments, remaining work (graded checkpoint) |
| 4 | Final report (≤5 pages, research style) + full code + results + reproduction instructions | **Tue Dec 22** | late penalty applies |
| 5 | Presentation + viva | Dec 25–26 | every member explains their own contribution |

The detailed rubric for the final report and presentation will be released separately.

## Work packages and owners

Fill in owners. The viva asks each member to explain their individual contribution.

| Package | Code | Owner |
|---|---|---|
| Data download, profiling, preprocessing, splits | `data/`, `notebooks/01_*` | |
| Baselines: persistence, linear, MLP, GRU-D | `models/baselines.py`, `models/grud.py` | |
| Temporal Transformer + GAT + graphs | `models/transformer.py`, `models/gat*.py`, `graphs/` | |
| Evaluation: utility check, alarms, bootstrap | `evaluation/` | |
| Experiments, ablations, results tables | `configs/`, `scripts/run_all.sh` | |
| Attention analysis before onset | `notebooks/03_*` | |
| Report and slides | | all |

## Towards Nov 30

- [ ] Download data and reproduce the dataset numbers from the proposal
- [ ] Verify `utility.py` against the official scorer
- [ ] Train persistence, linear, MLP, GRU-D and the Transformer (1 seed is fine)
- [ ] First GAT + Transformer run
- [ ] Draft the update: data prep, baseline table, preliminary comparison, remaining plan

## Towards Dec 22

- [ ] 3 seeds for core models, graph ablation, test_B
- [ ] Remaining ablations (priority order in `experiments.md`)
- [ ] Attention analysis
- [ ] Report (≤5 pages), README reproduction check from a clean clone

## Open decisions

See "Open decisions" in `methodology.md`. In addition:

- Calibration metric for the report (Brier and ECE are both implemented)
- Optional: cluster the forecast vital trajectories of septic patients into patterns, which links back to
  the npj Digital Medicine paper that motivates the project
