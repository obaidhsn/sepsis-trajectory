# Data

Nothing in this folder is committed (see `.gitignore`).

- `raw/training_setA/*.psv`: Hospital A, 20,336 patients
- `raw/training_setB/*.psv`: Hospital B, 20,000 patients
- `raw/evaluate_sepsis_score.py`: official Challenge scorer
- `processed/patients.pkl`: one dict of arrays per patient (see `src/sepsis/data/preprocess.py`)
- `processed/normalizer.json`: train-only means/stds
- `processed/splits.json`: patient ids for train / val / test_A / test_B

Get the raw data with `bash scripts/download_data.sh` (needs the Kaggle CLI and token, see `docs/data.md`), then run `python scripts/preprocess.py`.
Downloaded from the Kaggle mirror https://www.kaggle.com/datasets/farjanayesmin/the-physionet-challenge-2019-dataset
of https://physionet.org/content/challenge-2019/1.0.0/ (open access).
