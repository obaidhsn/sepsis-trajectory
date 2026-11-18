#!/usr/bin/env bash
# Download the PhysioNet/CinC Challenge 2019 training data (~40k .psv files)
# from the Kaggle mirror and lay it out as:
#   data/raw/training_setA/p0*.psv   (Hospital A, 20,336 patients)
#   data/raw/training_setB/p1*.psv   (Hospital B, 20,000 patients)
#
# Needs the Kaggle CLI and an API token (see docs/data.md):
#   pip install kaggle
#   put kaggle.json in ~/.kaggle/  (chmod 600)  or export KAGGLE_USERNAME / KAGGLE_KEY
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=farjanayesmin/the-physionet-challenge-2019-dataset
TMP=data/raw/_kaggle

command -v kaggle >/dev/null || { echo "kaggle CLI not found: pip install kaggle"; exit 1; }

mkdir -p "$TMP" data/raw/training_setA data/raw/training_setB
echo "Downloading $DATASET from Kaggle ..."
kaggle datasets download -d "$DATASET" -p "$TMP" --unzip

# The archive's folder layout may be nested (e.g. training_setA/training_setA/),
# so find every .psv wherever it is and sort it by patient id:
# Hospital A ids start with p0, Hospital B ids start with p1.
echo "Sorting .psv files into training_setA / training_setB ..."
find "$TMP" -type f -name 'p0*.psv' -exec mv -f -t data/raw/training_setA {} +
find "$TMP" -type f -name 'p1*.psv' -exec mv -f -t data/raw/training_setB {} +

LEFT=$(find "$TMP" -type f -name '*.psv' | wc -l)
[ "$LEFT" -eq 0 ] || echo "Warning: $LEFT .psv files did not match p0*/p1* and were left in $TMP"
[ "$LEFT" -eq 0 ] && rm -rf "$TMP"

# Official scorer, used to verify src/sepsis/evaluation/utility.py (scripts/verify_utility.py).
# The PhysioNet file path returns 404, so the challenge's GitHub repository is the primary source.
SCORER_URL=https://raw.githubusercontent.com/physionetchallenges/evaluation-2019/master/evaluate_sepsis_score.py
curl -fsSL -o data/raw/evaluate_sepsis_score.py "$SCORER_URL" \
  || echo "Could not fetch scorer; get it from https://github.com/physionetchallenges/evaluation-2019"

A=$(find data/raw/training_setA -name '*.psv' | wc -l)
B=$(find data/raw/training_setB -name '*.psv' | wc -l)
EMPTY=$(find data/raw/training_setA data/raw/training_setB -name '*.psv' -empty | wc -l)
echo "A: $A files (expect 20336), B: $B files (expect 20000), empty: $EMPTY"
