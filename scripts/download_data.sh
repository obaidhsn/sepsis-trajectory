#!/usr/bin/env bash
# Download the PhysioNet/CinC Challenge 2019 training data (~40k .psv files)
# and lay it out as:
#   data/raw/training_setA/p0*.psv   (Hospital A, 20,336 patients)
#   data/raw/training_setB/p1*.psv   (Hospital B, 20,000 patients)
#
# Two sources:
#   bash scripts/download_data.sh                    # Kaggle mirror (needs the Kaggle CLI + API token,
#                                                    #   see docs/data.md)
#   SOURCE=physionet bash scripts/download_data.sh   # PhysioNet directly, no account needed
#                                                    #   (40k small files fetched in parallel, ~10-20 min)
set -euo pipefail
cd "$(dirname "$0")/.."
SOURCE=${SOURCE:-kaggle}

if [ "$SOURCE" = physionet ]; then
  # The file listing comes from physionet.org; the files themselves are served from
  # PhysioNet's open S3 bucket (physionet.org answers each file with a 302 redirect there).
  LIST=https://physionet.org/files/challenge-2019/1.0.0/training
  FILES=https://physionet-open.s3.amazonaws.com/challenge-2019/1.0.0/training
  for s in A B; do
    mkdir -p "data/raw/training_set$s"
    find "data/raw/training_set$s" -type f -empty -delete
    echo "Listing and downloading training_set$s from PhysioNet ..."
    ( cd "data/raw/training_set$s" &&
      curl -fsSL "$LIST/training_set$s/" | grep -oE 'p[0-9]{6}\.psv' | sort -u |
      comm -23 - <(ls | sort) |
      sed "s#^#$FILES/training_set$s/#" |
      xargs -n 40 -P 32 curl -sfL --retry 5 --remote-name-all )
  done
  A=$(find data/raw/training_setA -name '*.psv' | wc -l)
  B=$(find data/raw/training_setB -name '*.psv' | wc -l)
  EMPTY=$(find data/raw/training_setA data/raw/training_setB -name '*.psv' -empty | wc -l)
  echo "A: $A files (expect 20336), B: $B files (expect 20000), empty: $EMPTY (re-run to fetch missing ones)"
fi

if [ "$SOURCE" = kaggle ]; then

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
fi

# Official scorer, used to verify src/sepsis/evaluation/utility.py.
# The PhysioNet file path returns 404, so the challenge's GitHub repository is the primary source.
SCORER_URL=https://raw.githubusercontent.com/physionetchallenges/evaluation-2019/master/evaluate_sepsis_score.py
curl -fsSL -o data/raw/evaluate_sepsis_score.py "$SCORER_URL" \
  || echo "Could not fetch scorer; get it from https://github.com/physionetchallenges/evaluation-2019"

A=$(find data/raw/training_setA -name '*.psv' | wc -l)
B=$(find data/raw/training_setB -name '*.psv' | wc -l)
EMPTY=$(find data/raw/training_setA data/raw/training_setB -name '*.psv' -empty | wc -l)
echo "A: $A files (expect 20336), B: $B files (expect 20000), empty: $EMPTY"
