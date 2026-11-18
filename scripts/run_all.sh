#!/usr/bin/env bash
# Baselines, the main comparison and the graph ablation over 3 seeds.
#
#   bash scripts/run_all.sh                     # everything, seeds 0 1 2
#   SEEDS="0" bash scripts/run_all.sh           # one seed
#   EXPS="transformer gat_transformer" bash scripts/run_all.sh
#   SUFFIX=_reg OVERRIDES="train.lr=3e-4 model.dropout=0.3" bash scripts/run_all.sh
#                                               # same grid with overrides, saved as <exp>_reg
#
# Each run logs to results/logs/<experiment>_seed<k>.log. Runs whose
# metrics.json already exists are skipped, so the script can be resumed.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
SEEDS=${SEEDS:-"0 1 2"}
BASELINES="persistence linear mlp grud transformer"
GRAPHS="gat_transformer gat_none gat_correlation gat_clinical_prior"
EXPS=${EXPS:-"$BASELINES $GRAPHS"}
SUFFIX=${SUFFIX:-""}
OVERRIDES=${OVERRIDES:-""}
mkdir -p results/logs
for s in $SEEDS; do
  for exp in $EXPS; do
    name="$exp$SUFFIX"
    if [ -f "results/$name/seed$s/metrics.json" ]; then
      echo "skip $name seed $s (done)"
      continue
    fi
    echo "$(date '+%F %T') start $name seed $s"
    # shellcheck disable=SC2086  # OVERRIDES is a space-separated list on purpose
    $PY scripts/train.py --config "configs/experiments/$exp.yaml" --seed "$s" --name "$name" \
      ${OVERRIDES:+--override $OVERRIDES} > "results/logs/${name}_seed${s}.log" 2>&1
    echo "$(date '+%F %T') done  $name seed $s"
  done
done
$PY scripts/summarize.py
