#!/usr/bin/env bash
# Mid-project evaluation: the baselines, Informer, Autoformer and the proposed models, one run each (seed 0).
#
#   bash scripts/run_mid_eval.sh                          # results/ (mid-project evaluation, old protocol)
#   OUT=results/per_task bash scripts/run_mid_eval.sh     # per-task selection rerun (current default config)
#   PY=.venv/bin/python bash scripts/run_mid_eval.sh
#
# Each run logs to results/logs/<experiment>_seed0.log. Runs whose
# metrics.json already exists are skipped, so the script can be resumed.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
SEED=0
EXPS="persistence linear mlp grud informer autoformer transformer gat_transformer gat_clinical_prior gat_correlation"
OUT=${OUT:-results}
mkdir -p "$OUT/logs"
for exp in $EXPS; do
  if [ -f "$OUT/$exp/seed$SEED/metrics.json" ]; then
    echo "skip $exp seed $SEED (done)"
    continue
  fi
  echo "$(date '+%F %T') start $exp seed $SEED"
  $PY scripts/train.py --config "configs/experiments/$exp.yaml" --seed "$SEED" --name "$exp" \
    --override paths.results_dir="$OUT" > "$OUT/logs/${exp}_seed${SEED}.log" 2>&1
  echo "$(date '+%F %T') done  $exp seed $SEED"
done
$PY scripts/summarize.py --results "$OUT"
