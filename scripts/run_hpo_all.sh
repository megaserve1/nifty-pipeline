#!/usr/bin/env bash
# run_hpo_all.sh -- optuna search for ALL 3 models on BOTH datasets. 6 searches, in order.
#
# each hpo.py call BLOCKS until its search finishes, so these run one after another. that is
# deliberate: all searches share the same agents, so running two at once just makes both slower.
#
# run:   bash scripts/run_hpo_all.sh
#        bash scripts/run_hpo_all.sh v4          # only v4
#        bash scripts/run_hpo_all.sh v5          # only v5
set -euo pipefail
cd "$(dirname "$0")/.."
PY=final_venv/bin/python

V4=53556d8907ce46e7b18e6ba72ddf89ee
V5=2526ababef67424eb71e95a9e3b2bbf7

CONCURRENT=3          # = your online agent count. higher just queues.

# per-model budget. the two boosters early-stop, so their trials are cheaper and we can afford
# more of them. random_forest has NO early stopping -- every trial builds all 3000 trees, ~4h --
# so it gets fewer trials and a much larger kill-switch. the default job_minutes of 90 would
# kill EVERY forest trial before it finished.
run_one () {          # $1=model  $2=dataset_id  $3=version
  local trials mins
  case "$1" in
    random_forest) trials=12; mins=300 ;;
    *)             trials=20; mins=180 ;;
  esac
  echo
  echo "=============================================================="
  echo " HPO  $1  on v$3   ($trials trials, $CONCURRENT at a time, kill at ${mins}m)"
  echo "=============================================================="
  $PY trainer/hpo.py \
      --model_type "$1" \
      --dataset_id "$2" \
      --dataset_version "$3" \
      --strategy optuna \
      --trials "$trials" \
      --concurrent "$CONCURRENT" \
      --job_minutes "$mins"
}

WHICH="${1:-all}"

if [ "$WHICH" = "all" ] || [ "$WHICH" = "v4" ]; then
  for m in xgboost catboost random_forest; do run_one "$m" "$V4" 4; done
fi
if [ "$WHICH" = "all" ] || [ "$WHICH" = "v5" ]; then
  for m in xgboost catboost random_forest; do run_one "$m" "$V5" 5; done
fi

echo
echo "all searches done. winners are written as best_params_<model>.json."
echo "promote one with:  $PY trainer/apply_hpo.py best_params_<model>.json"
