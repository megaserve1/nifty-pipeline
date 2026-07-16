#!/bin/bash
# run_tuned_overnight.sh -- HPO all 3 models, promote winners, train tuned + champion.
# unattended: kick off once, leave it overnight. see the launch note at the bottom.
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-final_venv/bin/python}"

DATASET_ID="${DATASET_ID:-${1:-}}"
DATASET_VERSION="${DATASET_VERSION:-${2:-v3}}"
TRIALS="${TRIALS:-15}"
CAP_MIN="${CAP_MIN:-150}" # per-model cap -- stops at TRIALS OR CAP_MIN, whichever comes first
                       # comes first. bounds the night so a few slow catboost trials can't run away.

if [ -z "$DATASET_ID" ]; then
  echo "usage: $0 <clearml-dataset-id> [dataset-version]" >&2
  echo "   or: DATASET_ID=<id> DATASET_VERSION=vN $0" >&2
  exit 2
fi

for M in random_forest xgboost catboost; do
  echo ""
  echo "############ HPO: $M   ($(date +%H:%M)) ############"
  $PY trainer/hpo.py --dataset_id "$DATASET_ID" --model_type "$M" \
      --dataset_version "$DATASET_VERSION" --trials "$TRIALS" --total_minutes "$CAP_MIN"
  # --force: promote unattended even if a winner sits at a range edge. REVIEW configs/tuned/*.json
  # in the morning -- a range-edge winner means the range was the binding constraint, not the data.
  $PY trainer/apply_hpo.py "best_params_${M}.json" --force
done

echo ""
echo "############ TRAIN TUNED + CHAMPION   ($(date +%H:%M)) ############"
$PY core/publish_version.py --version "$DATASET_VERSION"

echo ""
echo "############ ALL DONE   ($(date +%H:%M)) ############"
