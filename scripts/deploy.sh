#!/usr/bin/env bash
# deploy.sh -- push local code/config to the agents. RUN THIS AFTER EVERY change, or the 7 agents
# keep running stale code (they clone from git at the base task's commit, not your local files).
#   bash scripts/deploy.sh                 # default commit message
#   bash scripts/deploy.sh "my message"    # custom message
set -e
cd "$(dirname "$0")/.."
MSG="${1:-deploy: native-save (xgboost UBJ + catboost CBM), pins, labels/hyperparam fixes}"
PY="${PY:-final_venv/bin/python}"

echo "== 1/3  commit + push (agents clone THIS) =="
git add -A
git commit -m "$MSG" || echo "   (nothing new to commit -- already committed)"
git push
echo

echo "== 2/3  re-register the base tasks (they re-capture the code + requirements) =="
$PY trainer/register_base_trainer.py --force
echo

echo "== 3/3  DONE. now retrain so models are saved in the portable format: =="
echo "     $PY core/publish_version.py --version v4 --dry-run   # confirm ~513k rows"
echo "     $PY core/publish_version.py --version v4             # then v5, v6"
echo
echo "after that publish, XGBoost + CatBoost load on every machine -- the version issue is gone."
