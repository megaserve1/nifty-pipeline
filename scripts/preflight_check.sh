#!/usr/bin/env bash
# preflight_check.sh -- run on EVERY PC before the live run. Prints PASS/FAIL per check.
#   worker PC :  bash scripts/preflight_check.sh
#   controller:  bash scripts/preflight_check.sh --controller
# override the python if your venv is elsewhere:  PY=/path/to/python bash scripts/preflight_check.sh
set -u
PY="${PY:-final_venv/bin/python}"
BUCKET="live-nifty-pipeline"; PROJECT="mega-ml"; QUEUE="training"
pass=0; fail=0
ok(){ echo "  PASS  $1"; pass=$((pass+1)); }
no(){ echo "  FAIL  $1"; fail=$((fail+1)); }

echo "== 1. ClearML account (must be identical on every PC) =="
a=$(grep -E "api_server" ~/clearml.conf 2>/dev/null | head -1 | tr -d ' ')
[ -n "$a" ] && echo "  $a" || no "no api_server in ~/clearml.conf"

echo "== 2. GCS project line (the silent trap) =="
grep -q "$PROJECT" ~/clearml.conf 2>/dev/null && ok "google.storage.project present" \
  || no "add  google.storage.project $PROJECT  to ~/clearml.conf"

echo "== 3. pandas 3.x (must match the build machine) =="
v=$($PY -c "import pandas;print(pandas.__version__)" 2>/dev/null)
case "$v" in 3.*) ok "pandas $v";; *) no "pandas '$v' not 3.x -- pip install -r requirements.txt";; esac

echo "== 4. model libs (the ONLY ones the pipeline uses) =="
$PY -c "import xgboost,catboost,sklearn,shap,matplotlib" 2>/dev/null \
  && ok "xgboost catboost scikit-learn shap matplotlib" \
  || no "a model lib is missing -- pip install -r requirements.txt"

echo "== 5. GCS auth token valid =="
$PY -c "import google.auth;google.auth.default()" 2>/dev/null && ok "gcs auth ok" \
  || no "gcloud auth application-default login  (then: set-quota-project $PROJECT)"

echo "== 6. can this PC READ the live bucket? (auth+permission+project together) =="
$PY -c "from google.cloud import storage;list(storage.Client(project='$PROJECT').list_blobs('$BUCKET',max_results=1))" 2>/dev/null \
  && ok "read gs://$BUCKET" \
  || no "cannot read gs://$BUCKET -- 403=grant this identity roles/storage.objectAdmin ; 'project could not be determined'=check 2 failed"

echo "== 7. clearml-agent installed =="
clearml-agent --version >/dev/null 2>&1 && ok "clearml-agent installed" || no "pip install clearml-agent"

echo "== 8. training queue reachable =="
$PY -c "from clearml.backend_api.session.client import APIClient;exit(0 if APIClient().queues.get_all(name='$QUEUE') else 1)" 2>/dev/null \
  && ok "queue '$QUEUE' exists" \
  || echo "  WARN  queue '$QUEUE' not created yet -- the agent makes it with --create-queue"

if [ "${1:-}" = "--controller" ]; then
  echo "== C1. code committed (agents clone git, not your working files) =="
  [ -z "$(git status --porcelain 2>/dev/null)" ] && ok "working tree clean" \
    || no "uncommitted changes -- git add -A && commit && push, then: python trainer/register_base_trainer.py --force"
  echo "== C2. pushed to remote =="
  if git rev-parse --abbrev-ref '@{u}' >/dev/null 2>&1; then
    [ "$(git rev-parse HEAD)" = "$(git rev-parse @{u} 2>/dev/null)" ] && ok "local == remote" || no "local ahead -- git push"
  else no "no upstream branch set -- git push -u origin <branch>"; fi
  echo "== C3. offline tests =="
  $PY -m pytest tests/ -q >/dev/null 2>&1 && ok "pytest tests/ pass" || no "pytest failing -- run: $PY -m pytest tests/ -q"
  echo "== C4. can this PC WRITE the bucket? (dvc push needs this) =="
  $PY -c "from google.cloud import storage;b=storage.Client(project='$PROJECT').bucket('$BUCKET');x=b.blob('preflight/_wtest');x.upload_from_string('ok');x.delete()" 2>/dev/null \
    && ok "write+delete gs://$BUCKET (tiny test object, removed)" \
    || no "cannot WRITE gs://$BUCKET -- grant roles/storage.objectAdmin"
fi

echo
echo "==== $pass passed, $fail failed ===="
[ "$fail" -eq 0 ] && echo "READY" || { echo "FIX THE FAILURES ABOVE BEFORE THE RUN"; exit 1; }
