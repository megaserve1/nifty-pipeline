"""cleanup_hpo_shap.py -- abort + archive the SHAP tasks that HPO trials spawned by mistake.

the bug (fixed in train.py 2026-07-21): every finished HPO trial queued its own SHAP task. this
finds the QUEUED / PENDING shap tasks and clears them, so the board is clean. it touches ONLY
queued-or-pending shap tasks -- it never archives a completed SHAP you want to keep, and it never
touches a training or HPO task.

run:   final_venv/bin/python scripts/cleanup_hpo_shap.py            # show what it WOULD do
       final_venv/bin/python scripts/cleanup_hpo_shap.py --do       # actually do it
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as C
from clearml import Task

DO = "--do" in sys.argv

# THE DISCRIMINATOR: does this SHAP task explain an HPO TRIAL, or a REAL model?
# a pending shap task is NOT junk just because it is pending -- shap_random_forest v4 was legitimately
# running when this bug surfaced. the junk ones are the ones spawned by hpo trials. we tell them apart
# by the MODEL they were told to explain (Args/model_task_id): an hpo trial's name still carries
# "(base)" ("train_xgboost (base): Args/..."), a real run was renamed "train_xgboost v5".
# checking the target model, not the shap task's own status, is what stops us aborting a real SHAP.
def explains_an_hpo_trial(shap_task) -> bool:
    mid = (shap_task.get_parameters() or {}).get("Args/model_task_id", "")
    if not mid:
        return False                      # can't prove it's junk -> leave it alone
    mt = Task.get_task(task_id=mid)
    return bool(mt and "(base)" in (mt.name or ""))

victims, kept = [], []
for name in ("shap_xgboost", "shap_catboost", "shap_random_forest"):
    for t in Task.get_tasks(project_name=C.CLEARML_PROJECT, task_name=name,
                            task_filter={"page_size": 50, "page": 0,
                                         "order_by": ["-created"]}):
        if t.get_status() not in ("queued", "in_progress", "created"):
            continue
        (victims if explains_an_hpo_trial(t) else kept).append(t)

if kept:
    print(f"KEEPING {len(kept)} real SHAP task(s) (they explain a promoted model):")
    for t in kept:
        print(f"  {t.id[:8]}  {t.name:<22} {t.get_status()}   <- left running")
    print()

if not victims:
    print("nothing to clean -- no queued/pending shap tasks.")
    raise SystemExit(0)

print(f"{'ABORT+ARCHIVE' if DO else 'WOULD clear'} {len(victims)} pending shap task(s):")
for t in victims:
    print(f"  {t.id[:8]}  {t.name:<20} {t.get_status()}")
    if DO:
        try:
            t.mark_stopped(force=True)   # take it off the queue
        except Exception:
            pass
        t.set_archived(True)             # hide it from the default board

print("\ndone." if DO else "\nthis was a preview. re-run with --do to actually clear them.")
