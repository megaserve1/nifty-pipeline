"""
trainer/register_base_trainer.py -- run this ONCE, ever, before your first publish.

WHY IT EXISTS
    publish_version.py does not run the trainer directly. it CLONES a "base task" and changes
    its settings. so the base tasks have to exist first.

    running a script once with no dataset id makes ClearML record it -- the script, its
    arguments, its environment -- and then it exits. that recording IS the base task.

it creates these base tasks (select_champion only when config.RUN_CHAMPION is on -- it is off):
    train_xgboost (base)
    train_catboost (base)
    shap_explain (base)
    export_scored_tables (base)
    select_champion (base)   -- only if RUN_CHAMPION

without them, publish_version.py stops with a clear message instead of quietly publishing a
dataset that nothing ever trains on.

run:
    python trainer/register_base_trainer.py
    python trainer/register_base_trainer.py --force    # re-register after changing a script
"""
import argparse
import subprocess
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


def register(script: pathlib.Path, task_name: str, extra: list, force: bool) -> bool:
    from clearml import Task

    # On a FRESH account the project does not exist yet, and Task.get_task RAISES
    # ("No projects found") instead of returning None. That is not an error for us -- it just
    # means nothing is registered, so there is nothing to reuse. The first register() call
    # creates the project as a side effect.
    try:
        existing = Task.get_task(project_name=C.CLEARML_PROJECT, task_name=task_name)
    except ValueError:
        existing = None
    if existing is not None and not force:
        print(f"  already registered  {task_name}   ({existing.id})")
        return True
    if existing is not None and force:
        # ARCHIVE THE OLD ONE FIRST, and take its name away.
        # Task.init reuses a task that has the same project+name, and a reused task keeps the
        # git commit it was first recorded with -- so --force re-ran the script and changed
        # NOTHING. agents kept cloning a base pinned to an old commit while the repo moved on,
        # silently, with no error anywhere. renaming it means Task.init finds no match and
        # records today's commit.
        before = str(getattr(existing.data.script, "version_num", ""))[:12]
        try:
            existing.rename(f"{task_name} [superseded {before}]")
            existing.set_archived(True)
            print(f"  re-registering      {task_name}  (old one archived, was on {before})")
        except Exception as exc:
            print(f"  !! could not archive the old {task_name}: {exc}\n"
                  f"     delete it in the ClearML UI, or the new commit will not take effect.")
            return False

    print(f"  registering         {task_name} ...", flush=True)
    r = subprocess.run([sys.executable, str(script)] + extra, text=True,
                       capture_output=True)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()[-6:]
        print("     FAILED:\n       " + "\n       ".join(tail))
        return False

    t = Task.get_task(project_name=C.CLEARML_PROJECT, task_name=task_name)
    if t is None:
        print(f"     ran, but no task appeared. check ~/clearml.conf")
        return False
    got = str(getattr(t.data.script, "version_num", "")) or "?"
    print(f"     OK  {t.id}   commit {got[:12]}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-register even if the base task already exists "
                         "(do this after you change a trainer script)")
    a = ap.parse_args()

    print(f"registering base tasks in ClearML project '{C.CLEARML_PROJECT}'\n")

    ok = True
    # one base task per model type. the --model_type is baked in, and publish_version.py
    # overrides only the dataset.
    for mtype in C.MODEL_TYPES:
        ok &= register(HERE / "train.py", C.base_trainer_name(mtype),
                       ["--model_type", mtype], a.force)

    ok &= register(HERE / "shap_explain.py", C.BASE_SHAP_NAME, [], a.force)
    ok &= register(HERE / "export_scored_tables.py", C.BASE_EXPORT_NAME, [], a.force)
    # the SAME script registered a second time in OOS mode -- same trick as train.py being
    # registered once per model_type. with no --model_task_id it exits cleanly, which is what
    # makes it a clonable template.
    ok &= register(HERE / "export_scored_tables.py", C.BASE_OOS_NAME, ["--mode", "oos"], a.force)
    ok &= register(HERE / "deepchecks_report.py", C.BASE_DEEPCHECKS_NAME, [], a.force)
    ok &= register(HERE / "feature_analysis.py", C.BASE_FEATURE_ANALYSIS_NAME, [], a.force)
    if C.RUN_CHAMPION:
        ok &= register(HERE / "select_champion.py", C.BASE_CHAMPION_NAME, [], a.force)
    else:
        print("  select_champion NOT registered (champion is off: config.RUN_CHAMPION=False)")

    print()
    if not ok:
        raise SystemExit("some base tasks failed to register -- see the errors above")

    print("all base tasks registered.")
    print(f"\nremember: a worker must be listening or nothing will ever run:")
    print(f"    clearml-agent daemon --queue {C.TRAIN_QUEUE}")
    print(f"\none agent  -> the three models train one after another.")
    print(f"three agents (one per machine) -> they train at the same time.")


if __name__ == "__main__":
    main()
