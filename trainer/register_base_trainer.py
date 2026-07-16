"""
trainer/register_base_trainer.py -- run this ONCE, ever, before your first publish.

WHY IT EXISTS
    publish_version.py does not run the trainer directly. it CLONES a "base task" and changes
    its settings. so the base tasks have to exist first.

    running a script once with no dataset id makes ClearML record it -- the script, its
    arguments, its environment -- and then it exits. that recording IS the base task.

it creates FIVE base tasks:
    train_random_forest (base)
    train_xgboost (base)
    train_catboost (base)
    shap_explain (base)
    select_champion (base)

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
        print(f"  re-registering      {task_name}  (the old one stays in ClearML)")

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
    print(f"     OK  {t.id}")
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
    ok &= register(HERE / "select_champion.py", C.BASE_CHAMPION_NAME, [], a.force)

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
