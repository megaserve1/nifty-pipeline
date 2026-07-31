"""scripts/queue_oos.py -- score an OOS set with EVERY trained model. one table per model.

WHAT IT DOES
    it does NO scoring itself. for each trained model you name, it clones the 'score_oos (base)'
    task, points it at that model + the OOS dataset, and enqueues it. a worker then does the work:
        fetch model -> preflight -> score -> write table -> run the backtest -> print its output
    the backtest's numbers land in that task's CONSOLE tab in ClearML.

    one clone per model = one table per model, which is what the backtest wants.

RUN IT (you run it)
    # by model task ids (copy them from ClearML):
    final_venv/bin/python scripts/queue_oos.py --oos_dataset_id <ID> --oos_tag 2025h2 \
        --model_task_ids abc123,def456 --backtest scripts/backtest.py

    # or let it FIND the finished trainers for a dataset version:
    final_venv/bin/python scripts/queue_oos.py --oos_dataset_id <ID> --oos_tag 2025h2 \
        --version v5 --backtest scripts/backtest.py

NOTE the backtest script must be COMMITTED AND PUSHED -- an agent runs the repo snapshot, so a
file that only exists on your laptop will not be there.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as C   # noqa: E402


def find_model_tasks(Task, version: str) -> list:
    """the finished training tasks for a dataset version -> [(task_id, model_type), ...]."""
    out = []
    for mtype in C.MODEL_TYPES:
        name = f"train_{mtype} {version}"
        tasks = Task.get_tasks(project_name=C.CLEARML_PROJECT, task_name=name,
                               task_filter={"status": ["completed"]})
        if not tasks:
            print(f"  !! no completed task named '{name}' -- skipping {mtype}")
            continue
        t = sorted(tasks, key=lambda x: x.data.last_update or 0)[-1]   # newest
        out.append((t.id, mtype))
        print(f"  found {mtype}: {t.id}  ('{name}')")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos_dataset_id", required=True, help="ClearML dataset id of the OOS set")
    ap.add_argument("--oos_tag", default="", help="short name, e.g. 2025h2 (goes in the filename)")
    ap.add_argument("--model_task_ids", default="", help="comma-separated training task ids")
    ap.add_argument("--version", default="", help="instead of ids: find the trainers for this vN")
    # DEFAULT FROM CONFIG, not blank. with '' defaults, forgetting these flags queued tasks that
    # wrote the table, skipped the backtest, and still finished green -- config already knows both.
    ap.add_argument("--backtest", default=C.BACKTEST_SCRIPT,
                    help="backtest script to run on each table (default: config.BACKTEST_SCRIPT)")
    ap.add_argument("--price_dataset_id", default=C.PRICE_DATASET_ID,
                    help="ClearML dataset id of the OHLC prices (default: config.PRICE_DATASET_ID)")
    ap.add_argument("--queue", default=None, help=f"default: {C.OOS_QUEUE}")
    a = ap.parse_args()

    from clearml import Task

    queue = a.queue or C.OOS_QUEUE
    base = Task.get_task(project_name=C.CLEARML_PROJECT, task_name=C.BASE_OOS_NAME)
    if base is None:
        raise SystemExit(f"no base task '{C.BASE_OOS_NAME}'. run:  "
                         f"python trainer/register_base_trainer.py --force")

    if a.model_task_ids:
        pairs = [(t.strip(), "") for t in a.model_task_ids.split(",") if t.strip()]
    elif a.version:
        print(f"looking for the trained models of {a.version} ...")
        pairs = find_model_tasks(Task, a.version)
    else:
        raise SystemExit("give either --model_task_ids or --version.")
    if not pairs:
        raise SystemExit("no model tasks to score with.")

    tag = a.oos_tag or "oos"
    print(f"\nqueueing {len(pairs)} OOS scoring task(s) on '{queue}'")
    for tid, mtype in pairs:
        run = Task.clone(source_task=base, name=f"scored_oos_{tag}_{mtype or tid[:8]}")
        run.set_parameters({
            "Args/mode": "oos",
            "Args/model_task_id": tid,
            "Args/oos_dataset_id": a.oos_dataset_id,
            "Args/oos_tag": tag,
            "Args/backtest": a.backtest,
            "Args/price_dataset_id": a.price_dataset_id,
        })
        Task.enqueue(run, queue_name=queue)
        print(f"  queued  {run.name}   (model task {tid})")

    print(f"\ndone. watch them in ClearML -- each task's CONSOLE tab shows the backtest output, "
          f"and its ARTIFACTS tab holds that model's table.")
    if not a.backtest:
        print("(no --backtest given: the tables get written, but nothing is run on them.)")


if __name__ == "__main__":
    main()
