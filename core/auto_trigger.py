"""
core/auto_trigger.py -- OPTIONAL decoupled watcher.

publish_version.py already enqueues training in-process, deterministically, with an
exit code you can see. That is the PRIMARY path and it cannot silently die.

This watcher is the BACKUP path: it catches datasets published out-of-band (someone
clicks Publish in the ClearML UI, or another service publishes a version).

    *** RUN ONE OR THE OTHER, NEVER BOTH -- or every version trains twice. ***
    If you run this, publish with:  python core/publish_version.py --version vN --no-train

Two traps this file guards against:
  1. trigger_on_publish=True fires on status 'published'. Dataset.finalize() only reaches
     'completed'. publish_version.py calls .publish() -- if you ever stop doing that, this
     watcher goes silent with no error.
  2. add_dataset_trigger registers against the project's hidden '.datasets' subproject.
     If no dataset exists yet, the trigger registers against nothing. We assert it took.

Run (as a long-lived ClearML service, survives your laptop):
    python core/auto_trigger.py                # start_remotely -> the 'services' queue
    python core/auto_trigger.py --foreground   # blocking, for a quick local test
"""
import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import config as C                                            # noqa: E402
from publish_version import enqueue_all, enqueue_champion     # noqa: E402
# there is no function called `enqueue`. this file used to import one, so it died with an
# ImportError on this very line before it did anything at all -- and no test imported it, which
# is exactly why the suite stayed green while the whole backup path was dead.


def retrain(dataset_id):
    """Called by the scheduler with the id of the dataset that JUST published.

    We use that id directly. We never call Dataset.get(project, name) to find "the
    latest" -- that resolves by version ordering, races the 1-minute poll, and is how
    the demo pipeline kept training on v1.0 while new versions rolled past it.
    """
    # THIS RUNS ON THE SCHEDULER'S WORKER THREAD, AND A THREAD SWALLOWS SystemExit.
    # enqueue_all raises SystemExit when a base task is missing. on the main thread that is a
    # loud stop; on the TriggerScheduler's thread it kills ONLY the thread -- the watcher keeps
    # polling, looking perfectly healthy, having enqueued some-but-not-all models and no
    # champion. so everything is caught, printed in full, and re-raised as a real error the
    # scheduler will log.
    import traceback
    try:
        from clearml import Dataset
        ds = Dataset.get(dataset_id=dataset_id)
        print(f"[trigger] published: {ds.name} v{ds.version}  ({dataset_id})")
        # the same two calls publish_version makes on its own path: train all the models, then
        # queue the champion picker to WAIT for them (by name).
        ids = enqueue_all(dataset_id, ds.version, C.MODEL_TYPES)
        enqueue_champion(dataset_id, ds.version, list(ids))
    except BaseException as e:
        print(f"[trigger] FAILED for dataset {dataset_id}:")
        traceback.print_exc()
        raise RuntimeError(f"auto_trigger.retrain failed for {dataset_id}: {e}") from e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--foreground", action="store_true",
                    help="block in this terminal instead of handing off to the services queue")
    a = ap.parse_args()

    from clearml.automation import TriggerScheduler

    sched = TriggerScheduler(pooling_frequency_minutes=1.0)
    sched.add_dataset_trigger(
        schedule_function=retrain,
        trigger_project=C.CLEARML_PROJECT,
        trigger_name=C.CLEARML_DATASET,
        trigger_on_publish=True,      # 'published', not 'completed' -- see module docstring
        single_instance=True,         # never stack two trainings for one publish
        name="retrain-on-publish",
    )

    if not sched.get_triggers():
        raise SystemExit(
            "no trigger registered. ClearML attaches dataset triggers to the hidden\n"
            f"'{C.CLEARML_PROJECT}/.datasets' subproject, which does not exist until the\n"
            "first dataset is published. Publish one version first:\n"
            "    python core/publish_version.py --version v1 --no-train")

    print(f"watching {C.CLEARML_PROJECT}/{C.CLEARML_DATASET} for PUBLISHED versions")
    if a.foreground:
        print("(foreground: dies when this terminal closes)")
        sched.start()
    else:
        # hands the watcher to an always-on services agent, so it survives logout/reboot.
        # requires:  clearml-agent daemon --services-mode --queue services
        sched.start_remotely(queue="services")


if __name__ == "__main__":
    main()
