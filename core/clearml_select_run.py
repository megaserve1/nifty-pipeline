"""core/clearml_select_run.py -- the ClearML UI as your SELECTION SHEET (no extra app).

WHAT THIS IS
    a ClearML task whose CONFIGURATION panel is the selection sheet. you register it once, then in
    the ClearML web UI you: clone it -> edit its Args -> enqueue. an agent then runs the whole loop
    for you:  freeze recipe (make_version) -> build_dataset -> (optionally) publish_version + train.

    it is a THIN WRAPPER. it shells out to the exact CLI you already use -- it changes NOTHING in
    make_version / build_dataset / publish_version. delete this one file and the base task and the
    pipeline is exactly as before. (that is the "we can revert" guarantee.)

HONEST LIMITS (why this is the lean option, not the pretty one)
    * ClearML's Configuration panel is a TEXT / param editor -- NOT checkboxes. you type or paste
      feature names. for a real tick-menu with explanations you need a front-end (streamlit).
    * enqueuing with do_publish=true spends AGENT TIME (money). there is a confirm print, but the
      real guard is you: leave do_publish=false to just build, flip it to true only when you mean it.

REGISTER IT ONCE (from a machine with your clearml.conf; commit + push first so agents can clone):
    final_venv/bin/python core/clearml_select_run.py        # no args -> registers "select_and_run (base)"

THEN IN THE CLEARML UI
    find "select_and_run (base)" -> right-click Clone -> open the clone's CONFIGURATION -> set Args:
        from_version   clone an existing version and change it, e.g. "v4"   (blank = fresh selection)
        drop           features to REMOVE from from_version (comma list)    (derive mode only)
        add            features to ADD                                      (derive mode only)
        features       for a FRESH version: the exact feature list (comma)  (ignored if from_version set)
        labels         label set: L1 / L2 / L3 (blank = configured default)
        do_publish     "false" = build only  |  "true" = also publish + enqueue training
    -> Enqueue on the 'training' queue. watch it run in the UI.
"""
import argparse
import pathlib
import subprocess
import sys

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parent.parent
sys.path.insert(0, str(ROOT))
import config as C          # noqa: E402


def _versions() -> set:
    """the set of frozen recipe versions right now (names only, no lock files)."""
    return {p.name for p in C.VERSIONS_DIR.glob("dataset_v*.yaml")
            if not p.name.endswith(".lock.yaml")}


def main():
    # clearml MUST be imported before parse_args -- it hooks argparse at import so a cloned task's
    # Args/ overrides reach us. same rule as train.py / hpo.py.
    from clearml import Task   # noqa: F401
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_version", default="", help="clone this version and change it (e.g. v4)")
    ap.add_argument("--drop", default="", help="features to remove from from_version (comma list)")
    ap.add_argument("--add", default="", help="features to add")
    ap.add_argument("--features", default="", help="FRESH version: exact feature list (comma list)")
    ap.add_argument("--labels", default="", help="label set: L1/L2/L3 (blank = configured default)")
    ap.add_argument("--do_publish", default="false", help="'true' = publish + train; 'false' = build only")
    a = ap.parse_args()

    task = Task.init(project_name=C.CLEARML_PROJECT,
                     task_name="select_and_run (base)",
                     task_type=Task.TaskTypes.custom)

    # no selection -> this is the one-off REGISTRATION run that creates the clonable template.
    if not (a.from_version or a.features):
        print("no selection (from_version / features) given -- base registration run. exiting cleanly.\n"
              "clone this in the ClearML UI, set its Args, and enqueue.")
        task.close()
        return

    py = sys.executable

    def sh(cmd):
        print("  $ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=str(ROOT))

    # ---- 1. freeze the recipe (derive from a parent, or a fresh selection) --------------------
    before = _versions()
    mv = [py, "core/make_version.py"]
    if a.labels:
        mv += ["--labels", a.labels]
    if a.from_version:
        mv += ["--from", a.from_version]
        if a.drop:
            mv += ["--drop", a.drop]
        if a.add:
            mv += ["--add", a.add]
        print(f"[1/3] clone {a.from_version}  drop=[{a.drop}] add=[{a.add}] labels=[{a.labels or 'default'}]")
    else:
        mv += ["--feature", a.features]     # a fresh selection: exactly these features
        print(f"[1/3] fresh selection: {a.features}  labels=[{a.labels or 'default'}]")
    sh(mv)

    # which version did make_version just create? (dir diff -- robust, no stdout parsing)
    new = sorted(_versions() - before)
    if not new:
        raise SystemExit("make_version created no new recipe -- check the drop/add/features names.")
    version = "v" + new[-1].replace("dataset_v", "").replace(".yaml", "")
    print(f"      created recipe: {version}")
    task.set_parameter("Args/created_version", version)   # surfaced in the UI

    # ---- 2. build the dataset -----------------------------------------------------------------
    print(f"[2/3] build_dataset --version {version}   (this can take a while on the full data)")
    sh([py, "bridge/build_dataset.py", "--version", version])

    # ---- 3. publish + train (only if asked -- this is the step that costs agent time) ---------
    if str(a.do_publish).strip().lower() in ("1", "true", "yes"):
        print(f"[3/3] publish_version --version {version}  -> publishes + enqueues training (AGENT TIME).")
        sh([py, "core/publish_version.py", "--version", version])
    else:
        print(f"[3/3] do_publish=false -- built {version} but NOT published/trained. "
              f"re-run with do_publish=true (or publish it yourself) when ready.")

    print(f"\ndone. selection -> {version}. the ClearML Configuration panel was your selection sheet.")
    task.close()


if __name__ == "__main__":
    main()
