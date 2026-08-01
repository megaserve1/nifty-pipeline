"""ui/_shared.py -- one project picker and one model picker, used by every page.

WHY THIS EXISTS
    an overnight run finishes 2-3 models. each one produces a scored table, a backtest, a SHAP
    task and a deepchecks report. the pages used to list TABLES, so you picked "scored_tables_
    xgboost v7" and had to work out for yourself which model that was, which label it used, and
    whether the deepchecks report next to it belonged to the same run.

    so: PICK THE MODEL, and everything else is found from it. every child task carries
    Args/model_task_id pointing back at its trainer -- that is an exact link, not a name guess.

    and PICK THE PROJECT first, because config.CLEARML_PROJECT is where new runs go, but old runs
    sit in older projects and you still want to look at them.

    read-only. nothing here starts or changes anything.
"""
import pathlib
import sys

import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as C   # noqa: E402

MODEL_LIMIT = 12          # newest N. an overnight run adds 2-3; a week of runs would flood the box.


@st.cache_data(show_spinner="reading projects …", ttl=300)
def list_projects() -> list:
    """our ClearML projects, the configured one first.

    drops ClearML's own demo projects (the server ships ~25 of them) and the machine-made
    sub-projects -- /hpo holds optimiser trials and /.datasets holds dataset tasks, and neither
    contains a trained model you would ever want to open here.
    """
    from clearml import Task
    try:
        names = [getattr(p, "name", str(p)) for p in (Task.get_projects() or [])]
    except Exception:
        names = []
    keep = [n for n in names
            if n and not n.startswith("ClearML")
            and "/hpo" not in n and "/.datasets" not in n and not n.endswith("/hpo")]
    # configured project always at the top, and always present even if it has no tasks yet
    if C.CLEARML_PROJECT in keep:
        keep.remove(C.CLEARML_PROJECT)
    return [C.CLEARML_PROJECT] + sorted(keep)


def project_picker(key: str = "project") -> str:
    """the project dropdown. remembers the choice for the session, so switching pages keeps it."""
    try:
        projects = list_projects()
    except Exception as exc:
        st.warning(f"cannot reach ClearML: {exc}")
        return C.CLEARML_PROJECT
    remembered = st.session_state.get("_project", C.CLEARML_PROJECT)
    idx = projects.index(remembered) if remembered in projects else 0
    chosen = st.selectbox("Project", projects, index=idx, key=key,
                          help=f"new runs go to '{C.CLEARML_PROJECT}' (config.CLEARML_PROJECT). "
                               f"older runs live in older projects -- pick one to look back.")
    st.session_state["_project"] = chosen
    return chosen


SCAN_DEPTH = 150          # how many recent train_ tasks to look through to find `limit` models


def _val(x):
    """query_tasks returns hyperparameters as ParamsItem objects, not plain strings."""
    return getattr(x, "value", x)


@st.cache_data(show_spinner="looking for trained models …", ttl=180)
def list_models(project: str, limit: int = MODEL_LIMIT) -> list:
    """the trained models in a ClearML project, newest first.

    returns [{label, id, name, tags, when}] -- a model only counts if it actually SAVED a model
    artifact. a task that failed halfway is 'completed' in ClearML but has nothing to look at, and
    offering it would just produce an empty page further down.

    QUERY_TASKS, NOT GET_TASKS. get_tasks() builds a full Task object per row -- on 'Nifty
    Production' (240 tasks) that took 184 SECONDS, which is what made the dropdown look hung.
    query_tasks asks the server for the five fields we display and returns plain dicts: 1.5s.
    the artifact list it returns was checked against the full object, task by task, and matches.
    """
    from clearml import Task
    rows = Task.query_tasks(
        project_name=project, task_name="^train_",
        task_filter={"order_by": ["-last_update"], "page_size": SCAN_DEPTH, "page": 0},
        additional_return_fields=["id", "name", "tags", "status", "last_update",
                                  "execution.artifacts"]) or []
    out = []
    for r in rows:
        name = r.get("name") or ""
        # "(base)" catches both the template task AND every HPO trial, which clearml names
        # 'train_xgboost (base): Args/max_depth=10 ...'. those DO save a model artifact, and on a
        # project with a few hundred trials they would bury the real runs in the dropdown.
        if "(base)" in name:
            continue
        if not any(getattr(a, "key", "") == "model" for a in (r.get("execution.artifacts") or [])):
            continue                      # trained but saved nothing -- not worth offering
        tags = [x for x in (r.get("tags") or []) if x]
        when = str(r.get("last_update") or "")[:16].replace("T", " ")
        # the label carries what you would otherwise have to open the task to find out:
        # which model, which dataset version, which LABEL SET, which hyperparameter set.
        extra = " · ".join(x for x in tags if not x.startswith("train"))
        out.append({"label": f"{name}   [{extra}]   {when}" if extra else f"{name}   {when}",
                    "id": r.get("id"), "name": name, "tags": tags, "when": when,
                    "status": r.get("status"), "project": project})
    out.sort(key=lambda m: m["when"], reverse=True)
    # several runs of the same version share a name, so the id keeps the dropdown entries distinct
    seen = {}
    for m in out:
        seen[m["label"]] = seen.get(m["label"], 0) + 1
        if seen[m["label"]] > 1:
            m["label"] = f"{m['label']}   ({m['id'][:8]})"
    return out[:limit]


@st.cache_data(show_spinner="finding this model's outputs …", ttl=180)
def children_of(project: str, model_task_id: str) -> dict:
    """the tasks this model produced -> {"scored_tables": {...}, "shap": {...}, ...}

    matched on Args/model_task_id, NOT on the name. names look matchable (scored_tables_xgboost v7
    next to train_xgboost v7) right up until two runs of the same version exist, and then a name
    match silently pairs a table with the wrong model.

    query_tasks for the same reason as list_models -- and the name regex means only the four kinds
    of child task come back, not the whole project.
    """
    from clearml import Task
    rows = Task.query_tasks(
        project_name=project, task_name="^(scored_|shap_|deepchecks_)",
        task_filter={"order_by": ["-last_update"], "page_size": SCAN_DEPTH * 2, "page": 0},
        additional_return_fields=["id", "name", "status", "last_update",
                                  "hyperparams.Args.model_task_id"]) or []
    found = {}
    for r in rows:
        if _val(r.get("hyperparams.Args.model_task_id")) != model_task_id:
            continue
        n = (r.get("name") or "").lower()
        kind = ("scored_oos" if "scored_oos" in n else
                "scored_tables" if "scored_tables" in n else
                "shap" if "shap" in n else
                "deepchecks" if "deepchecks" in n else None)
        if kind:
            # newest wins -- a step that was re-queued after a fix should not be shadowed by the
            # broken original.
            when = str(r.get("last_update") or "")
            prev = found.get(kind)
            if prev is None or when > prev["when"]:
                found[kind] = {"id": r.get("id"), "name": r.get("name"),
                               "status": r.get("status"), "when": when}
    return found


def model_picker(key: str = "model", limit: int = MODEL_LIMIT, project: str = ""):
    """the dropdown. returns the chosen model dict, or None if there are no models yet."""
    project = project or st.session_state.get("_project") or C.CLEARML_PROJECT
    try:
        models = list_models(project, limit)
    except Exception as exc:
        st.warning(f"cannot reach ClearML: {exc}")
        return None
    if not models:
        st.info(f"no trained models in project '{project}' yet.", icon="⏳")
        return None
    labels = [m["label"] for m in models]
    picked = st.selectbox(f"Model  ({len(models)} most recent)", labels, key=key)
    return next(m for m in models if m["label"] == picked)


def artifact_path(task_id: str, artifact_name: str):
    """download one artifact and return its local path, or None."""
    from clearml import Task
    t = Task.get_task(task_id=task_id)
    art = (t.artifacts or {}).get(artifact_name)
    return art.get_local_copy() if art else None


@st.cache_data(show_spinner="fetching this run's files …", ttl=300)
def artifacts_named(task_id: str, prefix: str, report: bool = False):
    """{short name -> local path} for every artifact on a task starting with `prefix`.

    CACHED, because every widget click reruns the whole page. live files are cheap to re-fetch
    (clearml keeps them under ~/.clearml/cache) but a DEAD one costs a full 404 round trip each
    time, so flipping between V1/V2/V3 on a half-deleted run would stall for seconds a click.

    DEAD LINKS ARE DROPPED. clearml keeps the artifact entry in its own database, so a task can
    still advertise five files long after the bytes were removed from the bucket. get_local_copy()
    then returns None rather than raising, and a None quietly becomes an empty chart three
    functions later. so: only paths that actually downloaded come back.

    report=True -> (live, dead_names) so the page can say "deleted" instead of "none".
    """
    from clearml import Task
    t = Task.get_task(task_id=task_id)
    out, dead = {}, []
    for name, art in (t.artifacts or {}).items():
        if not name.startswith(prefix):
            continue
        short = name[len(prefix):].lstrip("_")
        try:
            p = art.get_local_copy()
        except Exception:
            p = None
        if p:
            out[short] = p
        else:
            dead.append(name)
    return (out, dead) if report else out
