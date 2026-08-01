"""ui/pages/2_Visualizer.py -- the target/predicted/features visualizer, as a page.

PAUSED while the visualizer team rewrites it to read from GCP (see ui/visualizer/DATA_SOURCES.md).
TO PUT IT BACK: set UNDER_DEVELOPMENT = False. the wrapper below is untouched.

it is a THIN WRAPPER. all the code lives in ui/visualizer/ (app.py + src/), untouched apart from
one line: its trailing `main()` is now guarded, so importing it here does not draw the page once
and then go blank. we import it once and call main() on every rerun, which is what streamlit needs.
"""
import pathlib
import sys

import pandas as pd
import streamlit as st

UNDER_DEVELOPMENT = True          # <- the one line to flip when the rewrite is merged

# our own page config, set BEFORE importing the app. app.py also calls set_page_config at import,
# but python caches the module so that only ever fires once -- this line is what runs every rerun.
st.set_page_config(page_title="Visualizer", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")


# the password gate. FIRST thing after set_page_config -- streamlit serves this file at its
# own URL, so a gate on Home.py alone would not protect it.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _auth import require_auth   # noqa: E402
require_auth()

VIZ = pathlib.Path(__file__).resolve().parent.parent / "visualizer"

if UNDER_DEVELOPMENT:
    st.title("📈  Visualizer")
    st.warning("**Under development.**", icon="🚧")
    st.stop()

# ------------------------------------------------------------------ the real page, when re-enabled
if not VIZ.exists():
    st.error(f"visualizer not found at {VIZ}")
    st.stop()

# the app does `from src import charts, ...`, so its OWN folder has to be importable.
if str(VIZ) not in sys.path:
    sys.path.insert(0, str(VIZ))

import app as visualizer_app          # noqa: E402  (must come after the sys.path insert)
from src import data_loader, schema   # noqa: E402  THEIR loaders -- we do not reimplement them

sys.path.insert(0, str(VIZ.parent.parent))
import config as C                    # noqa: E402

from _shared import model_picker, children_of, project_picker   # noqa: E402


@st.cache_data(show_spinner="downloading …", ttl=300)
def fetch_table(task_id: str, artifact: str) -> str:
    from clearml import Task
    return Task.get_task(task_id=task_id).artifacts[artifact].get_local_copy()


def scored_artifacts(task_id: str) -> list:
    from clearml import Task
    return [a for a in (Task.get_task(task_id=task_id).artifacts or {}) if a.startswith("scored_")]


# ---- our addition: PICK A MODEL, get its predictions on the chart.
# it feeds THEIR session_state["labels"] through THEIR prepare_labels, so the app below renders
# it exactly as if the file had been uploaded on the Price & Labels tab. their code is untouched.
with st.expander("📥  Load a model's predictions from the pipeline (ClearML)", expanded=False):
    st.caption("pick the MODEL, not the file — after an overnight run there are two or three, and "
               "the table's name alone does not say which label set or hyperparameters it used. "
               "its scored table has timestamp / true_label / predicted_label, which this app "
               "already reads as timestamp / target / predicted. no mapping needed.")
    col_p, col_m = st.columns([1, 2])
    with col_p:
        project = project_picker(key="viz_project")
    with col_m:
        model = model_picker(key="viz_model", project=project)
    if model is not None:
        kids = children_of(project, model["id"])
        task = kids.get("scored_tables") or kids.get("scored_oos")
        if task is None:
            st.info("this model has no scored table yet — queued, running, or the step failed.",
                    icon="⏳")
        else:
            arts = scored_artifacts(task["id"])
            col_a, col_b = st.columns([3, 1])
            with col_a:
                # a run writes scored_train AND scored_test; test is the honest one to look at
                default = next((i for i, a in enumerate(arts) if "test" in a or "oos" in a), 0)
                art = st.selectbox(f"Table from {task['name']}", arts, index=default) if arts else None
                if not arts:
                    st.warning("that task uploaded no scored table.")
            with col_b:
                st.write("")
                st.write("")
                if art and st.button("Load", width="stretch", type="primary"):
                    try:
                        path = fetch_table(task["id"], art)
                        raw = (pd.read_parquet(path) if str(path).endswith(".parquet")
                               else pd.read_csv(path))
                        mapping = schema.detect_labels(raw)      # finds them automatically
                        st.session_state["labels"] = data_loader.prepare_labels(raw, mapping)
                        st.session_state["labels_source"] = f"{model['name']} · {art}"
                        st.success(f"loaded {len(raw):,} rows from {model['name']} — "
                                   f"the markers are on the chart below.")
                    except Exception as exc:
                        st.error(f"could not load it: {exc}")


visualizer_app.main()
