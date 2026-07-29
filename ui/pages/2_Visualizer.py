"""ui/pages/2_Visualizer.py -- the target/predicted/features visualizer, as a page.

it is a THIN WRAPPER. all the code lives in ui/visualizer/ (app.py + src/), untouched apart from
one line: its trailing `main()` is now guarded, so importing it here does not draw the page once
and then go blank. we import it once and call main() on every rerun, which is what streamlit needs.
"""
import pathlib
import sys

import pandas as pd
import streamlit as st

# our own page config, set BEFORE importing the app. app.py also calls set_page_config at import,
# but python caches the module so that only ever fires once -- this line is what runs every rerun.
st.set_page_config(page_title="Visualizer", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

VIZ = pathlib.Path(__file__).resolve().parent.parent / "visualizer"
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


@st.cache_data(show_spinner="looking for scored tables …", ttl=300)
def scored_runs():
    """training runs that produced a scored table (the file the visualizer wants)."""
    from clearml import Task
    out = []
    for t in Task.get_tasks(project_name=C.CLEARML_PROJECT) or []:
        name = t.name or ""
        if not (name.startswith("scored_tables") or name.startswith("scored_oos")):
            continue
        for art in (t.artifacts or {}):
            if art.startswith("scored_"):
                out.append((f"{name} · {art}", t.id, art))
    return out


@st.cache_data(show_spinner="downloading …", ttl=300)
def fetch_table(task_id: str, artifact: str) -> str:
    from clearml import Task
    return Task.get_task(task_id=task_id).artifacts[artifact].get_local_copy()


# ---- our addition: pull a scored table straight in, so nobody downloads parquets by hand.
# it feeds THEIR session_state["labels"] through THEIR prepare_labels, so the app below renders
# it exactly as if the file had been uploaded on the Price & Labels tab.
with st.expander("📥  Load predictions from the pipeline (ClearML)", expanded=False):
    st.caption("a scored table has timestamp / true_label / predicted_label — which this app "
               "already recognises as timestamp / target / predicted. no mapping needed.")
    col_a, col_b = st.columns([3, 1])
    with col_a:
        try:
            runs = scored_runs()
        except Exception as exc:
            runs = []
            st.warning(f"cannot reach ClearML: {exc}")
        pick = st.selectbox("Scored table", [r[0] for r in runs]) if runs else None
        if not runs:
            st.info("no scored tables in ClearML yet.", icon="⏳")
    with col_b:
        st.write("")
        st.write("")
        if pick and st.button("Load", width="stretch", type="primary"):
            _, tid, art = next(r for r in runs if r[0] == pick)
            try:
                path = fetch_table(tid, art)
                raw = (pd.read_parquet(path) if str(path).endswith(".parquet")
                       else pd.read_csv(path))
                mapping = schema.detect_labels(raw)          # finds them automatically
                st.session_state["labels"] = data_loader.prepare_labels(raw, mapping)
                st.session_state["labels_source"] = pick
                st.success(f"loaded {len(raw):,} rows — the markers are on the chart below.")
            except Exception as exc:
                st.error(f"could not load it: {exc}")

visualizer_app.main()
