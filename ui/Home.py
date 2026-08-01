"""ui/Home.py -- the entry point of the pipeline UI. one address, several pages.

RUN IT (from the repo root, so relative paths resolve):
    cd /home/megaserve/Desktop/Gourav/final_pipeline
    final_venv/bin/streamlit run ui/Home.py                          # this machine only
    final_venv/bin/streamlit run ui/Home.py --server.address 0.0.0.0 # anyone on the network

pages live in ui/pages/ -- the number sets the sidebar order, underscores become spaces.
"""
import pathlib
import sys

import streamlit as st

st.set_page_config(page_title="Nifty pipeline", page_icon="🧭", layout="wide")

# the password gate. FIRST thing after set_page_config -- streamlit serves this file at its
# own URL, so a gate on Home.py alone would not protect it.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _auth import require_auth   # noqa: E402
require_auth()


ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as C   # noqa: E402

st.markdown("""
<style>
  /* NOTE: no .block-container width override here. that selector is GLOBAL, and an injected
     <style> can outlive a page switch -- constraining it squeezed the Visualizer's full-width
     charts. keep page-specific styling to our OWN classes (.h-card etc). */
  .h-card{ border:1px solid color-mix(in srgb, currentColor 15%, transparent);
           border-radius:14px; padding:1rem 1.2rem; margin-bottom:.8rem;
           background:color-mix(in srgb, currentColor 4%, transparent); }
  .h-card h3{ margin:0 0 .3rem; font-size:1.05rem; }
  .h-card p{ margin:0; font-size:.9rem;
             color:color-mix(in srgb, currentColor 60%, transparent); }
  .h-tag{ display:inline-block; font-size:.7rem; font-weight:700; padding:.15rem .5rem;
          border-radius:999px; margin-left:.4rem; vertical-align:middle; }
  .writes{ background:#b4231822; color:#e06a5c; }
  .reads { background:#06764722; color:#4ec38a; }
</style>""", unsafe_allow_html=True)

st.title("Nifty pipeline")
st.caption("pick features, build a dataset version, train — and look at what came out. "
           "runs locally; GCP is only ever touched by the pipeline scripts.")

st.markdown("""
<div class="h-card"><h3>Build Dataset <span class="h-tag writes">starts runs</span></h3>
<p>tick features, pick a label set, build a version — and optionally publish it and start training.</p></div>

<div class="h-card"><h3>Visualizer <span class="h-tag reads">read only</span></h3>
<p>target and predicted labels on the candles, with feature panels underneath on one shared
time axis. load a model and predict straight onto the chart.</p></div>
""", unsafe_allow_html=True)

st.info("Use the sidebar to switch pages.", icon="👈")

with st.expander("what this is connected to"):
    st.write(f"**ClearML project:** `{C.CLEARML_PROJECT}`")
    st.write(f"**Models:** `{', '.join(C.MODEL_TYPES)}`")
    st.write(f"**Repo:** `{ROOT}`")
    versions = sorted(p.stem.replace("dataset_", "") for p in C.VERSIONS_DIR.glob("dataset_v*.yaml")
                      if not p.name.endswith(".lock.yaml"))
    st.write(f"**Dataset versions:** {', '.join(versions) if versions else 'none yet'}")
