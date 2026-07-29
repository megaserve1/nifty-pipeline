"""ui/select_and_run.py -- a local Streamlit page: pick features -> build a dataset version -> run.

WHAT IT IS
    a front-end over your EXISTING pipeline. it reads registry.yaml (the feature menu) and
    data/labels/registry.yaml (the label menu), and on a button click it writes a recipe and calls
    your own scripts:
        write recipe -> bridge/build_dataset.py            (Build)
                     -> core/publish_version.py            (Build + Publish + Train)
    it reimplements NOTHING. delete this one file and the pipeline is unchanged.

SECURITY (the "no exposing" you asked for)
    runs LOCAL (streamlit on localhost). it NEVER touches GCP or any credential itself. GCP is
    reached only when publish_version.py runs as a subprocess, using your existing keyless auth.
    no key lives in, or passes through, this page.

RUN IT  (run from INSIDE final_pipeline -- the #1 reason it 'failed' before was a wrong-folder path)
    cd /home/megaserve/Desktop/Gourav/final_pipeline
    final_venv/bin/pip install streamlit          # once. resolver WARNINGS are fine, not errors.
    final_venv/bin/streamlit run ui/select_and_run.py
"""
import datetime
import json
import os
import pathlib
import subprocess
import sys

import streamlit as st
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent   # ui/pages/x.py -> final_pipeline/
sys.path.insert(0, str(ROOT))
import config as C                       # noqa: E402
from core import make_version as mkv     # next_major / next_minor / load_version  # noqa: E402

# the PIPELINE interpreter -- NOT sys.executable. this way the build/publish subprocess always uses
# the venv that has pandas/clearml/xgboost, even if this page is ever launched from another python.
PYBIN = str(ROOT / "final_venv" / "bin" / "python")
if not pathlib.Path(PYBIN).exists():
    PYBIN = sys.executable


# ---------------------------------------------------------------- data the page reads
@st.cache_data(show_spinner=False)
def registry() -> dict:
    return yaml.safe_load(C.REGISTRY.read_text()) or {}


@st.cache_data(show_spinner=False)
def label_menu() -> dict:
    """L1/L2/L3 -> plain-english note, from data/labels/registry.yaml."""
    p = C.LABELS_DIR / "registry.yaml"
    if not p.exists():
        return {}
    return {k: (v or {}).get("note", "") for k, v in (yaml.safe_load(p.read_text()) or {}).items()}


@st.cache_data(show_spinner=False)
def unique_features():
    """ONE deduped feature list across all registered files, plus which file each comes from.
    returns (sorted names, {feature: source_file}). first file that has a name wins -- this is the
    placeholder mapping until the real bucket/bucket_raw/raw groups are registered."""
    feat_src = {}
    for src, meta in (registry() or {}).items():
        for c in (meta or {}).get("columns") or []:
            feat_src.setdefault(c, src)
    return sorted(feat_src), feat_src


def versions() -> list:                  # not cached -- new versions appear as you build
    return sorted(p.stem.replace("dataset_", "")
                  for p in C.VERSIONS_DIR.glob("dataset_v*.yaml")
                  if not p.name.endswith(".lock.yaml"))


def version_info(v: str):
    """for clone mode: (source, raw column names it used, its label, manifest summary or None)."""
    doc = mkv.load_version(v)
    source = (doc.get("features") or [None])[0]
    full = doc.get("columns")
    man = C.DATASETS_DIR / v / "manifest.json"
    summary = None
    if not full and man.exists():                    # no explicit pick -> read what was built
        m = json.loads(man.read_text())
        full = m.get("feature_columns", [])
        summary = {"columns": len(full), "rows": m.get("rows"), "labels": m.get("labels_name")}
    prefix = f"{source}__"
    raw = [c[len(prefix):] for c in (full or []) if c.startswith(prefix)]
    return source, raw, (doc.get("labels_name") or ""), summary


def recipe_doc(features_list, columns_full, labels, parent, omit_columns=False):
    """build (but do not write) the recipe from the exact files + full 'source__col' names. when
    omit_columns is True (a clean whole-source clone) we drop 'columns' so build takes it all."""
    base_parent = parent.split("_", 1)[0] if parent else None   # 'v4.2_test' -> 'v4.2' for numbering
    v = mkv.next_minor(base_parent) if base_parent else mkv.next_major()
    doc = {
        "name": f"dataset_{v}", "kind": "variation" if parent else "selection", "parent": parent,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "author": "ui", "selected_by": "ui",
        "labels_name": labels or C.labels_name(), "date_range": "full",
        "features": features_list, "feature_clocks": {},
    }
    if not omit_columns:
        doc["columns"] = columns_full
    return v, doc


def write_recipe(v, doc):
    path = C.VERSIONS_DIR / f"dataset_{v}.yaml"
    path.write_text("# recipe written by the selection UI. immutable -- make a new version instead.\n"
                    + yaml.safe_dump(doc, sort_keys=False))
    return path


def _sig(features, columns, labels):
    """the CONTENT identity of a recipe: same features + same column pick + same label = same version.
    whole-source (no 'columns') collapses to 'ALL' so it matches an older whole-source version too."""
    return (frozenset(features or []),
            "ALL" if not columns else frozenset(columns),
            labels or C.labels_name())


def find_identical(doc):
    """an existing version whose recipe is content-identical to doc, else None. THIS is what stops a
    double-click (or re-picking the same thing) from minting a duplicate version like the v7/v8 pair."""
    want = _sig(doc.get("features"), doc.get("columns"), doc.get("labels_name"))
    for v in versions():
        try:
            d = mkv.load_version(v)
        except Exception:
            continue
        if _sig(d.get("features"), d.get("columns"), d.get("labels_name")) == want:
            return v
    return None


def stream(cmd, box) -> int:
    """run a pipeline command, streaming its output into a code box (throttled). returns exit code."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen([cmd[0], "-u", *cmd[1:]], cwd=str(ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    lines = []
    for i, line in enumerate(proc.stdout):
        lines.append(line.rstrip("\n"))
        if i % 8 == 0:                               # redraw every 8 lines, not every line
            box.code("\n".join(lines[-400:]))
    box.code("\n".join(lines[-400:]))
    proc.wait()
    return proc.returncode


CSS = """
<style>
:root{
  --accent:#4f46e5; --danger:#b42318; --danger-hover:#912018; --ok:#067647; --ok-hover:#055c38;
  --radius:14px;
  --card-bg:color-mix(in srgb, currentColor 4%, transparent);
  --card-border:color-mix(in srgb, currentColor 14%, transparent);
  --muted:color-mix(in srgb, currentColor 60%, transparent);
}
@media (prefers-color-scheme: dark){
  :root{ --card-bg:color-mix(in srgb, currentColor 8%, transparent);
         --card-border:color-mix(in srgb, currentColor 22%, transparent); --accent:#818cf8; }
}
/* NO .block-container width override: that selector is GLOBAL and an injected <style> can survive
   a page switch, which squeezed the Visualizer's full-width charts. own classes only. */
.fs-hero{ border-radius:var(--radius); padding:1.15rem 1.35rem; margin-bottom:1.2rem;
  background:linear-gradient(135deg,color-mix(in srgb,var(--accent) 22%,transparent),
  color-mix(in srgb,var(--accent) 6%,transparent)); border:1px solid var(--card-border); }
.fs-hero h1{ font-size:1.45rem; font-weight:700; margin:0; line-height:1.2; letter-spacing:-0.01em; }
.fs-hero p{ margin:.3rem 0 0; font-size:.92rem; color:var(--muted); }
.fs-step{ display:flex; align-items:center; gap:.55rem; font-size:.78rem; font-weight:700;
  text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin:.4rem 0 .5rem; }
.fs-step span.num{ display:grid; place-items:center; width:1.4rem; height:1.4rem; border-radius:50%;
  background:var(--accent); color:#fff; font-size:.72rem; font-weight:700; }
[data-testid="stVerticalBlockBorderWrapper"]{ background:var(--card-bg);
  border:1px solid var(--card-border) !important; border-radius:var(--radius) !important;
  padding:.6rem .8rem; }
[data-testid="stMultiSelect"] div[data-baseweb="select"]>div,
[data-testid="stSelectbox"] div[data-baseweb="select"]>div, .stTextInput input{
  border-radius:10px !important; border-color:var(--card-border) !important; }
[data-testid="stMultiSelect"] span[data-baseweb="tag"]{
  background:color-mix(in srgb,var(--accent) 20%,transparent) !important; border-radius:8px !important; }
.fs-badge{ display:inline-flex; align-items:center; gap:.4rem; padding:.28rem .7rem; border-radius:999px;
  font-size:.82rem; font-weight:700; background:color-mix(in srgb,var(--accent) 16%,transparent);
  color:var(--accent); border:1px solid color-mix(in srgb,var(--accent) 35%,transparent); }
.fs-badge.zero{ background:color-mix(in srgb,var(--danger) 12%,transparent); color:var(--danger);
  border-color:color-mix(in srgb,var(--danger) 30%,transparent); }
.stButton>button{ border-radius:10px !important; font-weight:600 !important; width:100%;
  padding:.55rem 1rem !important; border:1px solid transparent !important; transition:filter .15s ease; }
.stButton>button[kind="secondary"]{ background:color-mix(in srgb,var(--ok) 12%,transparent) !important;
  color:var(--ok) !important; border-color:color-mix(in srgb,var(--ok) 45%,transparent) !important; }
.stButton>button[kind="secondary"]:hover{ background:color-mix(in srgb,var(--ok) 20%,transparent) !important;
  border-color:var(--ok) !important; }
.stButton>button[kind="primary"]{ background:var(--accent) !important; color:#fff !important;
  box-shadow:0 2px 10px color-mix(in srgb,var(--accent) 35%,transparent); }
.stButton>button[kind="primary"]:hover{ filter:brightness(1.08) !important; }
.stButton>button[kind="primary"]:disabled{
  background:color-mix(in srgb,currentColor 15%,transparent) !important; color:var(--muted) !important;
  box-shadow:none !important; }
[data-testid="stExpander"] summary{ font-weight:600; }
/* (no #MainMenu/footer hiding either -- same bleed risk onto the other pages) */
</style>
"""


# ================================================================ the page
st.set_page_config(page_title="Nifty · feature selection", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="fs-hero"><h1>Feature selection → build → run</h1>'
            '<p>pick features, freeze a dataset version, then build the dataset — or '
            'build, publish and train. runs local; GCP only via publish_version.</p>'
            '</div>', unsafe_allow_html=True)

reg = registry()
labels = label_menu()


def step(n, title):
    st.markdown(f'<div class="fs-step"><span class="num">{n}</span>{title}</div>', unsafe_allow_html=True)


# ---- STEP 1 · mode ----------------------------------------------------------
step(1, "Mode")
mode = st.radio("mode", ["Fresh selection", "Clone a built version"],
                horizontal=True, label_visibility="collapsed")

# ---- STEP 2 · version (clone only) + label ---------------------------------
step(2, "Version & label")
parent = None
default_label = ""
clone_source = None
clone_prefilled = []
with st.container(border=True):
    left, right = st.columns([1, 1])
    with left:
        if mode == "Fresh selection":
            st.caption("a new version — pick features by group below.")
        else:
            v_clone = st.selectbox("Clone which version", versions() or ["(none built yet)"])
            clone_source, clone_prefilled, default_label, summary = version_info(v_clone)
            src_cols = list((reg.get(clone_source) or {}).get("columns") or []) or clone_prefilled
            clone_prefilled = [c for c in clone_prefilled if c in src_cols] or src_cols
            parent = v_clone
            note = f"source: {clone_source} · {len(clone_prefilled)} features"
            if summary:
                note += f" · built {summary.get('rows')} rows · label {summary.get('labels')}"
            st.caption(note)
    with right:
        lab_opts = ["(default)"] + list(labels.keys())
        idx = lab_opts.index(default_label) if default_label in lab_opts else 0
        lab_choice = st.selectbox("Label set", lab_opts, index=idx,
                                  format_func=lambda k: k if k == "(default)"
                                  else f"{k} — {labels.get(k, '')}")
        chosen_labels = "" if lab_choice == "(default)" else lab_choice

# ---- STEP 3 · features ------------------------------------------------------
step(3, "Features")
if mode == "Fresh selection":
    # ONE deduped list, shown through three GROUP dropdowns. groups aren't registered yet, so each
    # dropdown shows every feature for now; once bucket/bucket_raw/raw are registered each shows only
    # its own. the file a feature comes from is resolved behind the scenes (feat_src).
    all_feats, feat_src = unique_features()
    n_all = len(all_feats)
    st.caption(f"{n_all} unique features across {len(reg)} file(s). groups (bucket / bucket_raw / "
               "raw) aren't registered yet, so each dropdown shows all of them for now.")
    picked = set()
    with st.container(border=True):
        for g, col in zip(("bucket", "bucket_raw", "raw"), st.columns(3)):
            with col:
                st.markdown(f"**{g}**")
                if st.checkbox(f"select all — {g}", key=f"all_{g}"):
                    picked |= set(all_feats)
                    st.caption(f"all {n_all} selected")
                else:
                    sel = st.multiselect(g, options=all_feats, default=[], key=f"ms_{g}",
                                         label_visibility="collapsed", placeholder="type to search…")
                    picked |= set(sel)
    ticked = sorted(picked)
    # group the ticked features by the file they come from; build aligns those files then filters
    by_source = {}
    for f in ticked:
        by_source.setdefault(feat_src[f], []).append(f)
    features_list = sorted(by_source)
    columns_full = [f"{s}__{f}" for s in features_list for f in by_source[s]]
    omit_columns = False
else:
    # clone one built version: single source, change it by scope (whole / minus some / only some)
    source = clone_source
    all_raw = list((reg.get(source) or {}).get("columns") or []) or clone_prefilled
    n_all = len(all_raw)
    kept = set(clone_prefilled)
    if kept and kept != set(all_raw):
        default_scope = "Everything except…" if len(kept) >= n_all / 2 else "Only these…"
    else:
        default_scope = "Every feature"
    scopes = ["Every feature", "Everything except…", "Only these…"]
    with st.container(border=True):
        scope = st.radio("which features", scopes, index=scopes.index(default_scope),
                         horizontal=True, key=f"scope::{source}::{parent}")
        if scope == "Every feature":
            ticked = all_raw
            st.caption(f"the whole source — all {n_all} features.")
        elif scope == "Everything except…":
            excl_default = [c for c in all_raw if c not in kept] if kept else []
            drop = st.multiselect("features to leave OUT (type to search)", options=all_raw,
                                  default=[c for c in excl_default if c in all_raw],
                                  key=f"drop::{source}::{parent}")
            ticked = [c for c in all_raw if c not in set(drop)]
        else:
            keep_default = [c for c in clone_prefilled if c in all_raw] if kept != set(all_raw) else []
            keep = st.multiselect("features to INCLUDE (type to search)", options=all_raw,
                                  default=keep_default, key=f"keep::{source}::{parent}")
            ticked = keep
    features_list = [source]
    columns_full = [f"{source}__{c}" for c in ticked]
    omit_columns = (set(ticked) == set(all_raw))

# ---- selection summary ------------------------------------------------------
badge_cls = "fs-badge zero" if not ticked else "fs-badge"
scope_note = "whole source" if omit_columns else f"{len(ticked)} of {n_all}"
files_note = f" · {len(features_list)} files" if len(features_list) > 1 else ""
st.markdown(f'<span class="{badge_cls}">{len(ticked)} features</span>&nbsp;&nbsp;'
            f'<span style="color:var(--muted)">{scope_note}{files_note} · label {lab_choice} · '
            f'{"sub-version of " + parent if parent else "new version"}</span>',
            unsafe_allow_html=True)

if ticked:
    prev_v, prev_doc = recipe_doc(features_list, columns_full, chosen_labels, parent, omit_columns)
    with st.expander(f"preview the recipe that will be written  →  dataset_{prev_v}.yaml"):
        st.code(yaml.safe_dump(prev_doc, sort_keys=False), language="yaml")

# ---- STEP 4 · actions -------------------------------------------------------
step(4, "Build / run")
with st.container(border=True):
    c1, c2 = st.columns([1, 1])
    with c1:
        build = st.button("Build dataset", type="secondary")
        st.caption("builds the dataset locally under datasets/vN/.")
    with c2:
        ok = st.checkbox("Yes — publish, register the version, and start training")
        run = st.button("Build + Publish + Train", type="primary", disabled=not ok)
        st.caption("publishes the dataset, registers the version, and starts a training run "
                   "(an agent on the training queue picks it up).")

# ---- do it ------------------------------------------------------------------
if build or run:
    if not ticked:
        st.error("pick at least one feature.")
    else:
        v, doc = recipe_doc(features_list, columns_full, chosen_labels, parent, omit_columns)
        twin = find_identical(doc)                     # already exists? do NOT mint a duplicate
        built = bool(twin) and (C.DATASETS_DIR / twin / "manifest.json").exists()
        if twin:
            st.warning(f"this exact selection already exists as **{twin}** — reusing it, not making a copy.")
            v = twin
            if built and not run:
                st.info(f"{twin} is already built (datasets/{twin}/). nothing to do — "
                        f"change a feature or the label set to make a different version.")
                st.stop()
        else:
            write_recipe(v, doc)
            st.toast(f"wrote recipe {v}", icon="📝")
        with st.status(f"{'publishing' if run else 'building'} {v} …  (the page is locked until this finishes)",
                       expanded=True) as status:
            box = st.empty()
            rc = stream([PYBIN, "bridge/build_dataset.py", "--version", v], box)
            if rc != 0:
                status.update(label=f"build failed (exit {rc}) — see log", state="error")
            elif run:
                status.update(label=f"built {v} — publishing + enqueuing training …")
                rc2 = stream([PYBIN, "core/publish_version.py", "--version", v], box)
                status.update(label=f"published + enqueued {v}" if rc2 == 0
                              else f"publish failed (exit {rc2}) — see log",
                              state="complete" if rc2 == 0 else "error")
            else:
                status.update(label=f"built {v} (not published)", state="complete")
        if rc == 0:
            st.toast(f"done: {v}", icon="✅")
