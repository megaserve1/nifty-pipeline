"""Target / Predicted label + feature visualizer for NIFTY futures data.

Run with:  streamlit run app.py

Tab 1 puts target and predicted labels on the candles. Tab 2 stacks feature
panels underneath the same candles on a shared x-axis, so you can read
"target here, model said that, features were doing this" in a single hover.
"""

from __future__ import annotations

import logging
import re
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import streamlit as st

from src import charts, data_loader, exporter, metrics, predictor, schema, transforms


def _configure_logging() -> logging.Logger:
    """One-shot logger setup for the `visualizer.*` tree, streaming to stderr.

    Streamlit's own log formatter fires per record, and its default format hides
    the module and timestamp. We attach our own handler to the visualizer root so
    every load/predict step shows up in the terminal with a wall-clock stamp,
    without duplicating Streamlit's lines.
    """
    root = logging.getLogger("visualizer")
    if not any(getattr(h, "_visualizer", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S"
        ))
        handler._visualizer = True                       # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    root.propagate = False
    return root


log = _configure_logging().getChild("app")

APP_ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = APP_ROOT / "sample_data"
MAX_CHECKBOXES = 150
PANEL_LIMIT = 8
CHECKBOX_COLUMNS = 4

st.set_page_config(
    page_title="Target · Predicted · Features Visualizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 2rem;}
      div[data-testid="stMetricValue"] {font-size: 1.4rem;}
      div[data-testid="stCheckbox"] label p {font-size: 0.82rem;}
      section[data-testid="stSidebar"] {width: 345px !important;}

      /* Status pills, tinted for the royal-purple surface. A coloured dot
         carries the state so it never rests on the fill colour alone. */
      .pill {display:inline-block; padding:3px 11px; border-radius:999px;
             font-size:0.75rem; font-weight:600; margin:2px 6px 2px 0;
             border:1px solid transparent;}
      .pill::before {content:"●"; margin-right:6px; font-size:0.7rem;}
      .pill-ok   {background:#123524; color:#7BEFA8; border-color:#1E5C3C;}
      .pill-warn {background:#3A2A10; color:#FBBF24; border-color:#6B4E12;}

      /* Tabs read as chips against the deep purple plane. */
      button[data-baseweb="tab"] {font-weight:600;}
      div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {background:#A855F7;}
    </style>
    """,
    unsafe_allow_html=True,
)

STATE_DEFAULTS: dict = {
    "price": None,           # canonical price frame
    "labels": None,          # canonical label frame
    "features": None,        # canonical feature frame
    "price_source": "",
    "labels_source": "",
    "features_source": "",
    "panel_pending": {},     # panel key -> set[str]   live checkbox state
    "panel_applied": {},     # panel key -> list[str]  what the chart draws
    "panel_config": {},      # panel key -> per-panel display options
    "predictions": None,     # timestamp/target/predicted/confidence + proba_*
    "prediction_info": None, # BundleInfo of the model that produced them
    "prediction_source": "",
    "prediction_settings": None,  # {na_mode, renames, tl_col} at Run time
    "local_bundle_path": "", # set once the user presses "Load this bundle"
    "mapped_source": None,   # which file the OHLC mapping widgets belong to
    "_pending_build": None,  # set of pending build kinds ('excel' / 'csv')
    "loaded_model_name": "", # filename of the currently loaded model bundle
}


def init_state() -> None:
    for key, value in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = dict(value) if isinstance(value, dict) else value


def pending_for(panel_key: str) -> set:
    return st.session_state["panel_pending"].setdefault(panel_key, set())


def applied_for(panel_key: str) -> list:
    return st.session_state["panel_applied"].setdefault(panel_key, [])


def config_for(panel_key: str) -> dict:
    return st.session_state["panel_config"].setdefault(
        panel_key,
        {"normalise": "none", "chart_type": "line", "mark_events": False, "zero_line": False},
    )


# --------------------------------------------------------------- loading ---


@st.cache_data(show_spinner=False)
def read_bytes(payload: bytes, name: str) -> pd.DataFrame:
    """Parse uploaded CSV bytes; ``name`` only widens the cache key."""
    return data_loader.read_csv(payload)


@st.cache_data(show_spinner=False)
def read_path(path: str, mtime: float) -> pd.DataFrame:
    """Parse a CSV from disk; ``mtime`` invalidates the cache when it changes."""
    return data_loader.read_csv(path)


def load_upload(uploaded) -> Optional[pd.DataFrame]:
    if uploaded is None:
        return None
    try:
        return read_bytes(uploaded.getvalue(), uploaded.name)
    except data_loader.DataError as exc:
        st.error(f"**{uploaded.name}** — {exc}")
        return None


def sample_available() -> bool:
    return (SAMPLE_DIR / "nifty_combined.csv").exists()


def column_picker(label, options, default, key, allow_none=False, help=None):
    """Selectbox pre-set to the detected column, with an optional '— none —' entry."""
    choices = (["— none —"] if allow_none else []) + [str(c) for c in options]
    if not choices:
        return None
    default_str = str(default) if default is not None else None
    index = choices.index(default_str) if default_str in choices else 0
    picked = st.selectbox(label, choices, index=index, key=key, help=help)
    return None if picked == "— none —" else picked


def frame_badge(df: Optional[pd.DataFrame], name: str) -> None:
    if df is None or df.empty:
        st.markdown(
            f'<span class="pill pill-warn">{name} — not loaded</span>', unsafe_allow_html=True
        )
        return
    info = data_loader.describe_frame(df)
    span = ""
    if info["start"] is not None:
        span = f" · {info['start']:%d-%b-%y %H:%M} → {info['end']:%d-%b-%y %H:%M}"
    interval = f" · {info['interval']}" if info["interval"] is not None else ""
    st.markdown(
        f'<span class="pill pill-ok">{name} — {info["rows"]:,} rows{span}{interval}</span>',
        unsafe_allow_html=True,
    )


def load_sample(dayfirst: bool) -> None:
    """Populate every frame from the bundled synthetic dataset."""
    path = SAMPLE_DIR / "nifty_combined.csv"
    raw = read_path(str(path), path.stat().st_mtime)
    omap = schema.detect_ohlc(raw)
    lmap = schema.detect_labels(raw, omap.timestamp)
    st.session_state["price"] = data_loader.prepare_price(raw, omap, dayfirst)
    st.session_state["labels"] = data_loader.prepare_labels(raw, lmap, dayfirst)
    st.session_state["price_source"] = "sample · nifty_combined.csv"
    st.session_state["labels_source"] = "sample · nifty_combined.csv"
    reset_window_widgets()

    fpath = SAMPLE_DIR / "nifty_features.csv"
    if fpath.exists():
        fraw = read_path(str(fpath), fpath.stat().st_mtime)
        st.session_state["features"] = data_loader.prepare_features(
            fraw, "timestamp", dayfirst=dayfirst
        )
        st.session_state["features_source"] = "sample · nifty_features.csv"


WINDOW_KEYS = ("start_date", "end_date", "t_from", "t_to")


def reset_window_widgets() -> None:
    """Forget the chosen window so a newly loaded file gets its own default.

    Streamlit keeps a widget's stored value over a fresh default, and a date kept
    from the previous file can fall outside the new file's min/max entirely.
    """
    for key in WINDOW_KEYS:
        st.session_state.pop(key, None)


def trading_dates(price: pd.DataFrame) -> pd.DatetimeIndex:
    """The distinct sessions present in the data, in order."""
    return pd.DatetimeIndex(pd.Series(price["timestamp"]).dt.normalize().unique()).sort_values()


def initial_date_range(price: pd.DataFrame, days: int) -> tuple:
    """The most recent ``days`` *trading* sessions.

    Counted in sessions rather than calendar days so a Monday load does not open
    on an empty weekend window.
    """
    sessions = trading_dates(price)
    if len(sessions) == 0:
        stamp = price["timestamp"].max()
        return stamp.date(), stamp.date()
    first = sessions[max(0, len(sessions) - max(days, 1))]
    return first.date(), sessions[-1].date()


def window_bounds(price: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The start/end dates chosen on tab 1, falling back to the full price span.

    The end date is inclusive: picking 03-May means the whole of 03-May.
    """
    start = st.session_state.get("start_date")
    end = st.session_state.get("end_date")
    if start is None or end is None:
        return price["timestamp"].min(), price["timestamp"].max()
    return (
        pd.Timestamp(start),
        pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
    )


def apply_session_filter(view: pd.DataFrame) -> pd.DataFrame:
    """Re-apply tab 1's intraday session window."""
    t_from, t_to = st.session_state.get("t_from"), st.session_state.get("t_to")
    if t_from is None or t_to is None or t_from > t_to or view.empty:
        return view
    tod = view["timestamp"].dt.time
    return view.loc[(tod >= t_from) & (tod <= t_to)].reset_index(drop=True)


# --------------------------------------------------------------- sidebar ---


def render_sidebar() -> dict:
    with st.sidebar:
        st.title("📈 Visualizer")
        st.caption("Target vs predicted labels and features on one shared time axis.")

        st.subheader("Parsing")
        dayfirst = st.checkbox(
            "Day-first dates (DD-MM-YYYY)", value=True,
            help="Indian exports are usually day-first. Turn off for US-style MM-DD-YYYY.",
        )
        custom_format = st.text_input(
            "Explicit timestamp format", value="",
            placeholder="%Y%m%d %H:%M:%S — blank = auto-detect",
        ).strip() or None

        st.subheader("Chart")
        price_style = st.selectbox("Price style", charts.PRICE_STYLES)
        template = st.selectbox(
            "Chart theme",
            [charts.TEMPLATE_NAME, "plotly_dark", "plotly_white", "simple_white", "seaborn"],
            help="royal_purple is the tuned palette; the rest are Plotly's stock themes.",
        )
        collapse_gaps = st.checkbox(
            "Collapse non-trading gaps", value=True,
            help="Hide nights and weekends so the candles sit shoulder to shoulder.",
        )
        show_volume = st.checkbox("Volume panel", value=False)
        unified_hover = st.checkbox("Unified tooltip (x)", value=True)
        cross_hover = st.checkbox(
            "Tooltip spans all panels", value=True,
            help="One hover shows price, labels and every feature at that timestamp.",
        )
        show_spikes = st.checkbox("Crosshair spike lines", value=True)

        st.subheader("Label markers")
        marker_size = st.slider("Marker size", 3, 24, 7)
        marker_pad = st.slider("Offset from bar (% of range)", 0.0, 6.0, 1.5, 0.25)
        target_mode = st.selectbox("Target marker at", charts.MARKER_MODES, index=0)
        predicted_mode = st.selectbox("Predicted marker at", charts.MARKER_MODES, index=1)

        st.subheader("Performance")
        initial_days = st.number_input(
            "Sessions to open with", min_value=1, max_value=365, value=3, step=1,
            help="A freshly loaded file opens on its last N trading sessions, so a "
                 "multi-month file does not chart everything at once. Widen the "
                 "date range on the first tab to see more.",
        )
        max_bars = st.number_input(
            "Max bars to render", min_value=200, max_value=200_000, value=4_000, step=500,
            help="The newest N bars inside the selected window are drawn.",
        )
        price_height_px = st.slider(
            "Price chart height (px)", 300, 1600, 820, 20, key="price_height_px",
            help="Bigger candles are easier to read. Zoom does not shrink the "
                 "candles; only this slider does.",
        )
        row_height = st.slider("Feature panel height (px)", 120, 400, 220, 10)

        with st.expander("Reset"):
            if st.button("Clear all loaded data", key="clear_all", width="stretch"):
                for key, value in STATE_DEFAULTS.items():
                    st.session_state[key] = dict(value) if isinstance(value, dict) else value
                st.cache_data.clear()
                st.rerun()

    options = charts.ChartOptions(
        marker_size=marker_size,
        marker_pad_pct=marker_pad,
        target_mode=target_mode,
        predicted_mode=predicted_mode,
        show_volume=show_volume,
        collapse_gaps=collapse_gaps,
        template=template,
        unified_hover=unified_hover,
        cross_subplot_hover=cross_hover,
        show_spikes=show_spikes,
        row_height_px=row_height,
        price_style=price_style,
        price_height_px=price_height_px,
    )
    return {
        "dayfirst": dayfirst,
        "custom_format": custom_format,
        "max_bars": int(max_bars),
        "initial_days": int(initial_days),
        "options": options,
    }


# ------------------------------------------------------ tab 1: price/labels ---


def render_upload_section(settings: dict) -> tuple:
    """Pick the candle file. Returns ``(raw_frame, source_name)``.

    Candles only -- any label columns in this file are ignored. Labels come from
    the model on the Predict tab, or from a separate TS/TL/PL upload.
    """
    mode = st.radio(
        "Where are the candles?",
        ["From a file path", "Upload"],
        horizontal=True, key="load_mode",
        help="Only the OHLC columns are read from this file. Labels are joined "
             "separately, on timestamp.",
    )

    up_col, sample_col = st.columns([3, 1])
    price_raw = None
    src_price = ""

    with up_col:
        if mode == "From a file path":
            price_raw, src_price = table_from_path("price_src", "OHLC file")
        else:
            up = st.file_uploader(
                "OHLC file (timestamp + open/high/low/close)",
                type=["csv", "txt", "parquet", "xlsx"], key="up_combined",
            )
            if up is not None:
                try:
                    price_raw = read_table_cached(up.getvalue(), up.name)
                    src_price = up.name
                except data_loader.DataError as exc:
                    st.error(f"**{up.name}** — {exc}")

    with sample_col:
        st.write("")
        st.write("")
        if st.button(
            "📦 Load sample data", key="load_sample",
            width="stretch", disabled=not sample_available(),
            help="Synthetic NIFTY futures 5-min bars with labels and 25 features.",
        ):
            load_sample(settings["dayfirst"])
            st.rerun()

    return price_raw, src_price


OHLC_MAP_KEYS = ("map_ts", "map_o", "map_h", "map_l", "map_c", "map_v")


def render_mapping_section(price_raw, src_price, settings) -> None:
    """OHLC column mapping. Writes the canonical price frame into session state."""
    # Drop the previous file's mapping when the source changes. Streamlit keeps a
    # widget's stored value over the freshly-computed default, so without this a
    # column chosen for file A is silently applied to file B.
    if st.session_state.get("mapped_source") != src_price:
        for key in OHLC_MAP_KEYS:
            st.session_state.pop(key, None)
        st.session_state["mapped_source"] = src_price

    detected = schema.detect_ohlc(price_raw)
    with st.expander("2 · Map the columns", expanded=not detected.is_complete):
        cols = list(price_raw.columns)
        c = st.columns(6)
        with c[0]:
            ts_col = column_picker("Timestamp", cols, detected.timestamp, "map_ts")
        with c[1]:
            o_col = column_picker("Open", cols, detected.open, "map_o")
        with c[2]:
            h_col = column_picker("High", cols, detected.high, "map_h")
        with c[3]:
            l_col = column_picker("Low", cols, detected.low, "map_l")
        with c[4]:
            cl_col = column_picker("Close", cols, detected.close, "map_c")
        with c[5]:
            v_col = column_picker("Volume", cols, detected.volume, "map_v", allow_none=True)

        if st.button("✅ Apply mapping & load", key="apply_mapping", type="primary"):
            try:
                omap = schema.OhlcMapping(ts_col, o_col, h_col, l_col, cl_col, v_col)
                st.session_state["price"] = data_loader.prepare_price(
                    price_raw, omap, settings["dayfirst"], settings["custom_format"]
                )
                st.session_state["price_source"] = src_price
                reset_window_widgets()   # the old window may not exist in this file
                st.success(f"Loaded {len(st.session_state['price']):,} bars.")
            except data_loader.DataError as exc:
                st.error(str(exc))


def render_label_source(settings: dict, *, key_scope: str = "") -> None:
    """Optional TS/TL/PL upload, for anyone who is not producing labels here.

    Kept separate from the price loader on purpose: the price file is a *candle
    source*, and quietly promoting one of its columns to "target" is how a
    feature like `swing_label` ends up charted as ground truth by accident.

    ``key_scope`` prefixes every widget key so this can be rendered on more than
    one tab in the same session (Streamlit runs every tab body on every rerun,
    so unprefixed duplicate keys would crash). Default keeps the existing keys
    for the Price & Labels tab.
    """
    def k(base: str) -> str:
        return f"{key_scope}_{base}" if key_scope else base

    labels = st.session_state["labels"]
    source = st.session_state["labels_source"]

    if labels is not None:
        cols = st.columns([3, 1])
        with cols[0]:
            has_t = int(labels["target"].notna().sum())
            has_p = int(labels["predicted"].notna().sum())
            st.markdown(
                f'<span class="pill pill-ok">Labels · {source or "loaded"} — '
                f"{has_t:,} target · {has_p:,} predicted</span>",
                unsafe_allow_html=True,
            )
        with cols[1]:
            if st.button("Clear labels", key=k("clear_labels"), width="stretch"):
                st.session_state["labels"] = None
                st.session_state["labels_source"] = ""
                st.rerun()

    # The uploader is always rendered, even once labels exist: hiding it would
    # destroy its widget state mid-rerun, and it doubles as "replace them".
    title = (
        "Replace the labels with a TS / TL / PL file" if labels is not None
        else "Already have labels? Load a TS / TL / PL file"
    )
    with st.expander(title, expanded=False):
        st.caption(
            "Only needed if you are **not** generating labels on the "
            "**🔮 Predict from Model** tab. They are joined to the candles on timestamp."
        )
        raw, labels_source_name = file_source(
            k("labels"),
            label="labels file",
            uploader_label="Labels file (timestamp + target and/or predicted)",
            uploader_key=k("up_labels"),
            placeholder=r"C:\Users\Admin\data\labels_ts_tl_pl.csv",
        )
        if raw is None:
            return

        detected = schema.detect_labels(raw)
        cols = list(raw.columns)
        d = st.columns(3)
        with d[0]:
            lts = column_picker("Timestamp", cols, detected.timestamp, k("map_lts"))
        with d[1]:
            tgt = column_picker(
                "Target label", cols, detected.target, k("map_tgt"), allow_none=True
            )
        with d[2]:
            prd = column_picker(
                "Predicted label", cols, detected.predicted, k("map_prd"), allow_none=True
            )
        if st.button("✅ Load labels", key=k("apply_labels"), type="primary"):
            if not (tgt or prd):
                st.error("Pick at least one of the target / predicted columns.")
                return
            try:
                st.session_state["labels"] = data_loader.prepare_labels(
                    raw, schema.LabelMapping(lts, tgt, prd),
                    settings["dayfirst"], settings["custom_format"],
                )
                st.session_state["labels_source"] = labels_source_name or "labels"
                st.rerun()
            except data_loader.DataError as exc:
                st.error(str(exc))


def render_price_tab(settings: dict) -> None:
    st.subheader("1 · Load the candles (OHLC)")
    price_raw, src_price = render_upload_section(settings)

    if price_raw is not None:
        render_mapping_section(price_raw, src_price, settings)

    price = st.session_state["price"]
    if price is None:
        if price_raw is not None:
            # The file is read and mapped, but nothing is loaded until the button
            # is pressed -- say that, rather than asking for a file again.
            st.info(
                f"**{len(price_raw):,} rows read.** Check the mapping above, then press "
                "**✅ Apply mapping & load** to chart it.",
                icon="👆",
            )
        else:
            st.info(
                "Choose an OHLC file above (or press **📦 Load sample data**) to begin. "
                "Only `timestamp` + `open/high/low/close` are read from it — labels come "
                "from the **🔮 Predict from Model** tab, joined on timestamp."
            )
        return

    st.divider()
    st.subheader("2 · Labels")
    render_label_source(settings)
    labels = st.session_state["labels"]
    if labels is None:
        st.caption(
            "No labels yet. Generate them on the **🔮 Predict from Model** tab, or load a "
            "TS/TL/PL file above. The candles below are shown unlabelled."
        )

    st.divider()
    st.subheader("3 · Window & label display")

    tmin, tmax = price["timestamp"].min(), price["timestamp"].max()
    sessions = trading_dates(price)
    opens_on = initial_date_range(price, settings["initial_days"])

    w1, w2, w3, w4 = st.columns(4)
    # `value` is only the *initial* default -- once a widget exists Streamlit keeps
    # whatever the user picked, so a widened window sticks.
    with w1:
        start_date = st.date_input(
            "Start date", value=opens_on[0],
            min_value=tmin.date(), max_value=tmax.date(), key="start_date",
            help=f"Opens on the last {settings['initial_days']} trading session(s). "
                 "Move it back to load more history.",
        )
    with w2:
        end_date = st.date_input(
            "End date", value=opens_on[1],
            min_value=tmin.date(), max_value=tmax.date(), key="end_date",
        )
    with w3:
        st.time_input("Session from", value=tmin.time(), key="t_from", step=300)
    with w4:
        st.time_input("Session to", value=tmax.time(), key="t_to", step=300)

    if start_date > end_date:
        st.error(
            f"Start date ({start_date:%d-%b-%Y}) is after the end date "
            f"({end_date:%d-%b-%Y}). Swap them to see the chart."
        )
        return

    shown_all = start_date <= sessions[0].date() and end_date >= sessions[-1].date()
    if len(sessions) > 1 and not shown_all:
        st.caption(
            f"{len(sessions):,} trading sessions available "
            f"({sessions[0]:%d-%b-%Y} → {sessions[-1]:%d-%b-%Y}). "
            "Move the start date back to load more."
        )

    start, end = window_bounds(price)
    view = apply_session_filter(transforms.filter_time_range(price, start, end))

    if view.empty:
        st.warning("No bars in that window — widen the date range or the session times.")
        return

    trimmed = len(view) > settings["max_bars"]
    if trimmed:
        view = view.iloc[-settings["max_bars"]:].reset_index(drop=True)

    label_view = None
    if labels is not None:
        label_view = transforms.filter_time_range(
            labels, view["timestamp"].min(), view["timestamp"].max()
        )
        # Labels and candles come from different files, so the join is explicit.
        exact_hits = int(charts.attach_labels(view, label_view)["target"].notna().sum()) \
            + int(charts.attach_labels(view, label_view)["predicted"].notna().sum())
        if exact_hits == 0 and not label_view.empty:
            interval = data_loader.infer_bar_interval(price["timestamp"])
            snapped = transforms.align_to_price(
                view, label_view, how="nearest", tolerance=interval
            )
            if snapped[["target", "predicted"]].notna().any().any():
                st.info(
                    "The label timestamps do not match the candle timestamps exactly, "
                    "so they were snapped to the nearest bar at or before each label.",
                    icon="🔗",
                )
                label_view = snapped

    visible = charts.attach_labels(view, label_view)
    classes = metrics.label_classes(visible)

    l1, l2, l3, l4 = st.columns([1, 1, 2, 2])
    with l1:
        show_target = st.checkbox("Show target", value=True)
        show_pred = st.checkbox("Show predicted", value=True)
    with l2:
        show_mismatch = st.checkbox("Mark mismatches", value=True)
    with l3:
        tgt_classes = st.multiselect(
            "Target classes", classes, default=classes, key="tgt_classes",
            help="Hide a dominant class (e.g. HOLD) to declutter the chart.",
        )
    with l4:
        prd_classes = st.multiselect(
            "Predicted classes", classes, default=classes, key="prd_classes"
        )

    summary = metrics.summarise(visible)
    m = st.columns(5)
    m[0].metric("Bars shown", f"{len(view):,}", delta="trimmed" if trimmed else None)
    m[1].metric("Target labels", f"{summary.n_target:,}")
    m[2].metric("Predicted labels", f"{summary.n_predicted:,}")
    m[3].metric(
        "Agreement",
        f"{summary.accuracy:.1%}" if summary.accuracy is not None else "—",
        help="Share of bars where target == predicted, across bars carrying both.",
    )
    m[4].metric("Mismatches", f"{summary.n_mismatch:,}")

    # ---- feature panels (optional, tab-1 scoped) ---------------------------
    # After loading a TS/TL/PL file the user often wants to see indicators
    # underneath the candles too, without hopping to the Feature Panels tab.
    # This mirrors the Predict tab's "Visualize" section: load a feature file,
    # pick columns per panel, and the chart below turns into the combined
    # price + labels + features view.
    st.divider()
    st.subheader("4 · Feature panels (optional)")
    st.caption(
        "Load a feature file to draw indicators underneath the candles. "
        "Features loaded on the **Feature Panels** tab are reused."
    )
    panels_result = _render_price_tab_feature_panels(price, view, settings)

    opts = charts.ChartOptions(
        **{
            **vars(settings["options"]),
            "show_target": show_target,
            "show_predicted": show_pred,
            "show_mismatch": show_mismatch,
            "target_classes": tgt_classes,
            "predicted_classes": prd_classes,
            "price_height": 2.4 if panels_result else 3.0,
        }
    )
    if panels_result:
        panels_pl, aligned_pl, styles_pl = panels_result
        st.plotly_chart(
            charts.build_figure(
                view, label_view, aligned_pl, panels_pl, opts, styles_pl,
            ),
            key="price_chart",
            theme=None,
            config={"scrollZoom": True, "displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        )
        _render_cross_panel_hover_line()
    else:
        st.plotly_chart(
            charts.build_figure(view, label_view, options=opts),
            key="price_chart",
            theme=None,          # keep our own template; Streamlit's would override it
            config={"scrollZoom": True, "displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        )
    st.caption(
        "Solid markers = **target**, hollow markers = **predicted**, amber ✕ = disagreement. "
        "Drag to pan, scroll to zoom, double-click to reset, click a legend entry to hide it."
    )


def _render_price_tab_feature_panels(
    price: pd.DataFrame,
    view: pd.DataFrame,
    settings: dict,
) -> Optional[tuple]:
    """Inline feature loader + panel controls for the Price & Labels tab.

    Returns ``(panels, aligned, style_map)`` when the user has features loaded
    AND at least one panel has selected features to draw; otherwise ``None``
    (chart stays a plain price + labels chart).

    Widget keys use a ``pl_``/``tp`` prefix so they never collide with the
    Feature Panels tab, which uses ``features``/``p``. Session state for the
    features themselves is shared: loading on either tab shows on both.
    """
    fc1, fc2 = st.columns([3, 1.4])
    with fc1:
        pl_raw, pl_name = file_source(
            "pl_features",
            label="feature file",
            uploader_label="Feature file (timestamp + one column per feature)",
            uploader_key="pl_up_features",
        )
    with fc2:
        st.write("")
        st.write("")
        st.selectbox(
            "Align to price bars", ["exact", "nearest"], key="pl_align_mode",
            help="'nearest' back-fills from the last feature row at or before each bar.",
        )

    if pl_raw is not None:
        fcols = list(pl_raw.columns)
        detected_ts = schema.detect_ohlc(pl_raw).timestamp
        fcA, fcB = st.columns([1, 3])
        with fcA:
            pl_fts = column_picker("Timestamp column", fcols, detected_ts, "pl_map_fts")
        # numeric_only=False so text/categorical columns (gap_state, state, ...)
        # are offered too -- matches the Predict tab's panel picker, which draws
        # them as step lines over their categories.
        candidates = schema.feature_columns(
            pl_raw, exclude=[pl_fts], numeric_only=False,
        )
        numeric_count = sum(
            1 for c in candidates if pd.api.types.is_numeric_dtype(pl_raw[c])
        )
        with fcB:
            pl_chosen = st.multiselect(
                f"Feature columns to import ({len(candidates)} columns · "
                f"{numeric_count} numeric, {len(candidates) - numeric_count} categorical)",
                candidates, default=candidates, key="pl_import_feats",
            )
        if st.button("✅ Import features", key="pl_import_features", type="primary"):
            try:
                st.session_state["features"] = data_loader.prepare_features(
                    pl_raw, pl_fts, pl_chosen,
                    settings["dayfirst"], settings["custom_format"],
                )
                st.session_state["features_source"] = pl_name
                st.session_state["panel_applied"] = {}
                st.session_state["panel_pending"] = {}
                for key in [
                    k for k in st.session_state
                    if isinstance(k, str) and k.startswith("cb::")
                ]:
                    del st.session_state[key]
                st.success(f"Imported {len(pl_chosen)} features.")
            except data_loader.DataError as exc:
                st.error(str(exc))

    features = st.session_state["features"]
    if features is None:
        return None

    frame_badge(features, f"Features · {st.session_state['features_source'] or 'uploaded'}")

    align_mode = st.session_state.get("pl_align_mode", "exact")
    interval = data_loader.infer_bar_interval(price["timestamp"])
    aligned = transforms.align_to_price(
        view, features, how=align_mode,
        tolerance=interval if align_mode == "nearest" else None,
    )
    feature_names = [c for c in features.columns if c != "timestamp"]
    numeric_here = sum(
        1 for c in feature_names if pd.api.types.is_numeric_dtype(features[c])
    )

    pcA, pcB, pcC = st.columns([1, 1, 2])
    with pcA:
        n_panels = int(st.number_input(
            "Feature panels", 0, PANEL_LIMIT, 1, key="pl_n_panels"
        ))
    with pcB:
        panel_h = st.slider("Panel weight", 0.4, 2.5, 1.0, 0.1, key="pl_panel_weight")
    with pcC:
        st.write("")
        reload_clicked = st.button(
            "🔄  Reload with selected features",
            width="stretch", key="pl_reload",
        )
    st.caption(
        f"{len(feature_names):,} columns available — {numeric_here:,} numeric, "
        f"{len(feature_names) - numeric_here:,} categorical. "
        "Text features draw as a step line over their categories."
    )

    if n_panels == 0:
        return None

    panel_keys = [f"tp{i}" for i in range(n_panels)]
    dirty = False
    for pk, ptab in zip(panel_keys, st.tabs([f"Panel {i + 1}" for i in range(n_panels)])):
        with ptab:
            dirty |= render_panel_controls(pk, feature_names)

    if reload_clicked:
        for pk in panel_keys:
            st.session_state["panel_applied"][pk] = sorted(pending_for(pk))
        dirty = False
    elif dirty:
        st.info("Feature selection changed — press **Reload with selected features**.")

    panels = [
        charts.PanelSpec(
            features=[f for f in applied_for(pk) if f in aligned.columns],
            normalise=config_for(pk)["normalise"],
            chart_type=config_for(pk)["chart_type"],
            title=f"Panel {index + 1}",
            height=panel_h,
            mark_events=config_for(pk)["mark_events"],
            show_zero_line=config_for(pk)["zero_line"],
        )
        for index, pk in enumerate(panel_keys)
    ]
    if not any(p.features for p in panels):
        st.info(
            "No features applied yet — tick some boxes above, then press "
            "**🔄 Reload with selected features**."
        )
        return None

    return panels, aligned, charts.feature_style_map(feature_names)


# --------------------------------------------------- tab 2: feature panels ---


def render_feature_import(settings: dict) -> None:
    fu1, fu2 = st.columns([3, 1.4])
    with fu1:
        features_raw, feat_name = file_source(
            "features",
            label="feature file",
            uploader_label="Feature file (timestamp + one column per feature)",
            uploader_key="up_features",
        )
    with fu2:
        st.write("")
        st.write("")
        st.selectbox(
            "Align to price bars", ["exact", "nearest"], key="align_mode",
            help="'nearest' back-fills from the last feature row at or before each bar — "
                 "use it when the feature clock differs from the price clock.",
        )

    if features_raw is None:
        return

    fcols = list(features_raw.columns)
    detected_ts = schema.detect_ohlc(features_raw).timestamp
    cA, cB = st.columns([1, 3])
    with cA:
        f_ts = column_picker("Timestamp column", fcols, detected_ts, "map_fts")
    candidates = schema.feature_columns(features_raw, exclude=[f_ts])
    with cB:
        chosen = st.multiselect(
            f"Feature columns to import ({len(candidates)} numeric found)",
            candidates, default=candidates, key="import_feats",
        )
    if st.button("✅ Import features", key="import_features", type="primary"):
        try:
            st.session_state["features"] = data_loader.prepare_features(
                features_raw, f_ts, chosen, settings["dayfirst"], settings["custom_format"]
            )
            st.session_state["features_source"] = feat_name
            st.session_state["panel_applied"] = {}
            st.session_state["panel_pending"] = {}
            # Purge every per-checkbox key too, otherwise a feature that was
            # ticked under the previous file's search reappears in the pending
            # set on the next render -- users see phantom selections carried
            # from a file they no longer have loaded.
            for key in [
                k for k in st.session_state
                if isinstance(k, str) and k.startswith("cb::")
            ]:
                del st.session_state[key]
            st.success(f"Imported {len(chosen)} features.")
        except data_loader.DataError as exc:
            st.error(str(exc))


def render_panel_controls(panel_key: str, feature_names: list) -> bool:
    """Search box, per-panel options and the feature checkbox grid.

    Returns ``True`` when the live selection differs from what the chart shows.
    """
    cfg = config_for(panel_key)
    pending = pending_for(panel_key)

    s1, s2, s3, s4 = st.columns([2.2, 1, 1, 1])
    with s1:
        query = st.text_input(
            "🔎 Search features", key=f"search_{panel_key}",
            placeholder="e.g. rsi   ·   vol atr   ·   mom",
        )
    with s2:
        cfg["normalise"] = st.selectbox(
            "Normalise", transforms.NORMALISERS,
            index=transforms.NORMALISERS.index(cfg["normalise"]),
            key=f"norm_{panel_key}",
            help="Put features of wildly different magnitudes on one axis.",
        )
    with s3:
        cfg["chart_type"] = st.selectbox(
            "Style", charts.CHART_TYPES,
            index=charts.CHART_TYPES.index(cfg["chart_type"]),
            key=f"style_{panel_key}",
        )
    with s4:
        cfg["mark_events"] = st.checkbox(
            "Mark target events", value=cfg["mark_events"], key=f"ev_{panel_key}",
            help="Dotted vertical line on every bar carrying a target label.",
        )
        cfg["zero_line"] = st.checkbox(
            "Zero line", value=cfg["zero_line"], key=f"zl_{panel_key}"
        )

    visible = transforms.search_features(feature_names, query)
    b1, b2, b3 = st.columns([1, 1, 4])
    with b1:
        if st.button("Select shown", key=f"all_{panel_key}", width="stretch"):
            for name in visible[:MAX_CHECKBOXES]:
                st.session_state[f"cb::{panel_key}::{name}"] = True
                pending.add(name)
            st.rerun()
    with b2:
        if st.button("Clear", key=f"none_{panel_key}", width="stretch"):
            for name in feature_names:
                st.session_state[f"cb::{panel_key}::{name}"] = False
            pending.clear()
            st.rerun()
    with b3:
        shown = min(len(visible), MAX_CHECKBOXES)
        overflow = (
            "" if len(visible) <= MAX_CHECKBOXES
            else f" · narrow the search to reach the other {len(visible) - shown}"
        )
        st.caption(
            f"Showing {shown} of {len(feature_names)} features · "
            f"**{len(pending)} selected**{overflow}"
        )

    grid = st.columns(CHECKBOX_COLUMNS)
    for j, name in enumerate(visible[:MAX_CHECKBOXES]):
        key = f"cb::{panel_key}::{name}"
        if key not in st.session_state:
            st.session_state[key] = name in pending
        with grid[j % CHECKBOX_COLUMNS]:
            if st.checkbox(name, key=key):
                pending.add(name)
            else:
                pending.discard(name)

    changed = set(pending) != set(applied_for(panel_key))
    if changed:
        st.caption("⚠️ Selection changed — press **Reload charts** to redraw.")
    return changed


def render_features_tab(settings: dict) -> None:
    price = st.session_state["price"]
    if price is None:
        st.info("Load price data on the **Price & Labels** tab first.")
        return

    st.subheader("1 · Load the feature file")
    render_feature_import(settings)

    features = st.session_state["features"]
    if features is None:
        st.info(
            "Upload a feature CSV above, or press **📦 Load sample data** on the first "
            "tab to pull in 25 demo indicators."
        )
        return

    frame_badge(features, f"Features · {st.session_state['features_source'] or 'uploaded'}")

    st.divider()
    st.subheader("2 · Labels (optional)")
    st.caption(
        "Load a **TS / TL / PL file** here to overlay target + predicted markers on the "
        "price chart below, just like the **🔮 Predict from Model** tab does after a run. "
        "Skip this if you only want to see price · features."
    )
    render_label_source(settings, key_scope="feat")

    start, end = window_bounds(price)
    view = apply_session_filter(transforms.filter_time_range(price, start, end))
    if view.empty:
        st.warning("No bars in the window selected on the first tab.")
        return
    if len(view) > settings["max_bars"]:
        view = view.iloc[-settings["max_bars"]:].reset_index(drop=True)

    align_mode = st.session_state.get("align_mode", "exact")
    interval = data_loader.infer_bar_interval(price["timestamp"])
    aligned = transforms.align_to_price(
        view, features, how=align_mode,
        tolerance=interval if align_mode == "nearest" else None,
    )
    feature_names = [c for c in features.columns if c != "timestamp"]
    cover = transforms.coverage(aligned, feature_names)

    labels = st.session_state["labels"]
    label_view = (
        transforms.filter_time_range(
            labels, view["timestamp"].min(), view["timestamp"].max()
        )
        if labels is not None else None
    )

    i1, i2, i3 = st.columns(3)
    i1.metric("Bars in view", f"{len(view):,}")
    i2.metric("Features available", f"{len(feature_names):,}")
    i3.metric(
        "Timestamp coverage", f"{cover:.1%}",
        help="Share of visible bars that received feature values after alignment.",
    )
    if cover < 0.5:
        st.warning(
            f"Only {cover:.0%} of the visible bars matched a feature row. Try alignment "
            "mode **nearest**, or check the day-first setting in the sidebar."
        )

    st.divider()
    st.subheader("3 · Pick features per panel")

    pc1, pc2, pc3 = st.columns([1, 1, 2])
    with pc1:
        n_panels = int(st.number_input(
            "Parallel feature charts", 1, PANEL_LIMIT, 2, key="n_panels"
        ))
    with pc2:
        panel_h = st.slider("Panel weight", 0.4, 2.5, 1.0, 0.1, key="panel_weight")
    with pc3:
        st.write("")
        reload_clicked = st.button(
            "🔄  Reload charts with selected features",
            type="primary", width="stretch", key="reload_charts",
        )

    panel_keys = [f"p{i}" for i in range(n_panels)]
    dirty = False
    panel_tabs = st.tabs([f"Panel {i + 1}" for i in range(n_panels)])
    for panel_key, panel_tab in zip(panel_keys, panel_tabs):
        with panel_tab:
            dirty |= render_panel_controls(panel_key, feature_names)

    if reload_clicked:
        for panel_key in panel_keys:
            st.session_state["panel_applied"][panel_key] = sorted(pending_for(panel_key))
        dirty = False
    elif dirty:
        st.info("Feature selection has pending changes — press **Reload charts** to apply.")

    st.divider()
    panels = [
        charts.PanelSpec(
            features=[f for f in applied_for(panel_key) if f in aligned.columns],
            normalise=config_for(panel_key)["normalise"],
            chart_type=config_for(panel_key)["chart_type"],
            title=f"Panel {index + 1}",
            height=panel_h,
            mark_events=config_for(panel_key)["mark_events"],
            show_zero_line=config_for(panel_key)["zero_line"],
        )
        for index, panel_key in enumerate(panel_keys)
    ]

    if not any(panel.features for panel in panels):
        st.info(
            "No features applied yet — tick some boxes above, then press "
            "**🔄 Reload charts with selected features**."
        )
        return

    show_labels_here = st.checkbox(
        "Overlay target / predicted markers on the price chart", value=True,
        key="feat_show_labels",
    )
    opts = charts.ChartOptions(
        **{
            **vars(settings["options"]),
            "show_target": show_labels_here,
            "show_predicted": show_labels_here,
            "show_mismatch": show_labels_here,
            "price_height": 2.4,
        }
    )
    st.plotly_chart(
        charts.build_figure(
            view,
            label_view if show_labels_here else None,
            aligned, panels, opts,
            charts.feature_style_map(feature_names),
        ),
        key="feature_chart",
        theme=None,          # keep our own template; Streamlit's would override it
        config={"scrollZoom": True, "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )
    if panels:
        _render_cross_panel_hover_line()
    st.caption(
        "Price and every feature panel share one x-axis — hover anywhere to read the "
        "candle, the labels and each feature value at that timestamp."
    )

    with st.expander("Descriptive stats for the applied features"):
        applied_all = sorted({f for panel in panels for f in panel.features})
        st.dataframe(aligned[applied_all].describe().T.round(4), width="stretch")


# ---------------------------------------------------------- tab 3: predict ---


@st.cache_resource(show_spinner=False)
def load_bundle_from_path(path: str, mtime: float, size: int):
    """Load a bundle already on disk. Avoids re-uploading a 100 MB+ model."""
    return predictor.load_bundle(path)


def local_bundles() -> list[Path]:
    """Model bundles sitting next to the app, newest first."""
    found = list(APP_ROOT.glob("*.joblib")) + list(APP_ROOT.glob("models/*.joblib"))
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_resource(show_spinner=False)
def load_bundle_cached(payload: bytes, name: str):
    """Load a model bundle from uploaded bytes.

    The training repo's loader takes a path (it rebuilds XGBoost/CatBoost from
    native blobs via temp files), so the upload is spooled to disk and removed
    again. Cached as a *resource* because a fitted model is not plain data.
    """
    handle = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
    try:
        handle.write(payload)
        handle.close()
        return predictor.load_bundle(handle.name)
    finally:
        Path(handle.name).unlink(missing_ok=True)


@st.cache_data(show_spinner=False)
def read_table_cached(payload: bytes, name: str) -> pd.DataFrame:
    return data_loader.read_table(payload, name)


@st.cache_data(show_spinner=False)
def read_table_from_path(path: str, mtime: float) -> pd.DataFrame:
    """Read a CSV/Parquet already on disk. ``mtime`` busts the cache on edit."""
    return data_loader.read_table(path, path)


@st.cache_data(show_spinner=False)
def cached_missing_features(source_key: str, columns: tuple, features: tuple) -> list:
    """Which model features are absent from the file.

    Keyed on the *file's source* (path or upload name) and its column names, not
    the DataFrame itself — @st.cache_data pickles its return so the DataFrame's
    Python identity changes on every rerun, defeating a hash_funcs={id} key.
    Columns and features are cheap to hash and uniquely determine the answer.
    """
    have = set(columns)
    return [c for c in features if c not in have]


@st.cache_data(show_spinner=False)
def cached_rename_plan(source_key: str, columns: tuple, features: tuple):
    """A rename plan keyed on file source + column list + model features."""
    fake_df = pd.DataFrame({c: [] for c in columns})
    return predictor.suggest_renames(fake_df, {"features": list(features)})


@st.cache_data(show_spinner=False)
def cached_missing_categoricals(
    source_key: str,
    categorical: tuple,
    mapping: tuple,
    _df: pd.DataFrame,
) -> dict:
    """Cached per-column NaN counts. Object-dtype ``isna`` is a per-cell Python
    loop that costs 3+ seconds on the real 500k×100-categorical slice; without
    caching it would fire on every date change. ``_df`` is prefixed with ``_`` so
    Streamlit does not hash it — the ``source_key`` alone identifies its contents.
    """
    renamed = _df.rename(columns=dict(mapping), copy=False) if mapping else _df
    return predictor.count_missing_categoricals(
        renamed, {"categorical": list(categorical)}
    )


@st.cache_data(show_spinner=False)
def cached_summarise(source_key: str, n_rows: int, _result: pd.DataFrame):
    """Metric aggregation over the full prediction set (1s on 500k rows).

    Keyed on the file source + row count so a re-run with the same predictions
    hits the cache -- the DataFrame itself is not hashed (leading underscore).
    """
    return metrics.summarise(_result)


def _bundle_key(bundle: dict, path_or_name: str) -> str:
    """A stable identity for a loaded bundle, used only as a cache key."""
    return f"{path_or_name}|{bundle.get('dataset_version')}|{len(bundle.get('features') or [])}"


def _render_cross_panel_hover_line() -> None:
    """Inject a JS handler that draws a single dotted vertical line spanning
    every subplot at the hovered x.

    Plotly's per-subplot spike line only spans the subplot the mouse is over,
    so a 4-panel chart shows four disjoint dotted segments with visible gaps
    between them. A shape with ``xref='x'`` + ``yref='paper'`` spans the whole
    figure vertically -- this helper wires a `plotly_hover` event listener on
    the multi-panel chart in the parent document that keeps a shape's x anchor
    synced with the mouse. On unhover, the shape is removed.

    The iframe rendered by ``st.components.v1.html`` is same-origin with the
    parent Streamlit page, so ``parent.document`` reaches the plotly div.
    Only wires the LAST multi-panel chart in the DOM (there is only one visible
    at a time in a tab), and skips single-panel charts.
    """
    import streamlit.components.v1 as components

    script = """
    <script>
    (function() {
      let attempts = 0;
      function wire() {
        const doc = parent.document;
        const charts = doc.querySelectorAll('.js-plotly-plot');
        if (!charts.length || !parent.Plotly) {
          if (attempts++ < 40) setTimeout(wire, 100);
          return;
        }
        // Pick the LAST multi-panel chart (the most recently rendered one).
        let target = null;
        for (const c of charts) {
          const layout = c._fullLayout;
          if (!layout) continue;
          const xCount = Object.keys(layout).filter(k => k.startsWith('xaxis')).length;
          if (xCount > 1) target = c;
        }
        if (!target || target._crosslineWired) return;
        target._crosslineWired = true;

        const spikeColor = '#A855F7';
        const clear = () => {
          try { parent.Plotly.relayout(target, {shapes: []}); } catch (e) {}
        };
        target.on('plotly_hover', function(evt) {
          if (!evt.points || !evt.points.length) return;
          const x = evt.points[0].x;
          try {
            parent.Plotly.relayout(target, {
              shapes: [{
                type: 'line',
                xref: 'x', yref: 'paper',
                x0: x, x1: x, y0: 0, y1: 1,
                line: {color: spikeColor, width: 1, dash: 'dot'},
                layer: 'above',
              }],
            });
          } catch (e) {}
        });
        target.on('plotly_unhover', clear);
        // Also clear when the plot is redrawn / DOM changes, so stale shapes
        // from an old chart do not survive a Streamlit rerun.
        const obs = new MutationObserver(() => {
          if (!doc.contains(target)) { obs.disconnect(); }
        });
        obs.observe(doc.body, {childList: true, subtree: true});
      }
      wire();
    })();
    </script>
    """
    components.html(script, height=0)


# Characters that browsers reject in a Content-Disposition filename plus a few
# that are not rejected but make copy-pasted filenames awkward in shells.
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(name: str, max_len: int = 40) -> str:
    """Turn an arbitrary source name into a filesystem-safe short slug.

    ClearML-style artifact names run to 100+ chars and contain spaces and dots
    all over the place; letting them into a Content-Disposition filename can
    trigger browser rejection or truncation. This collapses runs of unsafe
    characters to a single underscore and caps the length. Empty input maps to
    ``"unknown"`` so callers never end up with ``predictions__on__.csv``.

    A trailing ``.ext`` is stripped only when it *looks* like a real file
    extension -- short and alphanumeric. That way ``v4.1_final-alpha`` is left
    alone (``Path("v4.1_final-alpha").stem`` mistakes ``.1_final-alpha`` for an
    extension and returns ``"v4"``).
    """
    text = str(name)
    path = Path(text)
    suffix = path.suffix                     # includes the leading dot
    # Strip the extension only if it looks like a real one: 2-8 characters
    # after the dot, all alphanumeric. That way a version like "4.1" survives
    # (`Path.stem` would otherwise cut off the ".1").
    if 3 <= len(suffix) <= 9 and suffix[1:].isalnum():
        stem = path.stem
    else:
        stem = text
    stem = _UNSAFE_FILENAME_CHARS.sub("_", stem).strip("_.")
    return (stem[:max_len].rstrip("_.-") or "unknown")


def _download_stem() -> str:
    """The filename stem for the two download buttons.

    Preferred form: ``predictions__<model>__on__<feature-file>``. If either
    piece is missing we fall back to ``predicted_labels`` so a truncated URL is
    still meaningful.
    """
    model_src = st.session_state.get("loaded_model_name") or ""
    feature_src = st.session_state.get("prediction_source") or ""
    info = st.session_state.get("prediction_info")

    # Prefer model_type + dataset_version over the raw model filename -- ClearML
    # bundle names are unreadable and shrunk versions lose the interesting bits.
    if info is not None and getattr(info, "model_type", ""):
        model_part = _slug(info.model_type, max_len=24)
        if getattr(info, "dataset_version", ""):
            model_part = f"{model_part}_v{_slug(str(info.dataset_version), 10)}"
    elif model_src:
        model_part = _slug(model_src)
    else:
        model_part = ""

    feature_part = _slug(feature_src) if feature_src else ""

    if model_part and feature_part:
        return f"predictions__{model_part}__on__{feature_part}"
    if model_part or feature_part:
        return f"predictions__{model_part or feature_part}"
    return "predicted_labels"


# Build requests are ferried through session_state instead of the button's
# return value. The `if st.button(...):` idiom silently loses clicks under
# certain rerun races (a second click while a spinner is active, an on_change
# on another widget firing during the same tick); flipping a session_state flag
# in an `on_click` callback is race-free -- Streamlit fires the callback before
# the rerun and preserves it across the rerun.
_BUILD_REQUEST_KEY = "_pending_build"


def _request_build(kind: str) -> None:
    """Callback: mark that the user asked to build ``kind`` ('excel' or 'csv').

    Uses a set (not a scalar) so two clicks in the same tick both survive --
    otherwise the second click's kind overwrites the first and the first
    request is silently dropped. In practice Streamlit serialises clicks per
    session, but keyboard-driven double-taps or programmatic clicks can
    trigger the race.
    """
    pending = st.session_state.get(_BUILD_REQUEST_KEY)
    if not isinstance(pending, set):
        pending = set()
    pending.add(kind)
    st.session_state[_BUILD_REQUEST_KEY] = pending
    log.info("build requested: %s (queued: %s)", kind, sorted(pending))


def _build_in_thread(
    build_fn: Callable[..., Any], *args, **kwargs
) -> tuple[dict[str, Any], threading.Thread]:
    """Run ``build_fn`` on a daemon thread with a progress-callback proxy.

    Returns the shared ``holder`` dict the worker writes to and the thread
    handle. The caller polls ``holder["done"]`` and paints the UI.

    Split out from :func:`_run_with_live_progress` so the threading contract is
    testable without a live Streamlit context.

    Contract:

    * ``daemon=True`` -- otherwise Windows hangs on Ctrl+C during a build.
    * The worker touches **no** ``st.*`` API. Its only external effect is
      writing to ``holder``. That way "missing ScriptRunContext" warnings from
      the ScriptRunner cannot appear.
    * ``holder`` starts with a "starting" phase and fraction 0.0 so the caller
      has something to paint before the first callback fires.
    """
    holder: dict[str, Any] = {
        "done": False,
        "result": None,
        "error": None,
        "progress": 0.0,
        "phase": "starting",
    }

    def _progress_cb(fraction: float, phase: str) -> None:
        # Single writer + single reader means we don't need a lock -- CPython
        # dict item writes are atomic under the GIL for the setattr moment.
        holder["progress"] = float(fraction)
        holder["phase"] = str(phase)

    def _worker() -> None:
        try:
            holder["result"] = build_fn(*args, progress_cb=_progress_cb, **kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised on the main thread
            # Deliberately NOT BaseException -- SystemExit and KeyboardInterrupt
            # are shutdown signals that must propagate. They cannot fire in a
            # worker thread on Windows anyway (only the main thread receives
            # console signals), but making the intent explicit avoids the class
            # of bug where a stray sys.exit() inside a build_fn masquerades as
            # a normal failure.
            holder["error"] = exc
        finally:
            holder["done"] = True

    thread = threading.Thread(target=_worker, name="export-build", daemon=True)
    thread.start()
    return holder, thread


def _run_with_live_progress(
    label: str,
    build_fn: Callable[..., Any],
    *args,
    poll_seconds: float = 0.5,
    **kwargs,
) -> Any:
    """Run ``build_fn`` in a worker thread while painting a live progress bar.

    The worker reports ``(fraction, phase)`` into a shared dict via a callback
    the ``build_fn`` must accept as ``progress_cb=``. The ScriptRunner thread
    polls the dict every ``poll_seconds`` and re-paints ``st.progress`` inside
    an ``st.status`` container -- so the user sees an elapsed timer and the
    current phase, not a static spinner.

    Design notes (validated by research):

    * ``st.status`` is chosen over ``st.spinner`` because the latter is
      indeterminate and its label is immutable after creation.
    * The visible fraction is capped at 0.99 until the worker sets
      ``done=True`` -- a bar that hits 100% while code is still running reads
      as "the app hung".
    * If the WebSocket disconnects mid-build, this loop dies with the
      ScriptRunner but the worker keeps running to completion and its result
      is dropped. That is acceptable for a 3-minute build; longer would need
      an out-of-session cache.
    """
    holder, _thread = _build_in_thread(build_fn, *args, **kwargs)

    start = time.monotonic()
    with st.status(label, expanded=True) as status:
        bar = st.progress(0.0, text="Starting…")
        while not holder["done"]:
            elapsed = time.monotonic() - start
            fraction = min(0.99, float(holder["progress"]))
            phase = str(holder["phase"])
            mm, ss = divmod(int(elapsed), 60)
            bar.progress(fraction, text=f"{phase} · elapsed {mm:02d}:{ss:02d}")
            time.sleep(poll_seconds)

        elapsed = time.monotonic() - start
        mm, ss = divmod(int(elapsed), 60)
        if holder["error"] is not None:
            status.update(
                label=f"{label} failed after {mm:02d}:{ss:02d} — {holder['error']}",
                state="error",
            )
            log.error("%s failed: %s", label, holder["error"])
            raise holder["error"]
        bar.progress(1.0, text=f"Done in {mm:02d}:{ss:02d}")
        status.update(label=f"{label} · finished in {mm:02d}:{ss:02d}",
                      state="complete")
    return holder["result"]


def render_lazy_downloads(result: pd.DataFrame) -> None:
    """Two-step download UI: build on click, then download.

    ``exporter.to_excel`` on 500k rows takes 3 minutes and produces 65 MB;
    ``to_csv`` takes 15 seconds and produces 100 MB. Neither can run on every
    rerun. Splitting into build-then-download means idle reruns are free, and
    once built the payload lives in session_state so a re-click is instant.
    """
    source = st.session_state["prediction_source"]
    excel_key = f"excel_blob::{source}::{len(result)}"
    csv_key = f"csv_blob::{source}::{len(result)}"
    info = st.session_state["prediction_info"]
    n_rows = len(result)

    # Service any pending request(s) FIRST so the progress bar and the resulting
    # swap to a download button happen in a single, obvious script pass. Each
    # build runs in a worker thread; the ScriptRunner thread polls its progress
    # and re-paints the bar every ~0.5s -- so the user sees a live timer plus
    # phase label rather than a static spinner.
    pending = st.session_state.pop(_BUILD_REQUEST_KEY, None) or set()
    if isinstance(pending, str):                          # forward-compat / old sessions
        pending = {pending}
    if "excel" in pending and excel_key not in st.session_state:
        log.info("building Excel for %s rows", f"{n_rows:,}")
        try:
            st.session_state[excel_key] = _run_with_live_progress(
                f"Writing xlsx ({n_rows:,} rows)", exporter.to_excel, result, info,
            )
            log.info("Excel built: %d MB",
                     len(st.session_state[excel_key]) // 1_000_000)
        except Exception as exc:                          # noqa: BLE001
            log.error("Excel build failed: %s", exc)
            st.error(f"Excel build failed: {exc}")
    if "csv" in pending and csv_key not in st.session_state:
        log.info("building CSV for %s rows", f"{n_rows:,}")
        try:
            st.session_state[csv_key] = _run_with_live_progress(
                f"Writing CSV ({n_rows:,} rows)", exporter.to_csv, result,
            )
            log.info("CSV built: %d MB",
                     len(st.session_state[csv_key]) // 1_000_000)
        except Exception as exc:                          # noqa: BLE001
            log.error("CSV build failed: %s", exc)
            st.error(f"CSV build failed: {exc}")

    stem = _download_stem()

    d1, d2 = st.columns(2)

    with d1:
        if excel_key in st.session_state:
            payload = st.session_state[excel_key]
            st.download_button(
                f"⬇️  Excel (TS/TL/PL) · {len(payload) / 1e6:.0f} MB",
                data=payload,
                file_name=f"{stem}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch", type="primary", key="dl_excel",
                help=f"Saves as {stem}.xlsx",
            )
        else:
            st.button(
                "📊  Build Excel", width="stretch", key="build_excel",
                help=f"Writes {n_rows:,} rows to xlsx (~3 min for 500k rows).",
                on_click=_request_build, args=("excel",),
            )

    with d2:
        if csv_key in st.session_state:
            payload = st.session_state[csv_key]
            st.download_button(
                f"⬇️  CSV · {len(payload) / 1e6:.0f} MB",
                data=payload,
                file_name=f"{stem}.csv",
                mime="text/csv",
                width="stretch", key="dl_csv",
                help=f"Saves as {stem}.csv",
            )
        else:
            st.button(
                "📄  Build CSV", width="stretch", key="build_csv",
                help=f"Writes {n_rows:,} rows to CSV (~15 s for 500k rows).",
                on_click=_request_build, args=("csv",),
            )


TYPE_A_PATH = "Type a path…"
CHOOSE_A_FILE = "— choose a file —"


def discover_data_files() -> list[Path]:
    """Candidate data files near the app, newest first, for the path picker."""
    found: list[Path] = []
    for folder in (APP_ROOT, APP_ROOT / "data", APP_ROOT / "sample_data"):
        if folder.is_dir():
            for pattern in ("*.csv", "*.parquet", "*.txt"):
                found.extend(folder.glob(pattern))
    return sorted(set(found), key=lambda p: p.stat().st_mtime, reverse=True)[:60]


def path_input(
    key_prefix: str,
    label: str,
    candidates: list[Path],
    placeholder: str,
) -> str:
    """Paste box (primary) plus a picker of nearby files. Returns a path or ``""``.

    A pasted path always wins, and the picker greys out while one is present, so
    which of the two is in force is never ambiguous.
    """
    c1, c2 = st.columns([2, 1.2])
    with c1:
        typed = st.text_input(
            f"Paste the full path to the {label}",
            key=f"{key_prefix}_path", placeholder=placeholder,
            help="Right-click the file in Explorer → 'Copy as path'. "
                 "The surrounding quotes are stripped for you.",
        ).strip().strip('"').strip("'")
    with c2:
        options = [CHOOSE_A_FILE] + [str(p) for p in candidates]
        picked = st.selectbox(
            "…or pick a nearby file", options, key=f"{key_prefix}_pick",
            format_func=lambda s: s if s == CHOOSE_A_FILE else _pretty_path(s),
            disabled=bool(typed),
            help="Files found in the app folder, data/ and sample_data/.",
        )
    if typed:
        return typed
    return "" if picked == CHOOSE_A_FILE else picked


def file_source(
    key_prefix: str,
    label: str,
    uploader_label: str,
    uploader_key: str,
    types: tuple[str, ...] = ("csv", "txt", "parquet", "xlsx"),
    placeholder: str = r"C:\Users\Admin\data\nifty_futures_1min.parquet",
) -> tuple[Optional[pd.DataFrame], str]:
    """Path-or-upload chooser for a data file. Returns ``(frame, source name)``.

    Path is offered first because an upload is buffered in memory three times
    over (browser, server, DataFrame) and Streamlit caps it; reading from disk
    does neither.
    """
    mode = st.radio(
        f"Where is the {label}?", ["From a file path", "Upload"],
        horizontal=True, key=f"{key_prefix}_mode",
        help="Large exports are far quicker from disk, and skip the upload limit.",
    )
    if mode == "From a file path":
        frame, path = table_from_path(key_prefix, label, placeholder=placeholder)
        return frame, (Path(path).name if path else "")

    up = st.file_uploader(uploader_label, type=list(types), key=uploader_key)
    if up is None:
        return None, ""
    try:
        return read_table_cached(up.getvalue(), up.name), up.name
    except data_loader.DataError as exc:
        st.error(f"**{up.name}** — {exc}")
        return None, ""


def table_from_path(
    key_prefix: str,
    label: str = "OHLC file",
    *,
    placeholder: str = r"C:\Users\Admin\data\nifty_futures_1min.parquet",
) -> tuple[Optional[pd.DataFrame], str]:
    """Choose a data file **by path** and read it. Returns ``(frame, path)``.

    Candles are the one input taken from disk rather than uploaded: an intraday
    NIFTY history is large and rarely changes, so re-uploading it every session
    is wasted time.
    """
    chosen = path_input(
        key_prefix, label.lower(), discover_data_files(), placeholder,
    )
    if not chosen:
        return None, ""

    target = Path(chosen)
    if not target.exists():
        st.error(f"No such file: `{target}`")
        return None, ""
    if not target.is_file():
        st.error(f"`{target}` is a folder, not a file.")
        return None, ""

    try:
        # Reject a model bundle (or other non-table) before the reader mangles
        # the message into a codec error.
        data_loader.check_readable(target.name)
        return read_table_from_path(str(target), target.stat().st_mtime), str(target)
    except data_loader.DataError as exc:
        st.error(str(exc) if str(exc).startswith("`") else f"**{target.name}** — {exc}")
        return None, ""


def price_from_path(key_prefix: str, settings: dict) -> Optional[pd.DataFrame]:
    """Path picker plus column mapping, returning the canonical price frame."""
    raw, chosen = table_from_path(key_prefix)
    if raw is None:
        return None
    target = Path(chosen)

    mapping = schema.detect_ohlc(raw)
    if not mapping.is_complete:
        st.warning(
            f"`{target.name}` is missing " + ", ".join(mapping.missing()) +
            ". Map the columns below."
        )
        cols = list(raw.columns)
        c = st.columns(5)
        fields = ("timestamp", "open", "high", "low", "close")
        for column, field in zip(c, fields):
            with column:
                setattr(
                    mapping, field,
                    column_picker(field.title(), cols, getattr(mapping, field),
                                  f"{key_prefix}_map_{field}"),
                )
        if not mapping.is_complete:
            return None

    try:
        price = data_loader.prepare_price(
            raw, mapping, settings["dayfirst"], settings["custom_format"]
        )
    except data_loader.DataError as exc:
        st.error(str(exc))
        return None

    info = data_loader.describe_frame(price)
    st.caption(
        f"`{target.name}` · {info['rows']:,} bars · "
        f"{info['start']:%d-%b-%Y %H:%M} → {info['end']:%d-%b-%Y %H:%M}"
        + (f" · {info['interval']}" if info["interval"] is not None else "")
    )
    return price


def _pretty_path(path: str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(APP_ROOT))
    except ValueError:
        return str(p)


def resolve_price_for_predictions(raw: pd.DataFrame, settings: dict) -> Optional[pd.DataFrame]:
    """Find candles to draw the predicted labels against.

    The model's features carry no raw OHLC, but the uploaded feature file often
    does as extra columns (they are ignored during prediction). Failing that we
    reuse whatever the Price & Labels tab already holds, or take a separate file.
    """
    detected = schema.detect_ohlc(raw)
    choices = ["From an OHLC file path"]
    if detected.is_complete:
        choices.insert(0, "From the feature file")
    if st.session_state["price"] is not None:
        choices.insert(len(choices) - 1, "Price loaded on the first tab")

    source = st.radio(
        "Candles to plot against", choices, horizontal=True, key="pred_price_source"
    )

    if source == "From the feature file":
        try:
            return data_loader.prepare_price(
                raw, detected, settings["dayfirst"], settings["custom_format"]
            )
        except data_loader.DataError as exc:
            st.error(f"Could not read OHLC from the feature file — {exc}")
            return None

    if source == "Price loaded on the first tab":
        return st.session_state["price"]

    return price_from_path("pred_ohlc", settings)


def prediction_feature_panels(
    result: pd.DataFrame,
    raw: pd.DataFrame,
    view: pd.DataFrame,
    exclude: set,
) -> tuple[list, Optional[pd.DataFrame], dict]:
    """Panel controls over the feature file that was just predicted on.

    Only the *applied* columns are materialised. The source frame can be half a
    million rows by three hundred columns, and copying all of it to draw four
    lines would cost a gigabyte for nothing.
    """
    names = [str(c) for c in raw.columns if str(c) not in exclude]
    if not names:
        return [], None, {}

    numeric = sum(1 for c in names if pd.api.types.is_numeric_dtype(raw[c]))
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        n_panels = int(st.number_input(
            "Feature panels", 0, PANEL_LIMIT, 1, key="pred_n_panels"
        ))
    with c2:
        panel_h = st.slider("Panel weight", 0.4, 2.5, 1.0, 0.1, key="pred_panel_weight")
    with c3:
        st.write("")
        reload_clicked = st.button(
            "🔄  Reload with selected features", width="stretch", key="pred_reload"
        )
    st.caption(
        f"{len(names):,} columns available — {numeric:,} numeric, "
        f"{len(names) - numeric:,} categorical. Text features draw as a step line "
        "over their categories."
    )

    if n_panels == 0:
        return [], None, {}

    keys = [f"pp{i}" for i in range(n_panels)]
    dirty = False
    for key, tab in zip(keys, st.tabs([f"Panel {i + 1}" for i in range(n_panels)])):
        with tab:
            dirty |= render_panel_controls(key, names)

    if reload_clicked:
        for key in keys:
            st.session_state["panel_applied"][key] = sorted(pending_for(key))
        dirty = False
    elif dirty:
        st.info("Feature selection changed — press **Reload with selected features**.")

    applied = sorted({f for key in keys for f in applied_for(key)})
    applied = [f for f in applied if f in raw.columns]
    if not applied:
        return [], None, {}

    # Materialise only what is drawn, stamped with the parsed timestamps.
    slim = pd.DataFrame({"timestamp": result["timestamp"].to_numpy()})
    for name in applied:
        slim[name] = raw[name].to_numpy()
    aligned = transforms.align_to_price(view, slim, how="exact")

    panels = [
        charts.PanelSpec(
            features=[f for f in applied_for(key) if f in aligned.columns],
            normalise=config_for(key)["normalise"],
            chart_type=config_for(key)["chart_type"],
            title=f"Panel {index + 1}",
            height=panel_h,
            mark_events=config_for(key)["mark_events"],
            show_zero_line=config_for(key)["zero_line"],
        )
        for index, key in enumerate(keys)
    ]
    panels = [p for p in panels if p.features]
    return panels, aligned, charts.feature_style_map(applied)


def render_prediction_chart(
    result: pd.DataFrame,
    raw: pd.DataFrame,
    settings: dict,
    exclude: set = frozenset(),
) -> None:
    """Candles with the predicted (and any target) labels, plus feature panels."""
    st.divider()
    st.subheader("4 · Visualize")

    price = resolve_price_for_predictions(raw, settings)
    if price is None or price.empty:
        return

    unfiltered = result[["timestamp", "target", "predicted"]].copy()
    labels = unfiltered.copy()

    c1, c2, c3 = st.columns([1.4, 1, 1])
    with c1:
        floor = st.slider(
            "Hide predictions below confidence", 0.0, 1.0, 0.0, 0.05,
            key="pred_conf_floor",
            help="Low-confidence calls are dropped from the chart, so you can see "
                 "only where the model was sure.",
        )
    hidden = 0
    if floor > 0 and "confidence" in result.columns:
        weak = pd.to_numeric(result["confidence"], errors="coerce") < floor
        hidden = int(weak.sum())
        labels.loc[weak.to_numpy(), "predicted"] = pd.NA
    with c2:
        show_target = st.checkbox(
            "Show target too", value=bool(labels["target"].notna().any()),
            key="pred_show_target", disabled=not labels["target"].notna().any(),
        )
    with c3:
        show_mismatch = st.checkbox("Mark mismatches", value=True, key="pred_show_mismatch")

    # Everything the candles and the predictions have in common.
    covered = transforms.filter_time_range(
        price, labels["timestamp"].min(), labels["timestamp"].max()
    )
    if covered.empty:
        st.warning(
            "The candles and the predictions do not overlap in time. Check the "
            "day-first setting, or that both files cover the same period."
        )
        return

    sessions = trading_dates(covered)
    opens_on = initial_date_range(covered, settings["initial_days"])
    first, last = covered["timestamp"].min().date(), covered["timestamp"].max().date()

    d1, d2 = st.columns(2)
    with d1:
        start_date = st.date_input(
            "Start date", value=opens_on[0], min_value=first, max_value=last,
            key="pred_start_date",
            help=f"Opens on the last {settings['initial_days']} predicted session(s). "
                 "Move it back to see more.",
        )
    with d2:
        end_date = st.date_input(
            "End date", value=opens_on[1], min_value=first, max_value=last,
            key="pred_end_date",
        )

    if start_date > end_date:
        st.error(
            f"Start date ({start_date:%d-%b-%Y}) is after the end date "
            f"({end_date:%d-%b-%Y}). Swap them to see the chart."
        )
        return

    view = transforms.filter_time_range(
        covered,
        pd.Timestamp(start_date),
        pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
    )
    if view.empty:
        st.warning("No candles in that window — widen the dates.")
        return

    if not (start_date <= sessions[0].date() and end_date >= sessions[-1].date()):
        st.caption(
            f"{len(sessions):,} predicted sessions available "
            f"({sessions[0]:%d-%b-%Y} → {sessions[-1]:%d-%b-%Y}). "
            "Move the start date back to see more."
        )

    if len(view) > settings["max_bars"]:
        st.caption(
            f"Window holds {len(view):,} bars; showing the newest "
            f"{settings['max_bars']:,} (raise **Max bars to render** in the sidebar)."
        )
        view = view.iloc[-settings["max_bars"]:].reset_index(drop=True)

    # Check alignment against the *unfiltered* labels, so a high confidence floor
    # is never misreported as a timestamp mismatch.
    aligned = int(charts.attach_labels(view, unfiltered)["predicted"].notna().sum())
    if aligned == 0:
        st.warning(
            "No predicted label lines up with a candle timestamp. The two files "
            "are probably on different bar intervals, or one is day-first and the "
            "other is not."
        )
        return

    matched = int(charts.attach_labels(view, labels)["predicted"].notna().sum())
    if matched == 0:
        st.info(
            f"Every prediction is below {floor:.0%} confidence, so none are drawn. "
            "Lower the threshold to see them."
        )

    st.markdown("###### Features to show underneath")
    panels, aligned, styles = prediction_feature_panels(result, raw, view, set(exclude))

    opts = charts.ChartOptions(
        **{
            **vars(settings["options"]),
            "show_target": show_target,
            "show_predicted": True,
            "show_mismatch": show_mismatch and show_target,
            "price_height": 3.0 if not panels else 2.4,
        }
    )
    st.plotly_chart(
        charts.build_figure(view, labels, aligned, panels, opts, styles or None),
        key="prediction_chart", theme=None,
        config={"scrollZoom": True, "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )
    if panels:
        _render_cross_panel_hover_line()
    st.caption(
        f"{matched:,} of {len(view):,} candles carry a prediction"
        + (f" · {hidden:,} hidden below the confidence floor" if hidden else "")
        + ". Hollow markers are predicted, solid are target. "
        "Hover reads the labels and every feature at that timestamp."
    )


def render_predict_tab(settings: dict) -> None:
    st.subheader("1 · Load the model bundle")
    st.warning(
        "**A `.joblib` runs code when it loads.** Only upload a model you built "
        "yourself or trust completely — never one that arrived by email or chat.",
        icon="⚠️",
    )

    source = st.radio(
        "Model source", ["From a path", "Upload"],
        horizontal=True, key="model_source",
        help="A trained bundle can be 100 MB+; loading it from disk avoids "
             "re-uploading it through the browser every session.",
    )

    bundle = info = None
    mc1, mc2 = st.columns([2, 2])

    if source == "From a path":
        picked = path_input(
            "model_src", "model bundle", local_bundles(),
            r"C:\Users\Admin\Downloads\model_xgboost.joblib",
        )
        if picked:
            target = Path(picked)
            if not target.exists():
                st.error(f"No such file: `{target}`")
            elif not target.is_file():
                st.error(f"`{target}` is a folder, not a file.")
            # Loading is explicit: these bundles run to 100 MB+, so it should not
            # happen just because the tab was opened.
            elif st.button(f"📂  Load {target.name[:48]}", key="load_local_bundle"):
                st.session_state["local_bundle_path"] = str(target)

        chosen = st.session_state.get("local_bundle_path")
        if chosen and Path(chosen).exists():
            try:
                stat = Path(chosen).stat()
                with st.spinner(f"Loading {Path(chosen).name}…"):
                    bundle = load_bundle_from_path(chosen, stat.st_mtime, stat.st_size)
                info = predictor.describe(bundle)
                st.session_state["loaded_model_name"] = Path(chosen).name
            except predictor.PredictionError as exc:
                st.error(str(exc))
    else:
        with mc1:
            up_model = st.file_uploader(
                "Model bundle (`model_*.joblib`)", type=["joblib", "pkl"], key="up_model"
            )
        if up_model is not None:
            try:
                bundle = load_bundle_cached(up_model.getvalue(), up_model.name)
                info = predictor.describe(bundle)
                st.session_state["loaded_model_name"] = up_model.name
            except predictor.PredictionError as exc:
                st.error(str(exc))

    with mc2:
        if info is not None:
            st.write("")
            st.markdown(
                f'<span class="pill pill-ok">{info.model_type}'
                f'{" · dataset v" + str(info.dataset_version) if info.dataset_version else ""}'
                f" · {info.n_features} features · {len(info.classes)} classes</span>",
                unsafe_allow_html=True,
            )
            if not info.has_proba:
                st.caption(
                    "This estimator has no `predict_proba`, so confidence and the "
                    "per-class probability columns will be blank."
                )

    if info is not None:
        with st.expander(f"Model expects {info.n_features} features · classes: "
                         f"{', '.join(info.classes)}"):
            st.dataframe(
                pd.DataFrame({"feature": info.features}), height=220,
                width="stretch", hide_index=True,
            )

    st.divider()
    st.subheader("2 · Load the feature data")
    raw, feat_source = file_source(
        "predict_features",
        label="feature file",
        uploader_label="Feature rows to predict on (CSV or Parquet)",
        uploader_key="up_predict_features",
    )

    if bundle is None or raw is None:
        st.info(
            "Load a model bundle and a feature file to predict. Already have a "
            "`TS / TL / PL` file? Load it directly on the **Price & Labels** tab — "
            "this tab is only for producing one."
        )
        return

    st.caption(f"{len(raw):,} rows × {raw.shape[1]} columns")

    # ---- schema check before anything is run --------------------------------
    # Both the missing-column scan and the rename planner iterate 315 feature
    # names against 320 file columns, and both fire on every rerun (every date
    # move, every checkbox tick). Caching keyed on the frame identity keeps them
    # off the widget hot path.
    bundle_id = _bundle_key(bundle, feat_source)
    feature_key = tuple(info.features)
    column_key = tuple(str(c) for c in raw.columns)
    source_key = f"{feat_source}|{bundle_id}"
    missing = cached_missing_features(source_key, column_key, feature_key)
    plan = (
        cached_rename_plan(source_key, column_key, feature_key) if missing else None
    )
    if plan is not None and not plan.helps:
        plan = None

    if missing and plan is not None:
        st.info(
            f"**{plan.n_renamed} of the {info.n_features} feature columns match once the "
            f"`{plan.prefix}` prefix is taken into account.** The model was trained on "
            "prefixed names; this export uses the bare ones.",
            icon="🔗",
        )
        # Ticked even for a partial match, so the error below names the columns
        # that are genuinely absent rather than the ones that merely look absent.
        use_renames = st.checkbox(
            f"Match columns by adding the `{plan.prefix}` prefix",
            value=True, key="use_renames",
        )
        with st.expander(f"Check the {plan.n_renamed} column matches before running"):
            st.dataframe(
                pd.DataFrame(
                    sorted(plan.mapping.items()), columns=["your column", "model feature"]
                ),
                hide_index=True, width="stretch", height=260,
            )
            if plan.ambiguous:
                st.warning(
                    f"{len(plan.ambiguous)} column(s) could match more than one feature "
                    "and were left alone: `" + "`, `".join(plan.ambiguous[:8]) + "`"
                )
        if not use_renames:
            plan = None
        else:
            missing = predictor.missing_features(
                predictor.apply_renames(raw, plan), bundle
            )
    else:
        plan = None

    if missing:
        st.error(
            f"**This file is missing {len(missing)} of the {info.n_features} feature "
            "columns the model was trained on**, so it cannot be used for prediction. "
            f"Missing: `{'`, `'.join(missing[:12])}`"
            + (f" …and {len(missing) - 12} more" if len(missing) > 12 else "")
        )
        return

    # The model's view of the data. Timestamp and label are still read from the
    # original frame, so the pickers below list the names the user actually has.
    model_view = predictor.apply_renames(raw, plan)
    renamed_away = set(plan.mapping) if plan else set()

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        ts_col = column_picker(
            "Timestamp column", list(raw.columns),
            predictor.DEFAULT_TIMESTAMP_COL if predictor.DEFAULT_TIMESTAMP_COL in raw.columns
            else schema.detect_ohlc(raw).timestamp,
            "pred_ts",
        )
    with cc2:
        label_options = [
            c for c in predictor.candidate_label_columns(raw, bundle)
            if c not in renamed_away
        ]
        default_label = (
            predictor.DEFAULT_LABEL_COL if predictor.DEFAULT_LABEL_COL in label_options else None
        )
        tl_col = column_picker(
            "Target label column (optional)", label_options, default_label,
            "pred_tl", allow_none=True,
            help="Prediction produces the label. A target is only needed if you "
                 "also want to score the model.",
        )
    with cc3:
        st.write("")
        run = st.button("🔮  Run prediction", type="primary", width="stretch", key="run_predict")

    st.success(f"All {info.n_features} model features present. Extra columns are ignored.")

    # ---- missing-categorical encoding --------------------------------------
    # Cache key includes the rename mapping so ticking/unticking the prefix
    # option produces the right count without recomputing on unrelated reruns.
    mapping = tuple(sorted(plan.mapping.items())) if plan else ()
    gaps = cached_missing_categoricals(
        source_key, tuple(info.categorical), mapping, raw,
    )
    na_mode = predictor.NA_MODE_PREDICT_PY
    if gaps:
        total = sum(gaps.values())
        st.warning(
            f"**{total:,} missing values across {len(gaps)} categorical column(s).** "
            "How these are encoded changes the prediction, because `predict.py` and the "
            "trainer disagree: the trainer filled text gaps with `MISSING` and encoded "
            "that (every `cat_map` in this bundle carries a MISSING code), while "
            "`predict.py` encodes them as `-1` — a value the model never saw in training.",
            icon="⚠️",
        )
        na_mode = st.radio(
            "Encode missing categories as",
            predictor.NA_MODES,
            horizontal=True, key="na_mode",
            format_func=lambda m: (
                "predict.py  →  -1  (matches your pipeline's output)"
                if m == predictor.NA_MODE_PREDICT_PY
                else "training  →  MISSING code  (matches how the model was fitted)"
            ),
        )
        with st.expander(f"Which columns have gaps ({len(gaps)})"):
            st.dataframe(
                pd.DataFrame(
                    sorted(gaps.items(), key=lambda kv: -kv[1]), columns=["column", "missing rows"]
                ),
                hide_index=True, width="stretch", height=240,
            )
    else:
        st.caption(
            "No missing values in the categorical features, so the `predict.py` / trainer "
            "encoding disagreement does not affect this file."
        )

    if run:
        log.info("run requested: %s rows, %s features, tl=%s",
                 f"{len(raw):,}", info.n_features, tl_col or "—")
        try:
            with st.spinner(f"Predicting {len(raw):,} rows…"):
                st.session_state["predictions"] = predictor.predict_frame(
                    raw, bundle, ts_col, tl_col,
                    dayfirst=settings["dayfirst"], na_mode=na_mode, renames=plan,
                )
            log.info("prediction complete: %s rows in session_state",
                     f"{len(st.session_state['predictions']):,}")
            st.session_state["prediction_info"] = info
            st.session_state["prediction_source"] = feat_source
            # Snapshot the settings this prediction was made under. Every rerun
            # compares against it so a user who flips na_mode or the rename
            # checkbox without re-clicking Run sees a "Settings changed" pill
            # and cannot download stale bytes.
            st.session_state["prediction_settings"] = {
                "na_mode": na_mode,
                "renames": tuple(sorted(plan.mapping.items())) if plan else (),
                "tl_col": tl_col,
            }
            # A window kept from the previous run can fall outside this one.
            for key in ("pred_start_date", "pred_end_date"):
                st.session_state.pop(key, None)
            # Previously-built Excel / CSV blobs refer to the OLD predictions
            # frame. Their cache key is (source_name, row_count), which is
            # identical when a user re-runs on the same file with a different
            # na_mode or prefix-rename choice -- so without this invalidation
            # the download button silently serves stale bytes.
            for key in [k for k in st.session_state
                        if isinstance(k, str)
                        and k.startswith(("excel_blob::", "csv_blob::"))]:
                st.session_state.pop(key, None)
        except predictor.PredictionError as exc:
            log.error("prediction failed: %s", exc)
            st.error(str(exc))

    result = st.session_state["predictions"]
    if result is None:
        return

    # Compare the current widget values against the snapshot taken at Run time.
    # A mismatch means the user changed na_mode / the prefix-rename tick / the
    # target label column without re-clicking Run; the cached predictions no
    # longer reflect what the UI says, so we surface a warning pill and skip
    # the download-button section rather than let stale bytes escape.
    snapshot = st.session_state.get("prediction_settings") or {}
    current = {
        "na_mode": na_mode,
        "renames": tuple(sorted(plan.mapping.items())) if plan else (),
        "tl_col": tl_col,
    }
    settings_changed = bool(snapshot) and snapshot != current

    if settings_changed:
        st.markdown(
            '<div style="text-align:right;margin-top:-8px;">'
            '<span class="pill pill-warn">settings changed since last Run — press '
            "🔮 Run prediction to refresh</span></div>",
            unsafe_allow_html=True,
        )
    else:
        # A subtle status pill so a rerun makes it visible that the current
        # predictions are the *previous* run's -- rather than leaving it ambiguous.
        st.markdown(
            f'<div style="text-align:right;margin-top:-8px;">'
            f'<span class="pill pill-ok">predictions cached '
            f"· {len(result):,} rows · {st.session_state['prediction_source'] or 'this session'}"
            "</span></div>",
            unsafe_allow_html=True,
        )

    # ---- results ------------------------------------------------------------
    st.divider()
    st.subheader("3 · Result")

    summary = cached_summarise(
        st.session_state["prediction_source"], len(result), result,
    )
    r = st.columns(5)
    r[0].metric("Rows predicted", f"{summary.n_rows:,}")
    r[1].metric("Classes predicted", f"{len(summary.predicted_counts)}")
    r[2].metric(
        "Mean confidence",
        f"{pd.to_numeric(result['confidence'], errors='coerce').mean():.1%}"
        if "confidence" in result.columns and result["confidence"].notna().any() else "—",
    )
    r[3].metric(
        "Accuracy vs target",
        f"{summary.accuracy:.1%}" if summary.accuracy is not None else "—",
        help="Only computed on rows that carry a ground-truth label.",
    )
    r[4].metric("Mismatches", f"{summary.n_mismatch:,}" if summary.has_comparison else "—")

    # Downloads are LAZY. Building a 65 MB xlsx from 513k rows takes ~3 minutes,
    # and doing that on every rerun (widget change, date move) hangs the whole
    # tab -- section 4 never gets to render. Now they're built only when the
    # user asks, and cached per file-source so a second click is instant.
    if settings_changed:
        st.info(
            "Downloads are hidden until the prediction reflects the current "
            "settings. Press 🔮 Run prediction to refresh."
        )
    else:
        render_lazy_downloads(result)

    if st.button("📈  Use these labels in the visualizer now", width="stretch",
                 key="use_predictions"):
        st.session_state["labels"] = result[["timestamp", "target", "predicted"]].copy()
        st.session_state["labels_source"] = (
            f"predicted · {st.session_state['prediction_source']}"
        )
        st.success(
            "Loaded into the **Price & Labels** tab — it needs price data to chart against."
        )

    st.caption(
        f"Saves as **`{_download_stem()}.csv`** / **`.xlsx`** — the name reflects the "
        "model and the feature file used. Headers are `timestamp` / `target_label` / "
        "`predicted_label`, so the file loads straight back into the "
        "**Price & Labels** tab."
    )

    with st.expander("Predicted class distribution", expanded=True):
        counts = pd.DataFrame(
            sorted(summary.predicted_counts.items(), key=lambda kv: -kv[1]),
            columns=["class", "rows"],
        )
        if summary.target_counts:
            counts["target rows"] = counts["class"].map(summary.target_counts).fillna(0).astype(int)
        st.dataframe(counts, hide_index=True, width="stretch")

    # Slice first, then format -- building the full 513k-row sheet just to slice
    # off 1,000 costs 400ms on every rerun.
    st.dataframe(
        exporter.build_sheet(result.head(1_000)), height=340,
        width="stretch", hide_index=True,
    )

    render_prediction_chart(
        result, raw, settings, exclude={ts_col, tl_col} - {None}
    )


# ------------------------------------------------------ tab 4: data/export ---


def render_data_tab() -> None:
    st.subheader("Loaded datasets")
    frame_badge(st.session_state["price"], f"Price · {st.session_state['price_source'] or '—'}")
    frame_badge(st.session_state["labels"], f"Labels · {st.session_state['labels_source'] or '—'}")
    frame_badge(
        st.session_state["features"], f"Features · {st.session_state['features_source'] or '—'}"
    )

    price = st.session_state["price"]
    if price is None:
        return

    merged = charts.attach_labels(price, st.session_state["labels"])
    features = st.session_state["features"]
    if features is not None:
        merged = merged.merge(features, on="timestamp", how="left", suffixes=("", "_feat"))

    st.divider()
    f1, f2 = st.columns([1, 3])
    with f1:
        only_mismatch = st.checkbox("Only mismatched rows", value=False)
    table = merged.loc[metrics.mismatch_mask(merged)] if only_mismatch else merged
    with f2:
        st.caption(f"{len(table):,} rows × {table.shape[1]} columns")

    st.dataframe(table.head(2_000), width="stretch", height=430, hide_index=True)
    st.download_button(
        "⬇️ Download the merged dataset (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="merged_price_labels_features.csv",
        mime="text/csv",
    )


# ------------------------------------------------------------------- main ---


def main() -> None:
    init_state()
    settings = render_sidebar()
    tab_price, tab_predict = st.tabs(
        [
            "📊  Price & Labels",
            "🔮  Predict from Model",
        ]
    )
    with tab_price:
        render_price_tab(settings)
    with tab_predict:
        render_predict_tab(settings)


# GUARDED so this file can be imported as a page of the multi-page app. python caches an imported
# module, so a bare main() at import time would draw the page ONCE and leave it blank on every
# rerun. ui/pages/2_Visualizer.py imports this and calls main() itself, every run.
if __name__ == "__main__":
    main()
