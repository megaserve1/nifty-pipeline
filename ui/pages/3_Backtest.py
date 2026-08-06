"""ui/pages/3_Backtest.py -- show the BACKTEST workbook a run produced, in one place.

READ ONLY. this page starts nothing and changes nothing. it only opens an .xlsx and draws it.

WHAT IT READS (changed 2026-08-05)
    the backtest engine now writes ONE multi-sheet Excel workbook instead of a scatter of CSVs.
    the sheets, and where each lands on this page:
        00_README         the workbook's own guide            -> Data & config tab
        01_Summary        executive V1/V2/V3 comparison        -> Overview tab (the stats table)
        02_Config         every setting the run used           -> Data & config tab
        03_Data_Quality   coverage / OHLC / churn checks       -> Data & config tab
        04_Metrics        all 103 metrics x V1/V2/V3           -> Full metrics tab
        05_Tradebook      one row per closed trade             -> Trades tab
        06_Yearly         year by year, per version            -> Year / Month tab
        07_Monthly        month by month, per version          -> Year / Month tab
        08_Daily_Equity   end-of-session equity per version    -> Equity & risk tab
        09_Missed_Trades  ideal trades the model skipped        -> Churn & missed tab

WHERE THE WORKBOOK COMES FROM  -- two sources, SAME file either way:
    * A folder on this machine  (what you use now: the engine writes the .xlsx there)
    * ClearML (a trained model) -- the GCP-backed path, UNCHANGED: same project / model pickers,
      it just fetches the .xlsx artifact instead of the old CSVs.

CORE vs OPTIONAL  (the workbook's own distinction, kept here):
    CORE metrics divide by NO capital -- rupees, points, counts, ratios. judge the signal on these.
    OPTIONAL metrics (return %, CAGR, Sharpe, Sortino, drawdown %, Calmar) divide by an ASSUMED
    capital, so the workbook reports them on TWO bases -- A: 1-lot notional, B: peak notional.
    pick the basis with the toggle; it only affects the OPTIONAL numbers.
"""
import math
import pathlib
import sys

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Backtest", page_icon="📉", layout="wide")

# the password gate. FIRST thing after set_page_config -- streamlit serves this file at its own
# URL, so a gate on Home.py alone would not protect it.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _auth import require_auth   # noqa: E402
require_auth()

# GCP / ClearML integration -- UNCHANGED. these are the same helpers every page uses.
from _shared import model_picker, children_of, artifacts_named, project_picker   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import config as C   # noqa: E402,F401

# the folder your local runs write into. new runs drop a new backtest_*/ subfolder here, and this
# page takes the NEWEST .xlsx under it -- so you never have to update the path.
DEFAULT_LOCAL = r"C:\Users\Admin\Downloads\backtest_results"
TRADES_SHOWN = 2000        # the tables are virtualised, but the payload still crosses the wire

VERSION_LABELS = {"V1": "V1 · SUPER entry", "V2": "V2 · SUB entry", "V3": "V3 · SUPER class"}
BASIS_BLOCK = {"A · 1-lot notional": "OPTIONAL A 1-lot", "B · peak notional": "OPTIONAL B peak"}

st.title("Backtest results")
st.caption("one workbook per run — the V1/V2/V3 signal-quality suite, priced on NIFTY futures with "
           "next-bar-open fills. read only; this page starts nothing.")

# TILE READABILITY. st.metric shows the value in a big ~2.25rem font on one no-wrap line, so a
# rupee figure in a narrow column was being clipped to "₹-12…". smaller value font + allow wrap
# (never ellipsis) + let the label wrap too. scoped to metric tiles only.
st.markdown("""
<style>
  [data-testid="stMetricValue"]{ font-size:1.5rem; line-height:1.25;
        white-space:normal; overflow-wrap:anywhere; overflow:visible; }
  [data-testid="stMetricValue"] > div{ white-space:normal; overflow:visible; text-overflow:clip; }
  [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p{ white-space:normal; }
</style>""", unsafe_allow_html=True)


# ============================== loading ====================================
def _engine():
    try:
        import python_calamine  # noqa: F401   fast Rust reader; silent fallback to openpyxl
        return {"engine": "calamine"}
    except ImportError:
        return {}


@st.cache_data(show_spinner="reading the workbook …", ttl=300)
def load_workbook(path: str, sig) -> dict:
    """every sheet -> {name: DataFrame}. cached on (path, sig) so an edited/replaced file reloads.

    sig is (mtime_ns, size); it is part of the cache key on purpose -- overwrite the .xlsx and the
    old sheets must not be served from cache.
    """
    return pd.read_excel(path, sheet_name=None, **_engine())


def newest_xlsx(folder: str):
    """the newest real .xlsx under `folder` (recursively), plus the full list. Excel's ~$ lock
    files are skipped -- they are 165-byte temp files, not workbooks."""
    p = pathlib.Path(folder).expanduser()
    if not p.exists():
        return None, []
    files = [f for f in p.rglob("*.xlsx") if not f.name.startswith("~$")]
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return (files[0] if files else None), files


def clearml_xlsx(task_id: str):
    """download the run's .xlsx artifact off ClearML (GCP-backed) and return its local path.

    the model/project pickers above are unchanged; only the artifact we pull is different -- the
    engine now uploads one workbook, not a pile of CSVs. we look at the artifact NAMES first and
    fetch only the workbook, so we never drag down the big scored-table parquet to find it.
    """
    from clearml import Task
    t = Task.get_task(task_id=task_id)
    arts = t.artifacts or {}
    names = list(arts)
    # prefer an obvious workbook: name mentions xlsx/backtest/workbook. fall back to trying each.
    ranked = sorted(names, key=lambda n: ("xlsx" not in n.lower(),
                                          "backtest" not in n.lower(), n))
    for n in ranked:
        try:
            p = arts[n].get_local_copy()
        except Exception:
            p = None
        if p and str(p).lower().endswith((".xlsx", ".xlsm")):
            return p, names
    return None, names


def sheet(wb: dict, needle: str):
    """a sheet by fuzzy name, so the numeric prefixes (04_Metrics) are not load-bearing."""
    for k in wb:
        if needle.lower() in str(k).lower():
            return wb[k]
    return None


# ============================== formatting =================================
def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def money(x):
    """COMPACT Indian format, so a tile never overflows: ₹-1.22 Cr, ₹9.58 L, ₹14.6 K, ₹-1,679.
    the full rupee figure is still shown exactly — on hover (tile tooltip) and in the tables."""
    v = _num(x)
    if v is None:
        return "—"
    a = abs(v)
    if a >= 1e7:
        return f"₹{v / 1e7:,.2f} Cr"
    if a >= 1e5:
        return f"₹{v / 1e5:,.2f} L"
    if a >= 1e4:
        return f"₹{v / 1e3:,.1f} K"
    return f"₹{v:,.0f}"


def money_exact(x):
    """the full rupee figure, for tooltips and anywhere there is room for it."""
    v = _num(x)
    return "—" if v is None else f"₹{v:,.0f}"


def pct(x, dp=2):
    v = _num(x)
    return "—" if v is None else f"{v:,.{dp}f}%"


def ratio(x, dp=2):
    v = _num(x)
    return "—" if v is None else f"{v:,.{dp}f}"


def cnt(x):
    v = _num(x)
    return "—" if v is None else f"{v:,.0f}"


def frac_pct(x, dp=1):
    """a 0..1 fraction shown as a percent (win_rate is stored as 0.177)."""
    v = _num(x)
    return "—" if v is None else f"{v * 100:,.{dp}f}%"


def smart_fmt(name: str, x):
    """best-effort unit from the metric NAME, for the big reference tables."""
    v = _num(x)
    if v is None:
        return "—"
    n = str(name).lower()
    if n in ("win_rate", "win_rate_gross"):
        return f"{v * 100:,.1f}%"
    if n.endswith("pct") or "pct" in n:
        return f"{v:,.2f}%"
    if n.endswith("points") or n in ("avg_points_per_lot", "breakeven_points_per_lot",
                                     "gross_pnl_points", "avg_delay_points"):
        return f"{v:,.2f}"
    if n in ("profit_factor", "profit_factor_gross", "edge_ratio", "calmar", "sharpe", "sortino",
             "sharpe_gross", "avg_delay_atr", "avg_delay_bars", "avg_holding_bars",
             "avg_lots_in_market", "years", "sessions_per_year", "annualisation_factor",
             "sharpe_gross"):
        return f"{v:,.2f}"
    if n.startswith("n_") or n in ("total_lots", "peak_lots", "n_drawdowns", "shadow_trades",
                                   "n_trading_days"):
        return f"{v:,.0f}"
    if n == "avg_holding_minutes":
        return f"{v:,.1f}"
    return f"₹{v:,.0f}"       # default: rupees


def fmt_table(df: pd.DataFrame, ver_cols=("V1", "V2", "V3")) -> pd.DataFrame:
    """format the V1/V2/V3 columns of a metric sheet for display."""
    d = df.copy()
    key = "metric" if "metric" in d.columns else d.columns[0]
    for vc in ver_cols:
        if vc in d.columns:
            d[vc] = [smart_fmt(m, v) for m, v in zip(d[key], d[vc])]
    return d


# ============================== metric lookup ==============================
def metric_index(metrics_df: pd.DataFrame) -> dict:
    """(block, metric) -> row. 04_Metrics carries CORE, OPTIONAL A 1-lot, OPTIONAL B peak."""
    idx = {}
    if metrics_df is None:
        return idx
    for _, r in metrics_df.iterrows():
        idx[(str(r.get("block")), str(r.get("metric")))] = r
    return idx


def mget(idx: dict, name: str, ver: str, block: str = "CORE"):
    r = idx.get((block, name))
    if r is None:
        return float("nan")
    try:
        return float(r[ver])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def tiles(specs, ncol=4):
    """draw a grid of st.metric tiles. specs = [(label, value_str[, exact_tooltip]), ...].

    FOUR per row, not six -- six squeezed a rupee figure into ~120px and streamlit clipped it to
    "₹-12…". the optional third item is the exact value, shown as the tile's hover tooltip."""
    for i in range(0, len(specs), ncol):
        row = specs[i:i + ncol]
        cols = st.columns(len(row))
        for c, spec in zip(cols, row):
            label, value = spec[0], spec[1]
            helptext = spec[2] if len(spec) > 2 else None
            c.metric(label, value, help=helptext)


# ============================== source =====================================
src = st.radio("Where is the workbook?",
               ["A folder on this machine", "ClearML (a trained model)"], horizontal=True)

xlsx_path, report_txt = None, None

if src.startswith("A folder"):
    folder = st.text_input("Folder", value=DEFAULT_LOCAL,
                           help="the folder your local backtest run wrote the .xlsx into. the "
                                "newest workbook under it is used.")
    xlsx_path, found = newest_xlsx(folder)
    if not xlsx_path:
        st.info(f"no .xlsx found under {folder}", icon="📂")
        st.stop()
    # the text report, if the run wrote one next to the workbook
    rt = pathlib.Path(xlsx_path).with_name("full_report.txt")
    report_txt = str(rt) if rt.exists() else None
    st.caption(f"reading **{pathlib.Path(xlsx_path).name}**"
               + (f"  ·  {len(found)} workbook(s) in this folder" if len(found) > 1 else ""))
else:
    col_p, col_m = st.columns([1, 2])
    with col_p:
        project = project_picker(key="bt_project")
    with col_m:
        model = model_picker(key="bt_model", project=project)
    if model is None:
        st.stop()
    kids = children_of(project, model["id"])
    task = kids.get("scored_oos")
    if task is None:
        st.info("this model has no OOS scoring task yet — queued, still running, or the step "
                "failed. check it in ClearML.", icon="⏳")
        st.stop()
    st.caption(f"reading **{task['name']}**  ({task['status']})  ·  out-of-sample")
    xlsx_path, names = clearml_xlsx(task["id"])
    if not xlsx_path:
        st.warning("that task has no backtest workbook (.xlsx) artifact yet. its ClearML console "
                   "says whether the backtest ran.", icon="⚠️")
        with st.expander("artifacts this task does have"):
            st.code("\n".join(names) or "(none)", language="text")
        st.stop()

# one read, cached on the file's own mtime+size
_st = pathlib.Path(xlsx_path).stat()
wb = load_workbook(str(xlsx_path), (_st.st_mtime_ns, _st.st_size))

metrics_df = sheet(wb, "Metrics")
summary_df = sheet(wb, "Summary")
config_df = sheet(wb, "Config")
idx = metric_index(metrics_df)

# ---- the two global controls: which version, which capital basis ----
cfg = dict(zip(config_df["setting"], config_df["value"])) if config_df is not None else {}
rule = {v: cfg.get(f"{v} rule", "") for v in ("V1", "V2", "V3")}

c1, c2 = st.columns([2, 1])
with c1:
    ver = st.radio("Version", ["V1", "V2", "V3"], horizontal=True,
                   format_func=lambda v: VERSION_LABELS.get(v, v))
with c2:
    basis_label = st.radio("Capital basis (optional metrics only)", list(BASIS_BLOCK), horizontal=True)
basis = BASIS_BLOCK[basis_label]
if rule.get(ver):
    st.caption(f"**{ver} rule:** {rule[ver]}")

TAB_OVERVIEW, TAB_METRICS, TAB_EQUITY, TAB_YM, TAB_TRADES, TAB_CHURN, TAB_DATA = st.tabs(
    ["Overview", "Full metrics", "Equity & risk", "Year / Month", "Trades",
     "Churn & missed", "Data & config"])


# ------------------------------------------------------------------ Overview
with TAB_OVERVIEW:
    st.subheader(f"Headline — {VERSION_LABELS.get(ver, ver)}")
    st.caption("CORE numbers — capital-independent. this is where the signal is judged.")
    def m_(name, ver_=ver, block="CORE"):
        v = mget(idx, name, ver_, block)
        return money(v), money_exact(v)      # (compact, exact-for-tooltip)

    tiles([
        ("Net PnL",            *m_("net_pnl")),
        ("Gross PnL",          *m_("gross_pnl")),
        ("Total charges",      *m_("total_charge")),
        ("Trades",             cnt(mget(idx, "n_trades", ver))),
        ("Win rate (net)",     frac_pct(mget(idx, "win_rate", ver))),
        ("Profit factor",      ratio(mget(idx, "profit_factor", ver))),
        ("Expectancy / trade", *m_("expectancy")),
        ("Avg points / lot",   ratio(mget(idx, "avg_points_per_lot", ver))),
        ("Edge ratio",         ratio(mget(idx, "edge_ratio", ver))),
        ("Max drawdown",       *m_("max_drawdown")),
        ("Churn rate",         pct(mget(idx, "churn_rate_pct", ver))),
        ("Avg hold (min)",     ratio(mget(idx, "avg_holding_minutes", ver), 0)),
    ])

    st.divider()
    st.subheader(f"Capital-dependent (optional) — basis {basis_label}")
    cap = mget(idx, "capital", ver, basis)
    st.caption(f"these divide by an ASSUMED capital of **{money(cap)}**. a straight NaN means the "
               f"net equity ended at or below zero, so the growth/ratio has no meaning on this basis.")
    tiles([
        ("Total return",  pct(mget(idx, "total_return_pct", ver, basis))),
        ("Gross return",  pct(mget(idx, "gross_return_pct", ver, basis))),
        ("CAGR (net)",    pct(mget(idx, "cagr_pct", ver, basis))),
        ("ARR",           pct(mget(idx, "arr_pct", ver, basis))),
        ("Max DD %",      pct(mget(idx, "max_drawdown_pct", ver, basis))),
        ("Sharpe",        ratio(mget(idx, "sharpe", ver, basis))),
        ("Sortino",       ratio(mget(idx, "sortino", ver, basis))),
        ("Calmar",        ratio(mget(idx, "calmar", ver, basis))),
    ], ncol=4)

    st.divider()
    st.subheader("The three versions, side by side")
    st.caption("same window, same 1-lot size, same costs — so the differences are the signal, "
               "not the sizing.")
    if summary_df is not None:
        show = [c for c in ("block", "metric", "what it is", "V1", "V2", "V3") if c in summary_df.columns]
        st.dataframe(fmt_table(summary_df)[show], width="stretch", hide_index=True, height=460)
    else:
        st.info("no Summary sheet in this workbook.", icon="ℹ️")


# ------------------------------------------------------------- Full metrics
with TAB_METRICS:
    st.caption("every metric, for every version. CORE first — then the two capital bases the "
               "OPTIONAL metrics are shown on.")
    if metrics_df is not None:
        cols = [c for c in ("metric", "what it is", "V1", "V2", "V3") if c in metrics_df.columns]
        for blk, note in [("CORE", "capital-independent — the honest ones"),
                          ("OPTIONAL A 1-lot", "return / risk on 1-lot notional capital"),
                          ("OPTIONAL B peak", "return / risk on peak-notional capital")]:
            part = metrics_df[metrics_df["block"] == blk]
            if not len(part):
                continue
            st.markdown(f"**{blk}** — {note}")
            st.dataframe(fmt_table(part)[cols], width="stretch", hide_index=True,
                         height=min(560, 40 + 35 * len(part)))
    else:
        st.info("no Metrics sheet in this workbook.", icon="ℹ️")


# ------------------------------------------------------------- Equity & risk
with TAB_EQUITY:
    eq = sheet(wb, "Daily_Equity")
    if eq is not None and f"{ver}_net" in eq.columns:
        eq = eq.copy()
        eq["date"] = pd.to_datetime(eq["date"], errors="coerce")
        eq = eq.dropna(subset=["date"]).set_index("date")

        st.subheader("Equity curve (net, after charges)")
        st.caption("the shape matters more than the endpoint — a straight climb is a strategy, "
                   "one vertical jump is one lucky day.")
        overlay = st.checkbox("overlay all three versions", value=False)
        net_cols = [c for c in ("V1_net", "V2_net", "V3_net") if c in eq.columns]
        st.line_chart(eq[net_cols] if overlay else eq[[f"{ver}_net"]], height=320)

        # underwater curve, computed from the selected version's net equity
        s = eq[f"{ver}_net"]
        underwater = s - s.cummax()
        st.subheader("Underwater (drawdown, ₹)")
        st.caption("distance below the running peak. it sits at zero only when the curve is making "
                   "new highs.")
        st.area_chart(underwater.rename("drawdown"), height=200)

        if "close" in eq.columns:
            st.subheader("NIFTY futures close (context)")
            st.line_chart(eq[["close"]], height=180)
    else:
        st.info("no Daily_Equity sheet in this workbook.", icon="ℹ️")

    st.divider()
    st.subheader(f"Risk — basis {basis_label}")
    tiles([
        ("Max drawdown ₹",   money(mget(idx, "max_drawdown", ver))),
        ("Max DD %",         pct(mget(idx, "max_drawdown_pct", ver, basis))),
        ("Avg drawdown ₹",   money(mget(idx, "avg_drawdown", ver))),
        ("DD episodes",      cnt(mget(idx, "n_drawdowns", ver))),
        ("Time underwater",  pct(mget(idx, "pct_time_in_dd", ver))),
        ("Calmar",           ratio(mget(idx, "calmar", ver, basis))),
    ])


# ------------------------------------------------------------- Year / Month
with TAB_YM:
    yr = sheet(wb, "Yearly")
    if yr is not None and "version" in yr.columns:
        y = yr[yr["version"] == ver].drop(columns=["version"])
        st.subheader("Year by year")
        st.dataframe(y, width="stretch", hide_index=True)
        if {"year", "net_pnl"} <= set(y.columns):
            st.caption("net PnL by year (after charges)")
            st.bar_chart(y.set_index("year")["net_pnl"], height=240)
    else:
        st.info("no Yearly sheet in this workbook.", icon="ℹ️")

    mo = sheet(wb, "Monthly")
    if mo is not None and "version" in mo.columns:
        m = mo[mo["version"] == ver]
        if {"month", "net_pnl"} <= set(m.columns):
            st.subheader("Monthly net PnL")
            mm = m.copy()
            mm["month"] = mm["month"].astype(str)
            st.bar_chart(mm.set_index("month")["net_pnl"], height=240)


# ------------------------------------------------------------------ Trades
with TAB_TRADES:
    st.caption("trade quality for the selected version.")
    tiles([
        ("Win rate (gross)", frac_pct(mget(idx, "win_rate_gross", ver))),
        ("Win rate (net)",   frac_pct(mget(idx, "win_rate", ver))),
        ("Avg win",          money(mget(idx, "avg_win", ver))),
        ("Avg loss",         money(mget(idx, "avg_loss", ver))),
        ("Avg MFE (pts)",    ratio(mget(idx, "avg_mfe_points", ver))),
        ("Avg MAE (pts)",    ratio(mget(idx, "avg_mae_points", ver))),
        ("MFE capture",      pct(mget(idx, "mfe_capture_pct", ver))),
        ("Ever in profit >cost", pct(mget(idx, "pct_trades_mfe_over_breakeven", ver))),
    ], ncol=4)

    tb = sheet(wb, "Tradebook")
    if tb is not None and "version" in tb.columns:
        t = tb[tb["version"] == ver]
        st.subheader(f"Tradebook ({len(t):,})")
        cols = [c for c in ("signal_time", "entry_signal", "entry_true_label", "is_false_trade",
                            "exit_true_label", "entry_time", "entry_price", "exit_time",
                            "exit_price", "exit_reason", "holding_minutes", "pnl_points",
                            "mae_points", "mfe_points", "pnl_money", "cum_net_pnl")
                if c in t.columns]
        if len(t) > TRADES_SHOWN:
            st.caption(f"showing the first {TRADES_SHOWN:,} — open the workbook for all {len(t):,}.")
        st.dataframe(t[cols].head(TRADES_SHOWN), width="stretch", height=380, hide_index=True)
    else:
        st.info("no Tradebook sheet in this workbook.", icon="ℹ️")


# ------------------------------------------------------------- Churn & missed
with TAB_CHURN:
    st.caption("what the over-trading and the mistakes cost — all CORE, all in rupees unless noted.")
    tiles([
        ("False trades",       cnt(mget(idx, "n_false_trades", ver))),
        ("Churn rate",         pct(mget(idx, "churn_rate_pct", ver))),
        ("Churn cost",         money(mget(idx, "churn_cost", ver))),
        ("Missed opportunity", money(mget(idx, "missed_opportunity_cost", ver))),
        ("Wrong direction",    money(mget(idx, "wrong_direction_cost", ver))),
        ("Delay cost",         money(mget(idx, "delay_cost", ver))),
        ("Total error cost",   money(mget(idx, "total_error_cost", ver))),
        ("Shadow trades",      cnt(mget(idx, "shadow_trades", ver))),
    ], ncol=4)

    ms = sheet(wb, "Missed")
    if ms is not None and "version" in ms.columns:
        m = ms[ms["version"] == ver]
        st.subheader(f"Missed trades ({len(m):,})")
        st.caption("ideal entries the model did not take — profit left on the table.")
        st.dataframe(m.drop(columns=["version"]).head(TRADES_SHOWN),
                     width="stretch", height=320, hide_index=True)
    else:
        st.info("no Missed_Trades sheet in this workbook.", icon="ℹ️")


# ------------------------------------------------------------- Data & config
with TAB_DATA:
    col_a, col_b = st.columns(2)
    with col_a:
        if config_df is not None:
            st.subheader("Config")
            st.caption("every setting this run used.")
            st.dataframe(config_df, width="stretch", hide_index=True, height=460)
    with col_b:
        dq = sheet(wb, "Data_Quality")
        if dq is not None:
            st.subheader("Data quality")
            st.caption("coverage, OHLC sanity, prediction-level churn.")
            st.dataframe(dq, width="stretch", hide_index=True, height=460)

    readme = sheet(wb, "README")
    if readme is not None:
        with st.expander("how to read this workbook (from the file itself)"):
            st.dataframe(readme, width="stretch", hide_index=True)

    if report_txt:
        with st.expander("the full text report"):
            try:
                st.code(pathlib.Path(report_txt).read_text(encoding="utf-8"), language="text")
            except Exception as exc:
                st.warning(f"could not read it: {exc}")

    with st.expander("workbook file"):
        st.code(str(xlsx_path), language="text")
        st.code("sheets: " + ", ".join(wb), language="text")
