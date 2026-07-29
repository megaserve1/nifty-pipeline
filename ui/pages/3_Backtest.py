"""ui/pages/3_Backtest.py -- read a backtest run and show it. READ ONLY, triggers nothing.

WHERE THE NUMBERS COME FROM
    the backtest step writes five files and export_scored_tables uploads each as a task artifact:
        backtest_metrics          net pnl, win rate, profit factor, drawdown, sharpe ...
        backtest_equity_curve     timestamp, equity, close, signal
        backtest_trades           one row per trade
        backtest_monthly_returns  month, net pnl
        backtest_report           the plain-text report
    so this page can open ANY model's backtest, not just the last one.

TWO SOURCES
    ClearML   pick a scored_tables / scored_oos task and it pulls that task's artifacts.
    Folder    point at a local backtest output directory (what a --backtest run writes on disk).
"""
import pathlib
import sys

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Backtest", page_icon="📉", layout="wide")

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import config as C   # noqa: E402

st.title("Backtest results")
st.caption("what the signals would have done on price. read only — this page starts nothing.")


# ---------------------------------------------------------------- loading
@st.cache_data(show_spinner="fetching from ClearML …", ttl=300)
def clearml_runs():
    """scored_tables / scored_oos tasks that actually carry backtest artifacts."""
    from clearml import Task
    out = []
    for t in Task.get_tasks(project_name=C.CLEARML_PROJECT) or []:
        name = t.name or ""
        if not (name.startswith("scored_tables") or name.startswith("scored_oos")):
            continue
        if not any(k.startswith("backtest_") for k in (t.artifacts or {})):
            continue
        out.append((f"{name}   ({t.id[:8]})", t.id))
    return out


@st.cache_data(show_spinner="downloading artifacts …", ttl=300)
def clearml_files(task_id: str) -> dict:
    """artifact stem -> local copy path, for the backtest_* artifacts of one task."""
    from clearml import Task
    t = Task.get_task(task_id=task_id)
    got = {}
    for name, art in (t.artifacts or {}).items():
        if name.startswith("backtest_"):
            got[name.replace("backtest_", "")] = art.get_local_copy()
    return got


def folder_files(folder: str) -> dict:
    """the CSVs a local --backtest run wrote (it nests them in a timestamped subfolder)."""
    p = pathlib.Path(folder).expanduser()
    if not p.exists():
        return {}
    runs = sorted(p.glob("backtest_*"), key=lambda d: d.name)
    base = runs[-1] if runs else p
    return {f.stem: str(f) for f in list(base.glob("*.csv")) + list(base.glob("*.txt"))}


def read_csv(files: dict, key: str):
    path = files.get(key)
    if not path:
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"could not read {key}: {exc}")
        return None


def money(x):
    try:
        return f"₹{float(x):,.0f}"
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------- source
src = st.radio("Where from", ["ClearML", "A folder on this machine"], horizontal=True)

files = {}
if src == "ClearML":
    try:
        runs = clearml_runs()
    except Exception as exc:
        st.error(f"cannot reach ClearML: {exc}")
        st.stop()
    if not runs:
        st.info("no backtest results in ClearML yet. a scored_tables / scored_oos task carries "
                "them once it has run with a backtest script and a price file.", icon="⏳")
        st.stop()
    label = st.selectbox("Run", [r[0] for r in runs])
    files = clearml_files(dict(runs)[label])
else:
    folder = st.text_input("Folder", value="/tmp/bt_check",
                           help="the --out folder of a local run; it holds a backtest/ subfolder")
    files = folder_files(folder)
    if not files:
        st.info(f"nothing found under {folder}", icon="📂")
        st.stop()

if not files:
    st.stop()

# ---------------------------------------------------------------- headline
metrics = read_csv(files, "metrics")
if metrics is not None and {"metric", "value"} <= set(metrics.columns):
    m = dict(zip(metrics["metric"], metrics["value"]))
    c = st.columns(6)
    c[0].metric("Net PnL", money(m.get("net_pnl")))
    c[1].metric("Trades", f"{float(m.get('n_trades', 0)):,.0f}")
    try:
        c[2].metric("Win rate", f"{float(m.get('win_rate')) * 100:.1f}%")
    except (TypeError, ValueError):
        c[2].metric("Win rate", "—")
    try:
        c[3].metric("Profit factor", f"{float(m.get('profit_factor')):.2f}")
    except (TypeError, ValueError):
        c[3].metric("Profit factor", "—")
    c[4].metric("Max drawdown", money(m.get("max_drawdown")))
    try:
        c[5].metric("Sharpe", f"{float(m.get('sharpe')):.2f}")
    except (TypeError, ValueError):
        c[5].metric("Sharpe", "—")

    net = float(m.get("net_pnl", 0) or 0)
    st.caption("🟢 net positive over this window" if net > 0
               else "🔴 net negative over this window — the strategy lost money here")

# ---------------------------------------------------------------- equity
eq = read_csv(files, "equity_curve")
if eq is not None and "equity" in eq.columns:
    st.subheader("Equity curve")
    st.caption("the shape matters more than the endpoint — a straight climb is a strategy, "
               "one vertical jump is one lucky day.")
    if "timestamp" in eq.columns:
        eq["timestamp"] = pd.to_datetime(eq["timestamp"], errors="coerce")
        st.line_chart(eq.set_index("timestamp")["equity"], height=320)
    else:
        st.line_chart(eq["equity"], height=320)

# ---------------------------------------------------------------- monthly
mo = read_csv(files, "monthly_returns")
if mo is not None and len(mo.columns) >= 2:
    st.subheader("Monthly net PnL")
    st.bar_chart(mo.set_index(mo.columns[0])[mo.columns[1]], height=260)

# ---------------------------------------------------------------- trades
tr = read_csv(files, "trades")
if tr is not None and len(tr):
    st.subheader(f"Trades ({len(tr):,})")
    if "tier" in tr.columns:
        tiers = sorted(tr["tier"].dropna().unique().tolist())
        keep = st.multiselect("Tier", tiers, default=tiers)
        tr = tr[tr["tier"].isin(keep)]
        if "pnl_money" in tr.columns and len(tr):
            st.caption("per tier — this is where you see which size actually pays")
            by = tr.groupby("tier")["pnl_money"].agg(["count", "sum", "mean"]).round(0)
            by.columns = ["trades", "net pnl", "avg pnl"]
            st.dataframe(by, width="stretch")
    st.dataframe(tr, width="stretch", height=340, hide_index=True)

# ---------------------------------------------------------------- raw report
if "report" in files:
    with st.expander("the full text report"):
        try:
            st.code(pathlib.Path(files["report"]).read_text(), language="text")
        except Exception as exc:
            st.warning(f"could not read the report: {exc}")
