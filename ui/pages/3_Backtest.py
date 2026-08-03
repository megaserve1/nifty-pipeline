"""ui/pages/3_Backtest.py -- pick a PROJECT and a MODEL, see what its signals would have earned.

READ ONLY. this page starts nothing and changes nothing.

PICK THE MODEL, NOT THE FILE. after an overnight run there are two or three models, each with its
own scored table, backtest, SHAP and deepchecks. choosing a table by name means working out which
model it came from; choosing the model means everything below it is the right thing by construction.

OOS ONLY. the test-split backtest is deliberately not shown -- under bundle_random that table is
thousands of scattered 15-minute bundles (13,088 breaks in continuity on v7), so its equity curve
is stitched across thousands of time gaps and cannot be read as a trading result.

TWO BACKTEST LAYOUTS are understood, because old runs are still worth opening:
  NEW (scripts/backtest_single.py, 3 versions) -- per-version files plus a comparison table
      V1  ENTRY_SUPER -> exit on EXIT_SUB or EXIT_SUPER   is the SUPER entry any good early?
      V2  ENTRY_SUB   -> exit on EXIT_SUB or EXIT_SUPER   does the SUB entry work on its own?
      V3  ENTRY_SUPER -> exit on EXIT_SUPER only          is the SUPER class self-consistent?
  OLD (one strategy, flat files) -- metrics / equity_curve / trades / monthly_returns / report
"""
import pathlib
import sys

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Backtest", page_icon="📉", layout="wide")


# the password gate. FIRST thing after set_page_config -- streamlit serves this file at its
# own URL, so a gate on Home.py alone would not protect it.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _auth import require_auth   # noqa: E402
require_auth()

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _shared import model_picker, children_of, artifacts_named, project_picker   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import config as C   # noqa: E402

st.title("Backtest results")
st.caption("out-of-sample only — what the signals would have done on days the model has never "
           "seen. read only; this page starts nothing.")

WANTED = ("yearly_returns", "monthly_returns", "equity_curve", "metrics", "trades")

# how many points to draw on a line chart. the equity curve is per-MINUTE (274,606 rows on the v7
# test table) and a browser cannot draw that -- the tab freezes. a 320px chart has ~1,300 pixels
# of width, so 3,000 points is already more resolution than the screen can show.
PLOT_POINTS = 3000
TRADES_SHOWN = 2000        # the table is virtualised but the payload still crosses the wire


def read_csv(path):
    if not path:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def money(x):
    try:
        return f"₹{float(x):,.0f}"
    except (TypeError, ValueError):
        return "—"


def collect(pairs) -> dict:
    """(key, path) pairs -> {'metrics': path, 'equity_curve': path, ...}

    matched on the END of the key, so it works for both layouts: the new suite names a file
    V1_SUPER_entry_quality_metrics, the old one just calls it metrics.
    """
    out = {}
    for k, p in pairs:
        for want in WANTED:
            if k.endswith(want):
                out.setdefault(want, p)
                break
    return out


def folder_files(folder: str) -> dict:
    """the files a LOCAL --backtest run wrote. it nests them in a timestamped folder, and the
    3-version suite adds one subfolder per version, so this walks the tree."""
    p = pathlib.Path(folder).expanduser()
    if not p.exists():
        return {}
    runs = sorted(p.glob("backtest_*"), key=lambda d: d.name)
    base = runs[-1] if runs else p
    out = {}
    for f in list(base.rglob("*.csv")) + list(base.rglob("*.txt")):
        out[str(f.relative_to(base).with_suffix("")).replace("/", "_")] = str(f)
    return out


def show_one(byname: dict, heading: str = ""):
    """render one strategy: the tiles, the curve, the years, the months, the trades."""
    m = read_csv(byname.get("metrics"))
    if m is not None and {"metric", "value"} <= set(m.columns):
        d = dict(zip(m["metric"], m["value"]))

        def num(k, fmt="{:,.2f}"):
            try:
                return fmt.format(float(d[k]))
            except (KeyError, TypeError, ValueError):
                return "—"

        c = st.columns(6)
        c[0].metric("Net PnL", money(d.get("net_pnl")))
        c[1].metric("Total return", f"{num('total_return_pct')} %")
        c[2].metric("CAGR", f"{num('cagr_pct')} %")
        c[3].metric("Max drawdown", f"{num('max_drawdown_pct')} %")
        c[4].metric("Trades", num("n_trades", "{:,.0f}"))
        c[5].metric("Profit factor", num("profit_factor"))

        c2 = st.columns(4)
        try:
            c2[0].metric("Win rate", f"{float(d['win_rate']) * 100:.1f} %")
        except Exception:
            c2[0].metric("Win rate", "—")
        c2[1].metric("Sharpe", num("sharpe"))
        c2[2].metric("Expectancy / trade", money(d.get("expectancy")))
        # c2[3].metric("vs benchmark", f"{num('vs_benchmark_pp')} pp")

        try:
            net = float(d.get("net_pnl", 0))
            # vsb = float(d.get("vs_benchmark_pp", 0))
            # st.caption(("🟢 net positive" if net > 0 else "🔴 net negative") +
            #            (" and beat buy & hold" if vsb > 0 else " and behind buy & hold"))
        except Exception:
            pass
    elif m is not None:
        st.dataframe(m, width="stretch", hide_index=True)

    eq = read_csv(byname.get("equity_curve"))
    if eq is not None and "equity" in eq.columns:
        st.subheader("Equity curve")
        st.caption("the shape matters more than the endpoint — a straight climb is a strategy, "
                   "one vertical jump is one lucky day.")
        # DOWNSAMPLE BEFORE PLOTTING. the curve is one row per MINUTE -- 274,606 of them on the
        # v7 test table. st.line_chart hands every point to vega-lite in the browser, and the tab
        # locks up ("page unresponsive"). a chart 320px tall cannot show more points than it has
        # pixels, so plotting all of them buys nothing and costs the page.
        # every Nth row PLUS the final one: the endpoint is the number people read off the chart
        # and striding can miss it.
        n = len(eq)
        if n > PLOT_POINTS:
            step = n // PLOT_POINTS + 1
            eq_plot = pd.concat([eq.iloc[::step], eq.iloc[[-1]]]).drop_duplicates()
            st.caption(f"drawn from {len(eq_plot):,} of {n:,} points (every {step}th minute) — "
                       f"the CSV artifact has all of them.")
        else:
            eq_plot = eq
        if "timestamp" in eq_plot.columns:
            eq_plot = eq_plot.copy()
            eq_plot["timestamp"] = pd.to_datetime(eq_plot["timestamp"], errors="coerce")
            st.line_chart(eq_plot.set_index("timestamp")["equity"], height=320)
        else:
            st.line_chart(eq_plot["equity"].reset_index(drop=True), height=320)

    yr = read_csv(byname.get("yearly_returns"))
    if yr is not None and "year" in yr.columns:
        st.subheader("Year by year")
        show = [c for c in ("year", "trades", "win_rate_pct", "net_pnl", "return_pct",
                            "bench_pct", "alpha_pp", "max_dd_pct") if c in yr.columns]
        st.dataframe(yr[show], width="stretch", hide_index=True)
        if "return_pct" in yr.columns:
            st.bar_chart(yr.set_index("year")["return_pct"], height=240)

    mo = read_csv(byname.get("monthly_returns"))
    if mo is not None and len(mo.columns) >= 2:
        st.subheader("Monthly net PnL")
        st.bar_chart(mo.set_index(mo.columns[0])[mo.columns[1]], height=240)

    tr = read_csv(byname.get("trades"))
    if tr is not None and len(tr):
        st.subheader(f"Trades ({len(tr):,})")
        if len(tr) > TRADES_SHOWN:
            st.caption(f"showing the first {TRADES_SHOWN:,} — download the artifact for all "
                       f"{len(tr):,}.")
        st.dataframe(tr.head(TRADES_SHOWN), width="stretch", height=320, hide_index=True)

    if not byname:
        st.info("no readable files for this one.", icon="ℹ️")


# ---------------------------------------------------------------- where from
src = st.radio("Source", ["ClearML (a trained model)", "A folder on this machine"], horizontal=True)

files = {}
if src.startswith("ClearML"):
    col_p, col_m = st.columns([1, 2])
    with col_p:
        project = project_picker(key="bt_project")
    with col_m:
        model = model_picker(key="bt_model", project=project)
    if model is None:
        st.stop()

    kids = children_of(project, model["id"])
    # OOS FIRST, ALWAYS. the backtest runs inside a scoring step, and there are two of them:
    #   score_oos      one continuous forward period -> a real, tradeable equity curve
    #   scored_tables  the TEST SPLIT, which under bundle_random is thousands of scattered
    #                  15-min bundles (measured: 13,088 breaks in continuity, largest gap 6 days)
    # this used to prefer scored_tables, which meant the honest backtest was the one you could
    # not see. config.BACKTEST_ON_TEST_TABLE is False now, so scored_tables carries no backtest
    # at all -- preferring it would have shown "no backtest files" and stopped there.
    # OOS ONLY. the other scoring task (scored_tables) covers the TEST SPLIT, and under
    # bundle_random that is thousands of scattered 15-minute bundles -- measured on v7: 274,605
    # rows with 13,088 BREAKS in continuity, largest gap 6 days 19 hours. a backtest walks a
    # table row by row, so an equity curve stitched across 13,088 time jumps is a number nobody
    # could have traded. showing it next to the OOS one only invites it to be quoted.
    # config.BACKTEST_ON_TEST_TABLE also stops that backtest being produced at all.
    task = kids.get("scored_oos")
    if task is None:
        if "scored_tables" in kids:
            st.info("This model has been scored on the test split, but **not on the OOS set** — "
                    "and only OOS is shown here.\n\nThe test split is thousands of scattered "
                    "15-minute bundles, so a backtest over it walks across thousands of time "
                    "gaps. It is a metric, not a tradeable result.", icon="📊")
            st.code(f"final_venv/bin/python scripts/queue_oos.py --version "
                    f"{model['name'].split()[-1]} --oos_tag 2025_2026", language="bash")
        else:
            st.info("this model has no OOS scoring task yet — queued, still running, or the step "
                    "failed. check it in ClearML.", icon="⏳")
        st.stop()
    st.caption(f"reading **{task['name']}**  ({task['status']})  ·  out-of-sample")

    files, dead = artifacts_named(task["id"], "backtest_", report=True)
    if dead and not files:
        # clearml keeps the artifact record forever; the bucket does not.
        st.error(f"this run advertises {len(dead)} backtest files but **the bytes are gone from "
                 f"the bucket** — they were deleted, so there is nothing left to draw. re-run the "
                 f"scoring step for this model to regenerate them.", icon="🗑️")
        with st.expander("which files ClearML still lists"):
            st.code("\n".join(sorted(dead)), language="text")
        st.stop()
    if dead:
        st.warning(f"{len(dead)} of this run's files were deleted from the bucket: "
                   f"{', '.join(sorted(dead))}", icon="⚠️")
    if not files:
        st.warning("that task produced the table but no backtest files. usually the price file was "
                   "unreachable on the worker, or BACKTEST_SCRIPT was not set. its ClearML console "
                   "says which.", icon="⚠️")
        st.stop()
else:
    folder = st.text_input("Folder", value="/tmp/bt_check",
                           help="the --out folder of a local run")
    files = folder_files(folder)
    if not files:
        st.info(f"nothing found under {folder}", icon="📂")
        st.stop()

# ---------------------------------------------------------------- comparison
cmp_df = read_csv(files.get("comparison"))
if cmp_df is not None and len(cmp_df.columns) > 1:
    st.subheader("The three versions, side by side")
    st.caption("same window, same 1-lot size, same costs — so the differences are the signal, "
               "not the sizing.")
    st.dataframe(cmp_df.set_index(cmp_df.columns[0]), width="stretch")

# ---------------------------------------------------------------- the results
# a key like V1_SUPER_entry_quality_metrics means the new 3-version suite. anything else is an old
# single-strategy run, and those files sit flat with no version prefix at all.
versions = sorted({k.split("_")[0] for k in files
                   if len(k) > 1 and k[0] == "V" and k[1].isdigit()})

if versions:
    pick = st.radio("Version", versions, horizontal=True,
                    format_func=lambda v: {"V1": "V1 · SUPER entry", "V2": "V2 · SUB entry",
                                           "V3": "V3 · SUPER class"}.get(v, v))
    show_one(collect((k, p) for k, p in files.items() if k.startswith(pick + "_")))
else:
    st.caption("this is an older single-strategy run — one result, no V1/V2/V3 split.")
    show_one(collect(files.items()))

# ---------------------------------------------------------------- raw report
report = files.get("full_report") or files.get("report")
if report:
    with st.expander("the full text report"):
        try:
            st.code(pathlib.Path(report).read_text(), language="text")
        except Exception as exc:
            st.warning(f"could not read it: {exc}")

with st.expander("every file in this run"):
    st.code("\n".join(f"{k:<40} {p}" for k, p in sorted(files.items())), language="text")
