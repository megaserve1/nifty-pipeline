"""
==============================================================================
 SIGNAL BACKTEST  -  3-VERSION SIGNAL QUALITY SUITE  -  ONE self-contained file
==============================================================================
 SIGNAL  = an Excel (.xlsx) file  (signal sheet auto-detected)
 PRICE   = a Parquet (.parquet) file with NIFTY futures OHLC
 RESULTS = CSV + TXT files written into a timestamped output folder

 RUN (from cmd):
     python backtest_single.py
 or override the three paths on the command line:
     python backtest_single.py  <signal.xlsx>  <price.parquet>  <output_dir>

 ---------------------------------------------------------------------------
 THE THREE VERSIONS  (SMALL family is NOT traded in any of them)
 ---------------------------------------------------------------------------
   VERSION 1  SUPER entry quality  : enter ENTRY_SUPER, exit on EXIT_SUB
                                     OR EXIT_SUPER (whichever fires first)
   VERSION 2  SUB   entry quality  : enter ENTRY_SUB,   exit on EXIT_SUB
                                     OR EXIT_SUPER (whichever fires first)
   VERSION 3  SUPER class quality  : enter ENTRY_SUPER, exit on EXIT_SUPER only

 Every version trades exactly 1 LOT = 65 quantity (equal sizing, so the three
 versions are directly comparable).

 Economics (all in the CONFIG block): lot size 65, charge 2000 per lot round
 trip, entry on the NEXT bar open, fire only on the first bar of a signal
 (on change).  CORE ENGINE LOGIC IS UNCHANGED - only the entry/exit map that
 each version is allowed to act on differs.

 Needs: pandas, numpy, pyarrow, and openpyxl OR python-calamine (for .xlsx).
==============================================================================
"""
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================== CONFIG =====================================
HERE = os.path.dirname(os.path.abspath(__file__))

# 1) ATTACH YOUR DATA
SIGNAL_FILE = r"C:\Users\Admin\Downloads\evaluation_catboost_v4.3.xlsx"
PRICE_FILE  = r"C:\Users\Admin\Downloads\09012015-31012025_dataset\Merged_file\Nifty_futures_base_data_09102015-24022026.parquet"
SIGNAL_SHEET = None        # None = auto-detect the signal sheet in the workbook

# 2) WHERE TO SAVE RESULTS (CSV files go into a timestamped folder here)
OUTPUT_DIR = r"C:\Users\Admin\Downloads\backtest_results"

# 3) TRADE ECONOMICS
LOT_SIZE       = 65        # index units per 1 lot  (1 lot = 65 quantity)
LOTS_PER_TRADE = 1         # EVERY tier trades exactly this many lots
CHARGE_PER_LOT = 2000.0    # round-trip ("rotation") charge per lot, in rupees
SLIPPAGE       = 0.0       # points lost on each fill

# 4) CAPITAL BASE used for total_return_% / CAGR / drawdown_%
#    None  -> auto = LOTS_PER_TRADE * LOT_SIZE * first close  (1-lot notional)
#    or set a number, e.g. 120000.0 to use the per-lot SPAN margin instead.
INITIAL_CAPITAL = None
# ===========================================================================

# ---------------------------- THE 3 VERSIONS -------------------------------
# qty   : which ENTRY signals this version is allowed to take, and how many lots
# exits : which EXIT signals this version honours, and which tiers each closes
_Q = LOTS_PER_TRADE
VERSIONS = [
    dict(key="V1", name="SUPER entry quality",
         desc="ENTRY_SUPER  ->  exit on EXIT_SUB or EXIT_SUPER (whichever comes first)",
         question="Is the SUPER entry signal any good when it is released early?",
         qty={"ENTRY_SUPER": _Q},
         exits={"EXIT_SUB": {"SUPER"}, "EXIT_SUPER": {"SUPER"}}),
    dict(key="V2", name="SUB entry quality",
         desc="ENTRY_SUB    ->  exit on EXIT_SUB or EXIT_SUPER (whichever comes first)",
         question="Is the SUB entry signal working on its own?",
         qty={"ENTRY_SUB": _Q},
         exits={"EXIT_SUB": {"SUB"}, "EXIT_SUPER": {"SUB"}}),
    dict(key="V3", name="SUPER class quality",
         desc="ENTRY_SUPER  ->  exit on EXIT_SUPER only",
         question="Is the SUPER class self-consistent (SUPER in, SUPER out)?",
         qty={"ENTRY_SUPER": _Q},
         exits={"EXIT_SUPER": {"SUPER"}}),
]

ALL_ENTRIES = {"ENTRY_SMALL", "ENTRY_SUB", "ENTRY_SUPER"}
ALL_EXITS = {"EXIT_SMALL", "EXIT_SUB", "EXIT_SUPER"}
ENTRY_TIER = {"ENTRY_SMALL": "SMALL", "ENTRY_SUB": "SUB", "ENTRY_SUPER": "SUPER"}
TRADING_DAYS = 252
W = 96                     # report width

TIME_COLS = ["timestamp", "datetime", "ts", "time", "date"]
SIG_COLS = ["predicted_label", "primary_label", "signal", "prediction", "y_pred", "label"]
OPEN_COLS = ["Nifty_Futures_Open", "open", "Open"]
HIGH_COLS = ["Nifty_Futures_High", "high", "High"]
LOW_COLS = ["Nifty_Futures_Low", "low", "Low"]
CLOSE_COLS = ["Nifty_Futures_Close", "close", "Close"]

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# fast Excel engine if available, else openpyxl
try:
    import python_calamine  # noqa: F401
    _XL = {"engine": "calamine"}
except ImportError:
    _XL = {}


# ============================ LOAD DATA ====================================

def _pick(cands, cols, what):
    for c in cands:
        if c in cols:
            return c
    raise SystemExit(f"ERROR: no {what} column found in {list(cols)[:12]}")


def _parse_time(s):
    """Datetime parse that will not silently mangle dd-mm-yyyy dates."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    a = pd.to_datetime(s, errors="coerce")
    b = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return b if b.isna().sum() < a.isna().sum() else a


def _pick_sheet(path):
    """Choose the workbook sheet that holds signals (has a time col AND a label
    col). A predictions workbook often has extra sheets (confusion matrices etc)."""
    xls = pd.ExcelFile(path, **_XL)
    sheets = xls.sheet_names
    if len(sheets) == 1:
        return sheets[0]
    best, best_score = sheets[0], -1
    for s in sheets:
        try:
            head = pd.read_excel(path, sheet_name=s, nrows=0, **_XL)
        except Exception:
            continue
        cols = set(head.columns)
        score = (2 * any(c in cols for c in TIME_COLS)
                 + 2 * any(c in cols for c in SIG_COLS))
        if score > best_score:
            best, best_score = s, score
    return best


def load_signal(path, sheet=None):
    if str(path).lower().endswith((".xlsx", ".xlsm", ".xls")):
        if sheet is None:
            sheet = _pick_sheet(path)
        print(f"   using Excel sheet: {sheet!r}")
        df = pd.read_excel(path, sheet_name=sheet, **_XL)
    else:
        df = pd.read_csv(path)
    tc = _pick(TIME_COLS, df.columns, "signal time")
    sc = _pick(SIG_COLS, df.columns, "signal")
    out = pd.DataFrame({"ts": _parse_time(df[tc]),
                        "sig": df[sc].astype(str).str.strip().str.upper()})
    return out.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts").reset_index(drop=True)


def load_price(path):
    df = pd.read_parquet(path) if str(path).lower().endswith(".parquet") else pd.read_csv(path)
    if "datetime" not in df.columns and df.index.name:
        df = df.reset_index()
    tc = _pick(TIME_COLS + ["datetime"], df.columns, "price time")
    o, h, l, c = (_pick(OPEN_COLS, df.columns, "open"), _pick(HIGH_COLS, df.columns, "high"),
                  _pick(LOW_COLS, df.columns, "low"), _pick(CLOSE_COLS, df.columns, "close"))
    out = pd.DataFrame({"ts": _parse_time(df[tc]), "open": df[o].astype(float),
                        "high": df[h].astype(float), "low": df[l].astype(float),
                        "close": df[c].astype(float)})
    return out.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts").reset_index(drop=True)


# ============================== ENGINE =====================================
# CORE LOGIC UNCHANGED.  The merge is factored out into prepare() so the three
# versions can share one merged frame, and the entry/exit tables are passed in
# instead of being read from globals.

def prepare(price_df, signal_df):
    """Merge the signal onto the price bars once, for all versions."""
    df = signal_df.merge(price_df, on="ts", how="inner").sort_values("ts").reset_index(drop=True)
    if df.empty:
        raise SystemExit("ERROR: no signal timestamp matched a price bar (check the dates/timezone).")
    return df


def run(df, qty_map, exit_map):
    """Simulate on the merged frame. Returns (trades_df, equity_money, fire_counts).

    Signal on bar T fills at bar T+1 OPEN. Fires only on the FIRST bar of a run
    (on signal change). Tiered: an exit closes only the tiers it is mapped to.
    Charge is per lot (x quantity), booked when the trade closes.
    Signals that are not in qty_map / exit_map for this version are ignored.
    """
    ts = df["ts"].to_numpy()
    sig = df["sig"].to_numpy()
    op = df["open"].to_numpy(float)
    cl = df["close"].to_numpy(float)
    n = len(df)

    fires = np.empty(n, dtype=bool)
    fires[0] = True
    fires[1:] = sig[1:] != sig[:-1]

    next_open = np.append(op[1:], np.nan)
    next_ts = np.append(ts[1:], ts[-1])

    book, trades = [], []
    realized_pts = 0.0
    charge_at = np.zeros(n)
    equity = np.zeros(n)

    def close(tiers, price, when, i, reason="signal"):
        nonlocal realized_pts, book
        keep, done = [], []
        for p in book:
            (done if p["tier"] in tiers else keep).append(p)
        book = keep
        for p in done:
            exit_px = price - SLIPPAGE
            pnl_pts = (exit_px - p["price"]) * p["qty"]
            charge = CHARGE_PER_LOT * p["qty"]
            realized_pts += pnl_pts
            charge_at[i] += charge
            trades.append(dict(
                tier=p["tier"], qty=p["qty"], entry_time=p["time"], entry_price=p["price"],
                exit_time=when, exit_price=exit_px, pnl_points=pnl_pts,
                pnl_money=pnl_pts * LOT_SIZE - charge, charge=charge,
                slippage_cost=2 * SLIPPAGE * p["qty"] * LOT_SIZE,
                entry_signal=p["signal"], exit_reason=reason))

    for i in range(n):
        s = sig[i]
        if fires[i] and s != "NO_TRADE":
            fill = next_open[i]
            if not np.isnan(fill):
                if s in qty_map:
                    book.append(dict(tier=ENTRY_TIER[s], qty=qty_map[s],
                                     price=fill + SLIPPAGE, time=next_ts[i], signal=s))
                elif s in exit_map and book:
                    close(exit_map[s], fill, next_ts[i], i)
        unreal = sum((cl[i] - p["price"]) * p["qty"] for p in book)
        equity[i] = (realized_pts + unreal) * LOT_SIZE - charge_at[:i + 1].sum()

    forced = len(book)
    if book:
        close({"SMALL", "SUB", "SUPER"}, cl[-1], ts[-1], n - 1, reason="end_of_data")
        equity[-1] = realized_pts * LOT_SIZE - charge_at.sum()

    tr = pd.DataFrame(trades)
    if len(tr):
        tr["entry_time"] = pd.to_datetime(tr["entry_time"])
        tr["exit_time"] = pd.to_datetime(tr["exit_time"])
        tr = tr.sort_values("exit_time").reset_index(drop=True)

    fired = pd.Series(sig[fires]).value_counts().to_dict()
    fire_counts = {"_forced_closes": forced, **{k: int(v) for k, v in fired.items()}}
    return tr, equity, fire_counts


# ============================== METRICS ====================================

def _dd_episodes(dd_pct):
    """Depth (in %) of every peak-to-trough drawdown episode."""
    in_dd = dd_pct < 0
    if not in_dd.any():
        return np.array([])
    prev = np.r_[False, in_dd[:-1]]
    nxt = np.r_[in_dd[1:], False]
    starts = np.flatnonzero(in_dd & ~prev)
    ends = np.flatnonzero(in_dd & ~nxt) + 1
    return np.array([dd_pct[s:e].min() for s, e in zip(starts, ends)])


def metrics(tr, equity, bars, capital):
    p = tr["pnl_money"].to_numpy() if len(tr) else np.array([])
    wins, losses = p[p > 0], p[p < 0]
    m = {}
    m["capital"] = float(capital)
    m["net_pnl"] = float(p.sum())
    m["n_trades"] = int(len(tr))
    m["n_wins"], m["n_losses"] = int(len(wins)), int(len(losses))
    m["win_rate"] = len(wins) / len(p) if len(p) else np.nan
    m["gross_profit"] = float(wins.sum())
    m["gross_loss"] = float(-losses.sum())
    m["profit_factor"] = (m["gross_profit"] / m["gross_loss"]) if m["gross_loss"] else np.inf
    m["avg_win"] = float(wins.mean()) if len(wins) else 0.0
    m["avg_loss"] = float(-losses.mean()) if len(losses) else 0.0
    m["total_charge"] = float(tr["charge"].sum()) if len(tr) else 0.0
    m["total_slippage"] = float(tr["slippage_cost"].sum()) if len(tr) else 0.0
    m["charge_pct_of_gross"] = (m["total_charge"] / m["gross_profit"] * 100
                               if m["gross_profit"] > 0 else np.nan)
    m["avg_bars_held"] = (float((tr["exit_time"] - tr["entry_time"]).dt.total_seconds().mean() / 60.0)
                          if len(tr) else np.nan)

    lr = 1 - m["win_rate"] if not np.isnan(m["win_rate"]) else np.nan
    m["expectancy"] = m["win_rate"] * m["avg_win"] - lr * m["avg_loss"]

    # ---- drawdown, in money and in % of the capital curve ----
    peak_money = np.maximum.accumulate(equity)
    m["max_drawdown"] = float((equity - peak_money).min())

    curve = capital + equity
    peak = np.maximum.accumulate(curve)
    dd_pct = (curve / peak - 1.0) * 100.0
    m["max_drawdown_pct"] = float(dd_pct.min())
    eps = _dd_episodes(dd_pct)
    m["avg_drawdown_pct"] = float(eps.mean()) if len(eps) else 0.0
    m["n_drawdowns"] = int(len(eps))
    m["pct_time_in_dd"] = float((dd_pct < 0).mean() * 100.0)

    # ---- returns on capital ----
    m["total_return_pct"] = m["net_pnl"] / capital * 100.0
    years = max((bars["ts"].iloc[-1] - bars["ts"].iloc[0]).days / 365.25, 1e-9)
    m["years"] = years
    m["arr"] = m["net_pnl"] / years
    m["arr_pct"] = m["arr"] / capital * 100.0
    growth = (capital + m["net_pnl"]) / capital
    m["cagr_pct"] = (growth ** (1.0 / years) - 1.0) * 100.0 if growth > 0 else np.nan
    m["calmar"] = (m["arr"] / abs(m["max_drawdown"])) if m["max_drawdown"] < 0 else np.inf

    # ---- risk-adjusted, from the daily equity path ----
    daily = pd.Series(equity, index=pd.to_datetime(bars["ts"])).resample("1D").last().dropna()
    d = daily.diff().dropna().to_numpy()
    if len(d) > 2 and d.std(ddof=1) > 0:
        m["sharpe"] = float(d.mean() / d.std(ddof=1) * np.sqrt(TRADING_DAYS))
        downside = np.sqrt(np.mean(np.minimum(d, 0.0) ** 2))
        m["sortino"] = float(d.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else np.inf
    else:
        m["sharpe"] = m["sortino"] = np.nan
    return m


def benchmark(bars, capital):
    """Buy & hold the price series for the same window, same 1-lot size."""
    c0 = float(bars["close"].iloc[0])
    c1 = float(bars["close"].iloc[-1])
    years = max((bars["ts"].iloc[-1] - bars["ts"].iloc[0]).days / 365.25, 1e-9)
    pnl = (c1 - c0) * LOT_SIZE * LOTS_PER_TRADE
    b = {"bench_open": c0, "bench_close": c1,
         "bench_points": c1 - c0, "bench_pnl": pnl,
         "bench_return_pct": pnl / capital * 100.0,
         "bench_price_return_pct": (c1 / c0 - 1.0) * 100.0}
    g = (capital + pnl) / capital
    b["bench_cagr_pct"] = (g ** (1.0 / years) - 1.0) * 100.0 if g > 0 else np.nan
    return b


def yearly_stats(tr, equity, bars, capital, bench_ret):
    """Year-by-year gain: PnL, return %, benchmark %, alpha, intra-year max DD."""
    d = pd.DataFrame({"ts": pd.to_datetime(bars["ts"]), "eq": np.asarray(equity, float),
                      "close": bars["close"].to_numpy(float)})
    d["year"] = d["ts"].dt.year
    ty = tr.assign(year=tr["exit_time"].dt.year) if len(tr) else None

    rows, prev_end = [], 0.0
    for y, g in d.groupby("year", sort=True):
        eq = g["eq"].to_numpy(float)
        end = float(eq[-1])
        pnl = end - prev_end
        # intra-year drawdown: peak seeded at the equity level the year opened on
        curve = np.r_[capital + prev_end, capital + eq]
        peak = np.maximum.accumulate(curve)
        dd = float(((curve / peak - 1.0) * 100.0).min())
        c = g["close"].to_numpy(float)
        bench = (c[-1] / c[0] - 1.0) * 100.0
        n_tr = wins = 0
        if ty is not None:
            sub = ty[ty["year"] == y]
            n_tr, wins = len(sub), int((sub["pnl_money"] > 0).sum())
        ret = pnl / capital * 100.0
        rows.append(dict(year=int(y), sessions=int(g["ts"].dt.normalize().nunique()),
                         trades=n_tr, wins=wins,
                         win_rate_pct=(wins / n_tr * 100.0 if n_tr else np.nan),
                         net_pnl=pnl, return_pct=ret, bench_pct=bench,
                         alpha_pp=ret - bench, max_dd_pct=dd))
        prev_end = end
    out = pd.DataFrame(rows)
    if len(out):
        out.attrs["avg_return_pct"] = float(out["return_pct"].mean())
        out.attrs["avg_bench_pct"] = float(out["bench_pct"].mean())
        out.attrs["best_year"] = int(out.loc[out["return_pct"].idxmax(), "year"])
        out.attrs["worst_year"] = int(out.loc[out["return_pct"].idxmin(), "year"])
        out.attrs["pos_years"] = int((out["net_pnl"] > 0).sum())
    return out


def monthly_returns(tr):
    if not len(tr):
        return pd.Series(dtype=float)
    t = tr.copy()
    t["month"] = t["exit_time"].dt.to_period("M")
    return t.groupby("month")["pnl_money"].sum()


def monthly_grid(tr):
    """year x month matrix of realised net PnL."""
    if not len(tr):
        return pd.DataFrame()
    t = tr.copy()
    t["y"] = t["exit_time"].dt.year
    t["m"] = t["exit_time"].dt.month
    g = t.pivot_table(index="y", columns="m", values="pnl_money", aggfunc="sum")
    return g.reindex(columns=range(1, 13)).sort_index()


# ============================== REPORT =====================================

def _money(x, w=18):
    return f"{x:>{w},.0f}" if np.isfinite(x) else f"{'n/a':>{w}}"


def _pct(x, w=18, dp=2):
    return f"{x:>{w - 2}.{dp}f} %" if np.isfinite(x) else f"{'n/a':>{w}}"


def _num(x, w=18, dp=2):
    if not np.isfinite(x):
        return f"{'inf' if x == np.inf else 'n/a':>{w}}"
    return f"{x:>{w}.{dp}f}"


def _rule(ch="=", w=W):
    return ch * w


def _kv(pairs, kw=30, vw=20, indent="  "):
    """Two-column boxed key/value table. A pair of (None, None) draws a rule."""
    bar = f"{indent}+{'-' * (kw + 2)}+{'-' * (vw + 2)}+"
    out = [bar]
    for k, v in pairs:
        if k is None:
            out.append(bar)
        else:
            out.append(f"{indent}| {k:<{kw}} | {str(v):>{vw}} |")
    out.append(bar)
    return out


def _grid(headers, rows, widths, indent="  "):
    """Boxed table. rows = list of list-of-preformatted-strings."""
    bar = indent + "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [bar, indent + "|" + "|".join(f" {h:^{w}} " for h, w in zip(headers, widths)) + "|"]
    out.append(bar)
    for r in rows:
        out.append(indent + "|" + "|".join(f" {c:>{w}} " for c, w in zip(r, widths)) + "|")
    out.append(bar)
    return out


def header_block(bars, capital, price_path, signal_path, sig_counts):
    L = ["", _rule("="), "  SIGNAL BACKTEST   -   3-VERSION SIGNAL QUALITY SUITE", _rule("="),
         f"  signal file : {os.path.basename(signal_path)}",
         f"  price file  : {os.path.basename(price_path)}", ""]
    L += _kv([
        ("window from", f"{bars['ts'].min():%Y-%m-%d %H:%M}"),
        ("window to", f"{bars['ts'].max():%Y-%m-%d %H:%M}"),
        ("matched bars", f"{len(bars):,}"),
        ("sessions", f"{bars['ts'].dt.normalize().nunique():,}"),
        ("years", f"{(bars['ts'].iloc[-1] - bars['ts'].iloc[0]).days / 365.25:.2f}"),
        (None, None),
        ("lot size", f"{LOT_SIZE} qty = 1 lot"),
        ("size per trade", f"{LOTS_PER_TRADE} lot ({LOTS_PER_TRADE * LOT_SIZE} qty)"),
        ("sizing rule", "EQUAL for every tier"),
        ("capital base", f"Rs {capital:,.0f}"),
        ("charge per lot", f"Rs {CHARGE_PER_LOT:,.0f} round trip"),
        ("slippage", f"{SLIPPAGE:g} pts per fill"),
        ("fill rule", "next bar OPEN"),
        ("firing rule", "on signal change only"),
        (None, None),
        ("SMALL family", "NOT TRADED (excluded)"),
    ], kw=18, vw=26)
    L.append("")
    L.append("  SIGNAL COUNT IN FILE  (raw rows, before on-change filtering)")
    lab = sorted(sig_counts, key=lambda k: -sig_counts[k])
    rows = [[k, f"{sig_counts[k]:,}", f"{sig_counts[k] / sum(sig_counts.values()) * 100:.2f} %"]
            for k in lab]
    L += _grid(["signal", "rows", "share"], rows, [14, 12, 10])
    return L


def version_block(v, m, b, yr, grid_df, fire_counts):
    key, name = v["key"], v["name"]
    L = ["", "#" * W,
         f"#  {key}   {name.upper()}",
         f"#  {v['desc']}",
         f"#  Q: {v['question']}",
         "#" * W, ""]

    # ---- rules actually applied ----
    ex = " | ".join(f"{k} closes {'+'.join(sorted(t))}" for k, t in v["exits"].items())
    en = " | ".join(f"{k} x {q} lot" for k, q in v["qty"].items())
    ign = sorted((ALL_ENTRIES | ALL_EXITS) - set(v["qty"]) - set(v["exits"]))
    L.append("  RULES APPLIED")
    L.append(f"    entries taken   : {en}")
    L.append(f"    exits honoured  : {ex}")
    L.append(f"    signals ignored : {', '.join(ign)}  (+ NO_TRADE)")
    fired = " | ".join(f"{k} {fire_counts.get(k, 0):,}"
                       for k in sorted(set(v['qty']) | set(v['exits'])))
    L.append(f"    on-change fires : {fired}")
    L.append(f"    forced closes   : {fire_counts.get('_forced_closes', 0)} "
             f"open position(s) squared off at the last bar")
    L.append("")

    if m["n_trades"] == 0:
        L.append("  >>> NO TRADES were generated for this version. <<<")
        return L

    # ---- headline: return / risk / benchmark ----
    L.append("  HEADLINE  -  RETURN ON CAPITAL")
    L += _kv([
        ("Net PnL (Rs)", _money(m["net_pnl"], 16)),
        ("total_return_%", _pct(m["total_return_pct"], 16)),
        ("CAGR_%", _pct(m["cagr_pct"], 16)),
        ("avg_return_% (per year)", _pct(yr.attrs.get("avg_return_pct", np.nan), 16)),
        ("ARR (Rs per year)", _money(m["arr"], 16)),
        (None, None),
        ("max_drawdown_%", _pct(m["max_drawdown_pct"], 16)),
        ("avg_DD_%", _pct(m["avg_drawdown_pct"], 16)),
        ("Max drawdown (Rs)", _money(m["max_drawdown"], 16)),
        ("drawdown episodes", f"{m['n_drawdowns']:,}"),
        ("time in drawdown", _pct(m["pct_time_in_dd"], 16)),
        (None, None),
        ("BENCHMARK buy & hold_%", _pct(b["bench_return_pct"], 16)),
        ("benchmark CAGR_%", _pct(b["bench_cagr_pct"], 16)),
        ("strategy vs benchmark", f"{m['total_return_pct'] - b['bench_return_pct']:>14.2f}pp"),
        ("beats benchmark?", "YES" if m["total_return_pct"] > b["bench_return_pct"] else "NO"),
    ], kw=24, vw=18)
    L.append("")

    # ---- trade stats ----
    L.append("  TRADE STATISTICS")
    L += _kv([
        ("Trades", f"{m['n_trades']:,}"),
        ("Wins / Losses", f"{m['n_wins']:,} / {m['n_losses']:,}"),
        ("Win rate", _pct(m["win_rate"] * 100, 16, 1)),
        ("Profit factor", _num(m["profit_factor"], 16)),
        ("Expectancy / trade (Rs)", _money(m["expectancy"], 16)),
        ("Avg win (Rs)", _money(m["avg_win"], 16)),
        ("Avg loss (Rs)", _money(m["avg_loss"], 16)),
        ("Gross profit (Rs)", _money(m["gross_profit"], 16)),
        ("Gross loss (Rs)", _money(m["gross_loss"], 16)),
        ("Avg hold (minutes)", _num(m["avg_bars_held"], 16, 1)),
        (None, None),
        ("Sharpe ratio", _num(m["sharpe"], 16)),
        ("Sortino ratio", _num(m["sortino"], 16)),
        ("Calmar ratio", _num(m["calmar"], 16)),
        (None, None),
        ("Total charges (Rs)", _money(m["total_charge"], 16)),
        ("charges / gross profit", _pct(m["charge_pct_of_gross"], 16, 1)),
        ("Total slippage (Rs)", _money(m["total_slippage"], 16)),
    ], kw=24, vw=18)
    L.append("")

    # ---- year by year ----
    L.append("  YEAR BY YEAR   (gain per year, benchmark = buy & hold same window)")
    rows = []
    for _, r in yr.iterrows():
        rows.append([f"{int(r['year'])}", f"{int(r['trades']):,}",
                     f"{r['win_rate_pct']:.1f}" if np.isfinite(r["win_rate_pct"]) else "-",
                     f"{r['net_pnl']:,.0f}", f"{r['return_pct']:.2f}",
                     f"{r['bench_pct']:.2f}", f"{r['alpha_pp']:+.2f}",
                     f"{r['max_dd_pct']:.2f}"])
    rows.append(["-" * 4, "-" * 6, "-" * 5, "-" * 12, "-" * 7, "-" * 7, "-" * 7, "-" * 7])
    rows.append(["TOTAL", f"{int(yr['trades'].sum()):,}",
                 f"{m['win_rate'] * 100:.1f}", f"{yr['net_pnl'].sum():,.0f}",
                 f"{m['total_return_pct']:.2f}", f"{b['bench_return_pct']:.2f}",
                 f"{m['total_return_pct'] - b['bench_return_pct']:+.2f}",
                 f"{m['max_drawdown_pct']:.2f}"])
    rows.append(["AVG/yr", f"{yr['trades'].mean():,.0f}", "",
                 f"{yr['net_pnl'].mean():,.0f}",
                 f"{yr.attrs.get('avg_return_pct', np.nan):.2f}",
                 f"{yr.attrs.get('avg_bench_pct', np.nan):.2f}",
                 f"{yr['alpha_pp'].mean():+.2f}", f"{yr['max_dd_pct'].mean():.2f}"])
    L += _grid(["year", "trades", "win%", "net PnL Rs", "ret %", "bench%", "alpha pp", "maxDD%"],
               rows, [6, 6, 5, 12, 7, 7, 8, 7])
    L.append(f"    positive years {yr.attrs.get('pos_years', 0)} / {len(yr)}"
             f"   |   best {yr.attrs.get('best_year', '-')}"
             f"   |   worst {yr.attrs.get('worst_year', '-')}")
    L.append("")

    # ---- monthly grid ----
    if len(grid_df):
        L.append("  MONTHLY NET PnL   (Rs thousands, by exit month)")
        rows = []
        for y, r in grid_df.iterrows():
            cells = [f"{v / 1e3:,.0f}" if pd.notna(v) else "-" for v in r.to_numpy()]
            tot = np.nansum(r.to_numpy())
            rows.append([str(int(y))] + cells + [f"{tot / 1e3:,.0f}"])
        L += _grid(["yr"] + MONTH_ABBR + ["year"], rows, [4] + [5] * 12 + [7])
        L.append("")
    return L


def comparison_block(results, b):
    L = ["", _rule("="), "  VERSION COMPARISON   (same window, same 1-lot size, same costs)", _rule("=")]
    keys = [v["key"] for v, _, _, _, _ in results]
    names = {v["key"]: v["name"] for v, _, _, _, _ in results}
    for k in keys:
        L.append(f"    {k} = {names[k]}")
    L.append("")

    def row(label, fn):
        return [label] + [fn(m, yr) for _, m, _, yr, _ in results]

    R = [
        row("Trades", lambda m, y: f"{m['n_trades']:,}"),
        row("Win rate %", lambda m, y: f"{m['win_rate'] * 100:.1f}" if m["n_trades"] else "-"),
        row("Profit factor", lambda m, y: f"{m['profit_factor']:.2f}" if np.isfinite(m["profit_factor"]) else "inf"),
        row("Expectancy / trade Rs", lambda m, y: f"{m['expectancy']:,.0f}" if m["n_trades"] else "-"),
        row("Net PnL Rs", lambda m, y: f"{m['net_pnl']:,.0f}"),
        row("total_return_%", lambda m, y: f"{m['total_return_pct']:.2f}"),
        row("avg_return_% per year", lambda m, y: f"{y.attrs.get('avg_return_pct', float('nan')):.2f}" if len(y) else "-"),
        row("CAGR_%", lambda m, y: f"{m['cagr_pct']:.2f}" if np.isfinite(m["cagr_pct"]) else "n/a"),
        row("max_drawdown_%", lambda m, y: f"{m['max_drawdown_pct']:.2f}"),
        row("avg_DD_%", lambda m, y: f"{m['avg_drawdown_pct']:.2f}"),
        row("vs benchmark pp", lambda m, y: f"{m['total_return_pct'] - b['bench_return_pct']:+.2f}"),
        row("Sharpe", lambda m, y: f"{m['sharpe']:.2f}" if np.isfinite(m["sharpe"]) else "n/a"),
        row("Calmar", lambda m, y: f"{m['calmar']:.2f}" if np.isfinite(m["calmar"]) else "inf"),
        row("Total charges Rs", lambda m, y: f"{m['total_charge']:,.0f}"),
        row("Positive years", lambda m, y: f"{y.attrs.get('pos_years', 0)} / {len(y)}" if len(y) else "-"),
    ]
    L += _grid(["metric"] + keys, R, [22] + [16] * len(keys))
    L.append("")
    L.append("  BENCHMARK (buy & hold 1 lot, gross of costs)")
    L += _kv([
        ("price at start", f"{b['bench_open']:,.2f}"),
        ("price at end", f"{b['bench_close']:,.2f}"),
        ("points", f"{b['bench_points']:,.2f}"),
        ("PnL (Rs)", f"{b['bench_pnl']:,.0f}"),
        ("return_% on capital", f"{b['bench_return_pct']:.2f} %"),
        ("CAGR_%", f"{b['bench_cagr_pct']:.2f} %"),
    ], kw=22, vw=18)

    best = max(results, key=lambda r: (r[1]["net_pnl"] if r[1]["n_trades"] else -np.inf))
    L.append("")
    L.append(f"  Highest net PnL : {best[0]['key']} - {best[0]['name']}")
    L.append(_rule("="))
    return L


# =============================== MAIN ======================================

METRIC_ORDER = ["capital", "net_pnl", "total_return_pct", "avg_return_pct", "cagr_pct",
                "arr", "arr_pct", "max_drawdown_pct", "avg_drawdown_pct", "max_drawdown",
                "n_drawdowns", "pct_time_in_dd", "bench_return_pct", "bench_cagr_pct",
                "vs_benchmark_pp", "n_trades", "win_rate", "profit_factor", "expectancy",
                "avg_win", "avg_loss", "gross_profit", "gross_loss", "n_wins", "n_losses",
                "avg_bars_held", "sharpe", "sortino", "calmar", "total_charge",
                "charge_pct_of_gross", "total_slippage", "years"]


def main():
    signal_file = sys.argv[1] if len(sys.argv) > 1 else SIGNAL_FILE
    price_file = sys.argv[2] if len(sys.argv) > 2 else PRICE_FILE
    output_dir = sys.argv[3] if len(sys.argv) > 3 else OUTPUT_DIR
    for pth, what in [(signal_file, "SIGNAL"), (price_file, "PRICE")]:
        if not os.path.exists(pth):
            raise SystemExit(f"ERROR: {what} file not found:\n   {pth}")

    print(_rule("="))
    print(" SIGNAL BACKTEST  -  3-VERSION SIGNAL QUALITY SUITE")
    print(_rule("="))
    print(f" signal : {signal_file}")
    print(f" price  : {price_file}")
    print(f" output : {output_dir}")
    print(_rule("-"))
    print(" loading price ...")
    price_df = load_price(price_file)
    print(" loading signal ...")
    signal_df = load_signal(signal_file, SIGNAL_SHEET)

    known = ALL_ENTRIES | ALL_EXITS | {"NO_TRADE"}
    unknown = set(signal_df["sig"].unique()) - known
    if unknown:
        raise SystemExit(f"ERROR: unrecognised signal values: {sorted(unknown)}\n"
                         f"   expected: {sorted(known)}")

    print(" merging signal onto price bars ...")
    bars = prepare(price_df, signal_df)
    capital = float(INITIAL_CAPITAL) if INITIAL_CAPITAL else \
        float(bars["close"].iloc[0]) * LOT_SIZE * LOTS_PER_TRADE
    bench = benchmark(bars, capital)
    sig_counts = bars["sig"].value_counts().to_dict()

    stem = os.path.splitext(os.path.basename(signal_file))[0]
    run_dir = os.path.join(output_dir, f"backtest_{stem}_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(run_dir, exist_ok=True)

    report = header_block(bars, capital, price_file, signal_file, sig_counts)
    results = []

    for v in VERSIONS:
        print(f" running {v['key']}  {v['name']} ...")
        tr, equity, fires = run(bars, v["qty"], v["exits"])
        m = metrics(tr, equity, bars, capital)
        m.update(bench)
        m["vs_benchmark_pp"] = m["total_return_pct"] - bench["bench_return_pct"]
        yr = yearly_stats(tr, equity, bars, capital, bench["bench_return_pct"])
        m["avg_return_pct"] = yr.attrs.get("avg_return_pct", np.nan) if len(yr) else np.nan
        gridf = monthly_grid(tr)
        report += version_block(v, m, bench, yr, gridf, fires)
        results.append((v, m, equity, yr, tr))

        # ---- per-version CSV output ----
        vdir = os.path.join(run_dir, f"{v['key']}_{v['name'].replace(' ', '_')}")
        os.makedirs(vdir, exist_ok=True)
        if len(tr):
            tr.to_csv(os.path.join(vdir, "trades.csv"), index=False)
            monthly_returns(tr).rename("net_pnl").reset_index().to_csv(
                os.path.join(vdir, "monthly_returns.csv"), index=False)
        if len(yr):
            yr.to_csv(os.path.join(vdir, "yearly_returns.csv"), index=False)
        pd.DataFrame({"timestamp": bars["ts"], "equity": equity,
                      "equity_on_capital": capital + equity,
                      "close": bars["close"], "signal": bars["sig"]}).to_csv(
            os.path.join(vdir, "equity_curve.csv"), index=False)
        pd.DataFrame([(k, m[k]) for k in METRIC_ORDER if k in m],
                     columns=["metric", "value"]).to_csv(
            os.path.join(vdir, "metrics.csv"), index=False)

    report += comparison_block(results, bench)
    text = "\n".join(report)
    print(text)

    # ---- combined outputs ----
    cmp_df = pd.DataFrame({v["key"]: {k: m.get(k) for k in METRIC_ORDER}
                           for v, m, _, _, _ in results})
    cmp_df.index.name = "metric"
    cmp_df.to_csv(os.path.join(run_dir, "comparison.csv"))
    with open(os.path.join(run_dir, "full_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + f"\n\ngenerated {datetime.now():%Y-%m-%d %H:%M}\n")

    print("")
    print(_rule("-"))
    print(" RESULTS SAVED TO:")
    print(f"   {run_dir}")
    print("     comparison.csv")
    print("     full_report.txt")
    for v in VERSIONS:
        print(f"     {v['key']}_{v['name'].replace(' ', '_')}\\  "
              f"(metrics.csv, trades.csv, yearly_returns.csv, "
              f"monthly_returns.csv, equity_curve.csv)")
    print(_rule("="))


if __name__ == "__main__":
    main()
