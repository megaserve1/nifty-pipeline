import json
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
LOT_SIZE       = 65        # index units per 1 lot
CHARGE_PER_LOT = 2000.0    # round-trip ("rotation") charge per lot, in rupees
SLIPPAGE       = 0.0       # points lost on each fill
QTY = {"ENTRY_SMALL": 1, "ENTRY_SUB": 2, "ENTRY_SUPER": 4}   # lots per tier
# ===========================================================================

EXIT_CLOSES = {"EXIT_SMALL": {"SMALL"}, "EXIT_SUB": {"SMALL", "SUB"},
               "EXIT_SUPER": {"SMALL", "SUB", "SUPER"}}
ENTRY_TIER = {"ENTRY_SMALL": "SMALL", "ENTRY_SUB": "SUB", "ENTRY_SUPER": "SUPER"}
TRADING_DAYS = 252

TIME_COLS = ["timestamp", "datetime", "ts", "time", "date"]
SIG_COLS = ["predicted_label", "primary_label", "signal", "prediction", "y_pred", "label"]
OPEN_COLS = ["Nifty_Futures_Open", "open", "Open"]
HIGH_COLS = ["Nifty_Futures_High", "high", "High"]
LOW_COLS = ["Nifty_Futures_Low", "low", "Low"]
CLOSE_COLS = ["Nifty_Futures_Close", "close", "Close"]

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

def run(price_df, signal_df):
    """Merge signal onto price and simulate. Returns (trades_df, equity_money, bars_df).

    Signal on bar T fills at bar T+1 OPEN. Fires only on the FIRST bar of a run
    (on signal change). Tiered: EXIT_SUB closes SMALL+SUB but leaves SUPER open.
    Charge is per lot (x quantity), booked when the trade closes.
    """
    df = signal_df.merge(price_df, on="ts", how="inner").sort_values("ts").reset_index(drop=True)
    if df.empty:
        raise SystemExit("ERROR: no signal timestamp matched a price bar (check the dates/timezone).")

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

    def close(tiers, price, when, i):
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
                slippage_cost=2 * SLIPPAGE * p["qty"] * LOT_SIZE))

    for i in range(n):
        s = sig[i]
        if fires[i] and s != "NO_TRADE":
            fill = next_open[i]
            if not np.isnan(fill):
                if s in QTY:
                    book.append(dict(tier=ENTRY_TIER[s], qty=QTY[s],
                                     price=fill + SLIPPAGE, time=next_ts[i]))
                elif s in EXIT_CLOSES and book:
                    close(EXIT_CLOSES[s], fill, next_ts[i], i)
        unreal = sum((cl[i] - p["price"]) * p["qty"] for p in book)
        equity[i] = (realized_pts + unreal) * LOT_SIZE - charge_at[:i + 1].sum()

    if book:
        close({"SMALL", "SUB", "SUPER"}, cl[-1], ts[-1], n - 1)
        equity[-1] = realized_pts * LOT_SIZE - charge_at.sum()

    tr = pd.DataFrame(trades)
    if len(tr):
        tr["entry_time"] = pd.to_datetime(tr["entry_time"])
        tr["exit_time"] = pd.to_datetime(tr["exit_time"])
        tr = tr.sort_values("exit_time").reset_index(drop=True)
    return tr, equity, df


# ============================== METRICS ====================================

def metrics(tr, equity, bars):
    p = tr["pnl_money"].to_numpy() if len(tr) else np.array([])
    wins, losses = p[p > 0], p[p < 0]
    m = {}
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

    lr = 1 - m["win_rate"] if not np.isnan(m["win_rate"]) else np.nan
    m["expectancy"] = m["win_rate"] * m["avg_win"] - lr * m["avg_loss"]

    peak = np.maximum.accumulate(equity)
    m["max_drawdown"] = float((equity - peak).min())

    daily = pd.Series(equity, index=pd.to_datetime(bars["ts"])).resample("1D").last().dropna()
    d = daily.diff().dropna().to_numpy()
    if len(d) > 2 and d.std(ddof=1) > 0:
        m["sharpe"] = float(d.mean() / d.std(ddof=1) * np.sqrt(TRADING_DAYS))
        downside = np.sqrt(np.mean(np.minimum(d, 0.0) ** 2))
        m["sortino"] = float(d.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else np.inf
    else:
        m["sharpe"] = m["sortino"] = np.nan

    years = max((bars["ts"].iloc[-1] - bars["ts"].iloc[0]).days / 365.25, 1e-9)
    m["years"] = years
    m["arr"] = m["net_pnl"] / years
    m["calmar"] = (m["arr"] / abs(m["max_drawdown"])) if m["max_drawdown"] < 0 else np.inf
    return m


def monthly_returns(tr):
    if not len(tr):
        return pd.Series(dtype=float)
    t = tr.copy()
    t["month"] = t["exit_time"].dt.to_period("M")
    return t.groupby("month")["pnl_money"].sum()


# ============================== REPORT =====================================

def build_report(m, bars, monthly, price_path, signal_path):
    money = lambda x: f"{x:>16,.0f}" if np.isfinite(x) else f"{'n/a':>16}"
    ratio = lambda x: f"{x:>16.2f}" if np.isfinite(x) else f"{'n/a':>16}"
    L = ["", "=" * 60, " SIGNAL BACKTEST", "=" * 60, " BASIC DETAILS"]
    L.append(f"   signal file    {os.path.basename(signal_path)}")
    L.append(f"   price file     {os.path.basename(price_path)}")
    L.append(f"   window         {bars['ts'].min()}  ->  {bars['ts'].max()}")
    L.append(f"   bars           {len(bars):,}   sessions "
             f"{bars['ts'].dt.normalize().nunique():,}   years {m['years']:.2f}")
    L.append(f"   lot size       {LOT_SIZE}   (1 quantity = 1 lot = {LOT_SIZE} index units)")
    L.append(f"   quantity       SMALL {QTY['ENTRY_SMALL']}  SUB {QTY['ENTRY_SUB']}"
             f"  SUPER {QTY['ENTRY_SUPER']}  (lots)")
    L.append(f"   charge / lot   Rs {CHARGE_PER_LOT:,.0f}  round trip (x quantity)")
    L.append(f"   slippage       {SLIPPAGE:g} points each fill")
    L.append(f"   entry / fire   next bar OPEN  |  first signal only (on change)")
    L.append("-" * 60)
    L.append(" RESULTS   (money in rupees)")
    L.append(f"   Net PnL              {money(m['net_pnl'])}")
    L.append(f"   Trades               {m['n_trades']:>16,}")
    L.append(f"   Win rate             {m['win_rate']*100:>15.1f}%"
             f"   ({m['n_wins']:,}W / {m['n_losses']:,}L)")
    L.append(f"   Profit factor        {ratio(m['profit_factor'])}")
    L.append(f"   Expectancy / trade   {money(m['expectancy'])}")
    L.append(f"   Avg win / avg loss   {money(m['avg_win'])} / {money(m['avg_loss'])}")
    L.append(f"   Max drawdown         {money(m['max_drawdown'])}")
    L.append(f"   Sharpe ratio         {ratio(m['sharpe'])}")
    L.append(f"   Sortino ratio        {ratio(m['sortino'])}")
    L.append(f"   Calmar ratio         {ratio(m['calmar'])}")
    L.append(f"   ARR (annualised)     {money(m['arr'])}   per year")
    L.append(f"   Total charges        {money(m['total_charge'])}")
    L.append(f"   Total slippage       {money(m['total_slippage'])}")
    L.append("-" * 60)
    L.append(" MONTHLY RETURN  (net PnL, no compounding)")
    for period, val in monthly.items():
        L.append(f"   {str(period)}          {val:>16,.0f}")
    L.append("=" * 60)
    return "\n".join(L)


# =============================== MAIN ======================================

def main():
    signal_file = sys.argv[1] if len(sys.argv) > 1 else SIGNAL_FILE
    price_file = sys.argv[2] if len(sys.argv) > 2 else PRICE_FILE
    output_dir = sys.argv[3] if len(sys.argv) > 3 else OUTPUT_DIR
    for pth, what in [(signal_file, "SIGNAL"), (price_file, "PRICE")]:
        if not os.path.exists(pth):
            raise SystemExit(f"ERROR: {what} file not found:\n   {pth}")

    print("=" * 60)
    print(" SIGNAL BACKTEST  (single file)")
    print("=" * 60)
    print(f" signal : {signal_file}")
    print(f" price  : {price_file}")
    print(f" output : {output_dir}")
    print("-" * 60)
    print(" loading price ...")
    price_df = load_price(price_file)
    print(" loading signal ...")
    signal_df = load_signal(signal_file, SIGNAL_SHEET)

    unknown = set(signal_df["sig"].unique()) - set(QTY) - set(EXIT_CLOSES) - {"NO_TRADE"}
    if unknown:
        raise SystemExit(f"ERROR: unrecognised signal values: {sorted(unknown)}\n"
                         f"   expected: {sorted(set(QTY) | set(EXIT_CLOSES) | {'NO_TRADE'})}")

    print(" running backtest ...")
    tr, equity, bars = run(price_df, signal_df)
    m = metrics(tr, equity, bars)
    monthly = monthly_returns(tr)
    report = build_report(m, bars, monthly, price_file, signal_file)
    print(report)

    # ---- write results as CSV into a fresh timestamped folder ----
    stem = os.path.splitext(os.path.basename(signal_file))[0]
    run_dir = os.path.join(output_dir, f"backtest_{stem}_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(run_dir, exist_ok=True)

    if len(tr):
        tr.to_csv(os.path.join(run_dir, "trades.csv"), index=False)
    pd.DataFrame({"timestamp": bars["ts"], "equity": equity,
                  "close": bars["close"], "signal": bars["sig"]}).to_csv(
        os.path.join(run_dir, "equity_curve.csv"), index=False)
    if len(monthly):
        monthly.rename("net_pnl").reset_index().to_csv(
            os.path.join(run_dir, "monthly_returns.csv"), index=False)
    # metrics as a tidy 2-column CSV
    order = ["net_pnl", "n_trades", "win_rate", "profit_factor", "expectancy",
             "avg_win", "avg_loss", "max_drawdown", "sharpe", "sortino", "calmar",
             "arr", "total_charge", "total_slippage", "gross_profit", "gross_loss",
             "n_wins", "n_losses", "years"]
    pd.DataFrame([(k, m[k]) for k in order if k in m],
                 columns=["metric", "value"]).to_csv(
        os.path.join(run_dir, "metrics.csv"), index=False)
    with open(os.path.join(run_dir, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write(report + f"\n\ngenerated {datetime.now():%Y-%m-%d %H:%M}\n")

    print("\n" + "-" * 60)
    print(" RESULTS SAVED TO:")
    print(f"   {run_dir}")
    for f in ["metrics.csv", "trades.csv", "monthly_returns.csv",
              "equity_curve.csv", "report.txt"]:
        if os.path.exists(os.path.join(run_dir, f)):
            print(f"     {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
