"""
==============================================================================
 NIFTY SIGNAL BACKTEST  -  3-VERSION SIGNAL QUALITY SUITE  -  one file
==============================================================================
 SIGNAL  = Excel (.xlsx) or CSV of per-bar predictions
 PRICE   = Parquet (or CSV) of NIFTY futures OHLC
 OUTPUT  = ONE Excel workbook with every result + one full_report.txt

 RUN:
     python backtest_single.py
     python backtest_single.py <signal.xlsx> <price.parquet> <output_dir>

 ---------------------------------------------------------------------------
 THE THREE VERSIONS  (the SMALL family is never traded)
 ---------------------------------------------------------------------------
   V1  SUPER entry quality : ENTRY_SUPER -> EXIT_SUB or EXIT_SUPER
   V2  SUB   entry quality : ENTRY_SUB   -> EXIT_SUB or EXIT_SUPER
   V3  SUPER class quality : ENTRY_SUPER -> EXIT_SUPER only

 SIZING: every on-change signal fires; the FIRST bar of a new signal run is
 traded at 1 lot. A fresh ENTRY while already long ADDS a lot - lots stack.

 ---------------------------------------------------------------------------
 THE REPORT IS SPLIT IN TWO
 ---------------------------------------------------------------------------
 A futures backtest has no unambiguous "capital": SPAN margin is ~1.2-1.5 lakh
 a lot against ~16 lakh of notional, and stacked entries change the lots held
 from minute to minute.  Any percentage OF capital is therefore an ASSUMPTION.

   CORE RESULTS (capital-INDEPENDENT)  <- the headline. Rupees, points, counts
       and ratios that divide by no capital: gross/net PnL, points per lot,
       win rate, profit factor, expectancy, MAE/MFE, drawdown IN RUPEES, time
       in drawdown, ARR in rupees, churn, exposure in lots.

   CAPITAL-DEPENDENT STATS (OPTIONAL, flagged)  total_return_%, ARR_%, CAGR,
       drawdown_%, Calmar, Sharpe, Sortino - on TWO assumed bases side by side
       (A = 1-lot notional, B = peak notional) so the sensitivity is visible.

 ---------------------------------------------------------------------------
 CHURN  (new)
 ---------------------------------------------------------------------------
 If the signal file carries a TRUE label column, the engine measures how often
 the model traded when it should have stood aside:
     false trade   = true label is NO_TRADE but the model predicted a trade
     churn cost Rs = false trades x round-trip charge
     churn rate %  = false trades / trades taken x 100
 Reported at two levels: PREDICTION level (model quality, version-independent)
 and EXECUTED level (economic impact, per version).

 Needs: pandas, numpy, pyarrow, xlsxwriter or openpyxl, and openpyxl or
 python-calamine to read .xlsx.
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
SIGNAL_FILE = r"C:\Users\Admin\Downloads\evaluation_catboost_v4.3.xlsx"
PRICE_FILE  = r"C:\Users\Admin\Downloads\09012015-31012025_dataset\Merged_file\Nifty_futures_base_data_09102015-24022026.parquet"
SIGNAL_SHEET = None        # None = auto-detect the signal sheet
OUTPUT_DIR = r"C:\Users\Admin\Downloads\backtest_results"

# --- trade economics -------------------------------------------------------
LOT_SIZE       = 65        # index units per lot
LOTS_PER_TRADE = 1         # lots added by EVERY on-change entry
CHARGE_PER_LOT = 2000.0    # round-trip charge per lot, rupees
SLIPPAGE       = 0.0       # points lost per fill

# --- the optional capital-dependent block ---------------------------------
SHOW_CAPITAL_STATS = True  # False = drop it entirely
INITIAL_CAPITAL = None     # None = basis A auto (1-lot notional at first close)
ROUND_CAPITAL_UP_TO = 0    # 0 = exact peak notional; 1e7 = round up to a crore
SHARED_PEAK_CAPITAL = True # one peak basis for all versions (comparable)

# --- engine ---------------------------------------------------------------
USE_DENSE_GRID   = True    # left-join signals onto EVERY price bar
MISSING_PREVIEW  = 8       # missing timestamps listed in the warning
MIN_YEARS_FOR_CAGR = 1.0   # shorter than this -> CAGR / Calmar are n/a
ANNUALISE_FROM_DATA = True # annualise with observed sessions, not 252
TRADING_DAYS     = 252     # fallback when ANNUALISE_FROM_DATA is False
TIME_FORMAT      = None    # e.g. "%d-%m-%Y %H:%M:%S" to force the parse
STRICT_MATCH     = False   # abort if a signal row has no price bar

# --- cost decomposition (needs a TRUE label column in the signal file) -----
COMPUTE_COST_DECOMP = True # churn / missed opportunity / wrong direction.
                           # Missed opportunity runs a SHADOW simulation on the
                           # TRUE labels, so this roughly doubles the runtime.
COMPUTE_DELAY_COST = True  # needs an ATR column in the price file (or one is
                           # computed with the Wilder recursion)
ATR_PERIOD = 14            # only used when the price file has no ATR column

# --- output ---------------------------------------------------------------
WRITE_BAR_EQUITY = False   # True adds a per-bar equity sheet. SLOW and huge
                           # (bars x versions rows) - daily equity is always
                           # written and is usually what you want.
# ===========================================================================

_Q = LOTS_PER_TRADE
VERSIONS = [
    dict(key="V1", name="SUPER entry quality",
         desc="ENTRY_SUPER -> exit on EXIT_SUB or EXIT_SUPER (whichever comes first)",
         question="Is the SUPER entry signal any good when released early?",
         qty={"ENTRY_SUPER": _Q},
         exits={"EXIT_SUB": {"SUPER"}, "EXIT_SUPER": {"SUPER"}}),
    dict(key="V2", name="SUB entry quality",
         desc="ENTRY_SUB   -> exit on EXIT_SUB or EXIT_SUPER (whichever comes first)",
         question="Is the SUB entry signal working on its own?",
         qty={"ENTRY_SUB": _Q},
         exits={"EXIT_SUB": {"SUB"}, "EXIT_SUPER": {"SUB"}}),
    dict(key="V3", name="SUPER class quality",
         desc="ENTRY_SUPER -> exit on EXIT_SUPER only",
         question="Is the SUPER class self-consistent (SUPER in, SUPER out)?",
         qty={"ENTRY_SUPER": _Q},
         exits={"EXIT_SUPER": {"SUPER"}}),
]

ALL_ENTRIES = {"ENTRY_SMALL", "ENTRY_SUB", "ENTRY_SUPER"}
ALL_EXITS = {"EXIT_SMALL", "EXIT_SUB", "EXIT_SUPER"}
ENTRY_TIER = {"ENTRY_SMALL": "SMALL", "ENTRY_SUB": "SUB", "ENTRY_SUPER": "SUPER"}
NO_TRADE = "NO_TRADE"
W = 96

TIME_COLS = ["timestamp", "datetime", "ts", "time", "date"]
SIG_COLS = ["predicted_label", "primary_label", "signal", "prediction", "y_pred", "label"]
TRUTH_COLS = ["target_label", "true_label", "actual_label", "y_true", "label_true"]
OPEN_COLS = ["Nifty_Futures_Open", "open", "Open"]
HIGH_COLS = ["Nifty_Futures_High", "high", "High"]
LOW_COLS = ["Nifty_Futures_Low", "low", "Low"]
CLOSE_COLS = ["Nifty_Futures_Close", "close", "Close"]
ATR_COLS = ["ATR_14_NEW_FUTURES", "atr_14", "ATR_14", "atr", "ATR"]
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

BREAKEVEN_POINTS = CHARGE_PER_LOT / LOT_SIZE

CAPITAL_DEPENDENT = ["total_return_pct", "gross_return_pct", "arr_pct",
                     "cagr_pct", "cagr_gross_pct", "max_drawdown_pct",
                     "avg_drawdown_pct", "max_drawdown_gross_pct", "calmar",
                     "sharpe", "sortino", "sharpe_gross"]

try:
    import python_calamine  # noqa: F401
    _XL = {"engine": "calamine"}
except ImportError:
    _XL = {}


# ---------------------------------------------------------------- metric doc
# every metric: what it is and exactly how it is worked out.  Written into the
# README and Metrics sheets so the workbook explains itself.
METRIC_DOC = {
    "gross_pnl": ("Total P&L BEFORE charges", "sum of pnl_money_gross = sum(pnl_points x 65)"),
    "total_charge": ("Total round-trip charges", "sum of charge = 2000 x lots closed"),
    "net_pnl": ("Total P&L AFTER charges", "gross_pnl - total_charge"),
    "arr_gross": ("Gross rupees per year", "gross_pnl / years"),
    "arr": ("Net rupees per year", "net_pnl / years"),
    "gross_pnl_points": ("Total index points won", "sum of pnl_points"),
    "avg_points_per_lot": ("Average market move per trade", "mean(pnl_points / qty)"),
    "breakeven_points_per_lot": ("Points needed just to pay the charge", "2000 / 65"),
    "charge_per_trade": ("Charge borne by one trade", "total_charge / n_trades"),
    "charge_pct_of_gross_pnl": ("Charges as a share of gross P&L", "total_charge / |gross_pnl| x 100"),
    "charge_pct_of_gross_wins": ("Charges vs winning trades only", "total_charge / gross_profit_won x 100"),
    "net_to_gross_pct": ("Share of gross P&L kept", "net_pnl / gross_pnl x 100 (only if gross > 0)"),
    "total_slippage": ("Slippage cost booked", "sum(2 x SLIPPAGE x qty x 65)"),
    "n_trades": ("Closed positions", "row count of the tradebook"),
    "total_lots": ("Lots traded in total", "sum of qty"),
    "win_rate_gross": ("Trades profitable BEFORE charges", "count(gross > 0) / n_trades"),
    "win_rate": ("Trades profitable AFTER charges", "count(net > 0) / n_trades"),
    "n_wins_gross": ("Winners before charges", "count(pnl_money_gross > 0)"),
    "n_losses_gross": ("Losers before charges", "count(pnl_money_gross < 0)"),
    "n_wins": ("Winners after charges", "count(pnl_money > 0)"),
    "n_losses": ("Losers after charges", "count(pnl_money < 0)"),
    "n_breakeven": ("Exactly flat trades", "n_trades - n_wins - n_losses"),
    "profit_factor_gross": ("Gross won / gross lost", "sum(gross>0) / |sum(gross<0)|"),
    "profit_factor": ("Net won / net lost", "sum(net>0) / |sum(net<0)|"),
    "expectancy_gross": ("Average trade before charges", "mean(pnl_money_gross)"),
    "expectancy": ("Average trade after charges", "mean(pnl_money)"),
    "avg_win_gross": ("Average winner before charges", "mean of positive pnl_money_gross"),
    "avg_loss_gross": ("Average loser before charges", "|mean of negative pnl_money_gross|"),
    "avg_win": ("Average winner after charges", "mean of positive pnl_money"),
    "avg_loss": ("Average loser after charges", "|mean of negative pnl_money|"),
    "gross_profit_won": ("Rupees made on gross winners", "sum of positive pnl_money_gross"),
    "gross_loss_lost": ("Rupees lost on gross losers", "|sum of negative pnl_money_gross|"),
    "gross_profit": ("Rupees made on NET winners", "sum of positive pnl_money"),
    "gross_loss": ("Rupees lost on NET losers", "|sum of negative pnl_money|"),
    "avg_holding_minutes": ("Average time in a trade", "mean(exit_time - entry_time) in minutes"),
    "avg_holding_bars": ("Average bars in a trade", "mean(bars_held)"),
    "avg_mfe_points": ("Avg best favourable move", "mean(mfe_points), a POSITIVE magnitude"),
    "avg_mae_points": ("Avg worst adverse move", "mean(mae_points), a POSITIVE magnitude"),
    "best_mfe_points": ("Largest favourable move seen", "max(mfe_points)"),
    "worst_mae_points": ("Largest adverse move seen", "max(mae_points)"),
    "avg_mfe_money": ("Avg favourable move in rupees", "mean(mfe_points x qty x 65)"),
    "avg_mae_money": ("Avg adverse move in rupees", "mean(mae_points x qty x 65)"),
    "mfe_capture_pct": ("Share of the favourable move banked", "sum(points_per_lot) / sum(mfe_points) x 100"),
    "edge_ratio": ("Reward vs pain while holding", "mean(mfe_points) / mean(mae_points)"),
    "pct_trades_mfe_over_breakeven": ("Trades ever in profit by more than the cost",
                                      "count(mfe_points > 30.77) / n_trades x 100"),
    "n_false_trades": ("Trades opened when the TRUE label was NO_TRADE",
                       "count of entries whose signal bar had true label NO_TRADE"),
    "churn_cost": ("CHURN - money burnt over-trading", "n_false_trades x 2000 x qty"),
    "churn_rate_pct": ("Share of trades that should never have happened",
                       "n_false_trades / n_trades x 100"),
    "shadow_trades": ("Trades a PERFECT model would have taken",
                      "same rules replayed on the TRUE labels"),
    "shadow_gross_pnl": ("Gross P&L of that perfect book", "sum of shadow pnl_money_gross"),
    "n_missed_trades": ("Ideal trades the model skipped",
                        "shadow trades whose signal bar the model called NO_TRADE"),
    "missed_opportunity_cost": ("MISSED OPPORTUNITY - profit left on the table",
                                "sum of max(0, gross P&L) over the skipped shadow trades"),
    "missed_gross_raw": ("Same sum WITHOUT the max(0,.) floor",
                         "sum of gross P&L over skipped trades - shown for contrast; "
                         "skipping a loser is a good call, so the floored figure is the metric"),
    "n_flip_trades": ("Trades taken on the wrong side",
                      "entered while truth said EXIT, or exited while truth said ENTRY"),
    "wrong_direction_cost": ("WRONG DIRECTION - damage from reversal calls",
                             "sum of |gross P&L| over those trades"),
    "n_delayed_trades": ("Correct calls that fired late",
                         "entries whose truth run had already started earlier"),
    "avg_delay_bars": ("Bars late, on average", "mean(signal bar - first bar of the true run)"),
    "avg_delay_points": ("Points of price given up by being late",
                         "mean(entry_price - open at the ideal fill bar)"),
    "avg_delay_atr": ("The same, in ATR units", "mean(delay_points / ATR at the ideal bar)"),
    "delay_cost": ("DELAY - rupees given up entering late",
                   "sum(delay_points x qty x 65). Negative means the late fill was better"),
    "total_error_cost": ("Churn + missed opportunity + wrong direction",
                         "the three costs added; delay is reported separately"),
    "max_drawdown": ("Deepest equity fall in rupees", "min(equity - running max(equity)), t0 = 0"),
    "max_drawdown_gross": ("Deepest gross equity fall", "same on the pre-charge curve"),
    "avg_drawdown": ("Average drawdown episode depth", "mean of each episode's trough, rupees"),
    "n_drawdowns": ("Peak-to-recovery episodes", "count of runs where equity < its running max"),
    "pct_time_in_dd": ("Share of bars underwater", "mean(equity < running max) x 100"),
    "peak_lots": ("Most lots open at once", "max(lots_open)"),
    "avg_lots_in_market": ("Avg lots while holding", "mean(lots_open where > 0)"),
    "pct_bars_in_market": ("Share of bars holding a position", "mean(lots_open > 0) x 100"),
    "pct_bars_multi_lot": ("Share of bars above 1 lot", "mean(lots_open > 1) x 100"),
    "peak_notional": ("Largest rupee exposure carried", "max(close x 65 x lots_open)"),
    "avg_notional_in_market": ("Avg rupee exposure while holding", "mean of that, where > 0"),
    "n_trading_days": ("Sessions with at least one bar", "distinct dates in the grid"),
    "sessions_per_year": ("Observed sessions per year", "n_trading_days / years"),
    "annualisation_factor": ("Factor used for Sharpe/Sortino", "sessions_per_year, or 252 if fixed"),
    "years": ("Window length", "(last ts - first ts).days / 365.25"),
    "cagr_suppressed": ("CAGR withheld as too short", "years < MIN_YEARS_FOR_CAGR"),
    # capital dependent
    "capital": ("Assumed capital for this basis", "A = first close x 65 x lots; B = peak notional"),
    "total_return_pct": ("Net return on assumed capital", "net_pnl / capital x 100"),
    "gross_return_pct": ("Gross return on assumed capital", "gross_pnl / capital x 100"),
    "arr_pct": ("Annual return on assumed capital", "(net_pnl / years) / capital x 100"),
    "cagr_pct": ("Compounded annual growth, net", "((capital + net_pnl)/capital)^(1/years) - 1"),
    "cagr_gross_pct": ("Compounded annual growth, gross", "same with gross_pnl"),
    "max_drawdown_pct": ("Deepest fall as % of capital", "min(wealth / running max(wealth) - 1) x 100"),
    "max_drawdown_gross_pct": ("Same on the pre-charge curve", "as above, gross equity"),
    "avg_drawdown_pct": ("Average episode depth in %", "mean of each episode's trough %"),
    "calmar": ("Growth per unit of drawdown", "cagr_pct / |max_drawdown_pct|"),
    "sharpe": ("Return per unit of volatility", "mean(daily return)/sd x sqrt(sessions per year)"),
    "sortino": ("Return per unit of DOWNSIDE volatility", "mean(daily return)/downside x sqrt(spy)"),
    "sharpe_gross": ("Sharpe on the pre-charge curve", "as sharpe, on gross equity"),
}


# ============================ LOAD DATA ====================================

def _pick(cands, cols, what):
    for c in cands:
        if c in cols:
            return c
    raise SystemExit(f"ERROR: no {what} column found in {list(cols)[:12]}")


def _parse_time(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if TIME_FORMAT:
        return pd.to_datetime(s, format=TIME_FORMAT, errors="coerce")
    a = pd.to_datetime(s, errors="coerce")
    b = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if a.isna().sum() == b.isna().sum() and not a.equals(b):
        print("   WARNING: the timestamp column is text and dd-mm vs mm-dd is "
              "AMBIGUOUS. Set TIME_FORMAT in CONFIG.")
    return b if b.isna().sum() < a.isna().sum() else a


def _dedupe(df, what):
    dup = df["ts"].duplicated(keep=False)
    if dup.any():
        cols = [c for c in df.columns if c != "ts"]
        conflict = df[dup].groupby("ts")[cols].nunique().max(axis=1)
        n_bad = int((conflict > 1).sum())
        if n_bad:
            bad = conflict[conflict > 1].index[:5]
            raise SystemExit(
                f"ERROR: {n_bad:,} {what} timestamps repeat with DIFFERENT values.\n"
                f"   first offenders: {[str(b) for b in bad]}")
        print(f"   note: dropped {int(df['ts'].duplicated().sum()):,} duplicate {what} rows")
    return df.drop_duplicates("ts")


def _pick_sheet(path):
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
    """Predictions, plus the TRUE label column if the file carries one."""
    used_sheet = ""
    low = str(path).lower()
    if low.endswith((".xlsx", ".xlsm", ".xls")):
        used_sheet = sheet or _pick_sheet(path)
        df = pd.read_excel(path, sheet_name=used_sheet, **_XL)
    elif low.endswith((".parquet", ".pq")):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # a scored-predictions table often carries train/val rows too - keep test
    if "split" in df.columns:
        sp = df["split"].astype(str).str.lower()
        kinds = sorted(sp.unique())
        if "test" in kinds and len(kinds) > 1:
            print(f"   split column found {kinds} - keeping 'test' only "
                  f"({int((sp == 'test').sum()):,} of {len(df):,} rows)")
            df = df[sp == "test"]
        else:
            print(f"   split column: {kinds}")
    tc = _pick(TIME_COLS, df.columns, "signal time")
    sc = _pick(SIG_COLS, df.columns, "signal")
    out = pd.DataFrame({"ts": _parse_time(df[tc]),
                        "sig": df[sc].astype(str).str.strip().str.upper()})
    truth_col = next((c for c in TRUTH_COLS if c in df.columns and c != sc), None)
    if truth_col:
        out["truth"] = df[truth_col].astype(str).str.strip().str.upper()
    out = out.dropna(subset=["ts"]).sort_values("ts")
    out = _dedupe(out, "signal").reset_index(drop=True)
    out.attrs["sheet"] = used_sheet
    out.attrs["truth_col"] = truth_col or ""
    return out


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
    acol = next((a for a in ATR_COLS if a in df.columns), None)
    out["atr"] = df[acol].astype(float) if acol else np.nan
    out = out.dropna(subset=["ts"]).sort_values("ts")
    out = _dedupe(out, "price").reset_index(drop=True)
    if acol is None and COMPUTE_DELAY_COST:
        out["atr"] = _wilder_atr(out, ATR_PERIOD)
    out.attrs["atr_col"] = acol or (f"computed Wilder({ATR_PERIOD})"
                                    if COMPUTE_DELAY_COST else "none")
    return out


def _wilder_atr(px, period=14):
    """Canonical Wilder ATR: SMA of the first `period` true ranges, then the
    recursion atr = (atr*(n-1) + tr) / n.  NOT an ewm - that is a different
    series and is badly wrong through the warm-up."""
    h, l, c = px["high"].to_numpy(float), px["low"].to_numpy(float), px["close"].to_numpy(float)
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    n = len(tr)
    atr = np.full(n, np.nan)
    if n <= period:
        return atr
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def validate_price(px):
    q = px[["open", "high", "low", "close"]].to_numpy(float)
    return {
        "null_ohlc": int(np.isnan(q).sum()),
        "non_finite": int((~np.isfinite(q)).sum()),
        "high<low": int((px["high"] < px["low"]).sum()),
        "high<open": int((px["high"] < px["open"]).sum()),
        "high<close": int((px["high"] < px["close"]).sum()),
        "low>open": int((px["low"] > px["open"]).sum()),
        "low>close": int((px["low"] > px["close"]).sum()),
        "price<=0": int((q <= 0).sum()),
    }


# ============================== ENGINE =====================================

def prepare(price_df, signal_df):
    """Signal onto the price grid once. Gap bars are flagged UNKNOWN."""
    lo, hi = signal_df["ts"].min(), signal_df["ts"].max()
    grid = price_df[(price_df["ts"] >= lo) & (price_df["ts"] <= hi)].reset_index(drop=True)
    matched = int(signal_df["ts"].isin(set(grid["ts"])).sum())
    unmatched = len(signal_df) - matched
    if matched == 0:
        raise SystemExit("ERROR: no signal timestamp matched a price bar (check dates/timezone).")
    if unmatched and STRICT_MATCH:
        raise SystemExit(f"ERROR: {unmatched:,} signal rows have no price bar (STRICT_MATCH).")

    has_truth = "truth" in signal_df.columns
    missing_ts = []
    if USE_DENSE_GRID:
        df = grid.merge(signal_df, on="ts", how="left")
        miss = df["sig"].isna()
        gaps = int(miss.sum())
        missing_ts = list(df.loc[miss, "ts"].head(MISSING_PREVIEW))
        df["is_gap"] = miss.to_numpy()
        df["sig"] = df["sig"].fillna(NO_TRADE)
    else:
        df = signal_df.merge(price_df, on="ts", how="inner")
        df["is_gap"] = False
        gaps = 0
    if has_truth:
        df["truth"] = df["truth"].fillna("")
    else:
        df["truth"] = ""
    df = df.sort_values("ts").reset_index(drop=True)

    # ---- prediction-level churn: model quality, independent of any version --
    churn = dict(has_truth=has_truth, truth_col=signal_df.attrs.get("truth_col", ""),
                 truth_has_no_trade=False, pred_trade_rows=0, false_pred_rows=0,
                 churn_rate_pred_pct=np.nan)
    if has_truth:
        known = ~df["is_gap"].to_numpy()
        t = df["truth"].to_numpy()
        s = df["sig"].to_numpy()
        churn["truth_has_no_trade"] = bool((t[known] == NO_TRADE).any())
        pred_trade = known & (s != NO_TRADE)
        false_pred = pred_trade & (t == NO_TRADE)
        churn["pred_trade_rows"] = int(pred_trade.sum())
        churn["false_pred_rows"] = int(false_pred.sum())
        churn["churn_rate_pred_pct"] = (churn["false_pred_rows"] / churn["pred_trade_rows"] * 100.0
                                        if churn["pred_trade_rows"] else np.nan)

    info = dict(price_bars_in_window=len(grid), signal_rows=len(signal_df),
                matched=matched, unmatched=unmatched, gap_bars=gaps,
                gap_pct=gaps / len(df) * 100.0 if len(df) else 0.0,
                missing_ts=missing_ts,
                coverage_pct=matched / len(grid) * 100.0 if len(grid) else np.nan,
                engine_bars=len(df), dense=USE_DENSE_GRID, churn=churn,
                sheet=signal_df.attrs.get("sheet", ""))
    return df, info


def missing_warning(info):
    """The ONE place this warning is produced. Emitted exactly once."""
    if not info["gap_bars"]:
        return []
    L = ["", "!" * W,
         f"!!  WARNING - {info['gap_bars']:,} PRICE BARS HAVE NO PREDICTION "
         f"({info['gap_pct']:.2f}% of the grid)",
         "!!",
         "!!  A missing row means one of two things and the script cannot tell which:",
         "!!      1. the correct signal genuinely was NO_TRADE, or",
         "!!      2. the model prediction is accidentally absent.",
         "!!",
         "!!  Handling: a missing bar NEVER opens or closes a position, and is treated",
         "!!  as UNKNOWN rather than a NO_TRADE state change - so a real signal after a",
         "!!  gap is compared to the last KNOWN signal and a gap can NOT manufacture a",
         "!!  fresh entry out of one continuing signal.",
         "!!",
         "!!  REVIEW THIS COUNT before trusting the result.",
         "!!  first missing: " + ", ".join(f"{pd.Timestamp(t):%Y-%m-%d %H:%M}"
                                           for t in info["missing_ts"][:4]),
         "!" * W]
    return L


def compute_fires(sig, gap):
    """Genuine signal changes. A gap never fires and never resets the state."""
    n = len(sig)
    fires = np.zeros(n, dtype=bool)
    last_known = None
    for i in range(n):
        if gap[i]:
            continue
        if last_known is None or sig[i] != last_known:
            fires[i] = True
        last_known = sig[i]
    naive = np.empty(n, dtype=bool)
    naive[0] = True
    naive[1:] = sig[1:] != sig[:-1]
    return fires, int((naive & ~fires & ~gap).sum())


def run(df, qty_map, exit_map, use_truth=False):
    """Simulate. Returns (tradebook, equity, equity_gross, lots_open, counters).

    Every bar i:  1) execute what bar i-1 queued, at bar i's OPEN
                  2) mark the book to bar i's CLOSE
                  3) stamp equity
                  4) read bar i's signal, queue it for bar i+1

    use_truth=True replays the identical rules on the TRUE labels instead of
    the predictions - the SHADOW run that says what a perfect model would have
    traded.  Its trades are what "missed opportunity" is measured against.
    """
    ts = df["ts"].to_numpy()
    truth = df["truth"].to_numpy()
    if use_truth:
        sig = truth
        gap = df["is_gap"].to_numpy(bool) | (truth == "")
    else:
        sig = df["sig"].to_numpy()
        gap = df["is_gap"].to_numpy(bool)
    op, hi, lo, cl = (df["open"].to_numpy(float), df["high"].to_numpy(float),
                      df["low"].to_numpy(float), df["close"].to_numpy(float))
    n = len(df)
    fires, suppressed = compute_fires(sig, gap)

    book, trades = [], []
    realized_pts = cum_charge = 0.0
    equity = np.zeros(n)
    equity_gross = np.zeros(n)
    lots_open = np.zeros(n)
    pending = None
    stacked = 0

    def close(tiers, price, when, i, at_close=False, reason="signal", sig_i=None):
        """MAE/MFE span the bars held PLUS the exit fill, and are POSITIVE
        MAGNITUDES: mfe = how far it went in favour, mae = how far against.
        Bound: -mae <= points_per_lot <= mfe.
        sig_i is the bar the EXIT signal fired on, so the true label at the
        moment of the exit decision can be recorded (wrong-direction cost)."""
        nonlocal realized_pts, cum_charge, book
        keep, done = [], []
        for p in book:
            (done if p["tier"] in tiers else keep).append(p)
        book = keep
        for p in done:
            exit_px = price - SLIPPAGE
            pnl_pts = (exit_px - p["price"]) * p["qty"]
            charge = CHARGE_PER_LOT * p["qty"]
            realized_pts += pnl_pts
            cum_charge += charge
            gross = pnl_pts * LOT_SIZE
            a = p["idx"]
            b = i if at_close else i - 1
            b = max(b, a)
            best = max(float(hi[a:b + 1].max()), exit_px)
            worst = min(float(lo[a:b + 1].min()), exit_px)
            mfe_pts = max(0.0, best - p["price"])
            mae_pts = max(0.0, p["price"] - worst)
            trades.append(dict(
                tier=p["tier"], qty=p["qty"],
                signal_idx=p["sig_idx"], signal_time=p["sig_time"],
                entry_signal=p["signal"], entry_true_label=p["truth"],
                exit_signal_idx=(sig_i if sig_i is not None else -1),
                exit_true_label=(truth[sig_i] if sig_i is not None else ""),
                entry_time=p["time"], entry_price=p["price"],
                exit_time=when, exit_price=exit_px, exit_reason=reason,
                bars_held=int(b - a + 1),
                points_per_lot=pnl_pts / p["qty"], pnl_points=pnl_pts,
                mfe_points=mfe_pts, mae_points=mae_pts,
                mfe_money=mfe_pts * p["qty"] * LOT_SIZE,
                mae_money=mae_pts * p["qty"] * LOT_SIZE,
                pnl_money_gross=gross, charge=charge, pnl_money=gross - charge,
                slippage_cost=2 * SLIPPAGE * p["qty"] * LOT_SIZE,
                is_false_trade=bool(p["truth"] == NO_TRADE)))

    for i in range(n):
        if pending is not None:
            s, si = pending
            pending = None
            if s in qty_map:
                tier = ENTRY_TIER[s]
                if any(p["tier"] == tier for p in book):
                    stacked += 1
                book.append(dict(tier=tier, qty=qty_map[s], idx=i,
                                 price=op[i] + SLIPPAGE, time=ts[i], signal=s,
                                 sig_idx=si, sig_time=ts[si], truth=truth[si]))
            elif s in exit_map and book:
                close(exit_map[s], op[i], ts[i], i, sig_i=si)

        unreal = lots = 0.0
        for p in book:
            unreal += (cl[i] - p["price"]) * p["qty"]
            lots += p["qty"]
        equity_gross[i] = (realized_pts + unreal) * LOT_SIZE
        equity[i] = equity_gross[i] - cum_charge
        lots_open[i] = lots

        s = sig[i]
        if fires[i] and s != NO_TRADE and (s in qty_map or s in exit_map):
            pending = (s, i)

    unexecuted = (pending[0], ts[pending[1]]) if pending is not None else None
    forced = len(book)
    if book:
        close({"SMALL", "SUB", "SUPER"}, cl[-1], ts[-1], n - 1, at_close=True,
              reason="end_of_data")
        equity_gross[-1] = realized_pts * LOT_SIZE
        equity[-1] = equity_gross[-1] - cum_charge
        lots_open[-1] = 0.0

    tr = pd.DataFrame(trades)
    if len(tr):
        for c in ("signal_time", "entry_time", "exit_time"):
            tr[c] = pd.to_datetime(tr[c])
        tr = tr.sort_values("exit_time").reset_index(drop=True)

    fired = pd.Series(sig[fires]).value_counts().to_dict()
    counters = {"_forced_closes": forced, "_stacked_entries": stacked,
                "_gap_suppressed": suppressed, "_unexecuted": unexecuted,
                **{k: int(v) for k, v in fired.items()}}
    return tr, equity, equity_gross, lots_open, counters


# ============================== METRICS ====================================

def _dd_episodes(dd):
    in_dd = dd < 0
    if not in_dd.any():
        return np.array([])
    prev = np.r_[False, in_dd[:-1]]
    nxt = np.r_[in_dd[1:], False]
    starts = np.flatnonzero(in_dd & ~prev)
    ends = np.flatnonzero(in_dd & ~nxt) + 1
    return np.array([dd[s:e].min() for s, e in zip(starts, ends)])


def _dd_money(equity):
    e0 = np.r_[0.0, np.asarray(equity, float)]
    return e0 - np.maximum.accumulate(e0), e0


def _dd_pct(e0, capital):
    curve = capital + e0
    return (curve / np.maximum.accumulate(curve) - 1.0) * 100.0


def _cagr(capital, pnl, years):
    if years < MIN_YEARS_FOR_CAGR:
        return np.nan
    g = (capital + pnl) / capital
    return (g ** (1.0 / years) - 1.0) * 100.0 if g > 0 else np.nan


def _daily_levels(equity, ts):
    """End-of-day equity for every session that has bars. Flat days are kept;
    weekends and holidays have no bars and drop out."""
    return (pd.Series(equity, index=pd.to_datetime(ts)).resample("1D")
            .last().dropna().to_numpy(float))


def _sharpe_sortino(daily_levels, capital, ann):
    """Conventional RETURN-based Sharpe/Sortino, first trading day included."""
    if len(daily_levels) < 3:
        return np.nan, np.nan
    wealth = capital + np.r_[0.0, daily_levels]
    if (wealth[:-1] <= 0).any():
        return np.nan, np.nan
    r = np.diff(wealth) / wealth[:-1]
    sd = r.std(ddof=1)
    if not sd > 0:
        return np.nan, np.nan
    sharpe = float(r.mean() / sd * np.sqrt(ann))
    dn = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    return sharpe, (float(r.mean() / dn * np.sqrt(ann)) if dn > 0 else np.inf)


def capital_stats(m, capital):
    """Everything that divides by an ASSUMED capital."""
    years, net, gross = m["years"], m["net_pnl"], m["gross_pnl"]
    c = {"capital": float(capital)}
    c["total_return_pct"] = net / capital * 100.0
    c["gross_return_pct"] = gross / capital * 100.0
    c["arr_pct"] = (net / years) / capital * 100.0
    c["cagr_pct"] = _cagr(capital, net, years)
    c["cagr_gross_pct"] = _cagr(capital, gross, years)
    dd = _dd_pct(m["_e0_net"], capital)
    c["max_drawdown_pct"] = float(dd.min())
    c["max_drawdown_gross_pct"] = float(_dd_pct(m["_e0_gross"], capital).min())
    eps = _dd_episodes(dd)
    c["avg_drawdown_pct"] = float(eps.mean()) if len(eps) else 0.0
    if not np.isfinite(c["cagr_pct"]):
        c["calmar"] = np.nan
    elif c["max_drawdown_pct"] < 0:
        c["calmar"] = c["cagr_pct"] / abs(c["max_drawdown_pct"])
    else:
        c["calmar"] = np.inf
    ann = m["annualisation_factor"]
    c["sharpe"], c["sortino"] = _sharpe_sortino(m["_daily_equity"], capital, ann)
    c["sharpe_gross"], _ = _sharpe_sortino(m["_daily_equity_gross"], capital, ann)
    return c


def metrics(tr, equity, equity_gross, lots, bars, churn_info):
    """CAPITAL-INDEPENDENT metrics only."""
    has = len(tr) > 0
    p = tr["pnl_money"].to_numpy() if has else np.array([])
    pg = tr["pnl_money_gross"].to_numpy() if has else np.array([])
    wins, losses = p[p > 0], p[p < 0]
    gwins, glosses = pg[pg > 0], pg[pg < 0]
    m = {}

    m["gross_pnl"] = float(pg.sum())
    m["total_charge"] = float(tr["charge"].sum()) if has else 0.0
    m["net_pnl"] = float(p.sum())
    m["total_slippage"] = float(tr["slippage_cost"].sum()) if has else 0.0
    m["gross_pnl_points"] = float(tr["pnl_points"].sum()) if has else 0.0
    m["n_trades"] = int(len(tr))
    m["total_lots"] = float(tr["qty"].sum()) if has else 0.0
    m["charge_per_trade"] = m["total_charge"] / m["n_trades"] if m["n_trades"] else np.nan
    m["breakeven_points_per_lot"] = BREAKEVEN_POINTS
    m["avg_points_per_lot"] = float(tr["points_per_lot"].mean()) if has else np.nan

    m["n_wins"], m["n_losses"] = int(len(wins)), int(len(losses))
    m["n_breakeven"] = int(m["n_trades"] - m["n_wins"] - m["n_losses"])
    m["win_rate"] = len(wins) / len(p) if len(p) else np.nan
    m["gross_profit"] = float(wins.sum())
    m["gross_loss"] = float(-losses.sum())
    m["profit_factor"] = (m["gross_profit"] / m["gross_loss"]) if m["gross_loss"] else np.inf
    m["avg_win"] = float(wins.mean()) if len(wins) else 0.0
    m["avg_loss"] = float(-losses.mean()) if len(losses) else 0.0
    m["expectancy"] = float(p.mean()) if len(p) else np.nan

    m["n_wins_gross"], m["n_losses_gross"] = int(len(gwins)), int(len(glosses))
    m["win_rate_gross"] = len(gwins) / len(pg) if len(pg) else np.nan
    m["gross_profit_won"] = float(gwins.sum())
    m["gross_loss_lost"] = float(-glosses.sum())
    m["profit_factor_gross"] = ((m["gross_profit_won"] / m["gross_loss_lost"])
                                if m["gross_loss_lost"] else np.inf)
    m["avg_win_gross"] = float(gwins.mean()) if len(gwins) else 0.0
    m["avg_loss_gross"] = float(-glosses.mean()) if len(glosses) else 0.0
    m["expectancy_gross"] = float(pg.mean()) if len(pg) else np.nan

    m["charge_pct_of_gross_pnl"] = (m["total_charge"] / abs(m["gross_pnl"]) * 100.0
                                    if m["gross_pnl"] else np.nan)
    m["net_to_gross_pct"] = (m["net_pnl"] / m["gross_pnl"] * 100.0
                             if m["gross_pnl"] > 0 else np.nan)
    m["charge_pct_of_gross_wins"] = (m["total_charge"] / m["gross_profit_won"] * 100.0
                                     if m["gross_profit_won"] > 0 else np.nan)
    m["avg_holding_minutes"] = (
        float((tr["exit_time"] - tr["entry_time"]).dt.total_seconds().mean() / 60.0)
        if has else np.nan)
    m["avg_holding_bars"] = float(tr["bars_held"].mean()) if has else np.nan

    # ---- MAE / MFE: positive magnitudes, no abs() needed anywhere ----------
    if has:
        mfe, mae = tr["mfe_points"].to_numpy(), tr["mae_points"].to_numpy()
        ppl = tr["points_per_lot"].to_numpy()
        m["avg_mfe_points"], m["avg_mae_points"] = float(mfe.mean()), float(mae.mean())
        m["best_mfe_points"], m["worst_mae_points"] = float(mfe.max()), float(mae.max())
        m["avg_mfe_money"] = float(tr["mfe_money"].mean())
        m["avg_mae_money"] = float(tr["mae_money"].mean())
        m["mfe_capture_pct"] = (float(ppl.sum() / mfe.sum() * 100.0) if mfe.sum() > 0 else np.nan)
        if mae.mean() > 0:
            m["edge_ratio"] = float(mfe.mean() / mae.mean())
        elif mfe.mean() > 0:
            m["edge_ratio"] = np.inf
        else:
            m["edge_ratio"] = np.nan
        m["pct_trades_mfe_over_breakeven"] = float((mfe > BREAKEVEN_POINTS).mean() * 100.0)
    else:
        for k in ("avg_mfe_points", "avg_mae_points", "best_mfe_points", "worst_mae_points",
                  "avg_mfe_money", "avg_mae_money", "mfe_capture_pct", "edge_ratio",
                  "pct_trades_mfe_over_breakeven"):
            m[k] = np.nan

    # ---- CHURN at the EXECUTED level: what over-trading actually cost ------
    if has and churn_info["has_truth"]:
        false_mask = tr["is_false_trade"].to_numpy(bool)
        m["n_false_trades"] = int(false_mask.sum())
        m["churn_cost"] = float(tr.loc[false_mask, "charge"].sum())
        m["churn_rate_pct"] = float(false_mask.mean() * 100.0)
    else:
        m["n_false_trades"] = 0 if churn_info["has_truth"] else -1
        m["churn_cost"] = np.nan
        m["churn_rate_pct"] = np.nan

    lots = np.asarray(lots, float)
    inmkt = lots[lots > 0]
    notional = bars["close"].to_numpy(float) * LOT_SIZE * lots
    m["peak_lots"] = float(lots.max()) if len(lots) else 0.0
    m["avg_lots_in_market"] = float(inmkt.mean()) if len(inmkt) else 0.0
    m["pct_bars_in_market"] = float((lots > 0).mean() * 100.0) if len(lots) else np.nan
    m["pct_bars_multi_lot"] = float((lots > LOTS_PER_TRADE).mean() * 100.0) if len(lots) else np.nan
    m["peak_notional"] = float(notional.max()) if len(notional) else 0.0
    m["avg_notional_in_market"] = (float(notional[notional > 0].mean())
                                   if (notional > 0).any() else 0.0)

    ddm, e0 = _dd_money(equity)
    ddmg, e0g = _dd_money(equity_gross)
    m["max_drawdown"] = float(ddm.min())
    m["max_drawdown_gross"] = float(ddmg.min())
    eps = _dd_episodes(ddm)
    m["avg_drawdown"] = float(eps.mean()) if len(eps) else 0.0
    m["n_drawdowns"] = int(len(eps))
    m["pct_time_in_dd"] = float((ddm[1:] < 0).mean() * 100.0)
    m["_e0_net"], m["_e0_gross"] = e0, e0g

    years = max((bars["ts"].iloc[-1] - bars["ts"].iloc[0]).days / 365.25, 1e-9)
    m["years"] = years
    m["cagr_suppressed"] = bool(years < MIN_YEARS_FOR_CAGR)
    m["arr"] = m["net_pnl"] / years
    m["arr_gross"] = m["gross_pnl"] / years

    m["_daily_equity"] = _daily_levels(equity, bars["ts"])
    m["_daily_equity_gross"] = _daily_levels(equity_gross, bars["ts"])
    m["n_trading_days"] = int(len(m["_daily_equity"]))
    m["sessions_per_year"] = m["n_trading_days"] / years
    m["annualisation_factor"] = (m["sessions_per_year"] if ANNUALISE_FROM_DATA
                                 else float(TRADING_DAYS))
    return m


def _label_run_onset(labels):
    """Index of the first bar of the contiguous run each bar belongs to."""
    n = len(labels)
    onset = np.zeros(n, dtype=int)
    for i in range(1, n):
        onset[i] = i if labels[i] != labels[i - 1] else onset[i - 1]
    return onset


def cost_decomposition(tr, shadow, bars, has_truth):
    """Where the model's mistakes actually cost money.

      CHURN            true = NO_TRADE, model traded  -> wasted round trips
      MISSED           true = a trade, model said NO_TRADE -> profit forgone,
                       measured as sum(max(0, gross P&L)) of the SHADOW trades
                       the model skipped.  max(0,.) because skipping a LOSING
                       trade is a good call, not a missed opportunity.
      WRONG DIRECTION  model entered while the truth said EXIT, or exited while
                       the truth said ENTRY -> sum |gross P&L| of those trades
      DELAY            model fired later than the truth did -> the price moved
                       between the ideal bar and the actual one, in points, in
                       ATR units and in rupees
    """
    c = {k: np.nan for k in
         ("n_missed_trades", "missed_opportunity_cost", "missed_gross_raw",
          "n_flip_trades", "wrong_direction_cost", "n_delayed_trades",
          "avg_delay_bars", "avg_delay_points", "avg_delay_atr", "delay_cost",
          "shadow_trades", "shadow_gross_pnl", "total_error_cost")}
    if not has_truth:
        return c

    # ---------------- missed opportunity, from the shadow book -------------
    sig_arr = bars["sig"].to_numpy()
    if shadow is not None and len(shadow):
        c["shadow_trades"] = int(len(shadow))
        c["shadow_gross_pnl"] = float(shadow["pnl_money_gross"].sum())
        si = shadow["signal_idx"].to_numpy()
        skipped = sig_arr[si] == NO_TRADE          # model said nothing there
        g = shadow["pnl_money_gross"].to_numpy()[skipped]
        c["n_missed_trades"] = int(skipped.sum())
        c["missed_opportunity_cost"] = float(np.maximum(g, 0.0).sum())
        c["missed_gross_raw"] = float(g.sum())     # for contrast, un-clipped
    else:
        c["shadow_trades"] = 0
        c["shadow_gross_pnl"] = 0.0
        c["n_missed_trades"] = 0
        c["missed_opportunity_cost"] = 0.0
        c["missed_gross_raw"] = 0.0

    if not len(tr):
        c["n_flip_trades"], c["wrong_direction_cost"] = 0, 0.0
        c["n_delayed_trades"], c["delay_cost"] = 0, 0.0
        c["total_error_cost"] = c["missed_opportunity_cost"]
        return c

    # ---------------- wrong direction --------------------------------------
    entry_flip = tr["entry_true_label"].isin(ALL_EXITS).to_numpy()
    exit_flip = tr["exit_true_label"].isin(ALL_ENTRIES).to_numpy()
    flip = entry_flip | exit_flip                  # union: never double counted
    c["n_flip_trades"] = int(flip.sum())
    c["wrong_direction_cost"] = float(np.abs(tr["pnl_money_gross"].to_numpy()[flip]).sum())

    # ---------------- delay -------------------------------------------------
    if COMPUTE_DELAY_COST and "atr" in bars.columns:
        truth_arr = bars["truth"].to_numpy()
        open_arr = bars["open"].to_numpy(float)
        atr_arr = bars["atr"].to_numpy(float)
        onset = _label_run_onset(truth_arr)
        n = len(bars)
        rows = []
        for k in range(len(tr)):
            if tr["entry_true_label"].iloc[k] not in ALL_ENTRIES:
                continue                            # not a late version of a
            si = int(tr["signal_idx"].iloc[k])       # correct call - skip
            k0 = int(onset[si])
            ideal_fill = min(k0 + 1, n - 1)
            ideal_price = open_arr[ideal_fill]
            dpts = float(tr["entry_price"].iloc[k]) - ideal_price
            atr0 = atr_arr[k0]
            rows.append((si - k0, dpts, dpts / atr0 if atr0 and atr0 > 0 else np.nan,
                         dpts * float(tr["qty"].iloc[k]) * LOT_SIZE))
        if rows:
            arr = np.array(rows, dtype=float)
            c["n_delayed_trades"] = int((arr[:, 0] > 0).sum())
            c["avg_delay_bars"] = float(arr[:, 0].mean())
            c["avg_delay_points"] = float(arr[:, 1].mean())
            c["avg_delay_atr"] = float(np.nanmean(arr[:, 2]))
            c["delay_cost"] = float(arr[:, 3].sum())
        else:
            c["n_delayed_trades"], c["delay_cost"] = 0, 0.0
            c["avg_delay_bars"] = c["avg_delay_points"] = c["avg_delay_atr"] = np.nan
    return c


def yearly_stats(tr, equity, equity_gross, bars, capital):
    d = pd.DataFrame({"ts": pd.to_datetime(bars["ts"]), "eq": np.asarray(equity, float),
                      "eqg": np.asarray(equity_gross, float)})
    d["year"] = d["ts"].dt.year
    ty = tr.assign(year=tr["exit_time"].dt.year) if len(tr) else None
    rows, prev, prev_g = [], 0.0, 0.0
    for y, g in d.groupby("year", sort=True):
        eq, eqg = g["eq"].to_numpy(float), g["eqg"].to_numpy(float)
        end, end_g = float(eq[-1]), float(eqg[-1])
        pnl, pnl_g = end - prev, end_g - prev_g
        curve = np.r_[capital + prev, capital + eq]
        dd_pct = float(((curve / np.maximum.accumulate(curve) - 1.0) * 100.0).min())
        arr = np.r_[prev, eq]
        dd_money = float((arr - np.maximum.accumulate(arr)).min())
        n_tr = wins = 0
        charge = 0.0
        if ty is not None:
            sub = ty[ty["year"] == y]
            n_tr, wins = len(sub), int((sub["pnl_money"] > 0).sum())
            charge = float(sub["charge"].sum())
        rows.append(dict(year=int(y), sessions=int(g["ts"].dt.normalize().nunique()),
                         trades=n_tr, wins=wins,
                         win_rate_pct=(wins / n_tr * 100.0 if n_tr else np.nan),
                         gross_pnl=pnl_g, charges=charge, net_pnl=pnl,
                         max_dd_money=dd_money,
                         gross_return_pct=pnl_g / capital * 100.0,
                         return_pct=pnl / capital * 100.0, max_dd_pct=dd_pct))
        prev, prev_g = end, end_g
    out = pd.DataFrame(rows)
    if len(out):
        out.attrs["pos_years"] = int((out["net_pnl"] > 0).sum())
        out.attrs["pos_years_gross"] = int((out["gross_pnl"] > 0).sum())
        out.attrs["best_year"] = int(out.loc[out["net_pnl"].idxmax(), "year"])
        out.attrs["worst_year"] = int(out.loc[out["net_pnl"].idxmin(), "year"])
        out.attrs["avg_return_pct"] = float(out["return_pct"].mean())
    return out


def monthly_stats(tr):
    if not len(tr):
        return pd.DataFrame()
    t = tr.copy()
    t["month"] = t["exit_time"].dt.to_period("M").astype(str)
    return (t.groupby("month").agg(trades=("pnl_money", "size"),
                                   gross_pnl=("pnl_money_gross", "sum"),
                                   charges=("charge", "sum"),
                                   net_pnl=("pnl_money", "sum"),
                                   wins=("pnl_money", lambda s: int((s > 0).sum())))
            .reset_index())


def monthly_grid(tr, col="pnl_money"):
    if not len(tr):
        return pd.DataFrame()
    t = tr.copy()
    t["y"] = t["exit_time"].dt.year
    t["m"] = t["exit_time"].dt.month
    g = t.pivot_table(index="y", columns="m", values=col, aggfunc="sum")
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


def _cell(x, dp=2):
    if isinstance(x, str):
        return x
    if np.isfinite(x):
        return f"{x:,.{dp}f}"
    return "inf" if x == np.inf else ("-inf" if x == -np.inf else "n/a")


def _h1(title):
    return ["", "=" * W, f"  {title}", "=" * W]


def _h2(title):
    return ["", f"  -- {title} " + "-" * max(W - 7 - len(title), 0)]


def _kv(pairs, kw=30, vw=20, indent="  "):
    bar = f"{indent}+{'-' * (kw + 2)}+{'-' * (vw + 2)}+"
    out = [bar]
    for k, v in pairs:
        out.append(bar if k is None else f"{indent}| {k:<{kw}} | {str(v):>{vw}} |")
    out.append(bar)
    return out


def _grid(headers, rows, widths, indent="  "):
    bar = indent + "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [bar, indent + "|" + "|".join(f" {h:^{w}} " for h, w in zip(headers, widths)) + "|", bar]
    for r in rows:
        out.append(indent + "|" + "|".join(f" {c:>{w}} " for c, w in zip(r, widths)) + "|")
    out.append(bar)
    return out


def setup_block(bars, price_path, signal_path, info):
    yrs = (bars["ts"].iloc[-1] - bars["ts"].iloc[0]).days / 365.25
    sheet = f"   [sheet: {info['sheet']}]" if info["sheet"] else ""
    return _h1("NIFTY SIGNAL BACKTEST   ·   3-VERSION SIGNAL QUALITY SUITE") + [
        f"  signal      {os.path.basename(signal_path)}{sheet}",
        f"  price       {os.path.basename(price_path)}",
        f"  window      {bars['ts'].min():%Y-%m-%d %H:%M}   ->   {bars['ts'].max():%Y-%m-%d %H:%M}",
        f"  span        {len(bars):,} bars   |   "
        f"{bars['ts'].dt.normalize().nunique():,} sessions   |   {yrs:.2f} years",
        f"  economics   1 lot = {LOT_SIZE} qty   |   Rs {CHARGE_PER_LOT:,.0f} per lot round trip"
        f"   |   breakeven {BREAKEVEN_POINTS:.2f} pts",
        f"  rules       fill = NEXT BAR open   |   fire on signal change (gap-aware)"
        f"   |   lots stack",
        f"  slippage    {SLIPPAGE:g} pts per fill",
    ]


def quality_block(info, pxv, capbook):
    ch = info["churn"]
    L = _h2("DATA QUALITY")
    L += _kv([
        ("price bars in window", f"{info['price_bars_in_window']:,}"),
        ("signal rows supplied", f"{info['signal_rows']:,}"),
        ("signal rows matched", f"{info['matched']:,}"),
        ("signal rows unmatched", f"{info['unmatched']:,}"),
        ("bar coverage by signal", f"{info['coverage_pct']:.2f} %"),
        ("bars with NO prediction", f"{info['gap_bars']:,}  ({info['gap_pct']:.2f} %)"),
        ("OHLC violations", "none" if not any(pxv.values())
         else ", ".join(f"{k}={n}" for k, n in pxv.items() if n)),
        (None, None),
        ("true-label column", ch["truth_col"] or "NOT PRESENT"),
        ("prediction-level churn", f"{ch['false_pred_rows']:,} / {ch['pred_trade_rows']:,} rows"
         if ch["has_truth"] else "n/a - no true labels"),
        ("  = churn rate on rows", f"{ch['churn_rate_pred_pct']:.2f} %"
         if ch["has_truth"] and np.isfinite(ch["churn_rate_pred_pct"]) else "n/a"),
    ], kw=26, vw=26)
    if ch["has_truth"] and not ch["truth_has_no_trade"]:
        L.append("    NOTE: the true-label column contains NO 'NO_TRADE' rows at all, so churn")
        L.append("          is structurally 0 here. That is a property of the evaluation file,")
        L.append("          not evidence that the model never over-trades.")
    L += missing_warning(info)                     # <- emitted ONCE, here only
    if SHOW_CAPITAL_STATS:
        L += _h2("ASSUMED CAPITAL BASES (used ONLY by the optional block)")
        L += _kv([
            ("A) 1-lot notional", f"Rs {capbook['lot']:,.0f}"),
            ("B) peak notional", f"Rs {capbook['peak']:,.0f}"),
            ("   peak reached at", capbook["peak_time"][:16] or "-"),
            ("   fund to next crore", f"Rs {capbook['fund_crore']:,.0f}"),
        ], kw=26, vw=26)
        L.append("    B is an UNLEVERAGED return on peak notional - NOT a margin return.")
        L.append("    SPAN blocks ~1.2-1.5 lakh a lot against ~16 lakh notional.")
    return L


def executive_summary(results, capbook):
    """The one table to read if you read nothing else."""
    keys = [v["key"] for v, _, _, _, _, _ in results]
    L = _h1("EXECUTIVE SUMMARY")
    L.append("  Everything here is CAPITAL-INDEPENDENT except the last two rows.")
    L.append("")

    def row(label, fn):
        return [label] + [fn(m, cs) for _, m, cs, _, _, _ in results]

    R = [
        row("Trades taken", lambda m, c: f"{m['n_trades']:,}"),
        row("Gross P&L Rs", lambda m, c: f"{m['gross_pnl']:,.0f}"),
        row("Charges Rs", lambda m, c: f"-{m['total_charge']:,.0f}"),
        row("NET P&L Rs", lambda m, c: f"{m['net_pnl']:,.0f}"),
        row("Net Rs per year", lambda m, c: f"{m['arr']:,.0f}"),
        ["-" * 22] + ["-" * 16 for _ in results],
        row("Avg pts won / lot", lambda m, c: _cell(m["avg_points_per_lot"])),
        row("Breakeven pts needed", lambda m, c: f"{BREAKEVEN_POINTS:.2f}"),
        row("SHORTFALL pts/trade", lambda m, c: _cell(m["avg_points_per_lot"] - BREAKEVEN_POINTS)),
        ["-" * 22] + ["-" * 16 for _ in results],
        row("Win rate gross %", lambda m, c: _cell(m["win_rate_gross"] * 100, 1)),
        row("Win rate net %", lambda m, c: _cell(m["win_rate"] * 100, 1)),
        row("Profit factor net", lambda m, c: _cell(m["profit_factor"])),
        row("Expectancy net Rs", lambda m, c: f"{m['expectancy']:,.0f}"),
        row("Max affordable charge", lambda m, c: f"{m['expectancy_gross']:,.0f}"),
        ["-" * 22] + ["-" * 16 for _ in results],
        row("Max drawdown Rs", lambda m, c: f"{m['max_drawdown']:,.0f}"),
        row("Time in drawdown %", lambda m, c: _cell(m["pct_time_in_dd"], 1)),
        row("Peak lots open", lambda m, c: f"{m['peak_lots']:.0f}"),
        row("Churn cost Rs", lambda m, c: _cell(m["churn_cost"], 0)),
        row("Missed opp Rs", lambda m, c: _cell(m["missed_opportunity_cost"], 0)),
        row("Wrong dir Rs", lambda m, c: _cell(m["wrong_direction_cost"], 0)),
        row("TOTAL error cost Rs", lambda m, c: _cell(m["total_error_cost"], 0)),
    ]
    if SHOW_CAPITAL_STATS:
        R += [["-" * 22] + ["-" * 16 for _ in results],
              row("total_return_% (A)", lambda m, c: _cell(c["lot"]["total_return_pct"])),
              row("Sharpe (A)", lambda m, c: _cell(c["lot"]["sharpe"]))]
    L += _grid(["metric"] + keys, R, [22] + [16] * len(keys))

    L.append("")
    L.append("  VERDICT")
    for v, m, cs, _, _, _ in results:
        short = m["avg_points_per_lot"] - BREAKEVEN_POINTS
        tag = "PROFITABLE" if m["net_pnl"] > 0 else "LOSS"
        L.append(f"    {v['key']}  {tag:<10}  net Rs {m['net_pnl']:>12,.0f}   "
                 f"|  edge {m['avg_points_per_lot']:.2f} pts vs {BREAKEVEN_POINTS:.2f} needed"
                 f"  ->  {'+' if short >= 0 else ''}{short:.2f} pts")
    return L


def version_block(v, m, cs, yr, grid_df, ctr):
    L = ["", "#" * W, f"#  {v['key']}   {v['name'].upper()}",
         f"#  {v['desc']}", f"#  Q: {v['question']}", "#" * W]

    L += _h2("RULES APPLIED")
    ex = " | ".join(f"{k}->{'+'.join(sorted(t))}" for k, t in v["exits"].items())
    en = " | ".join(f"{k} x {q}" for k, q in v["qty"].items())
    fired = " | ".join(f"{k} {ctr.get(k, 0):,}" for k in sorted(set(v['qty']) | set(v['exits'])))
    unex = ctr.get("_unexecuted")
    L += _kv([
        ("entries taken", en),
        ("exits honoured", ex),
        ("ignored", ", ".join(sorted((ALL_ENTRIES | ALL_EXITS) - set(v["qty"]) - set(v["exits"])))),
        ("on-change fires", fired),
        ("gap-suppressed fires", f"{ctr.get('_gap_suppressed', 0):,}"),
        ("stacked entries", f"{ctr.get('_stacked_entries', 0):,}"),
        ("forced closes", f"{ctr.get('_forced_closes', 0):,}"),
        ("unexecuted final signal",
         f"{unex[0]} @ {pd.Timestamp(unex[1]):%Y-%m-%d %H:%M}" if unex else "none"),
    ], kw=24, vw=44)

    if m["n_trades"] == 0:
        L.append("    >>> NO TRADES generated for this version. <<<")
        return L

    L += _h2("CORE  ·  P&L AND TRADE QUALITY   (capital-independent)")
    L += _grid(["metric", "GROSS (pre-charge)", "NET (post-charge)", "difference"], [
        ["P&L Rs", f"{m['gross_pnl']:,.0f}", f"{m['net_pnl']:,.0f}", f"-{m['total_charge']:,.0f}"],
        ["Rs per year", f"{m['arr_gross']:,.0f}", f"{m['arr']:,.0f}", ""],
        ["win rate %", f"{m['win_rate_gross'] * 100:.1f}", f"{m['win_rate'] * 100:.1f}",
         f"{(m['win_rate'] - m['win_rate_gross']) * 100:+.1f}"],
        ["wins / losses", f"{m['n_wins_gross']:,} / {m['n_losses_gross']:,}",
         f"{m['n_wins']:,} / {m['n_losses']:,}", ""],
        ["profit factor", _cell(m["profit_factor_gross"]), _cell(m["profit_factor"]), ""],
        ["expectancy Rs", f"{m['expectancy_gross']:,.0f}", f"{m['expectancy']:,.0f}",
         f"-{m['charge_per_trade']:,.0f}"],
        ["avg win Rs", f"{m['avg_win_gross']:,.0f}", f"{m['avg_win']:,.0f}", ""],
        ["avg loss Rs", f"{m['avg_loss_gross']:,.0f}", f"{m['avg_loss']:,.0f}", ""],
        ["max drawdown Rs", f"{m['max_drawdown_gross']:,.0f}", f"{m['max_drawdown']:,.0f}", ""],
    ], [20, 19, 19, 12])
    L += _kv([
        ("trades / lots", f"{m['n_trades']:,} / {m['total_lots']:,.0f}"),
        ("total points (1 lot)", _num(m["gross_pnl_points"], 14, 1)),
        ("avg pts per lot per trade", _num(m["avg_points_per_lot"], 14, 2)),
        ("breakeven pts needed", _num(BREAKEVEN_POINTS, 14, 2)),
        ("charges / |gross P&L|", _pct(m["charge_pct_of_gross_pnl"], 14, 1)),
        ("net kept of gross", _pct(m["net_to_gross_pct"], 14, 1)),
        ("avg holding minutes", _num(m["avg_holding_minutes"], 14, 1)),
        ("drawdown episodes", f"{m['n_drawdowns']:,}"),
        ("time in drawdown", _pct(m["pct_time_in_dd"], 14, 1)),
    ], kw=26, vw=16)

    L += _h2("CORE  ·  EXCURSION  (MAE / MFE, points per lot, POSITIVE magnitudes)")
    L += _kv([
        ("avg MFE - moved in favour", _num(m["avg_mfe_points"], 14, 2)),
        ("avg MAE - moved against", _num(m["avg_mae_points"], 14, 2)),
        ("largest MFE / MAE seen", f"{m['best_mfe_points']:.2f} / {m['worst_mae_points']:.2f}"),
        ("avg realised pts per lot", _num(m["avg_points_per_lot"], 14, 2)),
        ("MFE capture", _pct(m["mfe_capture_pct"], 14, 1)),
        ("edge ratio MFE / MAE", _num(m["edge_ratio"], 14, 2)),
        ("trades whose MFE beat cost", _pct(m["pct_trades_mfe_over_breakeven"], 14, 1)),
        ("avg MFE / MAE in Rs", f"{m['avg_mfe_money']:,.0f} / {m['avg_mae_money']:,.0f}"),
    ], kw=26, vw=16)

    if np.isfinite(m.get("total_error_cost", np.nan)):
        L += _h2("CORE  ·  COST OF MODEL ERROR   (vs the TRUE labels, all in rupees)")
        L += _grid(["cost", "condition", "trades", "rupees"], [
            ["CHURN", "true NO_TRADE, model traded", f"{m['n_false_trades']:,}",
             f"{m['churn_cost']:,.0f}"],
            ["MISSED OPPORTUNITY", "true traded, model said NO_TRADE",
             f"{m['n_missed_trades']:,}", f"{m['missed_opportunity_cost']:,.0f}"],
            ["WRONG DIRECTION", "ENTRY <-> EXIT flip", f"{m['n_flip_trades']:,}",
             f"{m['wrong_direction_cost']:,.0f}"],
            ["-" * 18, "-" * 32, "-" * 8, "-" * 14],
            ["TOTAL ERROR COST", "the three added", "",
             f"{m['total_error_cost']:,.0f}"],
            ["DELAY (separate)", "correct call, fired late",
             f"{m['n_delayed_trades']:,}", _cell(m["delay_cost"], 0)],
        ], [18, 32, 8, 14])
        L += _kv([
            ("churn rate %", _pct(m["churn_rate_pct"], 14, 2)),
            ("missed, un-floored Rs", _money(m["missed_gross_raw"], 14)),
            ("perfect-model trades", f"{m['shadow_trades']:,}"),
            ("perfect-model gross Rs", _money(m["shadow_gross_pnl"], 14)),
            ("avg delay bars", _num(m["avg_delay_bars"], 14, 2)),
            ("avg delay points", _num(m["avg_delay_points"], 14, 2)),
            ("avg delay in ATR", _num(m["avg_delay_atr"], 14, 3)),
        ], kw=26, vw=16)
        L.append("    MISSED uses max(0, gross P&L) per skipped trade - passing on a LOSING")
        L.append("    trade is a good call, not a missed opportunity, so it contributes 0.")
        L.append("    DELAY is negative when the late fill was actually the better price.")

    L += _h2("CORE  ·  CHURN AND EXPOSURE")
    L += _kv([
        ("false trades (truth flat)", f"{m['n_false_trades']:,}"
         if m["n_false_trades"] >= 0 else "n/a - no true labels"),
        ("churn cost Rs", _money(m["churn_cost"], 14)),
        ("churn rate %", _pct(m["churn_rate_pct"], 14, 2)),
        (None, None),
        ("peak lots open", f"{m['peak_lots']:.0f}"),
        ("avg lots while holding", f"{m['avg_lots_in_market']:.2f}"),
        ("% bars holding / >1 lot",
         f"{m['pct_bars_in_market']:.1f} / {m['pct_bars_multi_lot']:.1f}"),
        ("peak notional Rs", f"{m['peak_notional']:,.0f}"),
        ("avg notional Rs", f"{m['avg_notional_in_market']:,.0f}"),
    ], kw=26, vw=16)

    if SHOW_CAPITAL_STATS and cs:
        L += _h2("OPTIONAL  ·  CAPITAL-DEPENDENT  (an assumption, not a measurement)")
        a, b = cs["lot"], cs["peak"]
        L += _grid(["metric", "A) 1-lot notional", "B) peak notional"], [
            ["assumed capital Rs", f"{a['capital']:,.0f}", f"{b['capital']:,.0f}"],
            ["total_return_% net", _cell(a["total_return_pct"]), _cell(b["total_return_pct"])],
            ["gross_return_%", _cell(a["gross_return_pct"]), _cell(b["gross_return_pct"])],
            ["ARR_% per year", _cell(a["arr_pct"]), _cell(b["arr_pct"])],
            ["CAGR_% net", _cell(a["cagr_pct"]), _cell(b["cagr_pct"])],
            ["CAGR_% gross", _cell(a["cagr_gross_pct"]), _cell(b["cagr_gross_pct"])],
            ["max_drawdown_%", _cell(a["max_drawdown_pct"]), _cell(b["max_drawdown_pct"])],
            ["avg_drawdown_%", _cell(a["avg_drawdown_pct"]), _cell(b["avg_drawdown_pct"])],
            ["Calmar", _cell(a["calmar"]), _cell(b["calmar"])],
            ["Sharpe", _cell(a["sharpe"]), _cell(b["sharpe"])],
            ["Sortino", _cell(a["sortino"]), _cell(b["sortino"])],
        ], [24, 22, 22])
        if m["cagr_suppressed"]:
            L.append(f"    CAGR n/a - window is {m['years']:.2f} yr, under the "
                     f"{MIN_YEARS_FOR_CAGR:g}-yr minimum. Use Rs per year.")

    L += _h2("YEAR BY YEAR  (rupees - capital-independent)")
    rows = [[f"{int(r['year'])}", f"{int(r['trades']):,}",
             f"{r['win_rate_pct']:.1f}" if np.isfinite(r["win_rate_pct"]) else "-",
             f"{r['gross_pnl']:,.0f}", f"-{r['charges']:,.0f}", f"{r['net_pnl']:,.0f}",
             f"{r['max_dd_money']:,.0f}"] for _, r in yr.iterrows()]
    rows.append(["TOTAL", f"{int(yr['trades'].sum()):,}", f"{m['win_rate'] * 100:.1f}",
                 f"{yr['gross_pnl'].sum():,.0f}", f"-{yr['charges'].sum():,.0f}",
                 f"{yr['net_pnl'].sum():,.0f}", f"{m['max_drawdown']:,.0f}"])
    L += _grid(["year", "trades", "win%", "gross Rs", "charges", "net Rs", "maxDD Rs"],
               rows, [6, 7, 6, 13, 12, 13, 13])
    L.append(f"    positive years  net {yr.attrs.get('pos_years', 0)}/{len(yr)}"
             f"   gross {yr.attrs.get('pos_years_gross', 0)}/{len(yr)}"
             f"   |   best {yr.attrs.get('best_year', '-')}"
             f"   worst {yr.attrs.get('worst_year', '-')}")

    if len(grid_df):
        L += _h2("MONTHLY NET P&L  (Rs thousands, by exit month)")
        rows = []
        for y, r in grid_df.iterrows():
            cells = [f"{x / 1e3:,.0f}" if pd.notna(x) else "-" for x in r.to_numpy()]
            rows.append([str(int(y))] + cells + [f"{np.nansum(r.to_numpy()) / 1e3:,.0f}"])
        L += _grid(["yr"] + MONTH_ABBR + ["year"], rows, [4] + [5] * 12 + [7])
    return L


def comparison_block(results):
    keys = [v["key"] for v, _, _, _, _, _ in results]
    L = _h1("VERSION COMPARISON   (same window, same sizing, same costs)")
    for v, _, _, _, _, _ in results:
        L.append(f"    {v['key']} = {v['name']}")

    def row(label, fn):
        return [label] + [fn(m) for _, m, _, _, _, _ in results]

    L += _h2("CORE  (capital-independent)")
    L += _grid(["metric"] + keys, [
        row("Trades", lambda m: f"{m['n_trades']:,}"),
        row("Total lots", lambda m: f"{m['total_lots']:,.0f}"),
        row("GROSS P&L Rs", lambda m: f"{m['gross_pnl']:,.0f}"),
        row("Charges Rs", lambda m: f"-{m['total_charge']:,.0f}"),
        row("NET P&L Rs", lambda m: f"{m['net_pnl']:,.0f}"),
        row("Net Rs / year", lambda m: f"{m['arr']:,.0f}"),
        row("Net kept of gross %", lambda m: _cell(m["net_to_gross_pct"], 1)),
        row("Win rate gross %", lambda m: _cell(m["win_rate_gross"] * 100, 1)),
        row("Win rate net %", lambda m: _cell(m["win_rate"] * 100, 1)),
        row("Profit factor gross", lambda m: _cell(m["profit_factor_gross"])),
        row("Profit factor net", lambda m: _cell(m["profit_factor"])),
        row("Expectancy gross Rs", lambda m: f"{m['expectancy_gross']:,.0f}"),
        row("Expectancy net Rs", lambda m: f"{m['expectancy']:,.0f}"),
        row("Avg pts / lot", lambda m: _cell(m["avg_points_per_lot"])),
        row("avg MFE pts", lambda m: _cell(m["avg_mfe_points"])),
        row("avg MAE pts", lambda m: _cell(m["avg_mae_points"])),
        row("MFE capture %", lambda m: _cell(m["mfe_capture_pct"], 1)),
        row("Edge ratio", lambda m: _cell(m["edge_ratio"])),
        row("Avg holding minutes", lambda m: _cell(m["avg_holding_minutes"], 1)),
        row("Max drawdown Rs", lambda m: f"{m['max_drawdown']:,.0f}"),
        row("Time in drawdown %", lambda m: _cell(m["pct_time_in_dd"], 1)),
        row("False trades", lambda m: f"{m['n_false_trades']:,}" if m["n_false_trades"] >= 0 else "n/a"),
        row("CHURN cost Rs", lambda m: _cell(m["churn_cost"], 0)),
        row("Churn rate %", lambda m: _cell(m["churn_rate_pct"], 2)),
        row("Missed trades", lambda m: _cell(m["n_missed_trades"], 0)),
        row("MISSED OPP Rs", lambda m: _cell(m["missed_opportunity_cost"], 0)),
        row("Flip trades", lambda m: _cell(m["n_flip_trades"], 0)),
        row("WRONG DIR Rs", lambda m: _cell(m["wrong_direction_cost"], 0)),
        row("DELAY cost Rs", lambda m: _cell(m["delay_cost"], 0)),
        row("TOTAL ERROR Rs", lambda m: _cell(m["total_error_cost"], 0)),
        row("Peak lots", lambda m: f"{m['peak_lots']:.0f}"),
        row("Peak notional Rs", lambda m: f"{m['peak_notional']:,.0f}"),
    ], [22] + [16] * len(keys))

    if SHOW_CAPITAL_STATS:
        L += _h2("OPTIONAL  (assumed funding, not measured)")

        def crow(label, key, basis, dp=2):
            vals = [(f"{cs[basis][key]:,.0f}" if key == "capital" else _cell(cs[basis][key], dp))
                    for _, _, cs, _, _, _ in results]
            return [label] + vals

        L += _grid(["metric"] + keys, [
            crow("capital A Rs", "capital", "lot", 0),
            crow("total_return_% A", "total_return_pct", "lot"),
            crow("ARR_% A", "arr_pct", "lot"),
            crow("CAGR_% A", "cagr_pct", "lot"),
            crow("max_drawdown_% A", "max_drawdown_pct", "lot"),
            crow("Sharpe A", "sharpe", "lot"),
            crow("Calmar A", "calmar", "lot"),
            ["-" * 22] + ["-" * 16 for _ in results],
            crow("capital B Rs", "capital", "peak", 0),
            crow("total_return_% B", "total_return_pct", "peak"),
            crow("ARR_% B", "arr_pct", "peak"),
            crow("CAGR_% B", "cagr_pct", "peak"),
            crow("max_drawdown_% B", "max_drawdown_pct", "peak"),
            crow("Sharpe B", "sharpe", "peak"),
            crow("Calmar B", "calmar", "peak"),
        ], [22] + [16] * len(keys))

    best = max(results, key=lambda r: (r[1]["net_pnl"] if r[1]["n_trades"] else -np.inf))
    bestg = max(results, key=lambda r: (r[1]["gross_pnl"] if r[1]["n_trades"] else -np.inf))
    L += ["", f"  Best NET P&L   : {best[0]['key']} - {best[0]['name']}",
          f"  Best GROSS P&L : {bestg[0]['key']} - {bestg[0]['name']}", "=" * W]
    return L


# ============================== EXCEL ======================================

CORE_ORDER = [
    "gross_pnl", "total_charge", "net_pnl", "arr_gross", "arr",
    "gross_pnl_points", "avg_points_per_lot", "breakeven_points_per_lot",
    "charge_per_trade", "charge_pct_of_gross_pnl", "charge_pct_of_gross_wins",
    "net_to_gross_pct", "total_slippage",
    "n_trades", "total_lots", "win_rate_gross", "win_rate",
    "n_wins_gross", "n_losses_gross", "n_wins", "n_losses", "n_breakeven",
    "profit_factor_gross", "profit_factor", "expectancy_gross", "expectancy",
    "avg_win_gross", "avg_loss_gross", "avg_win", "avg_loss",
    "gross_profit_won", "gross_loss_lost", "gross_profit", "gross_loss",
    "avg_holding_minutes", "avg_holding_bars",
    "avg_mfe_points", "avg_mae_points", "best_mfe_points", "worst_mae_points",
    "avg_mfe_money", "avg_mae_money", "mfe_capture_pct", "edge_ratio",
    "pct_trades_mfe_over_breakeven",
    "n_false_trades", "churn_cost", "churn_rate_pct",
    "shadow_trades", "shadow_gross_pnl", "n_missed_trades",
    "missed_opportunity_cost", "missed_gross_raw",
    "n_flip_trades", "wrong_direction_cost",
    "n_delayed_trades", "avg_delay_bars", "avg_delay_points", "avg_delay_atr", "delay_cost",
    "total_error_cost",
    "max_drawdown", "max_drawdown_gross", "avg_drawdown", "n_drawdowns", "pct_time_in_dd",
    "peak_lots", "avg_lots_in_market", "pct_bars_in_market", "pct_bars_multi_lot",
    "peak_notional", "avg_notional_in_market",
    "n_trading_days", "sessions_per_year", "annualisation_factor",
    "years", "cagr_suppressed",
]

TRADE_COLS = ["version", "tier", "qty",
              "signal_time", "entry_signal", "entry_true_label", "is_false_trade",
              "exit_true_label", "is_direction_flip",
              "entry_time", "entry_price", "exit_time", "exit_price", "exit_reason",
              "bars_held", "holding_minutes",
              "points_per_lot", "pnl_points",
              "mfe_points", "mae_points", "mfe_money", "mae_money",
              "mfe_capture_pct", "beat_breakeven",
              "pnl_money_gross", "charge", "pnl_money", "slippage_cost",
              "cum_net_pnl"]


def build_tradebook(results):
    """One tradebook for every version, with the derived columns spelled out."""
    frames = []
    for v, _, _, _, _, tr in results:
        if not len(tr):
            continue
        t = tr.copy()
        t.insert(0, "version", v["key"])
        t["holding_minutes"] = (t["exit_time"] - t["entry_time"]).dt.total_seconds() / 60.0
        t["mfe_capture_pct"] = np.where(t["mfe_points"] > 0,
                                        t["points_per_lot"] / t["mfe_points"] * 100.0, np.nan)
        t["beat_breakeven"] = t["mfe_points"] > BREAKEVEN_POINTS
        t["is_direction_flip"] = (t["entry_true_label"].isin(ALL_EXITS)
                                  | t["exit_true_label"].isin(ALL_ENTRIES))
        t["cum_net_pnl"] = t["pnl_money"].cumsum()
        frames.append(t)
    if not frames:
        return pd.DataFrame(columns=TRADE_COLS)
    out = pd.concat(frames, ignore_index=True)
    return out[[c for c in TRADE_COLS if c in out.columns]]


def readme_frame(info, capbook):
    rows = [
        ("HOW TO READ THIS WORKBOOK", ""),
        ("", ""),
        ("01_Summary", "Executive summary + the full V1/V2/V3 comparison."),
        ("02_Config", "Every setting the run used. Change these in the .py CONFIG block."),
        ("03_Data_Quality", "Coverage, missing predictions, OHLC checks, prediction-level churn."),
        ("04_Metrics", "Every metric for every version, with its formula in the last column."),
        ("05_Tradebook", "One row per closed trade, all versions, every derived column."),
        ("06_Yearly", "Year by year, per version."),
        ("07_Monthly", "Month by month, per version."),
        ("08_Daily_Equity", "End-of-session equity per version (net, gross, charges)."),
        ("", ""),
        ("THE TWO KINDS OF NUMBER", ""),
        ("CORE (capital-independent)",
         "Rupees, points, counts, ratios. Divides by NO capital. Judge the signal on these."),
        ("OPTIONAL (capital-dependent)",
         "total_return_%, ARR_%, CAGR, drawdown_%, Calmar, Sharpe, Sortino. These divide by an "
         "ASSUMED capital, shown on two bases."),
        ("Basis A", f"1-lot notional = first close x {LOT_SIZE} x {LOTS_PER_TRADE} "
                    f"= Rs {capbook['lot']:,.0f}"),
        ("Basis B", f"peak notional = max(close x {LOT_SIZE} x lots open) "
                    f"= Rs {capbook['peak']:,.0f}"),
        ("WARNING on basis B",
         "This is an UNLEVERAGED return on peak notional. It is NOT a margin return - SPAN blocks "
         "roughly 1.2-1.5 lakh per lot against ~16 lakh of notional."),
        ("", ""),
        ("THE ENGINE IN SIX LINES", ""),
        ("1. grid", "Every price bar between the first and last signal timestamp is kept."),
        ("2. gaps", "A bar with no prediction is UNKNOWN: it never trades, and a real signal after "
                    "a gap is compared to the last KNOWN signal, so a gap cannot manufacture an entry."),
        ("3. firing", "Only the FIRST bar of a new signal run acts."),
        ("4. fill", "A signal on bar T is executed at bar T+1's OPEN."),
        ("5. sizing", f"Every entry adds {LOTS_PER_TRADE} lot. Entries stack - lots add up."),
        ("6. charge", f"Rs {CHARGE_PER_LOT:,.0f} per lot, booked once when the position closes. "
                      f"That is {BREAKEVEN_POINTS:.2f} index points a lot must gain to break even."),
        ("", ""),
        ("MAE / MFE SIGN CONVENTION", ""),
        ("mfe_points", "POSITIVE magnitude: how far the trade ever went in your favour."),
        ("mae_points", "POSITIVE magnitude: how far it ever went against you."),
        ("bound", "-mae_points <= points_per_lot <= mfe_points"),
        ("", ""),
        ("CHURN", ""),
        ("false trade", "A trade opened on a bar whose TRUE label was NO_TRADE."),
        ("churn_cost", f"false trades x Rs {CHARGE_PER_LOT:,.0f} - money burnt over-trading."),
        ("churn_rate_pct", "false trades / trades taken x 100 - how often it over-traded."),
        ("availability", info["churn"]["truth_col"] or
         "NO true-label column in the signal file - churn is n/a"),
    ]
    return pd.DataFrame(rows, columns=["item", "explanation"])


def config_frame(signal_file, price_file, info, capbook):
    rows = [
        ("signal file", signal_file), ("signal sheet", info["sheet"] or "-"),
        ("price file", price_file),
        ("LOT_SIZE", LOT_SIZE), ("LOTS_PER_TRADE", LOTS_PER_TRADE),
        ("CHARGE_PER_LOT", CHARGE_PER_LOT), ("SLIPPAGE", SLIPPAGE),
        ("BREAKEVEN_POINTS", round(BREAKEVEN_POINTS, 4)),
        ("USE_DENSE_GRID", USE_DENSE_GRID), ("SHOW_CAPITAL_STATS", SHOW_CAPITAL_STATS),
        ("INITIAL_CAPITAL", INITIAL_CAPITAL if INITIAL_CAPITAL is not None else "auto"),
        ("ROUND_CAPITAL_UP_TO", ROUND_CAPITAL_UP_TO),
        ("SHARED_PEAK_CAPITAL", SHARED_PEAK_CAPITAL),
        ("MIN_YEARS_FOR_CAGR", MIN_YEARS_FOR_CAGR),
        ("ANNUALISE_FROM_DATA", ANNUALISE_FROM_DATA), ("TRADING_DAYS fallback", TRADING_DAYS),
        ("capital basis A (Rs)", round(capbook["lot"], 2)),
        ("capital basis B (Rs)", round(capbook["peak"], 2)),
        ("generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    ]
    for v in VERSIONS:
        rows.append((f"{v['key']} rule", v["desc"]))
    return pd.DataFrame(rows, columns=["setting", "value"])


def quality_frame(info, pxv):
    ch = info["churn"]
    rows = [("price bars in window", info["price_bars_in_window"]),
            ("signal rows supplied", info["signal_rows"]),
            ("signal rows matched", info["matched"]),
            ("signal rows unmatched", info["unmatched"]),
            ("bar coverage %", round(info["coverage_pct"], 3)),
            ("bars with NO prediction", info["gap_bars"]),
            ("missing %", round(info["gap_pct"], 3)),
            ("grid mode", "DENSE (gaps = UNKNOWN)" if info["dense"] else "SPARSE")]
    rows += [(f"OHLC {k}", v) for k, v in pxv.items()]
    rows += [("true-label column", ch["truth_col"] or "NOT PRESENT"),
             ("true labels contain NO_TRADE", ch["truth_has_no_trade"]),
             ("prediction rows saying TRADE", ch["pred_trade_rows"]),
             ("of those, true was NO_TRADE", ch["false_pred_rows"]),
             ("prediction-level churn rate %",
              round(ch["churn_rate_pred_pct"], 3) if np.isfinite(ch["churn_rate_pred_pct"]) else "n/a")]
    rows += [(f"first missing ts {i + 1}", str(pd.Timestamp(t)))
             for i, t in enumerate(info["missing_ts"])]
    return pd.DataFrame(rows, columns=["check", "value"])


def metrics_frame(results):
    keys = [v["key"] for v, _, _, _, _, _ in results]
    rows = []
    for k in CORE_ORDER:
        d, f = METRIC_DOC.get(k, ("", ""))
        rows.append(["CORE", k, d] + [r[1].get(k) for r in results] + [f])
    if SHOW_CAPITAL_STATS:
        for tag, basis in (("OPTIONAL A 1-lot", "lot"), ("OPTIONAL B peak", "peak")):
            for k in ["capital"] + CAPITAL_DEPENDENT:
                d, f = METRIC_DOC.get(k, ("", ""))
                rows.append([tag, k, d] + [r[2][basis].get(k) for r in results] + [f])
    return pd.DataFrame(rows, columns=["block", "metric", "what it is"] + keys + ["how it is calculated"])


def daily_equity_frame(results, bars):
    idx = pd.to_datetime(bars["ts"])
    out = None
    for v, _, _, eq, _, _ in results:
        s = pd.Series(eq, index=idx).resample("1D").last().dropna()
        d = s.rename(f"{v['key']}_net").to_frame()
        if out is None:
            out = d
        else:
            out = out.join(d, how="outer")
    close = pd.Series(bars["close"].to_numpy(float), index=idx).resample("1D").last().dropna()
    out = out.join(close.rename("close"), how="left")
    return out.reset_index(names="date")


def _autofit(ws, df, wb, head_fmt, max_w=52):
    for j, col in enumerate(df.columns):
        try:
            body = df[col].astype(str).str.len().max()
        except Exception:
            body = 12
        width = min(max(int(body if body == body else 12), len(str(col))) + 2, max_w)
        ws.set_column(j, j, width)
    ws.freeze_panes(1, 0)
    ws.autofilter(0, 0, max(len(df), 1), max(len(df.columns) - 1, 0))
    for j, col in enumerate(df.columns):
        ws.write(0, j, str(col), head_fmt)


def write_excel(path, results, bars, info, pxv, capbook, signal_file, price_file,
                shadows=None):
    """ONE workbook with everything. No per-version CSVs."""
    sheets = {
        "00_README": readme_frame(info, capbook),
        "02_Config": config_frame(signal_file, price_file, info, capbook),
        "03_Data_Quality": quality_frame(info, pxv),
        "04_Metrics": metrics_frame(results),
        "05_Tradebook": build_tradebook(results),
        "06_Yearly": pd.concat([y.assign(version=v["key"]) for v, _, _, _, y, _ in results
                                if len(y)], ignore_index=True)
                     if any(len(y) for _, _, _, _, y, _ in results) else pd.DataFrame(),
        "07_Monthly": pd.concat([monthly_stats(t).assign(version=v["key"])
                                 for v, _, _, _, _, t in results if len(t)], ignore_index=True)
                      if any(len(t) for _, _, _, _, _, t in results) else pd.DataFrame(),
        "08_Daily_Equity": daily_equity_frame(results, bars),
    }
    # 01_Summary sits second in the tab order but is built from the comparison
    keys = [v["key"] for v, _, _, _, _, _ in results]
    srows = []
    for k in ["n_trades", "total_lots", "gross_pnl", "total_charge", "net_pnl", "arr",
              "avg_points_per_lot", "breakeven_points_per_lot", "win_rate_gross", "win_rate",
              "profit_factor", "expectancy", "expectancy_gross", "avg_mfe_points",
              "avg_mae_points", "mfe_capture_pct", "max_drawdown", "pct_time_in_dd",
              "n_false_trades", "churn_cost", "churn_rate_pct",
              "n_missed_trades", "missed_opportunity_cost",
              "n_flip_trades", "wrong_direction_cost", "delay_cost", "total_error_cost",
              "shadow_trades", "shadow_gross_pnl", "peak_lots", "peak_notional"]:
        d, f = METRIC_DOC.get(k, ("", ""))
        srows.append(["CORE", k, d] + [r[1].get(k) for r in results] + [f])
    if SHOW_CAPITAL_STATS:
        for tag, basis in (("OPTIONAL A", "lot"), ("OPTIONAL B", "peak")):
            for k in ["capital", "total_return_pct", "arr_pct", "cagr_pct",
                      "max_drawdown_pct", "sharpe", "sortino", "calmar"]:
                d, f = METRIC_DOC.get(k, ("", ""))
                srows.append([tag, k, d] + [r[2][basis].get(k) for r in results] + [f])
    sheets["01_Summary"] = pd.DataFrame(
        srows, columns=["block", "metric", "what it is"] + keys + ["how it is calculated"])

    if shadows and any(s is not None and len(s) for s in shadows):
        sh = []
        sig_arr = bars["sig"].to_numpy()
        for (v, _, _, _, _, _), s in zip(results, shadows):
            if s is None or not len(s):
                continue
            d = s.copy()
            d.insert(0, "version", v["key"])
            d["model_said_here"] = sig_arr[d["signal_idx"].to_numpy()]
            d["was_missed_by_model"] = d["model_said_here"] == NO_TRADE
            d["opportunity_value"] = d["pnl_money_gross"].clip(lower=0)
            sh.append(d[["version", "tier", "signal_time", "entry_signal",
                         "model_said_here", "was_missed_by_model",
                         "entry_time", "entry_price", "exit_time", "exit_price",
                         "points_per_lot", "pnl_money_gross", "opportunity_value"]])
        if sh:
            sheets["09_Missed_Trades"] = pd.concat(sh, ignore_index=True)

    if WRITE_BAR_EQUITY:
        be = pd.DataFrame({"timestamp": bars["ts"], "close": bars["close"],
                           "signal": bars["sig"], "true_label": bars["truth"],
                           "is_gap": bars["is_gap"]})
        for v, _, _, eq, _, _ in results:
            be[f"{v['key']}_equity"] = eq
        sheets["10_Bar_Equity"] = be

    order = ["00_README", "01_Summary", "02_Config", "03_Data_Quality", "04_Metrics",
             "05_Tradebook", "06_Yearly", "07_Monthly", "08_Daily_Equity",
             "09_Missed_Trades", "10_Bar_Equity"]
    try:
        engine = "xlsxwriter"
        import xlsxwriter  # noqa: F401
    except ImportError:
        engine = "openpyxl"

    with pd.ExcelWriter(path, engine=engine) as xw:
        for name in order:
            df = sheets.get(name)
            if df is None:
                continue
            df.to_excel(xw, sheet_name=name, index=False)
            if engine == "xlsxwriter":
                wb, ws = xw.book, xw.sheets[name]
                head = wb.add_format({"bold": True, "bg_color": "#1F3864",
                                      "font_color": "white", "border": 1,
                                      "text_wrap": True, "valign": "vcenter"})
                _autofit(ws, df, wb, head)
    return path


# =============================== MAIN ======================================

def main():
    signal_file = sys.argv[1] if len(sys.argv) > 1 else SIGNAL_FILE
    price_file = sys.argv[2] if len(sys.argv) > 2 else PRICE_FILE
    output_dir = sys.argv[3] if len(sys.argv) > 3 else OUTPUT_DIR
    for pth, what in [(signal_file, "SIGNAL"), (price_file, "PRICE")]:
        if not os.path.exists(pth):
            raise SystemExit(f"ERROR: {what} file not found:\n   {pth}")

    t0 = datetime.now()
    print("=" * W)
    print("  NIFTY SIGNAL BACKTEST   ·   3-VERSION SIGNAL QUALITY SUITE")
    print("=" * W)
    print(" [1/6] loading price  ...", end=" ", flush=True)
    price_df = load_price(price_file)
    pxv = validate_price(price_df)
    print(f"{len(price_df):,} bars")
    print(" [2/6] loading signal ...", end=" ", flush=True)
    signal_df = load_signal(signal_file, SIGNAL_SHEET)
    print(f"{len(signal_df):,} rows"
          + (f"   [sheet: {signal_df.attrs['sheet']}]" if signal_df.attrs.get("sheet") else ""))

    known = ALL_ENTRIES | ALL_EXITS | {NO_TRADE}
    unknown = set(signal_df["sig"].unique()) - known
    if unknown:
        raise SystemExit(f"ERROR: unrecognised signal values: {sorted(unknown)}\n"
                         f"   expected: {sorted(known)}")

    print(" [3/6] building grid ...", end=" ", flush=True)
    bars, info = prepare(price_df, signal_df)
    print(f"{info['engine_bars']:,} bars, {info['coverage_pct']:.1f}% carry a prediction")

    print(" [4/6] simulating    ...", end=" ", flush=True)
    sims = [(v,) + run(bars, v["qty"], v["exits"]) for v in VERSIONS]
    print(f"{len(VERSIONS)} versions done")

    do_costs = COMPUTE_COST_DECOMP and info["churn"]["has_truth"]
    print(" [5/6] shadow run    ...", end=" ", flush=True)
    if do_costs:
        shadows = [run(bars, v["qty"], v["exits"], use_truth=True)[0] for v in VERSIONS]
        print(f"{sum(len(s) for s in shadows):,} ideal trades on the TRUE labels")
    else:
        shadows = [None] * len(VERSIONS)
        print("skipped - no true-label column")

    first_close = float(bars["close"].iloc[0])
    if INITIAL_CAPITAL is not None:
        if float(INITIAL_CAPITAL) <= 0:
            raise SystemExit("ERROR: INITIAL_CAPITAL must be > 0 (or None).")
        cap_lot = float(INITIAL_CAPITAL)
    else:
        cap_lot = first_close * LOT_SIZE * LOTS_PER_TRADE
    close_arr = bars["close"].to_numpy(float)
    per_peak = [float((close_arr * LOT_SIZE * s[4]).max()) for s in sims]
    peak_raw = max(per_peak) if per_peak else 0.0
    peak_raw = peak_raw if peak_raw > 0 else cap_lot
    peak_use = (float(np.ceil(peak_raw / ROUND_CAPITAL_UP_TO) * ROUND_CAPITAL_UP_TO)
                if ROUND_CAPITAL_UP_TO > 0 else peak_raw)
    imax = int((close_arr * LOT_SIZE * sims[int(np.argmax(per_peak))][4]).argmax())
    capbook = dict(lot=cap_lot, peak=peak_use,
                   fund_crore=float(np.ceil(peak_raw / 1e7) * 1e7),
                   peak_time=str(pd.Timestamp(bars["ts"].to_numpy()[imax])))

    print(" [6/6] metrics + report ...", end=" ", flush=True)
    results = []
    for idx, (v, tr, eq, eqg, lots, ctr) in enumerate(sims):
        m = metrics(tr, eq, eqg, lots, bars, info["churn"])
        m.update(cost_decomposition(tr, shadows[idx], bars, do_costs))
        m["total_error_cost"] = (float(np.nansum([m["churn_cost"],
                                                  m["missed_opportunity_cost"],
                                                  m["wrong_direction_cost"]]))
                                 if do_costs else np.nan)
        peak_v = peak_use if SHARED_PEAK_CAPITAL else (per_peak[idx] or cap_lot)
        cs = ({"lot": capital_stats(m, cap_lot), "peak": capital_stats(m, peak_v)}
              if SHOW_CAPITAL_STATS else None)
        yr = yearly_stats(tr, eq, eqg, bars, cap_lot)
        results.append((v, m, cs, eq, yr, tr))
    print("ok")

    report = setup_block(bars, price_file, signal_file, info)
    report += quality_block(info, pxv, capbook)
    report += executive_summary(results, capbook)
    for idx, (v, m, cs, eq, yr, tr) in enumerate(results):
        report += version_block(v, m, cs, yr, monthly_grid(tr), sims[idx][5])
    report += comparison_block(results)
    text = "\n".join(report)
    print(text)

    # scored-artifact filenames carry long hashes; Windows caps a path at 260
    # chars, so keep the stem short and sanitised
    stem = os.path.splitext(os.path.basename(signal_file))[0]
    stem = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in stem)[:40].strip("_")
    run_dir = os.path.join(output_dir, f"backtest_{stem}_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(run_dir, exist_ok=True)
    xl = os.path.join(run_dir, f"BACKTEST_{stem}.xlsx")
    write_excel(xl, results, bars, info, pxv, capbook, signal_file, price_file, shadows)
    txt = os.path.join(run_dir, "full_report.txt")
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write(text + f"\n\ngenerated {datetime.now():%Y-%m-%d %H:%M}\n")

    print("")
    print("-" * W)
    print(f"  WORKBOOK   {xl}")
    print(f"  REPORT     {txt}")
    print(f"  elapsed    {(datetime.now() - t0).total_seconds():.1f}s")
    print("=" * W)


if __name__ == "__main__":
    main()
