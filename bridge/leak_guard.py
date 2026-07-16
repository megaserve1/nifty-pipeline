"""
bridge/leak_guard.py -- refuse the columns that must never become features.

WHY THIS FILE EXISTS
--------------------------------------------------------------------------------------
On 2026-07-14 the feature team dropped `merged_raw_1min.parquet` into data/features/.
195 columns. Among them:

    fwd_ret_1  fwd_ret_3  fwd_ret_5  fwd_ret_10
    signed_ret_1  signed_ret_3  signed_ret_5  signed_ret_10
    label_int  label_int_raw  dir_at_label
    session  t5

`fwd` means FORWARD. fwd_ret_10 is the return over the NEXT ten minutes. Measured on the real
file, fwd_ret_1 correlates +0.79 with the next minute's return and +0.02 with the last one. It
is not a feature. It is the answer.

Nothing in the pipeline would have stopped it. register.py would have put all 195 columns on the
ballot, the expert would have ticked some, and the model would have read the answer off the page.
Backtest: superb. Live: it does not exist, because nobody knows the next ten minutes.

That file was the feature team's WORKING sheet -- the raw inputs, the intermediate columns, the
forward returns they used to BUILD the labels, and the labels themselves, all in one place. That
is a perfectly normal thing for a working file to be. It is just not a feature file. And "please
remember not to put that file there" is not a control. This is.

THE THREE THINGS IT REFUSES
--------------------------------------------------------------------------------------
1. THE FUTURE.   A column computed from prices that have not happened yet.
2. THE ANSWER.   A label, or anything derived from one.
3. THE CALENDAR. A date or a row id. Not lookahead -- worse in one way. The model memorises
                 which DAY it is, scores brilliantly on train, and then meets a test set where
                 every single row carries an id it has never seen. The splits it learned are
                 worthless there. It looks like overfitting and it is very hard to find.

BY NAME **AND** BY BEHAVIOUR
--------------------------------------------------------------------------------------
Names are checked, because they are free and they catch the honest cases.

But a name check alone is theatre. The next leak will be called `score`, or `dir_at_label`, or
something nobody looks at twice. So every numeric column is also MEASURED against the price:

    a column built from the PAST correlates with PAST returns.
    a column built from the FUTURE correlates with FUTURE returns.

A real 1-minute alpha signal correlates with the next minute at about 0.02-0.05. That is what
edge looks like, and it is small. Anything above 0.10 is not edge. It is the answer.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------------------------------------------------------------
It does not flag `minute_of_day`, `minute_of_session`, `session_phase`, `zone_id`. Those are
time-OF-DAY, and they repeat every single day, so they cannot encode WHICH day it is. They are
legitimate and they are useful -- the market genuinely behaves differently at 09:20 and 14:50.

It cleared `anchored_high/low` and `frozen_high/low` too, and that mattered: a first, sloppier
test flagged them, because it compared each level against the running high OF THAT SESSION -- so
a legitimate level carried over from YESTERDAY looked like a violation. Tested against ALL
history, they never once hold a price that had not printed yet. A guard that cries wolf gets
switched off, so the test is deliberately hard to trip.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# --- 1. the future, and the answer -------------------------------------------------
# matched against the LOWERCASED column name.
FUTURE_PATTERNS = [
    r"^fwd[_.]", r"[_.]fwd$", r"^forward[_.]", r"^future[_.]", r"[_.]future$",
    r"^next[_.]", r"^ahead[_.]", r"[_.]ahead$", r"^lead[_.]", r"[_.]lead\d*$",
    r"^signed_ret", r"^fut[_.]",
]
# THE WORD "label" IS AMBIGUOUS IN THIS CODEBASE, AND GETTING THIS WRONG IS EXPENSIVE.
#
# The first version of this file banned anything containing "label". It caught the real targets
# -- and it also killed `label_combined`, which is 100% IDENTICAL to Stress_Signal, one of the
# twelve bucket features. And `Flow_State_Label`, which is just the text name of Flow_State
# (ACCUMULATION / DISTRIBUTION / NORMAL, one-to-one). Both are perfectly good features.
#
# In this project "label" usually means "the NAME of a state", not "the answer". A guard that
# silently deletes real features gets switched off, and then it guards nothing at all. So the
# BAN list is precise, and everything else that merely smells of a label becomes a loud SUSPECT
# that a human looks at.
LABEL_PATTERNS = [
    r"^label_int", r"^label$", r"^target", r"^y$", r"^y[_.]",
    r"^outcome", r"^primary_label$", r"^weight$", r"^weight_raw$",
    r"_at_label$", r"^is_label", r"^label_dir",
]
# smells like a label, is not provably one. printed loudly, NOT blocked.
SOFT_LABEL_PATTERNS = [r"label", r"target", r"class$", r"^y_"]
# --- 2. the calendar ----------------------------------------------------------------
CALENDAR_PATTERNS = [r"^session$", r"^date$", r"^datetime$", r"^day$", r"^t5$", r"^ts$"]

# --- 3. the numbers that decide the behavioural test --------------------------------
# real 1-min alpha is 0.02-0.05. 0.10 is not alpha.
FUTURE_CORR_LIMIT = 0.10
# and it must beat its own correlation with the PAST by this much -- a momentum feature is made
# OF past returns, so it correlates with them; the answer does not.
FUTURE_OVER_PAST  = 2.0
# a column that climbs monotonically with the row number IS the calendar, whatever it is called.
TIME_CORR_LIMIT   = 0.97
HORIZONS = (1, 5, 10)


def _numify(s: pd.Series):
    """turn any column into numbers so it can be correlated. None = cannot."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    if str(s.dtype).startswith("datetime"):
        return None                                   # handled by the calendar rule
    if str(s.dtype) in ("object", "str", "string", "category"):
        # SORTED categories, not order-of-appearance. factorize() numbers categories by which
        # row shows up first, so the SAME column gets different codes in differently-ordered
        # files -- and a correlation can appear or vanish with the ordering. sorted() makes the
        # encoding a property of the DATA, reproducible anywhere.
        cats = sorted(x for x in s.dropna().unique())
        mapping = {c: i for i, c in enumerate(cats)}
        return s.map(mapping).astype(float)
    return None


def _matches(name: str, patterns) -> bool:
    low = name.lower()
    return any(re.search(p, low) for p in patterns)


def screen(df: pd.DataFrame, price_col: str | None = None, allow: list | None = None) -> dict:
    """Look at every column. Say which ones must never be a feature, and why.

    Returns {"banned": {col: reason}, "suspect": {col: reason}, "ok": [cols]}

    BANNED  -> refused outright. it will not reach a dataset.
    SUSPECT -> it smells, but the evidence is not conclusive. printed loudly, NOT refused --
               because a guard that blocks good features on a hunch gets switched off, and then
               it blocks nothing at all.

    `allow` is the escape hatch, and it is deliberately explicit. A human, having LOOKED at a
    column, can force it through:

        registry.yaml:
          my_feature:
            allow_columns:
              - label_combined     # this is Stress_Signal under another name. checked 2026-07-14.

    It is on the record, in the file, with a name against it. That is the point -- an override
    nobody can see is the same as no guard.
    """
    banned, suspect = {}, {}
    allow = set(allow or [])

    # a price series to measure against. the feature team's files all carry one.
    if price_col is None:
        # broadened after the real files defeated it: adiii_merged_ml calls it src_close and
        # prateek_fade_risk has no price at all -- with the old 4-name list, 103 of 285 v2
        # columns silently SKIPPED the behavioural test and nobody knew. exact names first,
        # then anything that ENDS with 'close' (case-insensitive).
        for cand in ("close", "src_close", "Nifty_Futures_Close", "spot", "price", "px"):
            if cand in df.columns and pd.api.types.is_numeric_dtype(df[cand]):
                price_col = cand
                break
        if price_col is None:
            for c in df.columns:
                if c.lower().endswith("close") and pd.api.types.is_numeric_dtype(df[c]):
                    price_col = c
                    break
    if price_col is None and len(df.columns) > 0:
        # no price series ANYWHERE in the file -> the behavioural test cannot run. that is a
        # blind spot, not a pass -- say so once, loudly, instead of silently testing nothing.
        print(f"  ?? leak_guard: no price column in this file -- the measured-against-price "
              f"test is SKIPPED for all {len(df.columns)} column(s). name checks still apply.")

    fut = pas = None
    if price_col is not None:
        c = df[price_col].astype(float)
        fut = {h: (c.shift(-h) / c - 1) for h in HORIZONS}   # the return h minutes AHEAD
        pas = {h: (c / c.shift(h) - 1) for h in HORIZONS}    # the return h minutes BEHIND

    row_number = np.arange(len(df), dtype=float)

    for col in df.columns:
        s = df[col]

        # ---- 0. a human has looked at this one and forced it through.
        if col in allow:
            continue

        # ---- 1. the calendar. a date or a row id, whatever it is called.
        if str(s.dtype).startswith("datetime"):
            banned[col] = "a raw timestamp column -- the model would memorise the calendar"
            continue
        if _matches(col, CALENDAR_PATTERNS):
            banned[col] = "a date / session id -- the model would memorise which day it is"
            continue
        if str(s.dtype) in ("object", "str", "string"):
            v = s.dropna().astype(str)
            if len(v) and v.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.9:
                banned[col] = "a DATE written as text -- the model would memorise which day it is"
                continue

        # ---- 2. the future, and the answer, by name
        if _matches(col, FUTURE_PATTERNS):
            banned[col] = "the name says it looks FORWARD -- that is the future, not a feature"
            continue
        if _matches(col, LABEL_PATTERNS):
            banned[col] = "this is a LABEL (or made from one) -- it is the answer"
            continue
        if _matches(col, SOFT_LABEL_PATTERNS):
            # NOT banned. in this codebase "label" usually means "the name of a state" --
            # label_combined IS Stress_Signal, Flow_State_Label IS Flow_State's text name.
            # so we say it out loud and let a human look, instead of quietly deleting a feature.
            suspect[col] = ("the name contains 'label'/'target'. in this project that usually "
                            "means the NAME OF A STATE, not the answer -- but LOOK at it. if it "
                            "is fine, put it in allow_columns in registry.yaml.")

        x = _numify(s)
        if x is None or x.nunique(dropna=True) < 2:
            ok = True
        else:
            # ---- 3. a numeric column that climbs with the row number IS the calendar
            m = x.notna()
            # the unique-count gate is >= 20, not > 50. episode_id-style counters (a few
            # thousand uniques) passed the old gate, but gap_up_stack-style slow counters with
            # ~30 uniques and corr 0.97 slipped UNDER it. 20 uniques + near-perfect time corr
            # has no innocent explanation.
            if m.sum() > 1000 and x.nunique() >= 20:
                r = np.corrcoef(row_number[m.to_numpy()], x[m])[0, 1]
                if not np.isnan(r) and abs(r) > TIME_CORR_LIMIT:
                    banned[col] = (f"climbs straight up with time (corr={r:+.3f}) -- it is a "
                                   f"clock, and the model would memorise it")
                    continue
                # A RUNNING ID COUNTER (episode_id, pbsev_event_id): every event gets the next
                # integer, for five years. every test-period id is one training never saw --
                # a perfect memorisation vector. it is NOT monotonic (a -1 'no event' sentinel
                # interleaves) and its time-correlation can be as low as 0.17, so the two
                # obvious tests miss it. its real fingerprint, measured on the actual files:
                # the value is essentially ALWAYS sitting at its own RUNNING MAXIMUM (the
                # current, newest id) or at the sentinel minimum -- episode_id/pbsev_event_id
                # score exactly 1.000 on this; every honest column scored <= 0.013.
                xs = x[m]
                if x.nunique() >= 100 and np.allclose(xs, xs.round()):
                    cm = xs.cummax()
                    at_max = float(((xs >= cm) | (xs == xs.min())).mean())
                    late_growth = float((cm.iloc[-1] - cm.iloc[len(cm) // 2])
                                        / max(cm.iloc[-1] - cm.iloc[0], 1))
                    if at_max > 0.95 and late_growth > 0.2:
                        banned[col] = (
                            f"a running id counter: {at_max:.0%} of rows sit at the column's own "
                            f"running maximum and it keeps growing to the end of history. an id, "
                            f"not a feature -- every future value is out-of-range and the model "
                            f"just memorises which era each id belongs to")
                        continue

            # ---- 4. THE BEHAVIOURAL TEST. does it know the future?
            if fut is not None and col != price_col:
                best_f = best_p = 0.0
                for h in HORIZONS:
                    mm = x.notna() & fut[h].notna()
                    if mm.sum() > 1000:
                        r = abs(np.corrcoef(x[mm], fut[h][mm])[0, 1])
                        best_f = max(best_f, 0.0 if np.isnan(r) else r)
                    mm = x.notna() & pas[h].notna()
                    if mm.sum() > 1000:
                        r = abs(np.corrcoef(x[mm], pas[h][mm])[0, 1])
                        best_p = max(best_p, 0.0 if np.isnan(r) else r)

                # ABSOLUTE CEILING FIRST. the ratio test below can be beaten by a column that
                # correlates with the past AND the future (e.g. next-minute return PLUS a
                # momentum term). no honest 1-minute feature reaches 0.30 against the future --
                # real edge is 0.02-0.05 -- so past-correlation buys no forgiveness up here.
                if best_f > 0.30:
                    banned[col] = (
                        f"MEASURED: |corr| with the next 1-10 min return = {best_f:.3f}. real "
                        f"1-minute alpha is 0.02-0.05; {best_f:.3f} is the answer, whatever "
                        f"else the column also correlates with.")
                    continue
                if best_f > FUTURE_CORR_LIMIT and best_f > FUTURE_OVER_PAST * max(best_p, 1e-6):
                    banned[col] = (
                        f"MEASURED: it tracks the FUTURE. |corr| with the next 1-10 min return "
                        f"= {best_f:.3f}, but only {best_p:.3f} with the LAST 1-10 min. real "
                        f"1-minute alpha is 0.02-0.05. {best_f:.3f} is not alpha, it is the answer.")
                    continue
                if best_f > 0.05 and best_f > FUTURE_OVER_PAST * max(best_p, 1e-6):
                    suspect[col] = (f"leans toward the future (corr {best_f:.3f} ahead vs "
                                    f"{best_p:.3f} behind). not conclusive. worth an eyeball.")

    ok_cols = [c for c in df.columns if c not in banned]
    return {"banned": banned, "suspect": suspect, "ok": ok_cols}


def report(result: dict, where: str = "") -> None:
    """print it so a human cannot miss it."""
    b, s = result["banned"], result["suspect"]
    if b:
        print(f"\n  {'!' * 74}")
        print(f"  !! {len(b)} COLUMN(S) REFUSED{(' in ' + where) if where else ''}. "
              f"they will NOT go into any dataset.")
        for c, why in b.items():
            print(f"  !!   {c:26s} {why}")
        print(f"  {'!' * 74}")
    if s:
        print(f"\n  ?? {len(s)} column(s) look odd but are NOT blocked -- check them yourself:")
        for c, why in s.items():
            print(f"  ??   {c:26s} {why}")
