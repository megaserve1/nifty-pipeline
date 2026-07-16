"""
bridge/align.py -- THE no-peek join. Pure, importable, unit-tested.

THE RULE
--------
A bar stamped 09:15 on a 5-minute clock covers 09:15..09:19. It is only KNOWN once
it CLOSES, at 09:20. So the earliest label-minute allowed to see it is 09:20.

    bar_close(t) = (the start of t's bar) + clock

Every label-minute m receives the last bar whose CLOSE is <= m. Never a bar that is
still forming. This holds for any clock: a 1-min feature stamped 09:30 closes at
09:31; a 5-min feature stamped 09:15 closes at 09:20.

WHY NOT shift(1)
----------------
The old code did `feat.shift(1)` -- shift by ONE ROW. But these feature tables store a
5-minute value smeared across five 1-minute rows (verified: vwap_excursion_ratio is
constant within every 5-min bucket). One row back is one MINUTE back, not one BAR back,
so minute 09:16 still received the 09:15 bar -- a value computed from prices through
09:19. Four of every five minutes leaked 1-4 minutes of the future. A model trained on
that backtests beautifully and loses money live.

Correct for a 1-min feature by luck; catastrophically wrong for every 5-min feature.
This module fixes it by aligning on the bar CLOSE, driven by each feature's real clock.

TWO BUGS FOUND ON 2026-07-14, BOTH FIXED HERE. BOTH LET THE LEAK BACK IN.
------------------------------------------------------------------------
1. THE CLOCK DETECTOR RETURNED THE SMALLEST PERIOD, NOT THE LARGEST.
   The old rule: "a column's clock is the SMALLEST period under which it never moves
   inside a bucket." That premise is false. If a value is constant across every
   15-minute bar then it is AUTOMATICALLY constant inside every 3-minute bucket, because
   3 divides 15 and both floor from the same origin. So the loop hit 3 first and stopped.

       true clock 15min -> measured 3min  -> the bar is served 12 minutes before it closed
       true clock 30min -> measured 3min  -> 27 minutes early
       true clock 60min -> measured 3min  -> 57 minutes early

   The only two clocks it could get right were 1 and 5 -- 5 only by luck, because 3 does
   not divide 5. Those are also the only two the tests covered.

   The rule is now inverted: the clock is the LARGEST period under which the column never
   moves inside a bucket. When the evidence is thin the answer comes out SLOW, and slow is
   stale (honest). Fast is lookahead (a lie). See clock_of_column.

2. bar_close() FLOORED FROM MIDNIGHT, BUT THE SESSION OPENS AT 09:15.
   09:15 is 555 minutes past midnight. 555 = 3 x 5 x 37, so 1, 3, 5 and 15 divide it and
   the bars line up either way. 30 and 60 DO NOT. A 30-minute bar of the trading session
   runs 09:15..09:44 and truly closes at 09:45 -- but floor(09:15, 30min) from midnight is
   09:00, so the old code thought it closed at 09:30 and served it 15 minutes early. The
   arithmetic was self-consistent, so NoPeekViolation waved it straight through.

   bar_close() is now anchored at the session open (config.SESSION_ANCHOR_MINUTES).
   For 1/3/5/15 this changes nothing. For 30/60 it is the difference between a correct
   pipeline and a leaking one.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as C      # noqa: E402

# Every bar period we are willing to believe in. Ordered SMALL -> LARGE; the detector walks
# it backwards, because the answer we want is the LARGEST period that fits.
CANDIDATE_CLOCKS = (1, 3, 5, 15, 30, 60)


class NoPeekViolation(AssertionError):
    """Raised if any label-minute would receive a bar that had not yet closed."""


class TimezoneAwareFeature(ValueError):
    """Raised when a feature parquet arrives with tz-aware timestamps. See _clean_index."""


def _clean_index(df: pd.DataFrame) -> pd.DataFrame:
    """Datetime index, nanosecond unit, tz-naive, sorted, no NaT.

    A tz-AWARE index is REFUSED, not silently stripped. This used to call tz_localize(None),
    which drops the offset and keeps the wall clock. Hand it a UTC-aware parquet -- exactly what
    a pandas default write on the feature team's Windows box produces -- and "09:15 UTC" becomes
    naive "09:15", which then matches label-minute 09:15 IST with a distance of zero. That is
    5.5 HOURS of lookahead, delivered silently: merge_asof is happy, the staleness tolerance is
    happy, NoPeekViolation cannot see it (naive vs naive is internally consistent), and the
    manifest still swears no_peek.applied = true.

    There is no safe way to guess which timezone was meant. So we stop and make a human say.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        for c in ("timestamp", "datetime", "ts", "date"):
            if c in df.columns:
                df = df.set_index(pd.to_datetime(df[c])).drop(columns=[c])
                break
        else:
            raise ValueError("feature table has no datetime index or timestamp column")
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        raise TimezoneAwareFeature(
            f"this feature's timestamps are timezone-aware ({idx.tz}). the labels are naive IST.\n"
            f"  dropping the timezone would turn 09:15 UTC into 09:15 IST -- a 5.5 HOUR lookahead\n"
            f"  that nothing downstream can detect. convert to naive IST before saving:\n"
            f"      df.index = df.index.tz_convert('Asia/Kolkata').tz_localize(None)")
    df = df.copy()
    df.index = idx.as_unit("ns")            # avoid us-vs-ns MergeError against the labels
    df = df[df.index.notna()].sort_index()
    return df


def _is_constant_within(s: pd.Series, period: int, rtol: float = 1e-6) -> bool:
    """Does this column ever MOVE inside one `period`-minute bar?

    TWO THINGS THIS MUST GET RIGHT, AND BOTH BIT US:

    1. FLOAT JITTER. Comparing with nunique() is exact equality. A 5-minute value that has been
       through a float32 round-trip, or recomputed by a rolling window, can differ in the 12th
       decimal from row to row -- and then it reads as a 1-MINUTE column, and we serve a bar
       that has not closed. So numeric columns are compared with a TOLERANCE, not equality.

    2. EVERY BUCKET, not most. The old test accepted a column constant in 99.9% of buckets. Over
       5 years of minute data 0.1% is hundreds of bars, and each one is a bar served early.
    """
    b = _bar_start(pd.DatetimeIndex(s.index), period)
    g = s.groupby(b)

    # BOOL IS NOT A NUMBER, WHATEVER PANDAS SAYS.
    # is_numeric_dtype(bool) is True, so a bool column used to fall into the numeric branch
    # below and hit `g.max() - g.min()`. numpy refuses to subtract booleans:
    #     TypeError: numpy boolean subtract, the `-` operator, is not supported
    # merged_raw_1min has ~30 bool columns (raw_accumulation, acc_confirmed, ...), so the whole
    # file failed to register. a bool has no "spread" anyway -- it either changed inside the bar
    # or it did not. that is exactly the categorical test.
    if pd.api.types.is_bool_dtype(s):
        # NaN inside a bar is UNKNOWN, not movement. dropna=False counted the NaN as a second
        # distinct value, so a slow categorical with one missing minute measured as 1-minute --
        # the FAST direction, which is the leak direction. unknown must err SLOW.
        return bool(g.nunique(dropna=True).le(1).all())

    if pd.api.types.is_numeric_dtype(s):
        spread = (g.max() - g.min()).abs()         # how far it moves inside one bucket
        # the tolerance is RELATIVE to the column's own size -- but floored sensibly.
        # clip(lower=1.0) made the tolerance an ABSOLUTE 1e-9 for small-magnitude columns
        # (returns ~1e-4), where float32 round-trip jitter can exceed it -> false movement ->
        # fast clock -> lookahead. the floor is now the column's own typical magnitude.
        typical = float(s.abs().median()) if len(s) else 0.0
        scale = g.mean().abs().clip(lower=max(typical, 1e-9))
        return bool(((spread <= rtol * scale) | spread.isna()).all())

    return bool(g.nunique(dropna=True).le(1).all())


def clock_of_column(s: pd.Series, rtol: float = 1e-6) -> int:
    """The REAL bar period of ONE column, in minutes.

    THE RULE (inverted on 2026-07-14 -- read the module docstring, bug 1):

        the clock is the LARGEST candidate period under which the column never moves
        inside a bar.

    NOT the smallest. A value that is constant across every 15-minute bar is AUTOMATICALLY
    constant inside every 3-minute bucket, because 3 divides 15. The old rule took the smallest
    and so it answered 3min for every 15/30/60-minute feature -- and then served those bars 12,
    27 and 57 minutes before they had closed.

    WHICH WAY THIS ERRS, AND WHY THAT IS THE RIGHT WAY.
        A clock that is too LONG  -> the value is served late. It is STALE. Honest, and it costs
                                     us some signal.
        A clock that is too SHORT -> a bar is served before it closed. That is LOOKAHEAD. It
                                     backtests beautifully and loses money live.
        The two are not symmetric, so when the data cannot tell us apart we take the slow one.

    THE CASE THIS CANNOT SOLVE, AND DOES NOT PRETEND TO.
        A column that RARELY CHANGES is indistinguishable from a column on a SLOW CLOCK. Our own
        gap_state is NO_GAP 86% of the time, so it sits still inside a 60-minute bar and this
        function will call it 60min -- even though it is really a 1-minute column that just does
        not move much. That answer is SAFE (stale, never early) but it costs real signal.
        The measurement cannot fix this. Only the feature team knows. That is what the
        `clock` field in registry.yaml is for, and why clock_report() below hands back the
        EVIDENCE and not just the number -- so a human can see when the answer is a guess.
    """
    best = 1
    for period in CANDIDATE_CLOCKS:
        if period == 1:
            continue
        b = _bar_start(pd.DatetimeIndex(s.index), period)
        if pd.Index(b).nunique() < 2:                  # not enough bars to judge
            continue
        if _is_constant_within(s, period, rtol):
            best = period                              # keep going -- we want the LARGEST
    return best


def clock_report(s: pd.Series, rtol: float = 1e-6) -> dict:
    """The clock, plus the evidence behind it, so a human can tell a fact from a guess.

    `confident` is False when the column simply does not change often enough to prove anything.
    In that case the clock we return is the slowest one that fits -- safe, but probably too slow.
    """
    idx = pd.DatetimeIndex(s.index)
    # bool goes down the object path too -- see _is_constant_within. shifting a bool column
    # introduces NaN, which turns it into object anyway, and `ne` on that is what we want.
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        moved = s.ne(s.shift()) & s.notna() & s.shift().notna()
    else:
        o = s.astype("object")
        # a change is A -> B, never A -> NaN or NaN -> A. missing is unknown, not movement.
        moved = o.ne(o.shift()) & o.notna() & o.shift().notna()
        moved.iloc[:1] = False
    n_changes = int(moved.sum())
    clock = clock_of_column(s, rtol)

    # how many DISTINCT positions inside an hour does this column change at? a genuine 1-minute
    # column changes at many; a column that only ever moves at the open changes at one, and from
    # one position you cannot rule out any clock at all.
    offsets = sorted({int(m) for m in
                      ((idx[moved].hour * 60 + idx[moved].minute) - C.SESSION_ANCHOR_MINUTES) % 60})
    confident = n_changes >= 20 and len(offsets) >= 2
    return {"clock": clock, "n_changes": n_changes, "change_offsets": offsets,
            "confident": bool(confident)}


def column_clocks(feat: pd.DataFrame, cols=None, rtol: float = 1e-6) -> dict:
    """The bar period of EVERY column, separately.

    One parquet can hold columns on different clocks -- a 5-minute signal sitting next to the
    1-minute price it was computed from is an entirely ordinary shape. Forcing ONE clock on the
    whole file makes both choices wrong:

        the FASTEST clock -> the 5-minute column is served before its bar closes  = LOOKAHEAD
        the SLOWEST clock -> the 1-minute column is held back                     = needlessly stale

    So every column keeps its own clock, and each is aligned on its own terms.

    The whole column is measured, not a head sample. The old code sampled the first 200,000 rows,
    which on five years of minute data means the clock was decided from 2020-mid-2022 and simply
    assumed to hold through 2025.
    """
    feat = _clean_index(feat)
    cols = list(cols) if cols is not None else list(feat.columns)
    if not cols:
        return {}
    if len(feat) < 4:
        # the old code quietly answered 1min here -- the FAST, unsafe direction, on no evidence
        # at all. four rows cannot tell you a bar period. say so.
        raise ValueError(f"only {len(feat)} rows -- too few to measure a bar period. "
                         f"refusing to guess (guessing fast is a lookahead leak).")
    return {c: clock_of_column(feat[c], rtol) for c in cols}


def row_spacing_minutes(feat: pd.DataFrame) -> int:
    """The gap between consecutive rows, in minutes -- the clock of a SPARSE feature table.

    A feature delivered ONE ROW PER BAR (09:15, 09:20, 09:25 -- the natural output of a
    resample("5min")) defeats the repetition test entirely: there is nothing to repeat, every
    bucket holds at most one row, "constant inside the bucket" is trivially true, and the
    detector would answer with whatever the largest candidate is.

    For such a table the clock is not hidden at all -- it is the row spacing. So we read it
    directly, and it takes precedence.
    """
    idx = pd.DatetimeIndex(_clean_index(feat).index)
    if len(idx) < 3:
        return 1
    within_day = pd.Series(idx).groupby(pd.Series(idx).dt.normalize()).diff().dropna()
    if within_day.empty:
        # ONE ROW PER DAY -- a daily/EOD feature. answering 1min here (the old behavior) is the
        # FAST, unsafe direction: the value would be served the minute after its stamp, when it
        # is really only knowable the NEXT session. use the across-day spacing instead; the
        # aligner's anchored arithmetic handles a 1440-minute bar correctly (stamped Mon ->
        # served from Tue), which is stale-by-construction and therefore honest.
        across = pd.Series(idx).diff().dropna()
        if across.empty:
            return 1
        return max(1, int(round(across.dt.total_seconds().median() / 60)))
    gap = int(round(within_day.dt.total_seconds().median() / 60))
    return max(1, gap)


def infer_bar_minutes(feat: pd.DataFrame, cols=None) -> int:
    """The SLOWEST clock in the frame -- the safe SINGLE clock for the whole thing.

    Used for the registry (one number per feature is what a human can read and sanity-check) and
    as a conservative fallback. build_dataset aligns COLUMN BY COLUMN, so a mixed parquet is not
    dragged down to its slowest column.

    Taking the MAX and never the MIN is the whole point:
        a clock that is too LONG  -> the value is stale.  Honest.
        a clock that is too SHORT -> a bar is served before it closed.  LOOKAHEAD.
    When in doubt, be late, never early.

    The row spacing is a FLOOR: a table with one row every 5 minutes cannot be a 1-minute
    feature, whatever the repetition test thinks it sees.
    """
    per_col = column_clocks(feat, cols)
    spacing = row_spacing_minutes(feat)
    return max([spacing, *per_col.values()], default=1)


def _bar_start(index: pd.DatetimeIndex, bar_minutes: int,
               anchor_minutes: int | None = None) -> pd.DatetimeIndex:
    """The start of the bar that timestamp t belongs to, ANCHORED AT THE SESSION OPEN.

    This is not `index.floor(f"{n}min")`. floor() counts from MIDNIGHT. The session opens at
    09:15 = 555 minutes past midnight, and 555 is not divisible by 30 or 60 -- so for those
    clocks, flooring from midnight puts the bar boundary in the wrong place and the bar appears
    to close EARLIER than it really does. See the module docstring, bug 2.

        30-min bar of the session:  09:15 .. 09:44,  truly closes 09:45
        floor(09:15, "30min")    =  09:00,  + 30min =  09:30    <-- 15 minutes early. A LEAK.
        anchored at 555          =  09:15,  + 30min =  09:45    <-- correct.

    For 1, 3, 5 and 15 (all of which divide 555) the two agree exactly, so nothing changes for
    the clocks we use today. This makes 30 and 60 safe to use tomorrow.
    """
    if anchor_minutes is None:
        anchor_minutes = C.SESSION_ANCHOR_MINUTES
    idx = pd.DatetimeIndex(index).floor("1min")
    minutes_of_day = idx.hour * 60 + idx.minute
    into_bar = (minutes_of_day - anchor_minutes) % bar_minutes
    return idx - pd.to_timedelta(into_bar, unit="m")


def bar_close(index: pd.DatetimeIndex, bar_minutes: int,
              anchor_minutes: int | None = None) -> pd.DatetimeIndex:
    """The instant a bar becomes knowable: the start of its bar, plus one whole period."""
    return _bar_start(index, bar_minutes, anchor_minutes) + pd.Timedelta(minutes=bar_minutes)


def align_feature_to_labels(
    feat: pd.DataFrame,
    label_ts: pd.Series | pd.DatetimeIndex,
    bar_minutes: int,
    tolerance_bars: int = 3,
) -> pd.DataFrame:
    """Give every label-minute the last CLOSED bar of this feature. No lookahead.

    tolerance_bars caps how long a closed bar may forward-fill. Without it, a feature
    that stops updating fills a constant forever, and Friday 15:25's value leaps the
    weekend onto Monday 09:15. Beyond the tolerance the value becomes NaN -- which is
    honest: at 09:15 on a new session no 5-min bar has closed yet.

    Returns a frame with one row per label-minute, in label order, columns = feat's.
    """
    if bar_minutes < 1:
        raise ValueError(f"bar_minutes must be >= 1, got {bar_minutes}")

    feat = _clean_index(feat)
    if feat.empty:
        return pd.DataFrame(index=range(len(label_ts)), columns=feat.columns, dtype="float64")

    # THE FEATURE AND THE LABELS MUST DESCRIBE THE SAME STRETCH OF TIME.
    # Nothing downstream checks this. If a feature is handed over on a different clock-face --
    # a tz shift that got baked into the values, a date column parsed day-first when it was
    # month-first -- merge_asof does not complain, it just matches whatever is nearest and the
    # staleness tolerance quietly NaNs the rest. You get a dataset that is mostly empty, or
    # worse, one that is silently offset. So we look, once, before the join.
    #
    # THE COMPARISON IS IN BAR-CLOSE SPACE, NOT STAMP SPACE. A bar STAMPED 09:20 on a 5-min
    # clock is SERVED from 09:25 -- so a feature whose last stamp is 09:20 legitimately feeds a
    # label at 09:25 (or up to tolerance_bars later). The first version of this guard compared
    # raw stamps and refused exactly that legal join. A real mismatch (a 5.5-hour tz shift, a
    # day/month swap, a year-off parse) is thousands of minutes out and is still caught.
    lab_idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(label_ts).values))
    if lab_idx.isna().any():
        raise ValueError(f"{int(lab_idx.isna().sum())} label timestamp(s) are NaT -- refusing "
                         f"to align against a spine with holes in it")
    if lab_idx.tz is not None:
        # the same 5.5-hour-leak defence the FEATURE side gets. stripping the tz keeps the wall
        # clock -- "09:15 UTC" would silently match label minute 09:15 IST.
        raise TimezoneAwareFeature(
            f"the LABEL timestamps are timezone-aware ({lab_idx.tz}). convert to naive IST "
            f"before aligning:  ts.dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)")

    tol = pd.Timedelta(minutes=bar_minutes * tolerance_bars)
    first_close = bar_close(feat.index[:1], bar_minutes)[0]     # earliest minute it can serve
    last_close = bar_close(feat.index[-1:], bar_minutes)[0]     # latest bar it ever closes
    l_lo, l_hi = lab_idx.min(), lab_idx.max()
    if (last_close + tol) < l_lo or first_close > l_hi:
        raise ValueError(
            f"this feature and the labels do not overlap in time AT ALL.\n"
            f"  feature serves: {first_close}  ..  {last_close + tol}  (bar-close + tolerance)\n"
            f"  labels        : {l_lo}  ..  {l_hi}\n"
            f"  the usual cause is a timezone or a day/month parse that went wrong upstream.")

    g = feat.copy()
    g["__close"] = bar_close(g.index, bar_minutes)

    # Collapse each closed bar to ONE row. Keep the LAST row of the bucket: for a 5-min
    # value smeared on 1-min rows they are identical, and if they are not, the last row
    # is the only one whose information is complete at the close.
    g = (g.reset_index(drop=True)
           .sort_values("__close", kind="mergesort")
           .drop_duplicates("__close", keep="last"))

    lts = pd.DatetimeIndex(pd.to_datetime(pd.Series(label_ts).values)).as_unit("ns")
    left = pd.DataFrame({"__lts": lts})
    order = left["__lts"].argsort(kind="mergesort")
    left_sorted = left.iloc[order].reset_index(drop=True)

    out = pd.merge_asof(
        left_sorted, g,
        left_on="__lts", right_on="__close",
        direction="backward",                       # only bars that already closed
        allow_exact_matches=True,                   # a bar closing AT 09:20 is known at 09:20
        tolerance=pd.Timedelta(minutes=bar_minutes * tolerance_bars),
    )

    # BE HONEST ABOUT WHAT THIS ASSERT DOES AND DOES NOT PROVE.
    #
    # merge_asof(direction="backward") GUARANTEES right_on <= left_on. So this condition is
    # mathematically unreachable, and on its own it is a tautology -- it proves that merge_asof
    # did what merge_asof does. It says NOTHING about whether __close is the RIGHT close. Feed
    # it a wrong bar_minutes and __close is wrong, the assert passes cheerfully, and the
    # manifest still swears no_peek.applied = true. The real defence against that is
    # clock_of_column erring SLOW, not this line.
    #
    # It is kept because it is free and it catches the one thing it can: a future change to the
    # merge (someone "fixing" direction=, or allow_exact_matches=) that would let a forming bar
    # through. That is worth a cheap tripwire. It is not a proof of no-lookahead.
    late = out["__close"].notna() & (out["__close"] > out["__lts"])
    if bool(late.any()):
        raise NoPeekViolation(
            f"{int(late.sum())} label rows received a bar that had not closed yet")

    out = out.drop(columns=["__lts", "__close"])
    out.index = order.values                        # undo the sort -> original label order
    return out.sort_index().reset_index(drop=True)
