"""
trainer/purged_cv.py -- k-fold cross-validation that does not cheat on time series.

WHY PLAIN K-FOLD IS USELESS HERE, AND WHY TimeSeriesSplit IS NOT ENOUGH EITHER
------------------------------------------------------------------------------------
plain KFold scatters neighbouring minutes across train and test. minute 09:31 and 09:32
share nearly all of their 20-day rolling history, so the model is scored on rows it has
effectively memorised. the number that comes out is a fantasy.

sklearn's TimeSeriesSplit is better -- it keeps time order -- but read its source
(sklearn/model_selection/_split.py, TimeSeriesSplit._split):

    for test_start in test_starts:
        train_end = test_start - gap
        yield indices[:train_end], indices[test_start : test_start + test_size]

two things follow, and both matter:
  1. train is ALWAYS indices[:train_end] -- an expanding window that stops before the fold.
     it never uses the data AFTER the fold. on 5 folds that throws away most of the history.
  2. `gap` is counted in ROWS, and it is the only protection there is. sklearn has no purge
     and no embargo -- grep the whole package for "purg" or "embargo" and you get nothing.

we need k-fold that uses the data on BOTH sides of the fold. the moment you do that, you
have to cut both sides, and you have to cut them by DIFFERENT amounts. that is what this
file is for.


THE TWO CUTS. THEY ARE NOT THE SAME THING AND THEY ARE NOT THE SAME SIZE.
------------------------------------------------------------------------------------
write H for the LABEL HORIZON  -- how far FORWARD a label looks. our label is derived from
                                  what price does AFTER the minute, so it reaches forward.
write L for the FEATURE LOOKBACK -- how far BACKWARD a feature looks. ret_20d and
                                  drawdown_from_20d_high reach back 20 trading sessions.

the test fold covers [t_start, t_end]. take a training row at time s and ask, on each side,
"can this row possibly know anything about the fold?"

  s is BEFORE the fold (s < t_start):
      its FEATURES cover [s-L, s]. all of that is before t_start. they cannot touch the fold.
      its LABEL covers [s, s+H]. if s+H reaches t_start, the label was DECIDED by prices
      inside the fold. the row's answer is a function of the test period.
      -> the leak on this side is the LABEL. cut back by H.        this is PURGING.

  s is AFTER the fold (s > t_end):
      its LABEL covers [s, s+H]. all of that is after t_end. it cannot touch the fold.
      its FEATURES cover [s-L, s]. if s-L reaches back to t_end, the feature values were
      COMPUTED FROM prices inside the fold. the row's inputs are made of test data.
      -> the leak on this side is the FEATURE. cut forward by L.   this is EMBARGOING.

so:  before the fold the LABEL is the leak.  after the fold the FEATURE is the leak.
     they are opposite channels and they are sized by different numbers.
most implementations do one of the two and call it PurgedKFold. one of the two is not enough.

THE THIRD BIT, THE ONE THAT IS ALWAYS MISSED
    the test fold's OWN labels reach forward too. the last test row t_end has a label decided
    by prices over [t_end, t_end+H] -- which lies OUTSIDE the fold, in the rows just after it.
    a training row sitting there has features built from exactly those prices. so it carries
    the ingredients of a test answer.
    -> the cut after the fold is H + L, not L.

    left of the fold :  H
    right of the fold:  H + L
    it is ASYMMETRIC. a symmetric gap is wrong on one side or wasteful on the other.


WORKED EXAMPLE, ON REAL NUMBERS
------------------------------------------------------------------------------------
H = 1 session, L = 20 sessions. the fold is sessions 100..199.

    sessions  79..99    dropped   PURGE. a label stamped on session 99 is decided by what
                                  price does on session 100 -- which is inside the fold.
                                  (H=1, so only session 99 is strictly needed; see the
                                  config note about rounding up.)
    sessions 100..199   TEST
    sessions 200..220   dropped   EMBARGO. ret_20d on session 205 is computed from sessions
                                  185..205, and 185..199 are inside the fold. the feature
                                  is literally made of test prices.
                                  200 + H(1) + L(20) = 221, so 200..220 go.
    everything else                TRAIN

    session 221 is the first legal training row after the fold: its 20-session window starts
    at 201, which is clear of the fold, and its label was not decided by fold prices.


WHY SESSIONS AND NOT CALENDAR DAYS, AND NOT ROWS
------------------------------------------------------------------------------------
this is where the current pipeline is actually wrong, so it is worth being exact.

    CALENDAR DAYS are wrong. config.EMBARGO_DAYS = 21 is applied as
    `cut + pd.Timedelta(days=21)` -- 21 calendar days. measured on our own timestamps, a
    21-calendar-day window contains 14.1 trading sessions on average (min 9, max 16). it
    NEVER contains 20. so a 21-calendar-day gap does not cover a 20-session feature window,
    at any cut point in five years of data. to span 20 sessions you need 25 to 39 calendar
    days (mean 29.8). the number 21 was chosen to cover a "20-day" lookback and it covers
    three quarters of it.

    ROWS are wrong too. 20 sessions x 375 minutes = 7,500 rows, but 15 of our 1,372 sessions
    are short (one has 54 rows -- a half day). counting 7,500 rows back from a fold that sits
    next to a short session lands you INSIDE the window you were trying to skip.

    SESSIONS are right, because that is the unit the features are actually built in. ret_20d
    means 20 bars of a daily series, and a daily series has one bar per session. so we count
    the thing the feature counts. this is also safe under the other reading: if "20d" ever
    turned out to mean 20 calendar days, 20 sessions (~30 calendar days) still covers it.

so every cut in this file is measured in SESSIONS, taken from the trading calendar that is
observed in the data itself. no holiday table, no assumption about which days the market was
open -- we read it off the timestamps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class NotEnoughData(ValueError):
    """a fold left the training set empty, or nearly so. better to shout than to score noise."""


def sessions_of(ts: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """the trading calendar, read off the data. one entry per day the market was actually open.

    we never assume mon-fri. holidays, muhurat sessions and half days are whatever the
    timestamps say they are.
    """
    idx = pd.DatetimeIndex(pd.Series(ts).values)
    return np.array(sorted(pd.unique(idx.normalize())))


def embargo_end(ts, cut, embargo_sessions: int) -> pd.Timestamp:
    """the last instant INSIDE the embargo. rows after this are the next slice.

    THIS IS THE FUNCTION THAT FIXES THE CALENDAR-DAY BUG.

    train.py used to do this:

        embargo_end = cut + pd.Timedelta(days=21)

    21 CALENDAR days, to cover a feature lookback of "20 days". but "20 days" means 20 TRADING
    SESSIONS -- ret_20d is 20 bars of a DAILY series, and a daily series has one bar per session.
    the market is shut at weekends and on holidays, so 21 calendar days spans only about 14
    sessions. measured on the real label file: mean 14.1, min 9, max 16, and NEVER 20 at any cut
    point in five years. so the embargo was about 30% too short and the first ~6 sessions of
    every test set still carried features built from training-period prices.
    tests/test_purged_cv.py::test_calendar_days_are_NOT_trading_sessions proves it on real data.

    so we count SESSIONS, off the data's own calendar. no holiday table, no assumption about
    which days the market was open.

    the embargo swallows the `embargo_sessions` sessions that FOLLOW the cut's session. the next
    slice starts on the session after that.
    """
    ts = pd.to_datetime(pd.Series(ts))
    sess = sessions_of(ts)
    cut_day = pd.Timestamp(cut).normalize()

    i = int(np.searchsorted(sess, np.datetime64(cut_day)))      # the cut's own session
    j = i + int(embargo_sessions)                               # last session inside the embargo
    if j >= len(sess) - 1:
        raise NotEnoughData(
            f"an embargo of {embargo_sessions} sessions after {cut_day.date()} runs off the end "
            f"of the data ({len(sess)} sessions in total). there would be no rows left on the "
            f"other side of it.")

    # the last nanosecond of that session -- so `ts > embargo_end` starts cleanly on the next one
    return pd.Timestamp(sess[j]) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


class PurgedKFold:
    """k-fold over TIME, with a purge before each fold and an embargo after it.

    deliberately NOT a subclass of sklearn's BaseCrossValidator. that class implements
    split() as:

        train_index = indices[np.logical_not(test_index)]      # train = everything not test

    so if you inherit from it and define _iter_test_indices -- which is the obvious thing to
    do, and what most "PurgedKFold" snippets on the internet do -- you get a class called
    PurgedKFold that performs no purge whatsoever. it does not error. it just leaks, and the
    name on the tin tells you it did not.

    sklearn duck-types cross-validators (check_cv only asks `hasattr(cv, "split")`), so this
    plain class still works anywhere sklearn wants a cv object. we just own split() outright.

    params
        ts                the timestamp of every row, in the SAME ORDER as X. not sorted for
                          you -- if it is out of order that is a bug upstream and we say so.
        n_splits          k.
        purge_sessions    H. how far FORWARD the label looks, in trading sessions.
        embargo_sessions  L. how far BACKWARD the longest feature looks, in trading sessions.
    """

    def __init__(self, ts, n_splits: int = 5, purge_sessions: int = 1,
                 embargo_sessions: int = 20):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if purge_sessions < 0 or embargo_sessions < 0:
            raise ValueError("purge_sessions and embargo_sessions cannot be negative")

        self.ts = pd.DatetimeIndex(pd.Series(ts).values)
        if not self.ts.is_monotonic_increasing:
            raise ValueError(
                "timestamps are not sorted ascending. every cut in this file is positional in "
                "time, so unsorted rows would put the purge window in the wrong place and the "
                "leak would be silent. sort the dataset first.")

        self.n_splits = n_splits
        self.purge_sessions = purge_sessions
        self.embargo_sessions = embargo_sessions

        # KEEP THESE AS datetime64. BOTH LINES MATTER, AND THE REASON IS SPEED, NOT CORRECTNESS.
        #
        #   np.array(sorted(pd.unique(day)))   ->  dtype=OBJECT
        #
        # sorted() turns the array into a python list of pd.Timestamp objects, and np.array() of a
        # list of Timestamps cannot infer datetime64 -- it makes an object array. everything still
        # gives the RIGHT answer, so no test caught it. but np.isin() on an object array cannot use
        # its sorted/hashed fast path: it falls back to comparing python objects one at a time.
        #
        # measured, 200 sessions x 375 rows = 75,000 rows:
        #     object dtype      3.52  s per np.isin
        #     datetime64      0.0026  s per np.isin        -- 1350x
        #
        # split() calls np.isin twice per fold, so 5 folds = ~35 s on a TOY dataset. on the real
        # 513,611 rows x 1,370 sessions it is worse than linearly worse (the object path is
        # O(n*m)) -- hours. that is why the test suite stopped finishing.
        #
        # .values keeps it as a datetime64 numpy array, and np.sort keeps it that way.
        self._day = self.ts.normalize().values
        self._sessions = np.sort(pd.unique(self._day))

        if len(self._sessions) < n_splits:
            raise NotEnoughData(
                f"{len(self._sessions)} sessions cannot be cut into {n_splits} folds")

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X=None, y=None, groups=None):
        """yield (train_idx, test_idx) as integer positions into the rows.

        folds are cut on SESSION boundaries, never in the middle of a trading day. a fold that
        began at 11:03 would split a day in half, and our labels sit in blocks of 14 minutes or
        more (measured: min run 14, median 15, max 375 -- a whole session), so a mid-day cut
        would put the two halves of one label block on opposite sides of the wall.
        """
        n_rows = len(self.ts)
        if X is not None and len(X) != n_rows:
            raise ValueError(
                f"X has {len(X)} rows but the splitter was built with {n_rows} timestamps. "
                f"they must line up row for row, or the folds point at the wrong data.")

        n_sess = len(self._sessions)
        # contiguous, near-equal blocks of SESSIONS. np.array_split handles the remainder.
        blocks = np.array_split(np.arange(n_sess), self.n_splits)

        for k, block in enumerate(blocks):
            a, b = int(block[0]), int(block[-1]) + 1          # test = sessions[a:b]

            # left cut: PURGE. the label horizon H.
            lo = max(0, a - self.purge_sessions)
            # right cut: EMBARGO. the test fold's own label horizon H, plus the feature
            # lookback L. see the module docstring -- this side is the bigger one.
            hi = min(n_sess, b + self.purge_sessions + self.embargo_sessions)

            test_sess = self._sessions[a:b]
            train_sess = np.concatenate([self._sessions[:lo], self._sessions[hi:]])

            test_mask = np.isin(self._day, test_sess)
            train_mask = np.isin(self._day, train_sess)

            train_idx = np.flatnonzero(train_mask)
            test_idx = np.flatnonzero(test_mask)

            if len(train_idx) == 0:
                raise NotEnoughData(
                    f"fold {k}: the purge and embargo ate the whole training set.\n"
                    f"  sessions={n_sess}, n_splits={self.n_splits}, "
                    f"purge={self.purge_sessions}, embargo={self.embargo_sessions}.\n"
                    f"  each fold costs {self.purge_sessions * 2 + self.embargo_sessions} "
                    f"sessions of training data. use fewer folds, or a shorter lookback.")
            if len(test_idx) == 0:
                raise NotEnoughData(f"fold {k}: empty test fold")

            yield train_idx, test_idx

    def dropped_report(self) -> pd.DataFrame:
        """what each fold actually costs, in rows. print it once and never wonder again.

        this exists because the purge and the embargo are invisible. nothing crashes when they
        are set to zero, and nothing crashes when they eat half the data. so we show the bill.
        """
        rows = []
        n_rows = len(self.ts)
        for k, (tr, te) in enumerate(self.split()):
            dropped = n_rows - len(tr) - len(te)
            rows.append({
                "fold": k,
                "train_rows": len(tr),
                "test_rows": len(te),
                "dropped_rows": dropped,
                "dropped_%": round(100 * dropped / n_rows, 2),
                "test_from": str(pd.Timestamp(self.ts[te[0]]).date()),
                "test_to": str(pd.Timestamp(self.ts[te[-1]]).date()),
            })
        return pd.DataFrame(rows)


def assert_no_leak(ts, train_idx, test_idx, purge_sessions: int, embargo_sessions: int):
    """the guard. prove, on the indices actually handed to the model, that no train row can
    see the test fold.

    this is not decoration. the purge and the embargo are pure bookkeeping -- get them wrong
    and nothing raises, nothing looks odd, the model just scores better than it deserves.
    so after the split we go back and CHECK the property we claim to have, on the real arrays.

    checked in SESSION distance, because that is the unit the cuts were made in.
    """
    ts = pd.DatetimeIndex(pd.Series(ts).values)
    # same datetime64 discipline as __init__ (see the note there), plus: map day -> session number
    # with searchsorted instead of a dict comprehension and a per-row python loop. the old
    # `[pos[d] for d in ...]` walked all 513,611 rows in python, twice, on every fold -- and this
    # is the LEAK GUARD, the one thing that must stay cheap enough that nobody is ever tempted to
    # switch it off. sess is sorted and every day is in it, so searchsorted is exact.
    day = ts.normalize().values
    sess = np.sort(pd.unique(day))

    tr_s = np.searchsorted(sess, day[train_idx])
    te_s = np.searchsorted(sess, day[test_idx])
    lo, hi = te_s.min(), te_s.max()

    # a training row before the fold: its LABEL reaches forward H sessions.
    before = tr_s[tr_s < lo]
    if len(before) and before.max() > lo - purge_sessions - 1:
        raise AssertionError(
            f"PURGE FAILED: a training row sits {lo - before.max()} session(s) before the fold, "
            f"but the label horizon is {purge_sessions} session(s). its label was decided by "
            f"prices inside the test fold.")

    # a training row after the fold: its FEATURES reach back H+L sessions.
    after = tr_s[tr_s > hi]
    need = purge_sessions + embargo_sessions
    if len(after) and after.min() < hi + need + 1:
        raise AssertionError(
            f"EMBARGO FAILED: a training row sits {after.min() - hi} session(s) after the fold, "
            f"but its features reach back {need} session(s) (feature lookback "
            f"{embargo_sessions} + label horizon {purge_sessions}). those feature values were "
            f"computed from prices inside the test fold.")

    if np.intersect1d(train_idx, test_idx).size:
        raise AssertionError("a row is in BOTH train and test")
