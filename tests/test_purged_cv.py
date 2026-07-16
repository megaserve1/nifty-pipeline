"""
tests/test_purged_cv.py -- the tests that pin purged k-fold.

each test names the bug it exists to stop. they are arguments, not assertions.

Run:  final_venv/bin/python -m pytest tests/test_purged_cv.py -q
"""
import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from trainer.purged_cv import (  # noqa: E402
    PurgedKFold, NotEnoughData, assert_no_leak, sessions_of,
)

SESSION_MINUTES = 375          # 09:15..15:29 inclusive


def make_ts(n_sessions=200, start="2020-01-01", holidays=()):
    """minute timestamps shaped like the real data: 375 rows a day, weekdays, minus holidays.

    the holidays are the point. a purge counted in calendar days or in rows gets them wrong.
    """
    days, d = [], pd.Timestamp(start)
    while len(days) < n_sessions:
        if d.weekday() < 5 and d.normalize() not in set(pd.to_datetime(list(holidays))):
            days.append(d.normalize())
        d += pd.Timedelta(days=1)
    out = []
    for day in days:
        out.append(pd.date_range(day + pd.Timedelta(hours=9, minutes=15),
                                 periods=SESSION_MINUTES, freq="1min"))
    return pd.DatetimeIndex(np.concatenate(out))


# --------------------------------------------------------------------------- the two cuts
def test_purge_removes_train_rows_whose_LABEL_reaches_into_the_fold():
    """PIN: the left cut is sized by the LABEL HORIZON, not the feature lookback.

    a train row on the session right before the fold has a label decided by what price does
    NEXT session -- which is inside the fold. if it survives into train, the model is being
    taught with an answer copied out of the test set.
    """
    ts = make_ts(120)
    cv = PurgedKFold(ts, n_splits=4, purge_sessions=3, embargo_sessions=0)
    sess = sessions_of(ts)
    pos = {d: i for i, d in enumerate(sess)}

    for tr, te in cv.split():
        tr_s = np.array([pos[d] for d in pd.DatetimeIndex(ts[tr]).normalize()])
        te_s = np.array([pos[d] for d in pd.DatetimeIndex(ts[te]).normalize()])
        before = tr_s[tr_s < te_s.min()]
        if len(before):
            gap = te_s.min() - before.max()
            assert gap > 3, f"a train row is only {gap} session(s) before the fold; purge is 3"


def test_embargo_removes_train_rows_whose_FEATURES_reach_back_into_the_fold():
    """PIN: the right cut is the OTHER channel, and it is a DIFFERENT SIZE.

    ret_20d on a train row 5 sessions after the fold is computed from sessions -15..+5 around
    it -- and 15 of those sit inside the fold. the feature value is literally made of test
    prices. purging alone (which only looks at labels) does not touch this row.
    """
    ts = make_ts(200)
    cv = PurgedKFold(ts, n_splits=4, purge_sessions=1, embargo_sessions=20)
    sess = sessions_of(ts)
    pos = {d: i for i, d in enumerate(sess)}

    saw_an_after_side = False
    for tr, te in cv.split():
        tr_s = np.array([pos[d] for d in pd.DatetimeIndex(ts[tr]).normalize()])
        te_s = np.array([pos[d] for d in pd.DatetimeIndex(ts[te]).normalize()])
        after = tr_s[tr_s > te_s.max()]
        if len(after):
            saw_an_after_side = True
            gap = after.min() - te_s.max()
            # must clear the feature lookback (20) AND the fold's own label horizon (1)
            assert gap > 21, f"a train row is only {gap} session(s) after the fold; need > 21"
    assert saw_an_after_side, "this test is vacuous unless some fold has training data after it"


def test_the_two_cuts_are_ASYMMETRIC():
    """PIN: a symmetric gap is wrong. the right side must be bigger than the left.

    left  = H         (the train row's label reaching forward)
    right = H + L     (the train row's features reaching back, PLUS the fold's own labels
                       reaching forward out of the fold into those rows)

    anyone who 'simplifies' this to one gap on both sides either over-cuts the left (wasting
    data) or under-cuts the right (leaking). this test makes that refactor fail loudly.
    """
    ts = make_ts(300)
    H, L = 2, 20
    cv = PurgedKFold(ts, n_splits=5, purge_sessions=H, embargo_sessions=L)
    sess = sessions_of(ts)
    pos = {d: i for i, d in enumerate(sess)}

    tr, te = next(iter(cv.split()))       # fold 0 has no left side, so take a middle fold
    folds = list(cv.split())
    tr, te = folds[2]
    tr_s = np.array([pos[d] for d in pd.DatetimeIndex(ts[tr]).normalize()])
    te_s = np.array([pos[d] for d in pd.DatetimeIndex(ts[te]).normalize()])

    left_gap = te_s.min() - tr_s[tr_s < te_s.min()].max()
    right_gap = tr_s[tr_s > te_s.max()].min() - te_s.max()

    assert left_gap == H + 1, f"left gap should be exactly H={H} sessions of purge"
    assert right_gap == H + L + 1, f"right gap should be H+L={H+L} sessions"
    assert right_gap > left_gap, "the cuts are not symmetric -- that is the whole point"


# --------------------------------------------------------------------------- the units
def test_calendar_days_are_NOT_trading_sessions():
    """PIN: THE BUG IN config.EMBARGO_DAYS = 21.

    train.py does `cut + pd.Timedelta(days=21)` and the comment says it covers the 20-day
    feature lookback. it does not. ret_20d looks back 20 TRADING SESSIONS. a 21-CALENDAR-day
    window holds about 14 sessions -- weekends and holidays eat the rest.

    measured on the real labels file: a 21-calendar-day window contains 14.1 sessions on
    average (min 9, max 16) and NEVER 20, at any cut point in five years. you need 25-39
    calendar days (mean 29.8) to span 20 sessions.

    this test reproduces that on data with the same shape.
    """
    ts = make_ts(120, holidays=["2020-01-23", "2020-02-19", "2020-03-11"])
    sess = sessions_of(ts)

    covered = []
    for d in sess[:-30]:
        covered.append(int(((sess > d) & (sess <= d + pd.Timedelta(days=21))).sum()))
    covered = np.array(covered)

    assert covered.max() < 20, (
        f"a 21-CALENDAR-day gap covered {covered.max()} sessions at best -- it is being asked "
        f"to cover a 20-SESSION feature lookback and it cannot. this is why the cuts in "
        f"purged_cv.py are counted in sessions.")
    assert covered.mean() < 16


def test_row_counting_is_unsafe_because_sessions_are_not_all_375_rows():
    """PIN: 'just purge 20*375 rows' is wrong.

    15 of the 1,372 real sessions are short -- one has 54 rows (a half day). count 7,500 rows
    back from a fold that sits next to a short session and you land INSIDE the window you were
    trying to skip. the splitter must count SESSIONS, which it does.
    """
    ts = make_ts(60)
    # surgically shorten one session, exactly like a real half-day
    day = sessions_of(ts)[30]
    keep = ~((pd.DatetimeIndex(ts).normalize() == day) &
             (pd.DatetimeIndex(ts) > day + pd.Timedelta(hours=10)))
    ts_short = pd.DatetimeIndex(ts)[keep]

    counts = pd.Series(1, index=ts_short).groupby(pd.DatetimeIndex(ts_short).normalize()).size()
    assert (counts != SESSION_MINUTES).sum() == 1, "we meant to create exactly one short session"

    # the splitter still cuts whole sessions, short one included -- no row arithmetic anywhere
    cv = PurgedKFold(ts_short, n_splits=3, purge_sessions=1, embargo_sessions=5)
    for tr, te in cv.split():
        tr_days = set(pd.DatetimeIndex(ts_short[tr]).normalize())
        te_days = set(pd.DatetimeIndex(ts_short[te]).normalize())
        assert not (tr_days & te_days), "a session landed in BOTH train and test"


# --------------------------------------------------------------------------- the sklearn traps
def test_sklearn_TimeSeriesSplit_does_NOT_purge_and_does_NOT_embargo():
    """PIN: do not reach for TimeSeriesSplit and think you are safe.

    verified against the installed source (sklearn 1.9.0, model_selection/_split.py):

        for test_start in test_starts:
            train_end = test_start - gap
            yield indices[:train_end], indices[test_start : test_start + test_size]

    train is always indices[:train_end]. with gap=0 the training set runs right up to the first
    test row -- a row one minute before the fold, sharing 20 days of rolling history with it.
    and `gap` is in ROWS, so it cannot express a session-based lookback.

    grep the whole sklearn package for "purg" or "embargo": zero hits.
    """
    from sklearn.model_selection import TimeSeriesSplit

    n = 1000
    tss = TimeSeriesSplit(n_splits=4)                 # the default: gap=0
    for tr, te in tss.split(np.zeros((n, 2))):
        assert tr.max() == te.min() - 1, (
            "TimeSeriesSplit leaves NO gap by default -- the last train row is the one right "
            "before the first test row")

    # and even with gap set, there is never any training data AFTER the fold to embargo
    tss = TimeSeriesSplit(n_splits=4, gap=50)
    for tr, te in tss.split(np.zeros((n, 2))):
        assert tr.max() < te.min(), "train must be before test"
        assert not (tr > te.max()).any(), (
            "TimeSeriesSplit never puts training data after the fold, so it throws away all the "
            "history beyond the first fold -- which is exactly what purged k-fold recovers")


def test_subclassing_BaseCrossValidator_silently_defeats_the_purge():
    """PIN: THE MOST COMMON WAY 'PurgedKFold' IS WRITTEN WRONG.

    sklearn's BaseCrossValidator.split() is:

        train_index = indices[np.logical_not(test_index)]      # train = everything not in test

    so if you inherit from it and implement only _iter_test_indices -- the obvious thing, and
    what most snippets do -- the base class hands back EVERY row that is not in the test fold.
    your purge and your embargo are never applied. nothing raises. the class is still called
    PurgedKFold. it just leaks.

    that is why purged_cv.PurgedKFold owns split() outright and inherits nothing.
    """
    from sklearn.model_selection import BaseCrossValidator

    class NaivePurgedKFold(BaseCrossValidator):
        """what everyone writes. it purges nothing."""
        def __init__(self, n_splits=4):
            self.n_splits = n_splits

        def get_n_splits(self, X=None, y=None, groups=None):
            return self.n_splits

        def _iter_test_indices(self, X=None, y=None, groups=None):
            for block in np.array_split(np.arange(len(X)), self.n_splits):
                yield block

    X = np.zeros((400, 2))
    naive = NaivePurgedKFold(n_splits=4)
    folds = list(naive.split(X))
    tr, te = folds[1]                                   # a middle fold, so it has both sides

    assert tr.max() > te.max(), "middle fold should have training data after it"
    assert te.min() - tr[tr < te.min()].max() == 1, (
        "the naive subclass left ZERO purge -- the last train row is one row before the fold")
    assert tr[tr > te.max()].min() - te.max() == 1, (
        "the naive subclass left ZERO embargo -- the first train row is one row after the fold")

    # ours, on the same shape, cuts both sides
    ts = make_ts(80)
    ours = PurgedKFold(ts, n_splits=4, purge_sessions=1, embargo_sessions=10)
    otr, ote = list(ours.split())[1]
    sess = sessions_of(ts)
    pos = {d: i for i, d in enumerate(sess)}
    otr_s = np.array([pos[d] for d in pd.DatetimeIndex(ts[otr]).normalize()])
    ote_s = np.array([pos[d] for d in pd.DatetimeIndex(ts[ote]).normalize()])
    assert ote_s.min() - otr_s[otr_s < ote_s.min()].max() > 1, "ours purges the left"
    assert otr_s[otr_s > ote_s.max()].min() - ote_s.max() > 10, "ours embargoes the right"


def test_it_still_works_as_an_sklearn_cv_object():
    """PIN: we inherit nothing, so prove we did not break the sklearn contract.

    check_cv only asks `hasattr(cv, "split")`, so duck-typing is enough -- but if someone later
    adds an __init__ arg that shadows split(), this catches it.
    """
    from sklearn.model_selection import check_cv

    ts = make_ts(60)
    cv = PurgedKFold(ts, n_splits=3, purge_sessions=1, embargo_sessions=5)
    checked = check_cv(cv)
    assert checked is cv, "sklearn should accept our splitter as-is"
    assert checked.get_n_splits() == 3
    assert len(list(checked.split(np.zeros((len(ts), 2))))) == 3


# --------------------------------------------------------------------------- the guard
def test_assert_no_leak_catches_a_deliberately_broken_split():
    """PIN: the guard must actually fire. a guard that never fires is decoration.

    we hand it a split with NO purge and NO embargo and demand that it raises.
    """
    ts = make_ts(100)
    n_sess = len(sessions_of(ts))
    day = pd.DatetimeIndex(ts).normalize()
    sess = sessions_of(ts)

    test_days = sess[40:60]
    te = np.flatnonzero(np.isin(day, test_days))
    tr = np.flatnonzero(~np.isin(day, test_days))       # everything else -- the naive split

    with pytest.raises(AssertionError, match="PURGE FAILED"):
        assert_no_leak(ts, tr, te, purge_sessions=2, embargo_sessions=20)

    # and the real splitter's output passes the same guard
    cv = PurgedKFold(ts, n_splits=4, purge_sessions=2, embargo_sessions=20)
    for a, b in cv.split():
        assert_no_leak(ts, a, b, purge_sessions=2, embargo_sessions=20)


def test_unsorted_timestamps_are_refused():
    """PIN: every cut here is positional in time. unsorted rows put the purge in the wrong
    place, and the leak would be completely silent. so we refuse rather than guess."""
    ts = make_ts(30)
    shuffled = pd.DatetimeIndex(np.random.default_rng(0).permutation(ts.values))
    with pytest.raises(ValueError, match="not sorted"):
        PurgedKFold(shuffled, n_splits=3)


def test_too_much_purge_raises_instead_of_returning_a_tiny_train_set():
    """PIN: with a long lookback and many folds, the cuts can eat the training set alive.

    the failure we refuse to have: it silently returns 200 training rows, the model trains on
    them, and the fold score is pure noise that nobody questions because nothing crashed.
    """
    ts = make_ts(40)                              # only 40 sessions
    cv = PurgedKFold(ts, n_splits=2, purge_sessions=20, embargo_sessions=20)
    with pytest.raises(NotEnoughData, match="ate the whole training set"):
        list(cv.split())


def test_every_session_is_tested_exactly_once():
    """PIN: the folds must cover the data. if array_split ever drifts, some period is never
    tested and we would never know."""
    ts = make_ts(137)                             # deliberately not divisible by n_splits
    cv = PurgedKFold(ts, n_splits=5, purge_sessions=1, embargo_sessions=10)
    seen = []
    for tr, te in cv.split():
        seen.append(pd.DatetimeIndex(ts[te]).normalize().unique())
        assert not set(np.asarray(tr)) & set(np.asarray(te)), "a row is in train AND test"
    all_tested = np.concatenate(seen)
    assert len(all_tested) == len(np.unique(all_tested)), "a session was tested twice"
    assert len(np.unique(all_tested)) == len(sessions_of(ts)), "a session was never tested"
