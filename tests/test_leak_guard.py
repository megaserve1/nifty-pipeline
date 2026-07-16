"""
tests/test_leak_guard.py -- the guard that refuses the future, the answer, and the calendar.

WHY IT EXISTS. On 2026-07-14 the feature team dropped merged_raw_1min.parquet into
data/features/. 195 columns. Among them fwd_ret_1/3/5/10 -- the return over the NEXT n minutes.
Measured on the real file, fwd_ret_1 correlates +0.79 with the next minute's return and +0.02
with the last one. It is not a feature. It is the answer.

Nothing in the pipeline would have stopped it reaching a dataset.

Offline. Synthetic data only -- these tests must pass on a machine that has never seen the real
parquets.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bridge import leak_guard      # noqa: E402


def _prices(n=5000, seed=0):
    """a random walk on a 1-minute index. no signal in it at all."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 09:15", periods=n, freq="1min")
    close = 20000 + np.cumsum(rng.normal(0, 3, n))
    return idx, pd.Series(close, index=idx)


def test_it_bans_a_forward_return_even_if_the_name_is_innocent():
    """THE ONE THAT MATTERS. A name check alone is theatre -- the next leak will be called
    'score' or 'signal', not 'fwd_ret_10'. So the guard MEASURES every numeric column against
    the price: a column built from the past tracks past returns; a column built from the future
    tracks future ones."""
    idx, close = _prices()
    df = pd.DataFrame({
        "close": close,
        "innocent_looking_score": close.shift(-10) / close - 1,   # the future, badly disguised
    }, index=idx)

    r = leak_guard.screen(df)
    assert "innocent_looking_score" in r["banned"], (
        "a column holding the NEXT 10 minutes of return was not caught. the name gave nothing "
        "away -- the MEASUREMENT has to catch it, or the guard is decoration.")
    assert "MEASURED" in r["banned"]["innocent_looking_score"]


def test_it_does_NOT_ban_a_genuine_backward_looking_feature():
    """the guard must not eat good features. a momentum signal is MADE of past returns, so it
    correlates with them -- that is not a leak, that is the whole idea."""
    idx, close = _prices()
    df = pd.DataFrame({
        "close": close,
        "momentum_10": close / close.shift(10) - 1,        # the PAST. legitimate.
        "sma_20": close.rolling(20).mean(),                 # the PAST. legitimate.
        "vol_20": close.pct_change().rolling(20).std(),     # the PAST. legitimate.
    }, index=idx)

    r = leak_guard.screen(df)
    for c in ("momentum_10", "sma_20", "vol_20"):
        assert c not in r["banned"], (
            f"{c} is built from the PAST and was banned. a guard that deletes good features "
            f"gets switched off, and then it guards nothing.")


def test_it_bans_the_calendar():
    """not lookahead. worse in one way: the model memorises WHICH DAY it is, then meets a test
    set where every row carries an id it has never seen."""
    idx, close = _prices()
    df = pd.DataFrame({
        "close": close,
        "session": pd.Series(idx.strftime("%Y-%m-%d"), index=idx),   # the date, as text
        "t5": pd.Series(idx, index=idx),                              # a raw timestamp
        "row_counter": pd.Series(np.arange(len(idx), dtype=float), index=idx),  # a clock
    }, index=idx)

    r = leak_guard.screen(df)
    for c in ("session", "t5", "row_counter"):
        assert c in r["banned"], f"{c} is the calendar and must never be a feature"


def test_it_does_NOT_ban_time_of_day():
    """minute_of_day repeats every single day, so it cannot encode WHICH day it is. it is a
    legitimate and useful feature -- the market really does behave differently at 09:20."""
    idx, close = _prices()
    df = pd.DataFrame({
        "close": close,
        "minute_of_day": pd.Series(idx.hour * 60 + idx.minute, index=idx),
        "session_phase": pd.Series(np.where(idx.hour < 12, "OPEN", "CLOSE"), index=idx),
    }, index=idx)

    r = leak_guard.screen(df)
    assert "minute_of_day" not in r["banned"]
    assert "session_phase" not in r["banned"]


def test_label_named_columns_that_are_really_STATE_NAMES_are_not_banned():
    """THE FALSE POSITIVE THAT ALMOST COST A REAL FEATURE.

    The first version of this guard banned anything containing "label". It caught the real
    targets -- and it also killed `label_combined`, which is 100% IDENTICAL to Stress_Signal,
    one of the twelve bucket features. And `Flow_State_Label`, which is just the text name of
    Flow_State.

    In this project "label" usually means "the NAME of a state", not "the answer". So those are
    SUSPECT (said out loud) and not BANNED (silently deleted).
    """
    idx, close = _prices()
    df = pd.DataFrame({
        "close": close,
        "label_combined": pd.Series(np.random.default_rng(1).choice(
            ["CALM", "ELEVATED", "EXTREME", "BROAD"], len(idx)), index=idx),
        "Flow_State_Label": pd.Series(np.random.default_rng(2).choice(
            ["NORMAL", "ACCUMULATION", "DISTRIBUTION"], len(idx)), index=idx),
    }, index=idx)

    r = leak_guard.screen(df)
    assert "label_combined" not in r["banned"], (
        "label_combined IS Stress_Signal. banning it silently deletes a real feature.")
    assert "Flow_State_Label" not in r["banned"]
    # but it must SAY something -- a silent pass is how the real one gets through
    assert "label_combined" in r["suspect"]
    assert "Flow_State_Label" in r["suspect"]


def test_the_real_targets_are_still_banned_by_name():
    idx, close = _prices()
    df = pd.DataFrame({
        "close": close,
        "label_int": pd.Series(np.random.default_rng(3).choice([-1.0, 0.0, 1.0], len(idx)), index=idx),
        "dir_at_label": pd.Series(np.random.default_rng(4).choice([-1.0, 0.0, 1.0], len(idx)), index=idx),
        "target": pd.Series(np.zeros(len(idx)), index=idx),
        "fwd_ret_5": close.shift(-5) / close - 1,
        "signed_ret_3": (close.shift(-3) / close - 1).abs(),
    }, index=idx)

    r = leak_guard.screen(df)
    for c in ("label_int", "dir_at_label", "fwd_ret_5", "signed_ret_3"):
        assert c in r["banned"], f"{c} is the answer and must be refused"


def test_allow_columns_lets_a_human_force_one_through():
    """the escape hatch. explicit, in registry.yaml, with a name against it -- because an
    override nobody can see is the same as no guard at all."""
    idx, close = _prices()
    df = pd.DataFrame({"close": close, "session": pd.Series(idx.strftime("%Y-%m-%d"), index=idx)},
                      index=idx)

    assert "session" in leak_guard.screen(df)["banned"]
    assert "session" not in leak_guard.screen(df, allow=["session"])["banned"]


def test_a_file_of_pure_leaks_is_refused_entirely():
    """merged_raw is a WORKING file, not a feature file. if every column goes, say so."""
    idx, close = _prices()
    df = pd.DataFrame({
        "fwd_ret_1": close.shift(-1) / close - 1,
        "label_int": pd.Series(np.zeros(len(idx)), index=idx),
        "session": pd.Series(idx.strftime("%Y-%m-%d"), index=idx),
    }, index=idx)
    r = leak_guard.screen(df)
    assert len(r["ok"]) == 0
    assert len(r["banned"]) == 3
