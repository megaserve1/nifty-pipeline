"""
tests/test_alignment.py -- the tests that would have caught the leak.

Run:  pytest tests/ -q
"""
import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from bridge.align import (  # noqa: E402
    align_feature_to_labels, bar_close, infer_bar_minutes, NoPeekViolation,
)


def _five_min_feature_on_one_min_rows():
    """Exactly the shape of the real cached parquets: a 5-min VALUE, 1-min ROWS.

    Bucket 09:15 (covering 09:15..09:19, knowable only at 09:20) carries value 100.
    Bucket 09:20 carries 200, and so on.
    """
    idx = pd.date_range("2020-01-01 09:15", "2020-01-01 09:34", freq="1min")
    bucket = idx.floor("5min")
    val = pd.Series(bucket).map({t: 100 * (i + 1) for i, t in enumerate(sorted(set(bucket)))})
    return pd.DataFrame({"feat": val.values}, index=idx)


def test_five_min_value_is_not_visible_before_its_bar_closes():
    """THE regression test. Minute 09:16 must NOT see the 09:15 bucket (closes 09:20)."""
    feat = _five_min_feature_on_one_min_rows()
    labels = pd.date_range("2020-01-01 09:15", "2020-01-01 09:34", freq="1min")

    out = align_feature_to_labels(feat, labels, bar_minutes=5)
    got = pd.Series(out["feat"].values, index=labels)

    # 09:15..09:19 -> no bar has closed yet in this window -> NaN
    assert got.loc["2020-01-01 09:16"] != 100, "LEAK: saw the still-forming 09:15 bar"
    assert pd.isna(got.loc["2020-01-01 09:16"])

    # 09:20 is the instant the 09:15 bar closes -> now it is legal
    assert got.loc["2020-01-01 09:20"] == 100
    assert got.loc["2020-01-01 09:24"] == 100     # still the last closed bar
    assert got.loc["2020-01-01 09:25"] == 200     # 09:20 bar has now closed


def test_the_old_shift1_would_have_leaked():
    """Proves the bug we replaced: shift(1) hands 09:16 a value from 09:19."""
    feat = _five_min_feature_on_one_min_rows()
    labels = pd.date_range("2020-01-01 09:15", "2020-01-01 09:34", freq="1min")

    old = feat.shift(1)                                  # the previous "no-peek" rule
    merged = pd.merge_asof(
        pd.DataFrame({"ts": labels}), old.reset_index().rename(columns={"index": "ts"}),
        on="ts", direction="backward")
    leaked = merged.set_index("ts").loc["2020-01-01 09:16", "feat"]
    assert leaked == 100, "expected the old code to leak the 09:15 bucket into 09:16"

    new = align_feature_to_labels(feat, labels, bar_minutes=5)
    assert pd.isna(pd.Series(new["feat"].values, index=labels).loc["2020-01-01 09:16"])


def test_one_minute_feature_gets_previous_minute():
    idx = pd.date_range("2020-01-01 09:15", "2020-01-01 09:25", freq="1min")
    feat = pd.DataFrame({"feat": np.arange(len(idx), dtype=float)}, index=idx)
    out = align_feature_to_labels(feat, idx, bar_minutes=1)
    got = pd.Series(out["feat"].values, index=idx)
    # the 09:20 bar closes at 09:21, so label 09:21 sees it
    assert got.loc["2020-01-01 09:21"] == feat.loc["2020-01-01 09:20", "feat"]
    # a label never sees its own minute's bar (it has not closed)
    assert got.loc["2020-01-01 09:20"] == feat.loc["2020-01-01 09:19", "feat"]


def test_no_peek_invariant_holds_for_every_row():
    feat = _five_min_feature_on_one_min_rows()
    labels = pd.date_range("2020-01-01 09:15", "2020-01-01 09:34", freq="1min")
    for clock in (1, 5):
        out = align_feature_to_labels(feat, labels, bar_minutes=clock)
        assert len(out) == len(labels)
    # bar_close is always strictly after every timestamp inside the bar
    idx = pd.DatetimeIndex(["2020-01-01 09:15", "2020-01-01 09:19"])
    assert (bar_close(idx, 5) == pd.Timestamp("2020-01-01 09:20")).all()


def test_stale_value_does_not_leap_the_overnight_gap():
    """Friday's last bar must not become Monday morning's feature."""
    day1 = pd.date_range("2020-01-01 15:00", "2020-01-01 15:24", freq="1min")
    feat = pd.DataFrame({"feat": 1.0}, index=day1)
    labels = pd.DatetimeIndex(["2020-01-01 15:24", "2020-01-02 09:15", "2020-01-02 09:16"])
    out = align_feature_to_labels(feat, labels, bar_minutes=5, tolerance_bars=3)
    got = pd.Series(out["feat"].values, index=labels)
    assert not pd.isna(got.iloc[0])          # same session: fine
    assert pd.isna(got.iloc[1]), "stale value leaked across the overnight gap"
    assert pd.isna(got.iloc[2])


def test_infer_bar_minutes_detects_the_real_clock():
    assert infer_bar_minutes(_five_min_feature_on_one_min_rows()) == 5
    idx = pd.date_range("2020-01-01 09:15", periods=500, freq="1min")
    per_minute = pd.DataFrame({"feat": np.arange(500, dtype=float)}, index=idx)
    assert infer_bar_minutes(per_minute) == 1


def test_duplicate_timestamps_do_not_explode_the_join():
    idx = pd.DatetimeIndex(["2020-01-01 09:15", "2020-01-01 09:15", "2020-01-01 09:20"])
    feat = pd.DataFrame({"feat": [1.0, 2.0, 3.0]}, index=idx)
    labels = pd.DatetimeIndex(["2020-01-01 09:25"])
    out = align_feature_to_labels(feat, labels, bar_minutes=5)
    assert len(out) == 1                      # one row per label, never a fan-out


def test_rejects_bad_clock():
    feat = _five_min_feature_on_one_min_rows()
    with pytest.raises(ValueError):
        align_feature_to_labels(feat, feat.index, bar_minutes=0)


# --------------------------------------------------------------- the datetime-column trap
def test_time_can_arrive_as_a_column_not_only_as_the_index():
    """Feature parquets are inconsistent about where they put the time: some make it the
    index, some leave it in a 'datetime' column. Both must work, and the time must be
    anchored to the index BEFORE any column is touched -- otherwise a table whose only
    timestamp lives in a column can have it dropped, and align() then has nothing to
    align on. (That was a real crash.)"""
    from bridge.register import read_time_index

    idx = pd.date_range("2020-01-01 09:15", periods=10, freq="1min")

    as_column = pd.DataFrame({"datetime": idx, "real_feature": np.arange(10.0)})
    assert not isinstance(as_column.index, pd.DatetimeIndex)
    anchored = read_time_index(as_column)
    assert isinstance(anchored.index, pd.DatetimeIndex)
    assert list(anchored.columns) == ["real_feature"], "the time must leave the columns"

    out = align_feature_to_labels(anchored, idx, bar_minutes=5)   # must not raise
    assert len(out) == len(idx)

    as_index = pd.DataFrame({"real_feature": np.arange(10.0)}, index=idx)
    assert isinstance(read_time_index(as_index).index, pd.DatetimeIndex)


# ============================================================ THE CLOCK-DETECTOR LEAK
# the no-lookahead rule is only as good as the CLOCK it is given. these pin the two ways a
# wrong clock crept back in and served an unclosed bar on 4 of every 5 minutes.

def test_a_fast_column_must_not_drag_a_slow_one_into_a_lookahead():
    """REGRESSION. A parquet holding a 5-min signal NEXT TO a 1-min column is an ordinary shape
    (the signal, and the price it was computed from). The old detector required EVERY column to
    be constant in a bucket, so the 1-min column forced a 1-min clock on the whole file -- and
    the 5-min column was then served before its bar closed. Measured: 80% of minutes."""
    from bridge.align import column_clocks

    idx = pd.date_range("2020-01-01 09:15", periods=200, freq="1min")
    bucket = idx.floor("5min")
    sig5 = pd.Series(bucket).map({t: float(i) for i, t in enumerate(sorted(set(bucket)))}).values
    mix = pd.DataFrame({"sig_5min": sig5,                    # a genuine 5-MINUTE value
                        "close_1min": np.arange(200.0)},     # changes every minute
                       index=idx)

    clocks = column_clocks(mix)
    assert clocks["sig_5min"] == 5, "the 5-min column must keep its own 5-min clock"
    assert clocks["close_1min"] == 1, "...and the 1-min column must not be slowed down"

    # and the whole-file fallback must be the SLOWEST, never the fastest
    assert infer_bar_minutes(mix) == 5, (
        "the single safe clock is the SLOWEST column. taking the fastest serves a bar that has "
        "not closed -- that is lookahead."
    )

    # the 5-min column, aligned on its own clock, must not be visible before it closes
    out = align_feature_to_labels(mix[["sig_5min"]], idx, bar_minutes=5)
    got = pd.Series(out["sig_5min"].values, index=idx)
    assert pd.isna(got.loc["2020-01-01 09:16"]), "LEAK: served a bar that had not closed"
    assert got.loc["2020-01-01 09:20"] == 0.0, "at its close, it becomes legal"


def test_float_jitter_must_not_turn_a_5min_value_into_a_1min_one():
    """REGRESSION. Comparing with nunique() is EXACT equality. A 5-min value that has been
    through a float32 round-trip differs in the 12th decimal from row to row -- and then it
    reads as a 1-MINUTE column, and we serve a bar that has not closed."""
    from bridge.align import clock_of_column

    idx = pd.date_range("2020-01-01 09:15", periods=300, freq="1min")
    bucket = idx.floor("5min")
    clean = pd.Series(bucket).map({t: 100.0 + i for i, t in enumerate(sorted(set(bucket)))}).values
    jitter = clean * (1 + np.random.default_rng(0).normal(0, 1e-12, len(clean)))

    s_clean = pd.Series(clean, index=idx)
    s_jit = pd.Series(jitter, index=idx)

    assert clock_of_column(s_clean) == 5
    assert clock_of_column(s_jit) == 5, (
        "float noise in the 12th decimal must NOT be read as real movement. if it is, the "
        "column is treated as 1-minute and its 5-minute bar is served 4 minutes early."
    )


def test_a_column_constant_in_only_most_buckets_is_not_good_enough():
    """the old test accepted 99.9% constancy. over 5 years of minute data that is hundreds of
    bars served early. it must hold in EVERY bucket."""
    from bridge.align import clock_of_column

    idx = pd.date_range("2020-01-01 09:15", periods=500, freq="1min")
    bucket = idx.floor("5min")
    v = pd.Series(bucket).map({t: float(i) for i, t in enumerate(sorted(set(bucket)))}).values
    s = pd.Series(v, index=idx)
    s.iloc[7] += 50.0                      # ONE bucket where it genuinely moves

    assert clock_of_column(s) == 1, (
        "if the value genuinely moves inside even one bucket, it is NOT a 5-minute value. "
        "calling it one would hold a real 1-minute signal back."
    )
