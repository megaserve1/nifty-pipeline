"""
tests/test_the_four_bugs.py -- the regression tests for the four bugs found on 2026-07-14.

EVERY TEST IN HERE FAILS ON THE OLD CODE. That is the only thing that makes a test worth
having. The suite was 21-green while all four of these bugs were live, because it tested the
PIECES and never the PATHS that join them:

    bug 1  the clock detector returned the SMALLEST period, so a 15-minute feature was
           measured as 3-minute and served 12 minutes before its bar had closed.
           -> the tests only ever checked clocks 1 and 5, the only two the old rule got right.

    bug 2  publish_version.py skipped publish() entirely on the "already exists" path, so a
           run interrupted between finalize() and publish() left the dataset at 'completed'
           for ever and the trigger waited for it in silence.
           -> NO test called publish() at all. You could delete the line and stay green.

    bug 3  auto_trigger.py imported a function that does not exist and died on line 31.
           -> no test imported the module.

    bug 4  train.py's --max_depth default was 6 (a BOOSTING depth) and RandomForest read it.
           -> the test named after this bug tested build_model() and never the argparse
              default that actually reaches it, so it passed with the bug live.

Offline. No network, no ClearML, no GCS.
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C                                  # noqa: E402
from bridge import align                            # noqa: E402


SESSION_OPEN = "09:15"


def _session(day: str, minutes: int = 375) -> pd.DatetimeIndex:
    """one trading session of 1-minute stamps, 09:15 onwards -- the real NSE shape."""
    return pd.date_range(f"{day} {SESSION_OPEN}", periods=minutes, freq="1min")


_EPOCH = pd.Timestamp("2024-01-01")     # a FIXED, unit-free reference for encoding bar starts


def _smeared(idx: pd.DatetimeIndex, clock: int) -> pd.Series:
    """a feature on a `clock`-minute bar, stored smeared across 1-minute rows.

    the value of each bar is its bar-start encoded as MINUTES SINCE _EPOCH. so it is CONSTANT
    inside the bar and changes exactly on the bar boundary -- what a real 5-minute feature
    looks like on a 1-minute index -- and it can be decoded back to a timestamp EXACTLY.

    THE FIRST VERSION ENCODED RAW EPOCH INTEGERS (`starts.astype("int64") / 1e11`) AND THE
    DECODER ASSUMED NANOSECONDS. pandas 3 hands back MICROSECONDS here -- so every decoded bar
    start landed in JANUARY 1970, its close was 54 years before any label minute, and the
    invariant `close <= minute` was true for every row NO MATTER WHAT THE ALIGNER DID. the
    no-peek test was green while proving nothing. minutes-since-a-fixed-epoch cannot be
    misread, whatever unit pandas picks this year.
    """
    starts = align._bar_start(idx, clock)
    return pd.Series((starts - _EPOCH).total_seconds() / 60.0, index=idx)


def _decode(v: float) -> pd.Timestamp:
    """the exact inverse of _smeared's encoding."""
    return _EPOCH + pd.Timedelta(minutes=float(v))


# =============================================================================
# BUG 1 -- the clock detector. THE LEAK.
# =============================================================================
@pytest.mark.parametrize("clock", [1, 3, 5, 15, 30, 60])
def test_the_clock_detector_finds_the_REAL_bar_period(clock):
    """THE BUG: the old rule returned the SMALLEST period the column sits still inside.

    that premise is false. a value constant across every 15-minute bar is AUTOMATICALLY constant
    inside every 3-minute bucket, because 3 divides 15 and both floor from the same origin. so
    the old loop hit 3 first and stopped, and answered 3min for 15, 30 and 60.

    the old code passes this test for clock=1 and clock=5 and FAILS it for 15, 30 and 60 --
    which is exactly the shape of the bug, and exactly what the old test suite never checked.
    """
    idx = _session("2024-01-02")
    s = _smeared(idx, clock)
    got = align.clock_of_column(s)
    assert got == clock, (
        f"a {clock}-minute feature was measured as {got}-minute. "
        f"if got < clock, every bar is served {clock - got} minutes BEFORE IT CLOSED. that is "
        f"lookahead, and it is the bug this test exists for.")


def test_a_15min_feature_is_never_measured_as_3min():
    """the headline case, spelled out on its own so the failure message is unmissable."""
    s = _smeared(_session("2024-01-02"), 15)
    assert align.clock_of_column(s) != 3, "THE OLD BUG IS BACK: 15-minute feature read as 3-minute"
    assert align.clock_of_column(s) == 15


def test_when_the_evidence_is_thin_the_clock_comes_out_SLOW_not_fast():
    """a column that RARELY changes cannot be told apart from a column on a SLOW clock.

    gap_state is NO_GAP 86% of the time. the honest answer is "I cannot tell" -- and when we
    cannot tell we must choose the SLOW clock, because slow is stale (honest) and fast is
    lookahead (a lie). this test pins the DIRECTION of the guess.

    THE FIRST VERSION OF THIS TEST WAS WRONG, AND IT IS WORTH RECORDING HOW. it planted its one
    change at 09:16 -- a minute that sits on NO bar boundary except the 1-minute one. a change
    at 09:16 is not thin evidence; it is HARD PROOF of a 1-minute clock (no 3/5/15-minute bar
    changes value mid-bar), and the detector correctly answered 1min. the test then failed the
    detector for being right. thin evidence means: the few changes all land ON slow-bar
    boundaries, so every clock remains possible -- THAT is when the guess must come out slow.
    """
    idx = _session("2024-01-02")
    s = pd.Series(["NO_GAP"] * len(idx), index=idx)
    s.iloc[:60] = "GAP_UP"          # one change, at 10:15 -- ON the 3/5/15/30/60-min boundary,
                                    # so the evidence rules nothing out
    clock = align.clock_of_column(s)
    assert clock >= 5, (f"a column with almost no changes was called {clock}-minute. guessing "
                        f"FAST on ambiguous evidence is how a lookahead leak gets in.")
    rep = align.clock_report(s)
    assert rep["confident"] is False, "it should KNOW that it does not know"


def test_hard_evidence_of_a_fast_clock_beats_the_slow_default():
    """the counterpart: ONE change on a non-boundary minute is proof of a fast clock, and the
    detector must believe proof over caution -- otherwise every quiet 1-minute state column
    gets served an hour late."""
    idx = _session("2024-01-02")
    s = pd.Series(["NO_GAP"] * len(idx), index=idx)
    s.iloc[1:] = "GAP_UP"           # the change happens at 09:16 -- INSIDE every slow bar
    assert align.clock_of_column(s) == 1, (
        "a value that changed at 09:16 cannot be on a 3/5/15-minute clock -- no closed bar "
        "boundary sits there. the evidence proves 1min and must win.")


def test_a_sparse_one_row_per_bar_table_is_read_from_its_row_spacing():
    """the second way the old detector broke: a feature delivered ONE ROW PER BAR.

    resample("5min") gives 09:15, 09:20, 09:25 -- one row each. there is nothing to repeat, so
    every bucket holds at most one row, "constant inside the bucket" is trivially true, and the
    repetition test learns nothing at all. for such a table the clock is not hidden: it is the
    row spacing. this is a very likely delivery format, which is what makes it dangerous.
    """
    idx = pd.date_range(f"2024-01-02 {SESSION_OPEN}", periods=60, freq="5min")
    feat = pd.DataFrame({"x": np.arange(60, dtype=float)}, index=idx)
    assert align.row_spacing_minutes(feat) == 5
    assert align.infer_bar_minutes(feat) == 5, (
        "a table with one row every 5 minutes cannot be a 1-minute feature, whatever the "
        "repetition test thinks it sees")


# =============================================================================
# BUG 1b -- bar_close anchored at the session open, not midnight
# =============================================================================
def test_bar_close_is_anchored_at_the_session_open_not_midnight():
    """THE BUG: floor() counts from MIDNIGHT. the session opens at 09:15 = 555 minutes.

    555 = 3 x 5 x 37. so 1, 3, 5 and 15 divide it and the bars line up either way. 30 and 60
    DO NOT. a 30-minute bar of the session runs 09:15..09:44 and truly closes at 09:45 -- but
    floor(09:15, "30min") from midnight is 09:00, so the old code thought it closed at 09:30
    and served it FIFTEEN MINUTES EARLY. the arithmetic was self-consistent, so the
    NoPeekViolation assert waved it straight through.
    """
    t = pd.DatetimeIndex(["2024-01-02 09:15"])

    # what the old code did -- kept here so the bug is visible, not just described
    old = t.floor("30min") + pd.Timedelta(minutes=30)
    assert old[0] == pd.Timestamp("2024-01-02 09:30"), "this is the OLD, WRONG close"

    new = align.bar_close(t, 30)
    assert new[0] == pd.Timestamp("2024-01-02 09:45"), (
        "the 09:15-09:44 bar closes at 09:45. anything earlier serves it before it exists.")

    # and the clocks that already worked must not have moved
    for clock in (1, 3, 5, 15):
        assert align.bar_close(t, clock)[0] == t.floor(f"{clock}min")[0] + pd.Timedelta(
            minutes=clock), f"{clock}min must be unchanged -- {clock} divides 555"


def test_no_minute_ever_receives_a_bar_that_has_not_closed():
    """the invariant itself, checked row by row, on every clock. this is the whole point."""
    idx = _session("2024-01-02")
    for clock in (1, 3, 5, 15, 30, 60):
        feat = pd.DataFrame({"v": _smeared(idx, clock)})
        out = align.align_feature_to_labels(feat, idx, bar_minutes=clock, tolerance_bars=99)
        served = out["v"].to_numpy()
        # the value we were served IS a bar-start (we encoded it that way). recover it, and
        # check that bar had already CLOSED by the label minute.
        checked = 0
        for i, minute in enumerate(idx):
            if np.isnan(served[i]):
                continue
            bar_start = _decode(served[i])
            # NON-VACUITY: the decode must land in the session's own year. the old test decoded
            # into 1970 and then "passed" every row for 54-year-old bars.
            assert bar_start.year == minute.year, (
                f"decoded bar start {bar_start} is not even in {minute.year} -- "
                f"the encoding is broken and this test is proving nothing")
            close = bar_start + pd.Timedelta(minutes=clock)
            assert close <= minute, (
                f"clock={clock}: minute {minute} was served the bar starting {bar_start}, "
                f"which does not close until {close}. THAT IS LOOKAHEAD.")
            checked += 1
        assert checked > 300, (f"only {checked} rows actually checked for clock={clock} -- "
                               f"a no-peek test that checks nothing proves nothing")


def test_a_timezone_aware_feature_is_REFUSED_not_silently_stripped():
    """THE BUG: tz_localize(None) drops the offset and KEEPS THE WALL CLOCK.

    hand it a UTC-aware parquet -- exactly what a default pandas write on the feature team's
    Windows box produces -- and "09:15 UTC" becomes naive "09:15", which then matches label
    minute 09:15 IST with a distance of zero. that is 5.5 HOURS of lookahead, and nothing
    downstream can see it: merge_asof is happy, the tolerance is happy, NoPeekViolation cannot
    fire (naive vs naive is internally consistent), and the manifest still swears no_peek.
    """
    idx = pd.date_range("2024-01-02 09:15", periods=100, freq="1min", tz="UTC")
    feat = pd.DataFrame({"x": np.arange(100, dtype=float)}, index=idx)
    with pytest.raises(align.TimezoneAwareFeature):
        align.align_feature_to_labels(feat, _session("2024-01-02", 100), bar_minutes=1)


def test_a_feature_that_does_not_overlap_the_labels_is_refused():
    """no assert anywhere used to check that the feature and the labels describe the same time."""
    feat = pd.DataFrame({"x": [1.0, 2.0, 3.0]},
                        index=pd.date_range("2019-01-02 09:15", periods=3, freq="1min"))
    labels = _session("2024-01-02", 10)
    with pytest.raises(ValueError, match="do not overlap"):
        align.align_feature_to_labels(feat, labels, bar_minutes=1)


def test_too_few_rows_raises_instead_of_quietly_guessing_1min():
    """the old code answered 1min -- the FAST, unsafe direction -- on three rows of evidence."""
    feat = pd.DataFrame({"x": [1.0, 2.0, 3.0]},
                        index=pd.date_range("2024-01-02 09:15", periods=3, freq="1min"))
    with pytest.raises(ValueError, match="too few"):
        align.column_clocks(feat)


# =============================================================================
# BUG 4 -- the max_depth default. THE CRIPPLED FOREST.
# =============================================================================
def test_the_forest_gets_a_DEEP_tree_and_the_booster_a_SHALLOW_one():
    """THE BUG THIS GUARDS, AND WHY THE OLD TEST MISSED IT.

    the original bug: ONE argparse default (6) was shared by all three models, so RandomForest
    was silently built at depth 6 -- a BOOSTING depth. a forest wants DEEP trees (averaging
    cancels the noise); a booster wants SHALLOW ones (they correct each other). the crippling was
    that the forest got the booster's number.

    an earlier test asserted RF max_depth default == 0 (fully grown). the defaults now come from
    the proposed-hyperparameter PDF (1-min column): RF is a deep 22, xgboost a shallow 4. Both
    still satisfy the real invariant -- the forest is DEEP, the booster is SHALLOW, and they are
    NOT the same number. This test pins that invariant, not one specific value, so re-tuning the
    numbers cannot resurrect the crippled-forest bug.
    """
    from trainer import train as T
    from trainer import hyperparams

    rf_depth = hyperparams.defaults("random_forest")["max_depth"]
    xgb_def = hyperparams.defaults("xgboost")
    xgb_policy = str(xgb_def.get("grow_policy", "depthwise"))

    # THE FOREST INVARIANT IS UNCONDITIONAL. a forest must be DEEP (0 = unlimited, or a large
    # cap). NEVER a boosting depth of ~6. this is what the original crippled-forest bug violated,
    # and it holds regardless of what the boosters are doing.
    assert rf_depth == 0 or rf_depth >= 12, (
        f"the FOREST's max_depth default is {rf_depth} -- too shallow. a forest wants deep trees "
        f"(0=unlimited, or >=12). ~6 is a BOOSTING depth and cripples it.")
    rf = T.build_model("random_forest", 7, hyperparams.defaults("random_forest"))
    assert rf.max_depth is None or rf.max_depth >= 12, "the forest must build a DEEP tree"

    # THE BOOSTER INVARIANT DEPENDS ON THE GROWTH POLICY (loss-based support added 2026-07-21).
    if xgb_policy == "lossguide":
        # LOSS-BASED (leaf-wise): max_depth is NOT the shape control -- max_leaves is. so the
        # "booster shallower than the forest" test does not apply (the h3 overfit config
        # deliberately runs unlimited depth). the bug this file guards -- a boosting *depth*
        # silently crippling a *forest* -- cannot occur through the depth knob here. instead
        # pin that it is actually built loss-based. max_leaves 0 is LEGAL here: under xgboost
        # lossguide, 0 means "no leaf cap" -- the deliberate max-overfit setting, not a mistake.
        ml = int(xgb_def.get("max_leaves", 0) or 0)
        assert ml >= 0, f"max_leaves must be a non-negative int (0 = unlimited), got {ml}"
        xgb = T.build_model("xgboost", 7, xgb_def)
        assert xgb.get_params()["grow_policy"] == "lossguide"
        assert int(xgb.get_params()["max_leaves"]) == ml
    else:
        # DEPTH-BASED: the original guard. the booster must be SHALLOWER THAN THE FOREST (deep
        # boosters memorise noise), a bounded depth, and NOT the same number as the forest.
        xgb_depth = xgb_def["max_depth"]
        assert 1 <= xgb_depth <= 16, f"xgboost depth {xgb_depth} is not a bounded boosting depth"
        assert xgb_depth < rf_depth or rf_depth == 0, (
            f"the booster ({xgb_depth}) must be shallower than the forest ({rf_depth})")
        assert rf_depth != xgb_depth, "forest and booster must not share one depth (original bug)"
        xgb = T.build_model("xgboost", 7, xgb_def)
        assert 1 <= xgb.max_depth <= 16, "the booster must build a bounded tree"
        assert xgb.max_depth < (rf.max_depth or 10**6), "the booster must be shallower than the forest"


def test_no_hyperparameter_is_hardcoded_in_python_any_more():
    """the point of the YAML. a number buried in argparse is how the max_depth bug happened and
    how it stayed hidden -- nobody reads argparse defaults. one readable file, or it will happen
    again with a different number."""
    train_src = (ROOT / "trainer" / "train.py").read_text()
    hpo_src = (ROOT / "trainer" / "hpo.py").read_text()

    assert "hyperparams.all_param_names()" in train_src, (
        "train.py must GENERATE its parser from configs/hyperparams.yaml, not list defaults")
    assert 'default=6' not in train_src and 'default=300' not in train_src, (
        "there is still a hardcoded hyperparameter default in train.py")
    assert "hyperparams.search_space" in hpo_src, (
        "hpo.py must read the search space from the YAML, not hold its own copy. two copies of "
        "the same numbers, in two files, free to disagree -- and they did.")


def test_every_knob_hpo_searches_actually_EXISTS_on_the_trainer():
    """THE BUG: hpo.py searched 6 parameters train.py had never heard of.

    ClearML sets task parameters BY NAME. a name the trainer's parser does not have is a SILENT
    no-op -- clearml logs a warning nobody reads, the trial trains the DEFAULT model, reports the
    DEFAULT score, and the search concludes that nothing you changed made any difference.
    because nothing you changed WAS changed.

    this can no longer drift, because both sides read the same YAML -- but pin it anyway, since
    the whole point is that the two must never disagree again.
    """
    from trainer import hpo, hyperparams

    have = set(hyperparams.all_param_names())
    for mtype in C.MODEL_TYPES:
        wanted = {p.name.split("/", 1)[1] for p in hpo.search_space(mtype)}
        missing = sorted(wanted - have)
        assert not missing, (
            f"hpo.py searches over {missing} for {mtype}, but the trainer's parser has no such "
            f"argument. every trial would silently train the default model.")


def test_a_string_zero_from_clearml_does_not_become_a_depth_of_ZERO():
    """ClearML's optimiser sets task parameters as STRINGS.

    "0" is truthy. so `params["max_depth"] or None` on the string "0" gives "0", not None --
    and sklearn gets a max_depth of "0". the YAML default tells us the type, so the override is
    cast to int and the forest gets None. this is the sort of thing that produces a silently
    wrong model rather than an error.
    """
    from trainer import hyperparams

    p = hyperparams.merge("random_forest", {"max_depth": "0", "n_estimators": "900"})
    assert p["max_depth"] == 0 and isinstance(p["max_depth"], int)
    assert (p["max_depth"] or None) is None, "a STRING '0' would be truthy -- and wrong"
    assert p["n_estimators"] == 900 and isinstance(p["n_estimators"], int)


def test_the_optimiser_looks_for_a_scalar_the_trainer_actually_reports():
    """THE BUG: hpo.py optimised "val/trading_cost". train.py never reported it.

    train.py did not import objective.py at all -- there was no validation split. so every
    trial would score None, Objective.get_objective() would swallow the miss, and the search
    would hand back whatever it sampled first. green all the way.
    """
    from trainer import objective

    train_src = (ROOT / "trainer" / "train.py").read_text()
    assert "report_trading_cost(logger, \"val\"" in train_src, (
        f"train.py never reports {objective.OBJECTIVE_SERIES!r}, which is the scalar hpo.py "
        f"minimises. every trial would score None and the search would be theatre.")
    assert "three_way_split" in train_src, "train.py has no validation split to tune on"
    assert objective.OBJECTIVE_SERIES == "val/trading_cost"
    assert objective.OBJECTIVE_TITLE == "Summary"


# =============================================================================
# THE EMBARGO -- sessions, not calendar days
# =============================================================================
def test_the_embargo_is_counted_in_SESSIONS_not_calendar_days():
    """THE BUG: `cut + Timedelta(days=21)` to cover a 20-SESSION feature lookback.

    the market is shut at weekends and on holidays. measured on the real label file, 21 calendar
    days spans 14.1 sessions on average and NEVER 20. so the embargo was ~30% short and the
    first ~6 sessions of every test set still carried features built from training prices.
    """
    from trainer.purged_cv import embargo_end, sessions_of

    # 120 weekdays -- a realistic calendar with the weekends already missing
    days = pd.bdate_range("2024-01-01", periods=120)
    ts = pd.Series([d + pd.Timedelta(hours=9, minutes=15 + m)
                    for d in days for m in range(5)])

    cut = pd.Timestamp("2024-03-01 09:15")
    end = embargo_end(ts, cut, 25)

    sess = sessions_of(ts)
    swallowed = int(((sess > np.datetime64(cut.normalize())) & (sess <= np.datetime64(end))).sum())
    assert swallowed == 25, (
        f"the embargo swallowed {swallowed} sessions, not 25. it is counted in SESSIONS -- "
        f"the unit the features are actually built in.")

    # and it must be LONGER in wall-clock than the old calendar-day setting
    assert (end - cut) > pd.Timedelta(days=25), (
        "25 trading sessions spans MORE than 25 calendar days, because of the weekends. "
        "if this fails, the embargo is being applied as calendar days again.")


def test_config_no_longer_carries_the_calendar_day_embargo():
    """EMBARGO_DAYS was the bug. it must be gone, not just supplemented."""
    assert not hasattr(C, "EMBARGO_DAYS"), (
        "config.EMBARGO_DAYS still exists. it was CALENDAR days, applied to a lookback measured "
        "in SESSIONS. leaving it importable invites someone to use it again.")
    assert C.EMBARGO_SESSIONS >= 20, "the embargo must cover the 20-session feature lookback"
    assert C.SESSION_ANCHOR_MINUTES == 9 * 60 + 15, "NSE opens at 09:15"
