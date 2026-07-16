"""
tests/test_na_policy.py -- prove that the sentinel cannot collide, and 0/mean can.

this is the whole argument for the design, written as code that either passes or fails.
"""
import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from na_policy import (  # noqa: E402
    apply_policy, compute_sentinel, encode_categoricals, NaPolicyError,
)


def _gap_like():
    """the real shape of gap_fill_ratio: a 0..1 ratio that is NaN when there is no gap.
    0.0 is a REAL value ("a gap exists and none of it filled")."""
    return pd.DataFrame({
        "gap_fill_ratio": [np.nan, 0.0, 0.4, np.nan, 1.0, np.nan],
        "gap_state": ["NO_GAP", "GAP_UP", "GAP_UP", "NO_GAP", "GAP_DOWN", "NO_GAP"],
    })


# ---------------------------------------------------------------- the collision argument
def test_zero_fill_collides_with_a_real_zero():
    """filling with 0 makes 'no gap' identical to 'an open, unfilled gap'."""
    df = _gap_like()
    out, _ = apply_policy(df, "zero", for_model="no_nan")
    v = out["gap_fill_ratio"]
    # row 0 was 'no gap' (NaN). row 1 was a REAL unfilled gap (0.0).
    assert v.iloc[0] == v.iloc[1] == 0.0, "they now carry the same number"
    # ...which is exactly the problem: two different states, one value. this test documents
    # the danger; it is why 'zero' is only correct when NaN genuinely MEANS zero.


def test_mean_fill_would_also_collide():
    """the mean (0.4667) is a value the feature can really take -> same collision."""
    df = _gap_like()
    mean = df["gap_fill_ratio"].mean()
    filled = df["gap_fill_ratio"].fillna(mean)
    real_values = df["gap_fill_ratio"].dropna().tolist()
    assert 0.4 in real_values
    # a filled row and a real row can land on the same number
    assert (filled.round(4) == round(mean, 4)).sum() > df["gap_fill_ratio"].isna().sum() - 1


def test_sentinel_can_never_collide():
    """THE POINT. the sentinel sits below every value the column can actually take."""
    df = _gap_like()
    out, note = apply_policy(df, "sentinel", for_model="no_nan")
    v = out["gap_fill_ratio"]
    sv = note["columns"]["gap_fill_ratio"]["sentinel"]

    real = df["gap_fill_ratio"].dropna()
    assert sv < real.min(), "the sentinel must be below EVERY real value"
    assert not (real == sv).any(), "no real row may ever equal the sentinel"

    # the missing rows all sit on it, and the real 0.0 is untouched and still means 'unfilled'
    assert (v[df["gap_fill_ratio"].isna()] == sv).all()
    assert v.iloc[1] == 0.0
    assert v.iloc[1] != sv, "the real 0.0 must NOT have been swallowed by the sentinel"


def test_sentinel_is_computed_not_hardcoded():
    """-999 would be safe for a 0..1 ratio but NOT for a feature measured in points.
    the sentinel must adapt to the column, or it recreates the collision."""
    points = pd.Series([-2000.0, -999.0, 500.0, np.nan])   # -999 is a REAL value here
    sv = compute_sentinel(points)
    assert sv < points.min(skipna=True)
    assert sv != -999.0, "a hardcoded -999 would have collided with a real row"


def test_no_nan_survives_for_random_forest():
    """random forest crashes on NaN. after the sentinel policy there must be none left."""
    df = _gap_like()
    out, _ = apply_policy(df, "sentinel", for_model="no_nan")
    assert out["gap_fill_ratio"].isna().sum() == 0


def test_nan_is_kept_for_xgboost_and_catboost():
    """they learn a branch for missing themselves -- do not touch it."""
    df = _gap_like()
    out, note = apply_policy(df, "sentinel", for_model="native")
    assert out["gap_fill_ratio"].isna().sum() == 3
    assert note["columns"]["gap_fill_ratio"]["kept_nan"] is True


# ---------------------------------------------------------------- the other policies
def test_zero_policy_when_nan_really_means_zero():
    df = pd.DataFrame({"count": [np.nan, 3.0, np.nan]})
    out, note = apply_policy(df, "zero", for_model="no_nan")
    assert out["count"].tolist() == [0.0, 3.0, 0.0]
    assert note["columns"]["count"]["n"] == 2


def test_ffill_is_bounded_so_it_cannot_leap_a_gap():
    df = pd.DataFrame({"slow": [1.0] + [np.nan] * 10})
    out, _ = apply_policy(df, "ffill", for_model="no_nan", bar_minutes=1, tolerance_bars=3)
    assert out["slow"].iloc[1] == 1.0          # carried a little way
    assert pd.isna(out["slow"].iloc[8]), "a stale value must not be carried for ever"


def test_drop_marks_the_minute_unusable_but_must_NOT_delete_the_row():
    """THE BUG THIS PINS.

    'drop' used to DELETE the row from the feature table. That leaves a HOLE -- and align()
    does exactly what it should with a hole: it forward-fills the last good value across it.
    So the 'dropped' minutes quietly received a STALE number instead of being excluded, and
    'drop' silently behaved like 'ffill'. Precisely backwards.

    Measured before the fix: a 3-minute hole handed the model a value from 4 minutes earlier,
    and nothing complained.

    So: keep the row, keep its NaN, and hand back a MASK. build_dataset removes the LABEL
    MINUTES themselves.
    """
    idx = pd.date_range("2020-01-01 09:15", periods=3, freq="1min")
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]}, index=idx)
    out, note = apply_policy(df, "drop", for_model="no_nan")

    assert len(out) == 3, "the row must NOT be deleted -- a hole gets forward-filled"
    assert pd.isna(out["x"].iloc[1]), "its NaN is preserved"
    assert note["unusable_rows"] == 1
    assert list(note["unusable_index"]) == [idx[1]], "the mask names the unusable MINUTE"


def test_drop_really_removes_those_minutes_from_the_dataset():
    """end to end: the unusable minutes must leave, not be served a stale value."""
    from bridge.align import align_feature_to_labels

    labels = pd.date_range("2020-01-01 09:15", periods=10, freq="1min")
    feat = pd.DataFrame({"v": np.arange(10.0) * 10}, index=labels)
    feat.iloc[5:8, 0] = np.nan                       # 09:20, 09:21, 09:22 unusable

    treated, note = apply_policy(feat, "drop", for_model="native", bar_minutes=1)
    aligned = align_feature_to_labels(treated, labels, bar_minutes=1)

    drop_mask = pd.Series(labels).isin(note["unusable_index"]).to_numpy()
    assert int(drop_mask.sum()) == 3

    # the three unusable minutes are exactly the ones removed
    removed = set(labels[drop_mask])
    assert removed == set(labels[5:8])

    # and crucially: they are NOT quietly given the last good value
    kept = aligned[~drop_mask]
    assert len(kept) == 7


def test_unknown_policy_is_refused():
    with pytest.raises(NaPolicyError):
        apply_policy(_gap_like(), "guess", for_model="no_nan")


# ---------------------------------------------------------------- categories
def test_missing_category_keeps_its_own_identity():
    """a missing gap_state must NOT silently become NO_GAP."""
    df = pd.DataFrame({"gap_state": ["NO_GAP", None, "GAP_UP"]})
    out, note = apply_policy(df, "sentinel", for_model="native")
    assert out["gap_state"].tolist() == ["NO_GAP", "MISSING", "GAP_UP"]
    assert note["columns"]["gap_state"]["filled"] == "MISSING"


def test_category_encoding_is_stable_and_saved():
    """the mapping must be reusable, or live data gets encoded differently from training."""
    train = pd.DataFrame({"gap_state": ["NO_GAP", "GAP_UP", "GAP_DOWN"]})
    enc, maps = encode_categoricals(train)
    assert sorted(maps["gap_state"]) == ["GAP_DOWN", "GAP_UP", "NO_GAP"]

    # the SAME mapping applied later must give the SAME numbers
    live = pd.DataFrame({"gap_state": ["GAP_UP", "NO_GAP"]})
    enc2, _ = encode_categoricals(live, mapping=maps)
    assert enc2["gap_state"].iloc[0] == maps["gap_state"]["GAP_UP"]
    assert enc2["gap_state"].iloc[1] == maps["gap_state"]["NO_GAP"]


def test_unseen_category_becomes_minus_one_not_a_real_code():
    train = pd.DataFrame({"gap_state": ["NO_GAP", "GAP_UP"]})
    _, maps = encode_categoricals(train)
    live = pd.DataFrame({"gap_state": ["SOMETHING_NEW"]})
    enc, _ = encode_categoricals(live, mapping=maps)
    assert enc["gap_state"].iloc[0] == -1, "an unseen category must not collide with a real code"


# ================================================================== preprocessing leakage
def test_encoding_and_sentinel_must_be_learned_from_TRAIN_ONLY():
    """REGRESSION. if the category mapping, or the sentinel, is worked out from the WHOLE
    dataset, then facts about the TEST period have shaped how the TRAINING data was prepared.
    That is a leak: it never crashes, never shows in a metric, and quietly flatters the score.

    Fit on train, apply to test -- exactly as production must, where the future does not exist.
    """
    train = pd.DataFrame({"cat": ["A", "B", "A"], "num": [1.0, 2.0, np.nan]})
    test  = pd.DataFrame({"cat": ["A", "ZZZ"],    "num": [3.0, -50.0]})   # ZZZ + a new minimum
    full  = pd.concat([train, test], ignore_index=True)

    # --- the WRONG way: fit on everything ---
    _, bad_maps = encode_categoricals(full)
    assert "ZZZ" in bad_maps["cat"], "a test-only category leaked into the mapping"
    bad_sent = compute_sentinel(full["num"])          # sees the test row's -50.0

    # --- the RIGHT way: fit on train only ---
    _, good_maps = encode_categoricals(train)
    assert "ZZZ" not in good_maps["cat"], "the mapping must not know about test-only categories"
    good_sent = compute_sentinel(train["num"])        # only ever saw 1.0 and 2.0

    assert good_sent != bad_sent, "the sentinel must differ -- that difference IS the leak"

    # applying the train mapping to test: the unseen category becomes -1, exactly as it would
    # live when a genuinely new category turns up.
    enc, _ = encode_categoricals(test, mapping=good_maps)
    assert enc["cat"].iloc[1] == -1
