"""
bridge/na_policy.py -- what to do with a missing value.

THE IDEA IN ONE LINE
    NaN does not mean the same thing in every feature, so there is no single rule.
    The team that WROTE the feature declares what its NaN means; we obey and record it.

WHY 0 AND THE MEAN ARE WRONG (the thing worth understanding)
    Take gap_fill_ratio. It is a ratio, so a real value is always between 0 and 1:
        0.0  = there IS a gap and NONE of it has filled     <- a real, tradeable state
        1.0  = the gap has completely filled
    It is NaN when there is NO GAP AT ALL -- the question has no answer.

    Fill that NaN with 0 and you have told the model "there was a gap and none of it filled"
    for 86% of all minutes. Now two completely different states -- "no gap" and "an open,
    unfilled gap" -- carry the identical number. The model cannot tell them apart, and since
    the fake ones swamp the real ones it learns that 0 is boring and ignores it. You buried
    the signal.

    The mean (say 0.4) is the same disease: 0.4 is also a real value ("40% filled"), so it
    collides too -- and you have now claimed a gap existed on 86% of minutes when it never did.

    A SENTINEL cannot collide, because it is a value the feature can never actually take:

        -999                      0.0        0.5        1.0
          |                        |----------|----------|
          ^                        ^
        "no gap"              every real gap lives here, meanings intact

    The tree makes one cut ("is it below -900?") and "no gap" becomes its own branch.
    That is exactly what XGBoost and CatBoost do with NaN natively. RandomForest cannot take
    NaN at all, so the sentinel does the same job by hand.

    The sentinel is COMPUTED PER COLUMN, never hardcoded. A fixed -999 is fine for a 0..1
    ratio, but a feature measured in points could legitimately BE -999 -- and then we would
    have recreated the collision we were avoiding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class NaPolicyError(ValueError):
    pass


def compute_sentinel(s: pd.Series, margin: float = 1.0) -> float:
    """a number this column can never really take.

        sentinel = min - (max - min) - margin

    it sits a full range below the smallest real value, so no real row can ever land on it.
    for an all-NaN or constant column we still return something safely below.
    """
    lo, hi = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi):          # the column is entirely NaN
        return -1.0
    span = float(hi) - float(lo)
    if span == 0:                            # a constant column
        span = abs(float(lo)) if lo != 0 else 1.0
    return float(lo) - span - float(margin)


def apply_policy(
    df: pd.DataFrame,
    policy: str,
    *,
    for_model: str = "native",
    bar_minutes: int = 1,
    tolerance_bars: int = 3,
    sentinel_margin: float = 1.0,
    categorical_na_label: str = "MISSING",
    fixed_sentinel: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """apply one feature's NaN policy. returns (frame, what_we_did).

    for_model:
        'native'        -> xgboost / catboost. they handle NaN themselves, so we LEAVE IT.
        'no_nan'        -> random forest. it cannot train with NaN, so a sentinel is used.

    what_we_did is written into the manifest, so the treatment is never a mystery later.
    """
    if policy not in ("sentinel", "zero", "ffill", "drop"):
        raise NaPolicyError(f"unknown na_policy {policy!r}")

    out = df.copy()
    note: dict = {"policy": policy, "for_model": for_model, "columns": {}}

    # text columns are CATEGORIES. a missing category keeps its own identity -- it never gets
    # blended into a real one (a missing gap_state must not silently become 'NO_GAP').
    cat_cols = [c for c in out.columns
                if str(out[c].dtype) in ("object", "str", "string", "category", "bool")]
    for c in cat_cols:
        n = int(out[c].isna().sum())
        if n:
            out[c] = out[c].astype("object").where(out[c].notna(), categorical_na_label)
            note["columns"][c] = {"filled": categorical_na_label, "n": n}

    num_cols = [c for c in out.columns if c not in cat_cols]

    if policy == "zero":
        # the NaN honestly means zero -- a count of nothing.
        for c in num_cols:
            n = int(out[c].isna().sum())
            if n:
                out[c] = out[c].fillna(0)
                note["columns"][c] = {"filled": 0, "n": n}

    elif policy == "ffill":
        # a slow value carried on fast rows. bounded, so a stale value cannot leap the
        # overnight gap and become tomorrow morning's feature.
        limit = max(1, bar_minutes * tolerance_bars)
        for c in num_cols:
            n = int(out[c].isna().sum())
            if n:
                out[c] = out[c].ffill(limit=limit)
                note["columns"][c] = {"filled": f"ffill(limit={limit})",
                                      "n": n, "left_nan": int(out[c].isna().sum())}

    elif policy == "drop":
        # the row is genuinely unusable -> the LABEL MINUTE must leave the dataset.
        #
        # WE DO NOT DELETE THE ROW HERE, AND THAT IS THE WHOLE POINT.
        # deleting it leaves a HOLE in the feature table. align() then does what it is supposed
        # to do with a hole -- it forward-fills the last good value across it. so the unusable
        # minutes quietly receive a STALE number instead of being excluded, and 'drop' silently
        # behaves like 'ffill'. exactly backwards. (measured: a 3-minute hole handed the model a
        # value from 4 minutes earlier, and nothing complained.)
        #
        # so we KEEP the row, keep its NaN, and hand back a MASK of the unusable timestamps.
        # build_dataset uses that mask to remove the label minutes themselves.
        bad = out[num_cols].isna().any(axis=1) if num_cols else pd.Series(False, index=out.index)
        note["unusable_rows"] = int(bad.sum())
        note["unusable_index"] = out.index[bad]        # build_dataset drops THESE label minutes

    else:  # 'sentinel' -- the NaN is REAL INFORMATION
        if fixed_sentinel is not None:
            # a FLAT sentinel for every model (project decision). one value, no per-column math,
            # and NO NaN left -- so xgboost/catboost also see the fill instead of NaN. simpler to
            # explain than the computed sentinel, at the cost of the (here-impossible) collision.
            for c in num_cols:
                n = int(out[c].isna().sum())
                if n:
                    out[c] = out[c].fillna(fixed_sentinel)
                    note["columns"][c] = {"fixed_sentinel": fixed_sentinel, "n": n}
        elif for_model == "native":
            # xgboost / catboost: leave the NaN exactly as it is. they learn a branch for it.
            for c in num_cols:
                n = int(out[c].isna().sum())
                if n:
                    note["columns"][c] = {"kept_nan": True, "n": n}
        else:
            # random forest: it crashes on NaN, so give the missing rows a value nothing real
            # can reach. one clean cut then separates them, which is the same outcome.
            for c in num_cols:
                n = int(out[c].isna().sum())
                if n:
                    sv = compute_sentinel(out[c], sentinel_margin)
                    out[c] = out[c].fillna(sv)
                    note["columns"][c] = {"sentinel": sv, "n": n}

    return out, note


def encode_categoricals(df: pd.DataFrame, mapping: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """turn text columns into numbers, for the models that cannot read text.

    catboost reads text natively and never calls this. random forest and xgboost do.
    the mapping is SAVED with the model, so live data is encoded exactly the same way --
    if 'GAP_UP' is 1 at training it must still be 1 in production, for ever.
    an unseen category at prediction time becomes -1, never a silent collision with a real code.
    """
    out = df.copy()
    maps = dict(mapping or {})
    for c in out.columns:
        if str(out[c].dtype) not in ("object", "str", "string", "category", "bool"):
            continue
        if c not in maps:
            cats = sorted(str(v) for v in out[c].dropna().unique())
            maps[c] = {v: i for i, v in enumerate(cats)}
        out[c] = out[c].astype(str).map(maps[c]).fillna(-1).astype("int32")
    return out, maps
