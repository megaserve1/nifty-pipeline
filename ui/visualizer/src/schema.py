"""Column-name detection heuristics.

Uploaded CSVs come from many different exporters, so nothing here is assumed --
these functions only produce *best guesses* that the UI presents as pre-selected
defaults. The user can always override every mapping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import pandas as pd

# Alias lists are ordered best-first. Matching is done on a normalised form of
# the column name (lowercase, non-alphanumerics collapsed to "_").
TIMESTAMP_ALIASES = (
    "timestamp", "datetime", "date_time", "time", "bar_time", "candle_time",
    "date", "dt", "ts", "index",
)
OPEN_ALIASES = ("open", "open_price", "o", "op")
HIGH_ALIASES = ("high", "high_price", "h", "hi")
LOW_ALIASES = ("low", "low_price", "l", "lo")
CLOSE_ALIASES = ("close", "close_price", "adj_close", "last_price", "last", "ltp", "c")
VOLUME_ALIASES = ("volume", "vol", "traded_volume", "qty", "quantity", "v")
OPEN_INTEREST_ALIASES = ("open_interest", "oi", "openinterest")

# Detected *before* target so that "predicted_label" is never mistaken for a
# target column (both contain the word "label").
PREDICTED_ALIASES = (
    "predicted_label", "predicted", "prediction", "pred_label", "pred",
    "y_pred", "yhat", "y_hat", "model_label", "model_output", "forecast",
)
TARGET_ALIASES = (
    "target_label", "target", "true_label", "actual_label", "ground_truth",
    "label", "actual", "y_true", "ytrue", "y",
)

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Aliases shorter than this are matched only exactly or on a word boundary.
MIN_SUBSTRING_ALIAS = 3

# Words that mark a column as a *measurement about* a prediction rather than the
# prediction itself. Without this, "pred_confidence" satisfies the "pred" alias
# and a column of 0.597, 0.667, ... gets charted as if those were class names.
# Matched whole-word only, so "is_confirmed" is untouched.
LABEL_EXCLUDE_TOKENS = (
    "confidence", "conf", "proba", "probability", "prob",
    "score", "logit", "pct", "percent", "margin",
)

# A float column with more distinct values than this is a measurement, not a class.
MAX_LABEL_CARDINALITY = 32


def normalise(name: object) -> str:
    """Fold a column name to a comparable token, e.g. ``"Target Label" -> "target_label"``."""
    return _NON_ALNUM.sub("_", str(name).strip().lower()).strip("_")


def find_column(
    columns: Iterable[object],
    aliases: Sequence[str],
    exclude: Iterable[object] = (),
) -> Optional[object]:
    """Return the first column matching ``aliases``, or ``None``.

    Three passes, strongest evidence first: exact match, whole-word match, then
    substring. Running exact across *all* aliases before falling back keeps
    "close" from losing to "adj_close" merely because of column ordering.

    Short aliases ("o", "hi", ...) are barred from the substring pass -- without
    that guard "o" matches "volume" and a ``timestamp, volume, close`` file maps
    open to the volume column.
    """
    excluded = list(exclude)
    candidates = [
        (col, normalise(col))
        for col in columns
        if not any(col is e or col == e for e in excluded)
    ]
    if not candidates:
        return None

    for alias in aliases:
        for col, norm in candidates:
            if norm == alias:
                return col
    for alias in aliases:
        for col, norm in candidates:
            if alias in norm.split("_"):
                return col
    for alias in aliases:
        if len(alias) < MIN_SUBSTRING_ALIAS:
            continue
        for col, norm in candidates:
            if alias in norm:
                return col
    return None


@dataclass
class OhlcMapping:
    """Which uploaded columns hold the price series."""

    timestamp: Optional[object] = None
    open: Optional[object] = None
    high: Optional[object] = None
    low: Optional[object] = None
    close: Optional[object] = None
    volume: Optional[object] = None

    @property
    def is_complete(self) -> bool:
        return all(
            getattr(self, field) is not None
            for field in ("timestamp", "open", "high", "low", "close")
        )

    def missing(self) -> list[str]:
        return [
            field
            for field in ("timestamp", "open", "high", "low", "close")
            if getattr(self, field) is None
        ]


@dataclass
class LabelMapping:
    """Which uploaded columns hold the target / predicted labels."""

    timestamp: Optional[object] = None
    target: Optional[object] = None
    predicted: Optional[object] = None

    @property
    def has_any_label(self) -> bool:
        return self.target is not None or self.predicted is not None


def detect_ohlc(df: pd.DataFrame) -> OhlcMapping:
    """Guess the timestamp/OHLCV columns of ``df``."""
    cols = list(df.columns)
    timestamp = find_column(cols, TIMESTAMP_ALIASES)
    taken = [c for c in (timestamp,) if c is not None]

    # Claim open-interest columns FIRST so "Open Interest" cannot satisfy the
    # 'open' whole-word alias on the fallback pass. Without this, integer OI
    # counts get charted as opening prices.
    open_interest = find_column(cols, OPEN_INTEREST_ALIASES, exclude=taken)
    if open_interest is not None:
        taken.append(open_interest)

    mapping = OhlcMapping(timestamp=timestamp)
    for field, aliases in (
        ("open", OPEN_ALIASES),
        ("high", HIGH_ALIASES),
        ("low", LOW_ALIASES),
        ("close", CLOSE_ALIASES),
        ("volume", VOLUME_ALIASES),
    ):
        found = find_column(cols, aliases, exclude=taken)
        if found is not None:
            setattr(mapping, field, found)
            taken.append(found)

    if mapping.timestamp is None:
        # Fall back to the first column that actually parses as a datetime.
        for col in cols:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.8:
                mapping.timestamp = col
                break
    return mapping


def plausible_label_columns(df: pd.DataFrame) -> list[object]:
    """Columns that could hold a class label.

    Drops confidence/probability/score columns by name, and any float column with
    too many distinct values to be a class -- both are measurements *about* a
    prediction, not the prediction.
    """
    out: list[object] = []
    for col in df.columns:
        words = normalise(col).split("_")
        if any(token in words for token in LABEL_EXCLUDE_TOKENS):
            continue
        series = df[col]
        if pd.api.types.is_float_dtype(series):
            try:
                if series.nunique(dropna=True) > MAX_LABEL_CARDINALITY:
                    continue
            except TypeError:                       # unhashable contents
                continue
        out.append(col)
    return out


def detect_labels(df: pd.DataFrame, timestamp: Optional[object] = None) -> LabelMapping:
    """Guess the timestamp/target/predicted columns of ``df``.

    ``predicted`` is resolved first and then excluded from the ``target`` search,
    otherwise a file with only ``predicted_label`` would report it as the target.
    """
    cols = list(df.columns)
    if timestamp is None:
        timestamp = find_column(cols, TIMESTAMP_ALIASES)

    taken = [c for c in (timestamp,) if c is not None]
    cols = plausible_label_columns(df)
    predicted = find_column(cols, PREDICTED_ALIASES, exclude=taken)
    if predicted is not None:
        taken.append(predicted)
    target = find_column(cols, TARGET_ALIASES, exclude=taken)
    return LabelMapping(timestamp=timestamp, target=target, predicted=predicted)


def feature_columns(
    df: pd.DataFrame,
    exclude: Iterable[object] = (),
    numeric_only: bool = True,
) -> list[object]:
    """Every column usable as a plottable feature.

    Excludes the reserved (timestamp/OHLC/label) columns and, by default,
    anything non-numeric.
    """
    excluded = {normalise(c) for c in exclude if c is not None}
    out: list[object] = []
    for col in df.columns:
        if normalise(col) in excluded:
            continue
        if numeric_only and not pd.api.types.is_numeric_dtype(df[col]):
            continue
        out.append(col)
    return out


def looks_like_ohlc(df: pd.DataFrame) -> bool:
    """True when ``df`` carries a full price series (used to offer single-file mode)."""
    return detect_ohlc(df).is_complete
