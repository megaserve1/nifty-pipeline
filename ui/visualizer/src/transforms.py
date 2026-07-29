"""Reshaping helpers: time filtering, feature/price alignment, normalisation."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

NORMALISERS = ("none", "z-score", "min-max", "% change", "first=100", "rank")


def filter_time_range(
    df: pd.DataFrame,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    column: str = "timestamp",
) -> pd.DataFrame:
    """Inclusive time-window slice. ``None`` bounds are left open."""
    if df.empty or column not in df.columns:
        return df
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= df[column] >= pd.Timestamp(start)
    if end is not None:
        mask &= df[column] <= pd.Timestamp(end)
    return df.loc[mask].reset_index(drop=True)


def normalise_series(series: pd.Series, method: str = "none") -> pd.Series:
    """Rescale a feature so unrelated magnitudes can share one axis.

    Degenerate inputs (constant or all-NaN) return zeros rather than NaN/inf so
    the trace still renders as a flat line instead of vanishing.
    """
    values = pd.to_numeric(pd.Series(series), errors="coerce").astype(float)
    method = (method or "none").lower()

    if method in ("none", ""):
        return values
    if values.dropna().empty:
        return values

    if method == "z-score":
        std = values.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            return values - values.mean()
        return (values - values.mean()) / std
    if method == "min-max":
        lo, hi = values.min(), values.max()
        if not np.isfinite(hi - lo) or hi == lo:
            return pd.Series(np.zeros(len(values)), index=values.index)
        return (values - lo) / (hi - lo)
    if method == "% change":
        return values.pct_change().replace([np.inf, -np.inf], np.nan) * 100.0
    if method == "first=100":
        first = values.dropna()
        base = first.iloc[0] if len(first) else np.nan
        # A negative base flips the sign of every subsequent value in the ratio,
        # so the rebased series' *direction* no longer matches the raw series
        # (an upward move looks like a downward one). Fall back to raw values
        # rather than emit a misleading trace -- indexing to 100 only has a
        # coherent meaning for a strictly-positive base.
        if not np.isfinite(base) or base <= 0:
            return values
        return values / base * 100.0
    if method == "rank":
        return values.rank(pct=True) * 100.0

    raise ValueError(f"Unknown normalisation method: {method!r}")


def normalise_frame(
    df: pd.DataFrame,
    columns: Sequence[str],
    method: str = "none",
) -> pd.DataFrame:
    """Apply :func:`normalise_series` to ``columns``, leaving the rest untouched."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = normalise_series(out[col], method)
    return out


def align_to_price(
    price: pd.DataFrame,
    other: pd.DataFrame,
    how: str = "exact",
    tolerance: Optional[pd.Timedelta] = None,
    direction: str = "backward",
    column: str = "timestamp",
) -> pd.DataFrame:
    """Attach ``other``'s columns to every price bar, keyed on timestamp.

    ``how="exact"`` keeps only identical timestamps. ``how="nearest"`` uses
    :func:`pandas.merge_asof`, which rescues files whose clocks differ by a few
    seconds or that are stamped at a coarser interval than the price bars.
    """
    if column not in price.columns or column not in other.columns:
        raise ValueError(f"Both frames need a {column!r} column to be aligned.")

    # Only the timestamp is carried over from `price`, so every column of
    # `other` survives -- including a feature that happens to be named "close".
    # Callers keep the price frame separate, so nothing is shadowed.
    left = price[[column]].copy()
    right = other.copy()

    if right.empty or right.shape[1] <= 1:
        return left

    if how == "exact":
        merged = left.merge(right, on=column, how="left")
    elif how == "nearest":
        merged = pd.merge_asof(
            left.sort_values(column),
            right.sort_values(column),
            on=column,
            direction=direction,
            tolerance=tolerance,
        )
    else:
        raise ValueError(f"Unknown alignment mode: {how!r}")
    return merged.reset_index(drop=True)


def coverage(aligned: pd.DataFrame, columns: Iterable[str]) -> float:
    """Fraction of rows where at least one of ``columns`` is populated (0..1)."""
    cols = [c for c in columns if c in aligned.columns]
    if not cols or aligned.empty:
        return 0.0
    return float(aligned[cols].notna().any(axis=1).mean())


def downsample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Evenly thin a frame to at most ``max_rows``, always keeping the last row."""
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    step = int(np.ceil(len(df) / max_rows))
    keep = list(range(0, len(df), step))
    if keep[-1] != len(df) - 1:
        keep.append(len(df) - 1)
    return df.iloc[keep].reset_index(drop=True)


def label_events(
    labels: pd.DataFrame,
    column: str,
    include: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Rows of ``labels`` where ``column`` is set (optionally limited to ``include``)."""
    if column not in labels.columns:
        return labels.iloc[0:0]
    mask = labels[column].notna()
    if include is not None:
        mask &= labels[column].isin(list(include))
    return labels.loc[mask].reset_index(drop=True)


def search_features(names: Iterable[str], query: str) -> list[str]:
    """Case-insensitive filter. Space-separated terms are AND-ed together."""
    terms = [t for t in str(query or "").lower().split() if t]
    if not terms:
        return list(names)
    return [n for n in names if all(t in str(n).lower() for t in terms)]
