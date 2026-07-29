"""CSV reading and normalisation into the canonical frames the charts consume.

Canonical shapes produced here:

* price   -> ``timestamp, open, high, low, close[, volume]``  (sorted, unique)
* labels  -> ``timestamp, target, predicted``                 (labels as strings)
* features-> ``timestamp, <feature ...>``                     (numeric)
"""

from __future__ import annotations

import io
import logging
import time
import warnings
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from . import schema as schema_mod

log = logging.getLogger("visualizer.data_loader")

CsvSource = Union[str, bytes, bytearray, io.IOBase]

PRICE_FIELDS = ("open", "high", "low", "close")

# Epoch heuristics: (upper bound on |value|, pandas unit). Ordered ascending so
# the first match wins. Bounds sit between plausible ranges for years ~1973-2100.
_EPOCH_BUCKETS = (
    (1e11, "s"),
    (1e14, "ms"),
    (1e17, "us"),
    (np.inf, "ns"),
)


class DataError(ValueError):
    """Raised when an upload cannot be interpreted as the requested shape."""


def read_csv(source: CsvSource, **kwargs) -> pd.DataFrame:
    """Read a CSV from a path, raw bytes, or an uploaded file object."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(bytes(source))
    options = {"skipinitialspace": True}
    options.update(kwargs)
    try:
        df = pd.read_csv(source, **options)
    except pd.errors.EmptyDataError as exc:
        raise DataError("The CSV file is empty.") from exc
    except UnicodeDecodeError as exc:
        raise DataError(
            "This is a binary file, not a text CSV — check you pasted the right "
            "path. Supported here: .csv, .txt, .parquet, .xlsx."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim in the UI
        raise DataError(f"Could not parse the CSV: {exc}") from exc

    if df.empty:
        raise DataError("The CSV file contains no rows.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


# Extensions we recognise but cannot read as a table, with what to do instead.
WRONG_KIND = {
    ".joblib": "a model bundle — it belongs in the **Predict from Model** tab's "
               "model box, not here",
    ".pkl": "a pickled object — if it is your model, it belongs in the "
            "**Predict from Model** tab's model box",
    ".pt": "a PyTorch checkpoint, not a data table",
    ".h5": "a Keras/HDF5 model, not a data table",
    ".onnx": "an ONNX model, not a data table",
    ".zip": "an archive — extract it and point at the CSV/Parquet inside",
    ".json": "JSON — export it to CSV or Parquet first",
}

EXCEL_SUFFIXES = (".xlsx", ".xlsm", ".xls")
PARQUET_SUFFIXES = (".parquet", ".pq")


def check_readable(filename: str) -> None:
    """Raise a helpful :class:`DataError` for file types that are not tables.

    Pasting a path into the wrong box is easy; without this the user gets
    pandas' "'utf-8' codec can't decode byte 0x80" instead of being told they
    handed a table reader a model file.
    """
    suffix = Path(str(filename)).suffix.lower()
    if suffix in WRONG_KIND:
        raise DataError(f"`{Path(str(filename)).name}` is {WRONG_KIND[suffix]}.")


def read_table(source: CsvSource, filename: str = "", **kwargs) -> pd.DataFrame:
    """Read a CSV, Parquet or Excel file, dispatching on ``filename``'s extension.

    The training pipeline hands datasets around as Parquet, so every tab accepts
    it rather than forcing a conversion step.
    """
    check_readable(filename)
    lower = str(filename).lower()

    label = Path(str(filename)).name or "<inline>"
    start = time.perf_counter()

    if lower.endswith(EXCEL_SUFFIXES):
        log.info("reading Excel: %s", label)
        if isinstance(source, (bytes, bytearray)):
            source = io.BytesIO(bytes(source))
        try:
            df = pd.read_excel(source)
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim in the UI
            raise DataError(f"Could not read the Excel file: {exc}") from exc
        if df.empty:
            raise DataError("The Excel file contains no rows.")
        df.columns = [str(c).strip() for c in df.columns]
        log.info("read %s: %s rows x %d cols in %.2fs",
                 label, f"{len(df):,}", df.shape[1], time.perf_counter() - start)
        return df

    if lower.endswith(PARQUET_SUFFIXES):
        log.info("reading Parquet: %s", label)
        if isinstance(source, (bytes, bytearray)):
            source = io.BytesIO(bytes(source))
        try:
            df = pd.read_parquet(source)
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim in the UI
            raise DataError(f"Could not read the Parquet file: {exc}") from exc
        if df.empty:
            raise DataError("The Parquet file contains no rows.")
        # A DatetimeIndex is the pipeline's usual way of carrying the timestamp.
        if isinstance(df.index, pd.DatetimeIndex) and "timestamp" not in df.columns:
            df = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
        df.columns = [str(c).strip() for c in df.columns]
        log.info("read %s: %s rows x %d cols in %.2fs",
                 label, f"{len(df):,}", df.shape[1], time.perf_counter() - start)
        return df

    log.info("reading CSV: %s", label)
    df = read_csv(source, **kwargs)
    log.info("read %s: %s rows x %d cols in %.2fs",
             label, f"{len(df):,}", df.shape[1], time.perf_counter() - start)
    return df


def parse_timestamps(
    values: pd.Series,
    dayfirst: bool = False,
    fmt: Optional[str] = None,
) -> pd.Series:
    """Parse a column into tz-naive ``datetime64`` values, ``NaT`` where unparseable.

    Handles ISO/locale strings, ``YYYYMMDD`` integers, and epoch seconds through
    nanoseconds. Timezone-aware input keeps its *wall clock* time so that an IST
    session still renders at 09:15 rather than being shifted to UTC.
    """
    series = pd.Series(values)

    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = series
    elif fmt:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
    elif pd.api.types.is_numeric_dtype(series):
        parsed = _parse_numeric_timestamps(series)
    else:
        parsed = pd.to_datetime(series, dayfirst=dayfirst, errors="coerce", format="mixed")
        if parsed.isna().all():
            # "mixed" refuses some exotic layouts; retry with full inference.
            # The "could not infer format" notice is expected on this path.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(series, dayfirst=dayfirst, errors="coerce")

        # dayfirst is a *hint*, not a strict setting: pandas 2.x accepts a
        # dayfirst=True call on a US-style column, silently flipping months
        # into days. Detect the disagreement by parsing under both settings
        # and warning when they diverge on any row. We keep the user's
        # requested parse (they may know their locale better than us) but
        # surface the mismatch so the log makes the ambiguity findable.
        try:
            alt = pd.to_datetime(
                series, dayfirst=not dayfirst, errors="coerce", format="mixed",
            )
            diverge = int((parsed != alt).sum())
            if diverge:
                log.warning(
                    "dayfirst=%s: %d row(s) parse differently under the "
                    "opposite convention -- check the timestamp is not US-style "
                    "(MM-DD-YYYY)", dayfirst, diverge,
                )
        except Exception:                          # noqa: BLE001 - never block on the check
            pass

    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_localize(None)
    return parsed


def _parse_numeric_timestamps(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric.dropna()
    if finite.empty:
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    magnitude = float(finite.abs().median())

    # Excel serial dates: days since 1899-12-30. Dates 1968-2100 land in
    # ~25000-73000, comfortably inside 1e3-1e5. Values below 1e3 (before 1902)
    # and above 1e5 (past 2173) are implausible; this window is disjoint from
    # every epoch encoding (seconds > 1e8 for anything after 1973).
    if 1e3 <= magnitude < 1e5:
        return pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")

    # YYYYMMDD dates (e.g. 20230102) are 8 digits and would otherwise be read as
    # epoch seconds in 1970.
    if 1e7 <= magnitude < 1e8:
        as_dates = pd.to_datetime(
            numeric.astype("Int64").astype("string"), format="%Y%m%d", errors="coerce"
        )
        if as_dates.notna().mean() > 0.8:
            return as_dates

    unit = next(unit for bound, unit in _EPOCH_BUCKETS if magnitude < bound)
    return pd.to_datetime(numeric, unit=unit, errors="coerce")


def _timestamp_frame(
    df: pd.DataFrame,
    timestamp_col: object,
    dayfirst: bool,
    fmt: Optional[str],
) -> pd.Series:
    if timestamp_col not in df.columns:
        raise DataError(f"Timestamp column {timestamp_col!r} is not in the file.")
    parsed = parse_timestamps(df[timestamp_col], dayfirst=dayfirst, fmt=fmt)
    if parsed.isna().all():
        raise DataError(
            f"No value in {timestamp_col!r} could be parsed as a date/time. "
            "Pick a different column or set an explicit format in the sidebar."
        )
    return parsed


def _finalise(df: pd.DataFrame, sort: bool, drop_duplicates: bool) -> pd.DataFrame:
    out = df.dropna(subset=["timestamp"])
    if drop_duplicates:
        out = out.drop_duplicates(subset=["timestamp"], keep="last")
    if sort:
        out = out.sort_values("timestamp", kind="mergesort")
    return out.reset_index(drop=True)


def prepare_price(
    df: pd.DataFrame,
    mapping: schema_mod.OhlcMapping,
    dayfirst: bool = False,
    fmt: Optional[str] = None,
    drop_invalid: bool = True,
) -> pd.DataFrame:
    """Normalise ``df`` into the canonical price frame."""
    missing = mapping.missing()
    if missing:
        raise DataError("Missing required price column(s): " + ", ".join(missing))

    out = pd.DataFrame({"timestamp": _timestamp_frame(df, mapping.timestamp, dayfirst, fmt)})
    for field in PRICE_FIELDS:
        col = getattr(mapping, field)
        if col not in df.columns:
            raise DataError(f"Price column {col!r} is not in the file.")
        out[field] = pd.to_numeric(df[col], errors="coerce")
    if mapping.volume is not None and mapping.volume in df.columns:
        out["volume"] = pd.to_numeric(df[mapping.volume], errors="coerce")

    if drop_invalid:
        out = out.dropna(subset=list(PRICE_FIELDS))

    out = _finalise(out, sort=True, drop_duplicates=True)
    if out.empty:
        raise DataError("No usable price rows remain after parsing (check the column mapping).")
    return out


def prepare_labels(
    df: pd.DataFrame,
    mapping: schema_mod.LabelMapping,
    dayfirst: bool = False,
    fmt: Optional[str] = None,
) -> pd.DataFrame:
    """Normalise ``df`` into ``timestamp, target, predicted``.

    Labels are cast to trimmed strings so that ``1`` and ``"1"`` from two
    different files compare equal; blanks become ``NA``.
    """
    if not mapping.has_any_label:
        raise DataError("Select at least one of the target / predicted label columns.")

    out = pd.DataFrame({"timestamp": _timestamp_frame(df, mapping.timestamp, dayfirst, fmt)})
    for field in ("target", "predicted"):
        col = getattr(mapping, field)
        if col is None:
            out[field] = pd.Series(pd.NA, index=out.index, dtype="string")
            continue
        if col not in df.columns:
            raise DataError(f"Label column {col!r} is not in the file.")
        out[field] = _as_label(df[col])

    out = _finalise(out, sort=True, drop_duplicates=True)
    if out.empty:
        raise DataError("No usable label rows remain after parsing.")
    return out


def _as_label(values: pd.Series) -> pd.Series:
    """Cast a label column to clean strings (``1.0`` -> ``"1"``, ``""`` -> ``NA``)."""
    series = pd.Series(values)
    if pd.api.types.is_float_dtype(series):
        whole = series.dropna() % 1 == 0
        if len(whole) and bool(whole.all()):
            series = series.astype("Int64")
    text = series.astype("string").str.strip()
    return text.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NaN": pd.NA})


def prepare_features(
    df: pd.DataFrame,
    timestamp_col: object,
    feature_cols: Optional[Sequence[object]] = None,
    dayfirst: bool = False,
    fmt: Optional[str] = None,
    numeric_only: bool = False,
) -> pd.DataFrame:
    """Normalise ``df`` into ``timestamp`` plus its feature columns.

    Text columns are kept as text rather than coerced to NaN: a third of a real
    feature set is categorical (gap_state, session_phase, ...), and silently
    dropping those would hide exactly the features that explain a label.
    """
    out = pd.DataFrame({"timestamp": _timestamp_frame(df, timestamp_col, dayfirst, fmt)})

    if feature_cols is None:
        feature_cols = schema_mod.feature_columns(
            df, exclude=[timestamp_col], numeric_only=numeric_only
        )
    if not list(feature_cols):
        raise DataError("No feature columns found in this file.")

    for col in feature_cols:
        if col not in df.columns:
            raise DataError(f"Feature column {col!r} is not in the file.")
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            out[str(col)] = series
        elif numeric_only:
            out[str(col)] = pd.to_numeric(series, errors="coerce")
        else:
            out[str(col)] = series.astype("string").str.strip()

    out = _finalise(out, sort=True, drop_duplicates=True)
    if out.empty:
        raise DataError("No usable feature rows remain after parsing.")
    return out


def infer_bar_interval(timestamps: pd.Series) -> Optional[pd.Timedelta]:
    """Modal gap between consecutive bars, or ``None`` when undeterminable."""
    ts = pd.Series(timestamps).dropna().sort_values()
    if len(ts) < 3:
        return None
    deltas = ts.diff().dropna()
    positive = deltas[deltas > pd.Timedelta(0)]
    if positive.empty:
        return None
    return positive.mode().iloc[0]


def describe_frame(df: pd.DataFrame) -> dict:
    """Small summary used for the 'what did I just upload?' expanders."""
    ts = df["timestamp"] if "timestamp" in df.columns else None
    interval = infer_bar_interval(ts) if ts is not None else None
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "start": None if ts is None or ts.empty else ts.min(),
        "end": None if ts is None or ts.empty else ts.max(),
        "interval": interval,
    }
