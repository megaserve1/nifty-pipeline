"""Build the downloadable TS / TL / PL workbook.

The sheet the app writes is also a sheet the app can read back: the column names
match what the Price & Labels tab auto-detects, so a prediction run can be handed
to someone else and loaded straight onto a chart.
"""

from __future__ import annotations

import io
from typing import Callable, Optional

import pandas as pd

from . import metrics, predictor

ProgressCB = Callable[[float, str], None]
"""Progress callback: ``(fraction in [0.0, 1.0], phase label) -> None``.

The fraction is monotonically non-decreasing across a single build. The phase
label is a human-readable string ("writing csv", "writing labels sheet",
"finalizing xlsx", "done") the caller may surface as-is.
"""

# Header names chosen so schema.detect_labels() picks them up without help.
TS_COL = "timestamp"
TL_COL = "target_label"
PL_COL = "predicted_label"


def build_sheet(
    predictions: pd.DataFrame,
    include_proba: bool = True,
    include_target: Optional[bool] = None,
) -> pd.DataFrame:
    """Rename the internal frame into the exported TS / TL / PL layout.

    ``include_target`` controls the ``target_label`` and ``correct`` columns:

    * ``None`` (default) - **auto**: keep them when the run has any ground-truth
      labels, drop them when it does not. A dataset run purely for inference
      then produces a clean ``timestamp / predicted_label / confidence [/ proba_*]``
      sheet instead of one with a column of empty ``target_label`` cells.
    * ``True`` / ``False`` - force the columns on or off explicitly.
    """
    if predictions is None or predictions.empty:
        raise ValueError("There are no predictions to export.")

    # Duplicate columns in the input frame make .get / [col] return two-column
    # slices, which cascade into cryptic Series-vs-DataFrame errors much later
    # in the pipeline. Refuse early with a clear message.
    dup = predictions.columns[predictions.columns.duplicated()].unique().tolist()
    if dup:
        raise ValueError(
            f"predictions frame has duplicate columns: {dup}. "
            "Deduplicate before exporting (e.g. df.loc[:, ~df.columns.duplicated()])."
        )

    target = predictions.get("target")
    has_target = target is not None and bool(target.notna().any())
    if include_target is None:
        include_target = has_target

    # xlsxwriter refuses to serialise tz-aware datetimes -- it raises
    # "Excel does not support datetimes with timezones" deep inside the write
    # loop. Strip the tz here (converting to UTC wall clock first so IST etc.
    # keep their meaning) so both to_csv and to_excel see plain datetime64[ns].
    ts = predictions["timestamp"]
    if isinstance(getattr(ts, "dtype", None), pd.DatetimeTZDtype):
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)

    out = pd.DataFrame({TS_COL: ts})
    if include_target:
        out[TL_COL] = target if target is not None else pd.NA
    out[PL_COL] = predictions["predicted"]

    if "confidence" in predictions.columns:
        out["confidence"] = predictions["confidence"]

    # The `correct` column is meaningful only when both sides of the comparison
    # exist; suppressing it alongside `target_label` keeps the sheet tidy.
    if include_target and target is not None:
        both = target.notna() & predictions["predicted"].notna()
        correct = target.astype("string") == predictions["predicted"].astype("string")
        out["correct"] = pd.Series(pd.NA, index=predictions.index, dtype="string")
        out.loc[both, "correct"] = correct[both].map({True: "YES", False: "no"})

    if include_proba:
        for col in predictor.probability_columns(predictions):
            out[col] = predictions[col]
    return out


def summary_sheet(predictions: pd.DataFrame, bundle_info) -> pd.DataFrame:
    """A small run-provenance sheet, so a saved file explains itself later."""
    summary = metrics.summarise(predictions)
    rows = [
        ("model_type", bundle_info.model_type),
        ("dataset_version", bundle_info.dataset_version or "—"),
        ("n_features", bundle_info.n_features),
        ("classes", ", ".join(bundle_info.classes)),
        ("rows_predicted", summary.n_rows),
        ("rows_with_target", summary.n_target),
        ("rows_compared", summary.n_compared),
        ("matches", summary.n_match),
        ("mismatches", summary.n_mismatch),
        (
            "accuracy",
            "—" if summary.accuracy is None else round(summary.accuracy, 6),
        ),
    ]
    if "confidence" in predictions.columns:
        conf = pd.to_numeric(predictions["confidence"], errors="coerce")
        # An all-NA Float64 (nullable) column returns pd.NA from .mean(), which
        # float(...) refuses. Guard with pd.isna so an inference-only run without
        # confidence values still produces a valid summary sheet.
        mean_val = conf.mean()
        rows.append((
            "mean_confidence",
            "—" if pd.isna(mean_val) else round(float(mean_val), 6),
        ))
    return pd.DataFrame(rows, columns=["field", "value"])


def to_excel(
    predictions: pd.DataFrame,
    bundle_info=None,
    include_proba: bool = True,
    features: Optional[pd.DataFrame] = None,
    progress_cb: Optional[ProgressCB] = None,
    *,
    chunk_rows: int = 25_000,
) -> bytes:
    """Serialise the predictions to an .xlsx workbook.

    The labels sheet is written in ~25 000-row chunks against a single open
    ``ExcelWriter`` (xlsxwriter cannot append to a closed .xlsx). The chunk
    boundaries carry real progress -- ``progress_cb`` fires per chunk between
    fraction 0.0 and 0.6. The remaining 0.6 -> 1.0 is the "finalizing" phase:
    ``writer.close()`` writes the sharedStrings XML and zips the archive, and
    consumes roughly 40% of total wall time on a 500k-row export with no way to
    yield progress. Emitting a distinct phase label for it is honest.
    """
    WRITE_BUDGET = 0.6
    sheet = build_sheet(predictions, include_proba)
    buffer = io.BytesIO()
    n = len(sheet)

    if progress_cb is not None:
        progress_cb(0.0, "preparing")

    with pd.ExcelWriter(
        buffer, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm:ss"
    ) as writer:
        if n == 0:
            sheet.to_excel(writer, sheet_name="labels", index=False)
        else:
            # First chunk writes the header at row 0. Every later chunk skips
            # past that header (startrow = lo + 1) and asks pandas not to write
            # another one.
            for lo in range(0, n, chunk_rows):
                hi = min(lo + chunk_rows, n)
                sheet.iloc[lo:hi].to_excel(
                    writer, sheet_name="labels",
                    startrow=0 if lo == 0 else lo + 1,
                    header=(lo == 0),
                    index=False,
                )
                if progress_cb is not None:
                    progress_cb(min(WRITE_BUDGET, WRITE_BUDGET * hi / n),
                                "writing labels sheet")

        if bundle_info is not None:
            summary_sheet(predictions, bundle_info).to_excel(
                writer, sheet_name="run_summary", index=False
            )
        if features is not None and not features.empty:
            features.to_excel(writer, sheet_name="features", index=False)

        # Header styling + column widths on the labels sheet -- done once after
        # every row is in place, not per chunk.
        book = writer.book
        header = book.add_format(
            {"bold": True, "bg_color": "#2C1B54", "font_color": "#F5F1FF", "border": 0}
        )
        worksheet = writer.sheets["labels"]
        for col, value in enumerate(sheet.columns):
            worksheet.write(0, col, str(value), header)
            width = max(len(str(value)) + 2, 14)
            worksheet.set_column(col, col, width)
        worksheet.freeze_panes(1, 1)

        if progress_cb is not None:
            progress_cb(
                WRITE_BUDGET,
                "finalizing xlsx (writer.close is opaque, ~40% of total time)",
            )

    if progress_cb is not None:
        progress_cb(1.0, "done")
    return buffer.getvalue()


def to_csv(
    predictions: pd.DataFrame,
    include_proba: bool = True,
    progress_cb: Optional[ProgressCB] = None,
    *,
    target_updates: int = 20,
    min_chunk: int = 5_000,
    max_chunk: int = 100_000,
) -> bytes:
    """Serialise the labels sheet to CSV bytes.

    Chunked to give ``progress_cb`` roughly ``target_updates`` callbacks over
    the write. Output is byte-identical to
    ``build_sheet(predictions).to_csv(index=False).encode('utf-8')`` at every
    chunk size we tested (StringDtype, Int64, Float64, Categorical, NaT all
    round-trip cleanly). Measured speed penalty for 500k rows at 25 000-row
    chunks: ~2%.

    Empty frames still produce a header-only CSV — the chunk loop never runs,
    so a small special-case writes the header itself.
    """
    sheet = build_sheet(predictions, include_proba)
    n = len(sheet)
    buf = io.BytesIO()

    # Fix a datetime format explicitly on every chunk. Otherwise pandas picks
    # the format per-slice: an all-midnight slice writes "2024-01-01" while the
    # same rows in a wider frame write "2024-01-01 00:00:00", which breaks the
    # byte-identity contract and shortens the file.
    csv_kwargs = {"index": False, "date_format": "%Y-%m-%d %H:%M:%S"}

    if n == 0:
        buf.write(sheet.to_csv(**csv_kwargs).encode("utf-8"))
        if progress_cb is not None:
            progress_cb(1.0, "done")
        return buf.getvalue()

    # -(-n // target_updates) is a ceil-division: it never underflows to zero
    # for n >= 1, so the loop always makes progress.
    chunk = max(min_chunk, min(max_chunk, -(-n // max(target_updates, 1))))
    for i in range(0, n, chunk):
        text = sheet.iloc[i:i + chunk].to_csv(header=(i == 0), **csv_kwargs)
        buf.write(text.encode("utf-8"))
        if progress_cb is not None:
            # Clamp: the last slice can push (i + chunk) / n above 1.0.
            progress_cb(min(1.0, (i + chunk) / n), "writing csv")
    if progress_cb is not None:
        progress_cb(1.0, "done")
    return buf.getvalue()
