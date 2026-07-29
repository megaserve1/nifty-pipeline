"""Run a saved nifty-pipeline model bundle over feature data.

This is the app-side equivalent of ``scripts/predict.py`` in the training repo.
The preprocessing step (:func:`prepare`) is that script's own ``prepare()``,
copied verbatim -- see :mod:`src.vendor` for why nothing here is reimplemented.

What this module adds on top of the script:

* the timestamp is carried through from the feature data (``predict.py`` drops
  it, because its output is positionally aligned to its input)
* results come back as a frame instead of being written to disk
* the target label, when present, is passed through as ``target``

No training, no re-labelling, no fitting -- the bundle already carries the exact
feature list, the categorical maps and the label encoder.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import data_loader, vendor

log = logging.getLogger("visualizer.predictor")

# Defaults from nifty-pipeline/config.py (LABEL_TS_COL / LABEL_COL).
DEFAULT_TIMESTAMP_COL = "timestamp"
DEFAULT_LABEL_COL = "primary_label"

REQUIRED_BUNDLE_KEYS = ("features", "label_encoder")

# How a missing *categorical* value is encoded before it reaches the model.
#
# The trainer fills text NaN with CATEGORICAL_NA_LABEL ("MISSING") and then
# encodes it, which is why every cat_map in the v4.1 bundle carries a MISSING
# code. predict.py skips that fill, so a missing category encodes to -1 instead
# -- a value the model never saw in training. Both behaviours are offered because
# they answer different questions:
#
#   "predict.py"  reproduce the pipeline's output exactly, bug included
#   "training"    encode the way the model was actually trained
NA_MODE_PREDICT_PY = "predict.py"
NA_MODE_TRAINING = "training"
NA_MODES = (NA_MODE_PREDICT_PY, NA_MODE_TRAINING)


class PredictionError(ValueError):
    """Raised when the bundle and the data cannot be reconciled."""


@dataclass
class BundleInfo:
    """The human-readable summary shown after a model is loaded."""

    model_type: str = "unknown"
    dataset_version: Optional[str] = None
    n_features: int = 0
    classes: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    has_proba: bool = False


def _not_a_bundle(obj) -> str:
    return (
        "That file is not a nifty-pipeline model bundle "
        f"(expected a dict, got {type(obj).__name__}). Upload the "
        "model_*.joblib the trainer produced."
    )


def _load_failure_message(path: str, exc: Exception) -> str:
    """Turn a loader crash into something actionable.

    The vendored loader reaches for ``bundle.get(...)`` immediately, so a file
    that unpickles to the wrong type dies with an opaque AttributeError. Only on
    the failure path do we re-read it to say what it actually is.
    """
    try:
        import joblib

        raw = joblib.load(path)
    except Exception:  # noqa: BLE001 - the original error is the useful one
        return f"Could not load the model bundle: {exc}"
    if not isinstance(raw, dict):
        return _not_a_bundle(raw)
    return f"Could not load the model bundle: {exc}"


def load_bundle(path: str) -> dict:
    """Load a ``model_*.joblib`` bundle.

    Delegates to the training repo's own loader, which rebuilds XGBoost and
    CatBoost models from the portable native blobs the trainer stores instead of
    pickling the estimator. A plain ``joblib.load`` returns ``model: None`` for
    those bundles.
    """
    start = time.perf_counter()
    log.info("loading model bundle: %s", path)
    try:
        bundle = vendor.load_model_bundle(path)
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim in the UI
        log.exception("model bundle failed to load")
        raise PredictionError(_load_failure_message(path, exc)) from exc

    if not isinstance(bundle, dict):
        raise PredictionError(_not_a_bundle(bundle))
    missing = [key for key in REQUIRED_BUNDLE_KEYS if key not in bundle]
    if missing:
        raise PredictionError(
            "The bundle is missing " + ", ".join(missing) +
            ". Expected the joblib produced by the trainer."
        )
    if bundle.get("model") is None:
        raise PredictionError(
            "The bundle carries no usable model (neither a pickled estimator nor "
            "a native xgb_model_ubj / cb_model_cbm blob)."
        )
    classes = getattr(bundle.get("label_encoder"), "classes_", None)
    if classes is None or len(classes) == 0:
        # An encoder with zero classes crashes only at inverse_transform time
        # with a cryptic "unseen labels" error deep inside predict_frame.
        # Fail loudly at load so the user knows to re-export the bundle.
        raise PredictionError(
            "The bundle's label_encoder carries no classes -- the model has "
            "nothing to decode predictions into. Re-export the bundle from "
            "the trainer."
        )
    log.info(
        "bundle loaded in %.2fs: %s v%s | %d features | %d classes",
        time.perf_counter() - start,
        bundle.get("model_type"), bundle.get("dataset_version"),
        len(bundle.get("features") or []),
        len(classes),
    )
    return bundle


def describe(bundle: dict) -> BundleInfo:
    """Summarise a loaded bundle for display."""
    encoder = bundle.get("label_encoder")
    classes = [str(c) for c in getattr(encoder, "classes_", [])]
    features = [str(c) for c in bundle.get("features", [])]
    return BundleInfo(
        model_type=str(bundle.get("model_type", "unknown")),
        dataset_version=bundle.get("dataset_version"),
        n_features=len(features),
        classes=classes,
        features=features,
        categorical=[str(c) for c in (bundle.get("categorical") or [])],
        has_proba=hasattr(bundle.get("model"), "predict_proba"),
    )


def missing_features(df: pd.DataFrame, bundle: dict) -> list[str]:
    """Feature columns the model needs that the uploaded data does not have."""
    return [c for c in bundle.get("features", []) if c not in df.columns]


def common_prefix(names: Sequence[str]) -> str:
    """Longest ``__``-delimited prefix shared by every name, e.g. ``"bucket_raw_v4__"``.

    Delimited rather than character-wise on purpose: a character-wise prefix would
    happily cut a name mid-word and produce matches that only look right.
    """
    names = [str(n) for n in names]
    if not names:
        return ""
    parts = names[0].split("__")
    for cut in range(len(parts) - 1, 0, -1):
        candidate = "__".join(parts[:cut]) + "__"
        if all(n.startswith(candidate) for n in names):
            return candidate
    return ""


@dataclass
class RenamePlan:
    """How to reconcile an export's column names with the model's."""

    mapping: dict[str, str] = field(default_factory=dict)   # incoming -> model
    prefix: str = ""
    still_missing: list[str] = field(default_factory=list)
    ambiguous: list[str] = field(default_factory=list)
    already_matched: int = 0

    @property
    def n_renamed(self) -> int:
        return len(self.mapping)

    @property
    def resolves_everything(self) -> bool:
        return bool(self.mapping) and not self.still_missing and not self.ambiguous

    @property
    def helps(self) -> bool:
        return bool(self.mapping)


def suggest_renames(df: pd.DataFrame, bundle: dict) -> RenamePlan:
    """Work out which incoming columns are the model's features under another name.

    Only the model's own shared prefix is considered, and a candidate is used only
    when exactly one incoming column claims exactly one model feature. Anything
    contested is reported instead of guessed -- a wrong rename would feed the model
    the wrong series and still produce confident-looking labels.
    """
    wanted = [str(c) for c in bundle.get("features", [])]
    have = {str(c) for c in df.columns}
    plan = RenamePlan(already_matched=sum(1 for f in wanted if f in have))

    missing = [f for f in wanted if f not in have]
    if not missing:
        return plan

    plan.prefix = common_prefix(wanted)
    if not plan.prefix:
        plan.still_missing = missing
        return plan

    # Stripping one shared prefix from distinct features always yields distinct
    # names, so two features can never contest the same column. What *can* go
    # wrong is a duplicated column name in the export -- renaming that would
    # silently hand the model whichever copy pandas resolves last.
    occurrences = Counter(str(c) for c in df.columns)

    for feature in missing:
        short = feature[len(plan.prefix):]
        if short not in have:
            continue
        if occurrences[short] > 1:
            plan.ambiguous.append(short)
            continue
        plan.mapping[short] = feature

    claimed = set(plan.mapping.values())
    contested = set(plan.ambiguous)
    plan.still_missing = [
        f for f in missing
        if f not in claimed and f[len(plan.prefix):] not in contested
    ]
    return plan


def apply_renames(df: pd.DataFrame, plan: Optional[RenamePlan]) -> pd.DataFrame:
    """Return ``df`` with the plan's columns renamed to the model's names.

    ``rename(copy=False)`` returns a *view* that shares data with the input --
    a full-copy rename runs about a second on the real 500k x 320 feature file
    and would fire on every rerun through :func:`count_missing_categoricals`.
    Safe here because callers only read from the returned frame.
    """
    if plan is None or not plan.mapping:
        return df
    return df.rename(columns=plan.mapping, copy=False)


def count_missing_categoricals(df: pd.DataFrame, bundle: dict) -> dict[str, int]:
    """Per-column count of missing values in the model's categorical features.

    Tells the user whether the ``na_mode`` choice actually affects their file:
    with nothing missing, the two modes produce identical input.

    Object-dtype NaN detection is genuinely slow in pandas (per-cell Python
    loop), so callers on rerun-heavy paths should cache the result instead of
    re-running it every time the user moves a widget. This function does the
    computation; caching is the caller's job.
    """
    wanted = [c for c in (bundle.get("categorical") or []) if c in df.columns]
    counts: dict[str, int] = {}
    for col in wanted:
        n = int(df[col].isna().sum())
        if n:
            counts[col] = n
    return counts


def _fill_categorical_na(X: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Fill text NaN with CATEGORICAL_NA_LABEL, exactly as na_policy.apply_policy does."""
    out = X.copy()
    for c in columns:
        if c in out.columns and out[c].isna().any():
            out[c] = out[c].astype("object").where(
                out[c].notna(), vendor.CATEGORICAL_NA_LABEL
            )
    return out


def prepare(
    df: pd.DataFrame,
    bundle: dict,
    na_mode: str = NA_MODE_PREDICT_PY,
) -> pd.DataFrame:
    """apply the SAME preprocessing the trainer did, using the maps saved in the bundle.

    Copied from nifty-pipeline/scripts/predict.py, with two deliberate changes:
    :class:`PredictionError` instead of ``SystemExit`` so the app can show the
    message rather than killing the server process, and ``na_mode``.

    ``na_mode`` decides how a *missing categorical* is encoded:

    ``"predict.py"`` (default)
        Reproduce upstream exactly. A missing category becomes -1, because
        ``encode_categoricals`` does ``.astype(str)`` first, turning NaN into the
        string "nan", which is not a key in the saved map. The catboost branch's
        ``.astype(str).fillna(...)`` is a no-op for the same reason.
    ``"training"``
        Fill text NaN with CATEGORICAL_NA_LABEL before encoding, which is what
        the trainer did -- that is why every cat_map carries a MISSING code. The
        model has never seen -1 for these columns, so this is the encoding it was
        actually fitted on.
    """
    if na_mode not in NA_MODES:
        raise PredictionError(f"Unknown na_mode {na_mode!r}; expected one of {NA_MODES}.")

    feats = bundle["features"]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise PredictionError(
            f"the incoming data is missing {len(missing)} feature column(s) the model needs, "
            f"e.g. {missing[:6]}.\n  the model can only predict on the columns it was trained on.")
    X = df[feats].copy()                       # exact columns, exact order (extra columns ignored)

    if na_mode == NA_MODE_TRAINING:
        X = _fill_categorical_na(X, bundle.get("categorical", []) or [])

    mtype = bundle.get("model_type")
    maps = bundle.get("cat_maps", {}) or {}
    if mtype == "catboost":
        # catboost reads text natively -- just make the text columns str and fill missing.
        for c in bundle.get("categorical", []):
            if c in X.columns:
                X[c] = X[c].astype(str).fillna(vendor.CATEGORICAL_NA_LABEL)
    else:
        # xgboost / random forest: encode text with the SAVED map (unseen category -> -1),
        # then fill the numeric sentinels the trainer recorded.
        #
        # Guard against a fabricated encoding: encode_categoricals builds a
        # brand-new map from the data whenever a text column has no saved map,
        # which silently produces a fresh, train/serve-skewed encoding. Any
        # column the bundle marks as categorical MUST have a map -- refuse to
        # invent one from live data.
        saved_map = {k: v for k, v in maps.items() if k != "_sentinels"}
        cat_cols = list(bundle.get("categorical", []) or [])
        text_cat_cols = [c for c in cat_cols if c in X.columns
                         and str(X[c].dtype) in vendor_CAT_DTYPES]
        without_map = [c for c in text_cat_cols if c not in saved_map]
        if without_map:
            raise PredictionError(
                "The bundle marks " + ", ".join(without_map[:5])
                + (f" and {len(without_map) - 5} more" if len(without_map) > 5 else "")
                + " as categorical but has no saved cat_map for them. "
                "Encoding from live data would silently fabricate a mapping the "
                "model was never trained on -- refusing to predict."
            )
        X, _ = vendor.encode_categoricals(X, mapping=saved_map)
        for c, sv in (maps.get("_sentinels") or {}).items():
            if c in X.columns:
                X[c] = X[c].fillna(sv)
    return X


# The dtypes na_policy.encode_categoricals treats as text. Kept as a module-level
# tuple so the guard above and any future callers agree with the vendored code.
vendor_CAT_DTYPES = ("object", "str", "string", "category", "bool")


def _clean_labels(values: pd.Series) -> pd.Series:
    """Trim label text. Upstream notes that the labels carry trailing spaces, so
    without this ``NO_TRADE`` and ``NO_TRADE `` become two classes on the chart."""
    text = pd.Series(values).astype(str).str.strip()
    return text.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})


def predict_frame(
    df: pd.DataFrame,
    bundle: dict,
    timestamp_col: str = DEFAULT_TIMESTAMP_COL,
    label_col: Optional[str] = None,
    include_proba: bool = True,
    dayfirst: bool = False,
    na_mode: str = NA_MODE_PREDICT_PY,
    renames: Optional[RenamePlan] = None,
) -> pd.DataFrame:
    """Predict over ``df`` and return ``timestamp, target, predicted, confidence[, proba_*]``.

    ``label_col`` is optional: prediction produces the label, so a target is only
    needed if you also want to score. Rows keep the order of ``df``.

    The timestamp is parsed to ``datetime64`` rather than passed through as-is:
    a CSV round-trip leaves it a string, and a string timestamp silently fails to
    merge onto the price bars, so the labels would just never appear on the chart.
    """
    if df is None or df.empty:
        raise PredictionError("The feature data is empty.")
    if timestamp_col not in df.columns:
        raise PredictionError(
            f"Timestamp column {timestamp_col!r} is not in the feature file. "
            "The chart needs a timestamp for every predicted row."
        )

    model = bundle["model"]
    encoder = bundle["label_encoder"]

    # Renaming happens on a copy of the *feature* view only; the timestamp and
    # label columns are read from the original frame below.
    X = prepare(apply_renames(df, renames), bundle, na_mode=na_mode)

    log.info(
        "predicting %s rows x %d features (na_mode=%s%s)",
        f"{len(X):,}", X.shape[1], na_mode,
        f", {len(renames.mapping)} column(s) renamed" if renames and renames.mapping else "",
    )
    t = time.perf_counter()
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        pred_idx = proba.argmax(axis=1)
        confidence = proba.max(axis=1)
    else:
        proba = None
        pred_idx = np.asarray(model.predict(X))
        confidence = np.full(len(X), np.nan)

    predicted = encoder.inverse_transform(pred_idx)
    log.info(
        "predicted %s rows in %.2fs (%s rows/s)",
        f"{len(X):,}", time.perf_counter() - t,
        f"{len(X) / max(time.perf_counter() - t, 1e-6):,.0f}",
    )

    stamps = data_loader.parse_timestamps(df[timestamp_col], dayfirst=dayfirst)
    if stamps.isna().all():
        raise PredictionError(
            f"No value in {timestamp_col!r} could be parsed as a date/time. "
            "Pick a different column, or set an explicit format in the sidebar."
        )
    # Partial NaT is also a hard error: silently keeping a NaT row would drop
    # it from every downstream merge (chart, exporter) without warning, so a
    # partial bad file would ship a partial silent-loss prediction.
    bad = stamps.isna()
    if bad.any():
        bad_positions = list(np.where(bad.to_numpy())[0][:5])
        raise PredictionError(
            f"{int(bad.sum())} of {len(stamps)} row(s) in {timestamp_col!r} "
            f"could not be parsed as a date/time (e.g. rows {bad_positions}). "
            "Fix the file, choose a different column, or set an explicit "
            "format in the sidebar."
        )

    out = pd.DataFrame(
        {
            "timestamp": stamps.to_numpy(),
            "target": pd.Series(pd.NA, index=range(len(df)), dtype="string"),
            "predicted": _clean_labels(pd.Series(predicted)).astype("string"),
            "confidence": confidence,
        }
    )
    if label_col and label_col in df.columns:
        out["target"] = _clean_labels(df[label_col]).astype("string").to_numpy()

    if include_proba and proba is not None:
        for i, cls in enumerate(getattr(encoder, "classes_", [])):
            out[f"proba_{cls}"] = proba[:, i]
    return out


def probability_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if str(c).startswith("proba_")]


def candidate_label_columns(df: pd.DataFrame, bundle: dict) -> list[str]:
    """Columns that could hold the ground-truth label.

    Anything the model consumes as a feature is excluded -- a feature can never
    also be the target, and offering one would silently leak.
    """
    feats = {str(c) for c in bundle.get("features", [])}
    return [
        str(c) for c in df.columns
        if str(c) not in feats and not pd.api.types.is_float_dtype(df[c])
    ]
