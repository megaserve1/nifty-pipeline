"""Verbatim copies of the three nifty-pipeline functions the predictor depends on.

Source repo: C:/Users/Admin/nifty-pipeline

    load_model_bundle    <- trainer/train.py   (line ~93)
    encode_categoricals  <- na_policy.py       (line ~166)
    CATEGORICAL_NA_LABEL <- config.py          (line ~334)

These are COPIED, NOT REWRITTEN, and must stay that way. The whole point of using
the training repo's own code is that preprocessing at prediction time is
bit-identical to preprocessing at training time; a "tidied up" reimplementation
is exactly how you get predictions that look plausible and are wrong.

``tests/test_vendor_sync.py`` re-reads the originals and fails if they drift, so
an upstream edit surfaces as a red test instead of a silent behaviour change.
Re-sync by copying the function bodies across again -- do not hand-merge.
"""

from __future__ import annotations

import pandas as pd

# --- from nifty-pipeline/config.py -----------------------------------------
# a missing CATEGORY keeps its own identity rather than being blended into a real one
CATEGORICAL_NA_LABEL = "MISSING"


# --- from nifty-pipeline/trainer/train.py ----------------------------------
def load_model_bundle(path):
    """joblib.load a model bundle, REBUILDING the xgboost model from its portable UBJ.

    WHY THIS EXISTS. XGBoost pickles its booster as a VERSION-SPECIFIC binary buffer. a bundle
    pickled under one xgboost dies with "XGBoostError: input stream corrupted" the moment another
    xgboost tries to unpickle it -- and clearml-agents build a FRESH venv per task, so the SHAP or
    champion task easily resolves a different xgboost than the trainer did. pinning is fragile
    against that. so the trainer no longer pickles the xgboost object at all: it stores the model's
    NATIVE UBJ (cross-version-portable) under 'xgb_model_ubj' and leaves 'model' None. here we load
    the (now version-safe) bundle and rebuild the classifier from that UBJ. non-xgboost bundles are
    untouched. older bundles (no 'xgb_model_ubj') load exactly as before.
    """
    import joblib, tempfile, pathlib
    b = joblib.load(path)
    if b.get("model") is None and b.get("xgb_model_ubj") is not None:
        import xgboost as xgb
        tf = tempfile.NamedTemporaryFile(suffix=".ubj", delete=False); tf.write(b["xgb_model_ubj"]); tf.close()
        clf = xgb.XGBClassifier()
        clf.load_model(tf.name)                 # restores num_class/objective -> predict_proba works
        pathlib.Path(tf.name).unlink()
        b["model"] = clf
    elif b.get("model") is None and b.get("cb_model_cbm") is not None:
        from catboost import CatBoostClassifier
        tf = tempfile.NamedTemporaryFile(suffix=".cbm", delete=False); tf.write(b["cb_model_cbm"]); tf.close()
        clf = CatBoostClassifier()
        clf.load_model(tf.name, format="cbm")   # restores classes + cat features -> ShapValues works
        pathlib.Path(tf.name).unlink()
        b["model"] = clf
    return b


# --- from nifty-pipeline/na_policy.py --------------------------------------
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
