"""scripts/predict.py -- run a saved model on NEW data. no re-training, no re-labeling.

the model bundle (model_xgboost.joblib) already carries everything needed to preprocess incoming
data EXACTLY as training did:
    features    the exact columns, in the exact order
    cat_maps    the text->number encoding ('GAP_UP'->1), so live data encodes identically
    label_encoder  to turn the model's numeric output back into ENTRY_SUPER / NO_TRADE / ...

so you do NOT re-run the pipeline and you do NOT need labels. you only need the incoming data to
have the same FEATURE columns. this script applies the three transforms and predicts.

LABELS ARE NOT REQUIRED. prediction PRODUCES the label. you only need a label column if you also
want to SCORE the predictions (accuracy) -- pass --label to do that.

run:
    final_venv/bin/python scripts/predict.py --bundle ~/Downloads/model_xgboost.joblib \
        --data datasets/v4/dataset_v4.parquet --out preds.csv
"""
import sys, argparse, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
import config as C
from trainer.train import load_model_bundle


def prepare(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """apply the SAME preprocessing the trainer did, using the maps saved in the bundle."""
    feats = bundle["features"]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise SystemExit(
            f"the incoming data is missing {len(missing)} feature column(s) the model needs, "
            f"e.g. {missing[:6]}.\n  the model can only predict on the columns it was trained on.")
    X = df[feats].copy()                       # exact columns, exact order (extra columns ignored)

    mtype = bundle.get("model_type")
    maps = bundle.get("cat_maps", {}) or {}
    if mtype == "catboost":
        # catboost reads text natively -- just make the text columns str and fill missing.
        for c in bundle.get("categorical", []):
            if c in X.columns:
                X[c] = X[c].astype(str).fillna(C.CATEGORICAL_NA_LABEL)
    else:
        # xgboost / random forest: encode text with the SAVED map (unseen category -> -1),
        # then fill the numeric sentinels the trainer recorded.
        from na_policy import encode_categoricals
        X, _ = encode_categoricals(X, mapping={k: v for k, v in maps.items() if k != "_sentinels"})
        for c, sv in (maps.get("_sentinels") or {}).items():
            if c in X.columns:
                X[c] = X[c].fillna(sv)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="the model_xgboost.joblib you downloaded")
    ap.add_argument("--data", required=True, help="parquet or csv of NEW rows to predict")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--label", default="", help="optional: a true-label column, to also SCORE")
    a = ap.parse_args()

    bundle = load_model_bundle(a.bundle)
    model, le = bundle["model"], bundle["label_encoder"]
    print(f"model: {bundle.get('model_type')}   trained on v{bundle.get('dataset_version')}   "
          f"{len(bundle['features'])} features   classes: {list(le.classes_)}")

    p = pathlib.Path(a.data)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    print(f"incoming: {len(df):,} rows")

    X = prepare(df, bundle)
    proba = np.asarray(model.predict_proba(X))
    pred_idx = proba.argmax(axis=1)
    pred_name = le.inverse_transform(pred_idx)

    out = pd.DataFrame({"predicted": pred_name, "confidence": proba.max(axis=1)})

    # IF a true-label column was given, put ACTUAL next to PREDICTED and mark the match -- this
    # makes the output file itself the "actual vs predicted, compare" table.
    if a.label and a.label in df.columns:
        actual = df[a.label].astype(str).str.strip().values     # labels carry trailing spaces
        out.insert(0, "actual", actual)
        out.insert(2, "correct", np.where(out["predicted"].values == actual, "YES", "no"))

    for i, cls in enumerate(le.classes_):          # one probability column per class
        out[f"proba_{cls}"] = proba[:, i]
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out}  ({len(out):,} rows)")

    if "correct" in out.columns:
        acc = float((out["correct"] == "YES").mean())
        print(f"actual vs predicted: {int((out['correct']=='YES').sum()):,}/{len(out):,} "
              f"correct  (accuracy {acc:.4f})")
        print(out[["actual", "predicted", "correct"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
