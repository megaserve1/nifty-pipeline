"""scripts/split_leak_experiment.py -- does a random split leak? measure it, don't argue it.

this settles ONE question with a number instead of a debate:

    if the label is really just a tag on the CURRENT state that the features already explain,
    then a random split and a time split should score the SAME. if the random split scores
    BETTER, that gap is the future leaking through the label -- skill the model looks like it
    has but does not, and that dies the moment it trades live.

it trains the SAME xgboost model THREE ways. only the split changes. everything else -- the
features, the weights, the hyperparameters, the seed, the test size, the preprocessing -- is
identical, so any difference in the score is caused by the split and nothing else.

    arm A  TIME + embargo   the honest baseline. train on the past, test on the future, with a
                            25-session gap so no test row shares a rolling window with training.
                            this is exactly what trainer/objective.three_way_split does.
    arm B  RANDOM by ROW    the naive random split. any minute can land in train or test.
    arm C  RANDOM by BUNDLE your proposal. group the 15 one-minute rows of each 15-min candle
                            and send the WHOLE bundle to one side. no minute is split across
                            train and test. this removes within-bundle leakage -- the thing you
                            were right about -- and tests whether that is enough.

read the result like this:
    A ~= B ~= C   -> the label was a function of the current state. random split is safe.
                     you were right; use it.
    B, C  <<  A   -> (trading_cost is a COST, lower=better; so "better" = lower cost / higher
                     macro_f1). the random arms only look better because the test answer's
                     ingredients sat in training. bundling (C) did NOT remove it. the gap is
                     the leak, in the model's own numbers.

NOTE ON SPEED. the three arms share one capped tree count (--cap_trees, default 1200) so each
arm finishes in a few minutes. the ABSOLUTE score is not the point -- the GAP BETWEEN ARMS is,
and the gap is not created by using a smaller model as long as all three use the SAME one. pass
--cap_trees 0 to use the real configs/hyperparams.yaml value (much slower, three full trainings).

run (you run it, not me):
    final_pipeline/final_venv/bin/python scripts/split_leak_experiment.py

    # faster smoke pass:
    final_pipeline/final_venv/bin/python scripts/split_leak_experiment.py --cap_trees 400
"""
import sys, argparse, pathlib
_here = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_here.parent.parent))     # repo root: config, trainer, na_policy
sys.path.insert(0, str(_here.parent))            # scripts/
import json
import numpy as np
import pandas as pd

import config as C
from trainer.objective import three_way_split, trading_cost
from trainer.train import build_model, report_metrics, full_proba
from trainer import hyperparams
from na_policy import encode_categoricals


# ------------------------------------------------------------------ the three splits
def split_time(ts):
    """arm A -- the honest one. TRAIN | embargo | VAL | embargo | TEST, by time."""
    tr, va, te, info = three_way_split(ts, C.VAL_FRACTION, C.TEST_FRACTION, C.EMBARGO_SESSIONS)
    return (np.flatnonzero(tr.to_numpy()),
            np.flatnonzero(va.to_numpy()),
            np.flatnonzero(te.to_numpy()), info)


def split_random_rows(n, val_frac, test_frac, seed):
    """arm B -- shuffle every ROW, then cut. any minute can go anywhere."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    n_te = int(round(n * test_frac))
    n_va = int(round(n * val_frac))
    te = order[:n_te]
    va = order[n_te:n_te + n_va]
    tr = order[n_te + n_va:]
    return np.sort(tr), np.sort(va), np.sort(te)


def split_random_bundles(bundle_id, val_frac, test_frac, seed):
    """arm C -- YOUR proposal. shuffle whole 15-min BUNDLES, then cut. the 15 minutes of one
    candle always travel together, so no minute is ever split across train and test."""
    rng = np.random.RandomState(seed)
    bundles = pd.unique(bundle_id)
    order = rng.permutation(len(bundles))
    bundles = bundles[order]
    n_te = int(round(len(bundles) * test_frac))
    n_va = int(round(len(bundles) * val_frac))
    te_b = set(bundles[:n_te])
    va_b = set(bundles[n_te:n_te + n_va])
    is_te = bundle_id.isin(te_b).to_numpy()
    is_va = bundle_id.isin(va_b).to_numpy()
    tr = np.flatnonzero(~is_te & ~is_va)
    va = np.flatnonzero(is_va)
    te = np.flatnonzero(is_te)
    return tr, va, te


# ------------------------------------------------------------------ features, the xgboost way
def prepare_xgb(df, feat_cols, tr_idx):
    """encode the text columns to numbers, LEARNING the map from the training rows only (a fact
    about the test period must never shape how training data is prepared -- that is itself a
    leak). xgboost keeps NaN: it learns a branch for 'missing' on its own, so no sentinel."""
    X = df[feat_cols].copy()
    _, cat_maps = encode_categoricals(X.iloc[tr_idx])          # learn from train only
    X, _ = encode_categoricals(X, mapping=cat_maps)            # apply to all
    return X


def run_arm(name, df, feat_cols, y, w, classes, params, tr_idx, va_idx, te_idx):
    """train one xgboost on this arm's split, score it on this arm's test rows. returns the row
    for the comparison table."""
    n = len(classes)
    print(f"\n{'='*78}\n{name}\n{'='*78}")
    print(f"  train {len(tr_idx):>9,}   val {len(va_idx):>9,}   test {len(te_idx):>9,}")

    X = prepare_xgb(df, feat_cols, tr_idx)
    Xtr, Xva, Xte = X.iloc[tr_idx], X.iloc[va_idx], X.iloc[te_idx]
    ytr, yva, yte = y[tr_idx], y[va_idx], y[te_idx]
    wtr = w[tr_idx]

    model = build_model("xgboost", n, params, cat_idx=None, has_val=len(va_idx) > 0)
    model.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], verbose=False)

    proba = full_proba(model, model.predict_proba(Xte), n)
    pred = proba.argmax(axis=1)
    m = report_metrics(yte, pred, proba, classes, logger=None, split="test")
    cost = trading_cost(yte, pred, classes)
    del X, Xtr, Xva, Xte, model                          # free memory before the next arm
    return {"arm": name, "n_test": len(te_idx),
            "macro_f1": m["macro_f1"], "mean_pr_auc": m["mean_pr_auc"],
            "trading_cost": cost, "never_predicted": m["never_predicted"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/v4.1/dataset_v4.1.parquet")
    ap.add_argument("--manifest", default="datasets/v4.1/manifest.json",
                    help="to read the exact feature columns; falls back to the '__' heuristic")
    ap.add_argument("--cap_trees", type=int, default=1200,
                    help="same tree ceiling for ALL three arms (fair + fast). 0 = use the real "
                         "configs/hyperparams.yaml value (slow: three full trainings)")
    ap.add_argument("--bundle_freq", default="15min", help="candle size for arm C bundles")
    ap.add_argument("--seed", type=int, default=None, help="default: the yaml seed")
    ap.add_argument("--arms", default="A,B,C", help="which arms to run, comma-separated")
    a = ap.parse_args()

    # ---- data + the exact feature list the real trainer would use ----
    df = pd.read_parquet(a.data)
    print(f"loaded {a.data}: {len(df):,} rows x {df.shape[1]} cols")
    man = json.loads(pathlib.Path(a.manifest).read_text()) if pathlib.Path(a.manifest).exists() else None
    feat_cols = man["feature_columns"] if man else [c for c in df.columns if "__" in c]
    print(f"features: {len(feat_cols)}   (source: {'manifest' if man else 'heuristic'})")

    # ---- labels, weights, classes -- exactly as trainer/train.py does ----
    from sklearn.preprocessing import LabelEncoder
    y_raw = df[C.LABEL_COL].astype(str).str.strip()          # labels carry trailing spaces
    le = LabelEncoder().fit(y_raw)
    classes = list(le.classes_)
    y = le.transform(y_raw)
    if getattr(C, "CLASS_WEIGHTS", None):                    # per-class weight, mapped BY NAME
        w = y_raw.map(C.CLASS_WEIGHTS).astype(float).to_numpy()
    else:
        w = (df[C.WEIGHT_COL].fillna(0.0)).to_numpy()
    print(f"classes: {classes}")

    # ---- the ONE model, capped so the three arms are quick and comparable ----
    params = hyperparams.merge("xgboost", {})
    if a.cap_trees and int(params.get("n_estimators", 0)) > a.cap_trees:
        print(f"capping n_estimators {params['n_estimators']} -> {a.cap_trees} for all arms")
        params["n_estimators"] = a.cap_trees
    if a.seed is not None:
        params["seed"] = a.seed
    seed = int(params.get("seed", 42))

    ts = pd.to_datetime(df[C.LABEL_TS_COL])

    # ---- sanity check YOUR premise: do the 15 minutes of a candle share one label? ----
    bundle_id = ts.dt.floor(a.bundle_freq)
    grp = pd.DataFrame({"b": bundle_id.values, "y": y_raw.values})
    per = grp.groupby("b")["y"].nunique()
    print(f"\nbundle check ({a.bundle_freq}): {len(per):,} bundles, "
          f"{(per == 1).mean()*100:.1f}% have ALL rows on ONE label "
          f"(your premise: the 15 minutes live together on the same label)")

    # ---- build each arm's split (cheap) ----
    want = [s.strip().upper() for s in a.arms.split(",")]
    splits = {}
    if "A" in want:
        trA, vaA, teA, info = split_time(ts)
        print(f"\narm A time split: train <= {info['train_end']}   test >= {info['test_start']}   "
              f"embargo {C.EMBARGO_SESSIONS} sessions, {info['n_embargoed']:,} rows thrown away")
        splits["A  TIME + embargo   (honest baseline)"] = (trA, vaA, teA)
    if "B" in want:
        splits["B  RANDOM by ROW    (naive random)"] = split_random_rows(
            len(df), C.VAL_FRACTION, C.TEST_FRACTION, seed)
    if "C" in want:
        splits["C  RANDOM by BUNDLE (your proposal)"] = split_random_bundles(
            bundle_id, C.VAL_FRACTION, C.TEST_FRACTION, seed)

    # ---- run them ----
    rows = []
    for name, (tr, va, te) in splits.items():
        rows.append(run_arm(name, df, feat_cols, y, w, classes, params, tr, va, te))

    # ---- the verdict, in one table ----
    print(f"\n\n{'#'*78}\n#  RESULT -- same model, same everything, only the split differs\n{'#'*78}")
    tab = pd.DataFrame(rows)[["arm", "n_test", "macro_f1", "mean_pr_auc", "trading_cost"]]
    print(tab.to_string(index=False,
          formatters={"macro_f1": "{:.4f}".format, "mean_pr_auc": "{:.4f}".format,
                      "trading_cost": "{:.2f}".format, "n_test": "{:,}".format}))
    print("\n(trading_cost is a COST -> LOWER is better.  macro_f1 / mean_pr_auc -> HIGHER is better.)")

    base = next((r for r in rows if r["arm"].startswith("A")), None)
    if base:
        print("\ngap vs the honest time split (arm A):")
        for r in rows:
            if r["arm"].startswith("A"):
                continue
            df1 = r["macro_f1"] - base["macro_f1"]
            dpr = r["mean_pr_auc"] - base["mean_pr_auc"]
            dco = r["trading_cost"] - base["trading_cost"]
            flag = "  <-- looks BETTER than reality = LEAK" if (df1 > 0.02 or dco < -1) else \
                   "  <-- matches the honest number = no leak"
            print(f"  {r['arm']:<38} macro_f1 {df1:+.4f}   pr_auc {dpr:+.4f}   "
                  f"trading_cost {dco:+.2f}{flag}")
        print("\nif B and C look better than A, that difference is the future leaking through the")
        print("label -- and note whether C (your bundle split) closed the gap or not.")


if __name__ == "__main__":
    main()
