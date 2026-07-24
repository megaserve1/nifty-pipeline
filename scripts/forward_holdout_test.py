"""scripts/forward_holdout_test.py -- the COMPLETE, fair test of bundle+random.

the quick experiment (split_leak_experiment.py) showed random-by-bundle inflates the ranking
metrics ~2x over a time split. this script settles the follow-up honestly: DOES a model built the
random way actually hold up forward, and does it beat the time way where it counts?

it does that by judging both strategies on the ONE thing that mimics live -- a forward slice
neither of them ever touched.

    1. reserve the LAST `--holdout_frac` of time as H, the forward referee. an embargo sits
       before it so nothing in development leaks into it.
    2. on everything before H, build the model TWO ways:
         R (bundle + RANDOM):  develop on randomly-assigned whole bundles   <- your strategy
         T (bundle + TIME):    develop by time, with the purge/embargo       <- the fallback
    3. score BOTH on the same H.

what the numbers mean:
    * R's own-dev score vs R-on-H:  if dev is much higher than H, the random dev number was
      inflated -- the leak, proven at full scale, on a forward set.
    * R-on-H vs T-on-H:  the ONLY comparison that decides a pivot. same forward rows, both models.
      R wins forward -> the random strategy genuinely generalises better -> pivot, with proof.
      T wins forward -> revert to bundle + time + purge.

run (you run it):
    final_venv/bin/python scripts/forward_holdout_test.py
    final_venv/bin/python scripts/forward_holdout_test.py --cap_trees 400     # faster first look
"""
import sys, argparse, pathlib, json
_here = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_here.parent.parent))     # repo root
sys.path.insert(0, str(_here.parent))            # scripts/ (reuse prepare_xgb)
import numpy as np
import pandas as pd

import config as C
from split_leak_experiment import prepare_xgb            # the EXACT training preprocessing
from trainer.train import build_model, full_proba        # the EXACT model builder
from trainer.objective import three_way_split, trading_cost
from trainer import hyperparams


def quiet_metrics(y, pred, proba, classes):
    """macro_f1, mean PR-AUC, trading_cost -- computed without printing a per-class wall."""
    from sklearn.metrics import f1_score, average_precision_score
    macro_f1 = float(f1_score(y, pred, labels=range(len(classes)), average="macro", zero_division=0))
    onehot = np.eye(len(classes))[y]
    prs = [average_precision_score(onehot[:, i], proba[:, i])
           for i in range(len(classes)) if onehot[:, i].sum() > 0]
    mean_pr = float(np.mean(prs)) if prs else 0.0
    cost = trading_cost(y, pred, classes)
    return {"macro_f1": macro_f1, "mean_pr_auc": mean_pr, "trading_cost": cost}


def train_and_judge(tag, df, feats, y, w, classes, params, tr_idx, va_idx, H_idx):
    """train xgboost on this regime's train (early-stop on its val), then score on its OWN val
    (the number it would REPORT during development) and on the forward hold-out H (the truth)."""
    print(f"  [{tag}] train {len(tr_idx):,}  val {len(va_idx):,}  -> scoring forward H {len(H_idx):,}")
    X = prepare_xgb(df, feats, tr_idx)                    # encode, fitting the map on THIS train only
    model = build_model("xgboost", len(classes), params, cat_idx=None, has_val=True)
    model.fit(X.iloc[tr_idx], y[tr_idx], sample_weight=w[tr_idx],
              eval_set=[(X.iloc[va_idx], y[va_idx])], verbose=False)

    def score(idx):
        proba = full_proba(model, model.predict_proba(X.iloc[idx]), len(classes))
        return quiet_metrics(y[idx], proba.argmax(axis=1), proba, classes)

    own = score(va_idx)          # what this strategy tells you while developing
    fwd = score(H_idx)           # what actually happens on the future it never saw
    del X, model
    return {"tag": tag, "own_dev": own, "forward_H": fwd}


def _idx(mask):
    return np.flatnonzero(np.asarray(mask))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/v4.1/dataset_v4.1.parquet")
    ap.add_argument("--manifest", default="datasets/v4.1/manifest.json")
    ap.add_argument("--holdout_frac", type=float, default=0.25,
                    help="the LAST fraction of time reserved as the forward referee H")
    ap.add_argument("--cap_trees", type=int, default=1200,
                    help="same tree ceiling for BOTH regimes. 0 = real hyperparams.yaml value (slow)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--bundle_freq", default="15min")
    a = ap.parse_args()

    df = pd.read_parquet(a.data)
    man = json.loads(pathlib.Path(a.manifest).read_text()) if pathlib.Path(a.manifest).exists() else None
    feats = man["feature_columns"] if man else [c for c in df.columns if "__" in c]
    print(f"loaded {a.data}: {len(df):,} rows   features {len(feats)}")

    from sklearn.preprocessing import LabelEncoder
    y_raw = df[C.LABEL_COL].astype(str).str.strip()
    le = LabelEncoder().fit(y_raw)
    classes = list(le.classes_)
    y = le.transform(y_raw)
    w = (y_raw.map(C.CLASS_WEIGHTS).astype(float).to_numpy()
         if getattr(C, "CLASS_WEIGHTS", None) else df[C.WEIGHT_COL].fillna(0.0).to_numpy())

    params = hyperparams.merge("xgboost", {})
    if a.cap_trees and int(params.get("n_estimators", 0)) > a.cap_trees:
        print(f"capping n_estimators {params['n_estimators']} -> {a.cap_trees} for both regimes")
        params["n_estimators"] = a.cap_trees
    if a.seed is not None:
        params["seed"] = a.seed
    seed = int(params.get("seed", 42))

    ts = pd.to_datetime(df[C.LABEL_TS_COL])
    bundle_id = ts.dt.floor(a.bundle_freq)

    # ---- 1. reserve the forward referee H (last holdout_frac of time, with an embargo before it) ----
    dev_m, _, H_m, hinfo = three_way_split(ts, 0.0, a.holdout_frac, C.EMBARGO_SESSIONS)
    dev_pos = _idx(dev_m.to_numpy())
    H_idx = _idx(H_m.to_numpy())
    print(f"\nforward referee H: last {a.holdout_frac:.0%} of time, >= {hinfo['test_start']}  "
          f"({len(H_idx):,} rows)   embargo {C.EMBARGO_SESSIONS} sessions before it")
    print(f"development pool (everything before H): {len(dev_pos):,} rows\n")

    # timestamps / bundles restricted to the development pool
    ts_dev = ts.iloc[dev_pos].reset_index(drop=True)
    bun_dev = bundle_id.iloc[dev_pos].reset_index(drop=True)

    # ---- 2a. regime T: bundle + TIME. carve val as the last VAL_FRACTION of dev, with embargo ----
    trT_m, _, vaT_m, _ = three_way_split(ts_dev, 0.0, C.VAL_FRACTION, C.EMBARGO_SESSIONS)
    trT_idx = dev_pos[_idx(trT_m.to_numpy())]
    vaT_idx = dev_pos[_idx(vaT_m.to_numpy())]

    # ---- 2b. regime R: bundle + RANDOM. val = a random VAL_FRACTION of the dev BUNDLES ----
    rng = np.random.RandomState(seed)
    bundles = pd.unique(bun_dev)
    bundles = bundles[rng.permutation(len(bundles))]
    n_va = int(round(len(bundles) * C.VAL_FRACTION))
    va_bundles = set(bundles[:n_va])
    is_va = bun_dev.isin(va_bundles).to_numpy()
    trR_idx = dev_pos[_idx(~is_va)]
    vaR_idx = dev_pos[_idx(is_va)]

    # ---- 3. train both, judge both on the SAME forward H ----
    print("training both regimes (same model, same forward referee -- only development split differs):")
    resT = train_and_judge("T  bundle+time ", df, feats, y, w, classes, params, trT_idx, vaT_idx, H_idx)
    resR = train_and_judge("R  bundle+random", df, feats, y, w, classes, params, trR_idx, vaR_idx, H_idx)

    # ---- 4. the verdict ----
    def line(res):
        o, f = res["own_dev"], res["forward_H"]
        return (f"  {res['tag']:<16}  dev(macro_f1 {o['macro_f1']:.3f}, cost {o['trading_cost']:.1f})"
                f"   FORWARD-H(macro_f1 {f['macro_f1']:.3f}, pr_auc {f['mean_pr_auc']:.3f}, "
                f"cost {f['trading_cost']:.1f})")

    print(f"\n{'#'*82}\n#  COMPLETE TEST -- judged on the forward hold-out H\n{'#'*82}")
    print(line(resT))
    print(line(resR))

    # question 1: was R's development number inflated vs its own forward truth?
    ro, rf = resR["own_dev"]["macro_f1"], resR["forward_H"]["macro_f1"]
    print(f"\nQ1 does bundle+random inflate?  R dev macro_f1 {ro:.3f}  vs  R forward {rf:.3f}  "
          f"-> {'INFLATED (dev overstates live by ' + format(ro-rf,'+.3f') + ')' if ro - rf > 0.02 else 'honest (dev ~ forward)'}")

    # question 2: the pivot decision -- who wins on the SAME forward rows?
    fR, fT = resR["forward_H"], resT["forward_H"]
    r_better = (fR["macro_f1"] > fT["macro_f1"] + 0.005) and (fR["trading_cost"] < fT["trading_cost"] - 0.5)
    t_better = (fT["macro_f1"] > fR["macro_f1"] + 0.005) or (fT["trading_cost"] < fR["trading_cost"] - 0.5)
    print(f"\nQ2 who wins FORWARD (the only thing that mimics live)?")
    print(f"   R forward: macro_f1 {fR['macro_f1']:.3f}  cost {fR['trading_cost']:.1f}")
    print(f"   T forward: macro_f1 {fT['macro_f1']:.3f}  cost {fT['trading_cost']:.1f}")
    if r_better:
        print("   -> R (bundle+random) wins forward. your idea is validated -- PIVOT, with proof.")
    elif t_better:
        print("   -> T (bundle+time+purge) wins forward. random buys nothing live -- KEEP time, revert.")
    else:
        print("   -> roughly tied forward. no reason to take on the random inflation -- KEEP time.")


if __name__ == "__main__":
    main()
