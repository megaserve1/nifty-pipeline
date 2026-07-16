"""
trainer/local_check.py -- DO THESE FEATURES CARRY ANY SIGNAL AT ALL?

This is not the pipeline. It touches no ClearML, no GCS, no DVC, no git, and it costs nothing.
It reads one built dataset off the disk and answers ONE question, honestly:

    is there a pattern in here, or am I looking at noise?

run:
    python trainer/local_check.py --version v2
    python trainer/local_check.py --version v2 --models xgboost          # just one, faster
    python trainer/local_check.py --version v2 --rows 150000             # a quick look


WHY THIS FILE EXISTS RATHER THAN "JUST TRAIN IT AND LOOK AT THE ACCURACY"
--------------------------------------------------------------------------------------
Three ways this question gets answered wrong, and all three feel fine at the time.

1. ACCURACY IS A LIE HERE.
   NO_TRADE is 53% of the rows. A model that says NO_TRADE and nothing else scores 53% and is
   completely worthless. Any number you are tempted to quote must be compared against that.
   So we print the majority-class baseline next to every model, every time.

2. "BETTER THAN THE BASELINE" IS NOT THE SAME AS "THERE IS A PATTERN".
   A model with enough capacity will find structure in pure noise, especially with 500k rows
   and a time-ordered split. The only honest test is to DESTROY the relationship and re-run:
   shuffle the labels, keep everything else identical, and train again.

       if the shuffled model scores about the same as the real one -> THERE IS NO SIGNAL.
       whatever the real model "learned", it would have learned from noise.

   That is the control experiment, and it is the single most important number this file prints.
   It is also the one people skip, because it can tell you something you did not want to hear.

3. THE WEIGHT BUG WILL HIDE THE ANSWER.
   The labels file gives NO_TRADE a weight of 0, and rows with weight 0 contribute NOTHING to
   the loss. So the model never learns to stay out -- it wants to trade every single minute.
   If we only train weighted, a bad result is ambiguous: bad features, or broken weights?

   So we run BOTH:
       weighted    -- what the real pipeline does today. shows you the over-trading.
       unweighted  -- every row counts the same. THIS is the one that tells you whether the
                      FEATURES carry information, with the weight bug taken out of the way.

WHAT A HONEST RESULT LOOKS LIKE
--------------------------------------------------------------------------------------
On 1-minute index futures, real edge is SMALL. Do not expect a beautiful number.

    macro-F1 a few points above the shuffled control       -> plausible, keep going
    per-class recall on the ENTRY classes clearly above 0  -> the rare signals are findable
    PR-AUC on a 1.2% class of 0.03-0.08                    -> that is what real edge looks like
    the model matches its own shuffled control             -> there is nothing here. stop.

If a model looks BRILLIANT -- 90% accuracy, macro-F1 of 0.8 -- do not celebrate. Go and look for
a leak. On this problem, a brilliant score means a column is telling the model the answer.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as C                                              # noqa: E402
from trainer.objective import three_way_split, trading_cost      # noqa: E402
from trainer.train import build_model                            # noqa: E402


def load(version: str, rows: int | None) -> pd.DataFrame:
    p = C.DATASETS_DIR / version / f"dataset_{version}.parquet"
    if not p.exists():
        have = sorted(x.name for x in C.DATASETS_DIR.glob("v*")) if C.DATASETS_DIR.exists() else []
        raise SystemExit(f"no dataset at {p}\n  built versions: {have or 'NONE'}\n"
                         f"  build one first:  python bridge/build_dataset.py --version {version}")
    df = pd.read_parquet(p)
    if rows and rows < len(df):
        df = df.tail(rows).reset_index(drop=True)      # the MOST RECENT rows, never a sample
        print(f"  (--rows {rows:,}: using the most recent {rows:,} minutes only)")
    return df


def scores(y_true, y_pred, proba, classes) -> dict:
    from sklearn.metrics import f1_score, average_precision_score, recall_score
    oh = np.eye(len(classes))[y_true]
    pr = [average_precision_score(oh[:, i], proba[:, i])
          for i in range(len(classes)) if oh[:, i].sum() > 0]
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mean_pr_auc": float(np.mean(pr)) if pr else 0.0,
        "cost": float(trading_cost(y_true, y_pred, classes)),
        "recall": recall_score(y_true, y_pred, average=None, zero_division=0,
                               labels=range(len(classes))),
        "n_predicted": int(len(np.unique(y_pred))),
        "traded_pct": float((y_pred != list(classes).index("NO_TRADE")).mean() * 100)
        if "NO_TRADE" in classes else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, metavar="vN")
    ap.add_argument("--models", nargs="+", default=["xgboost", "random_forest"],
                    choices=C.MODEL_TYPES)
    ap.add_argument("--rows", type=int, default=0, help="use only the most recent N minutes")
    ap.add_argument("--n_estimators", type=int, default=200)
    ap.add_argument("--shap", action="store_true",
                    help="explain the best model with SHAP (adds a few minutes)")
    ap.add_argument("--shap_rows", type=int, default=4000,
                    help="how many TEST rows to explain. SHAP is O(rows); 4000 is plenty")
    a = ap.parse_args()

    from sklearn.preprocessing import LabelEncoder
    from na_policy import encode_categoricals, compute_sentinel

    print(f"\n{'='*84}\nDO THE FEATURES CARRY SIGNAL?   dataset {a.version}\n{'='*84}")
    df = load(a.version, a.rows)

    feat_cols = [c for c in df.columns
                 if c not in (C.LABEL_TS_COL, C.LABEL_COL, C.WEIGHT_COL, C.WEIGHT_RAW_COL)]
    ts = pd.to_datetime(df[C.LABEL_TS_COL])
    y_raw = df[C.LABEL_COL].astype(str)
    w = df[C.WEIGHT_COL].fillna(0.0) if C.WEIGHT_COL in df.columns else pd.Series(1.0, index=df.index)

    le = LabelEncoder().fit(y_raw)
    classes = list(le.classes_)
    y = le.transform(y_raw)

    print(f"\n  {len(df):,} minutes   {len(feat_cols)} features   {len(classes)} classes")
    print(f"  {ts.min()}  ->  {ts.max()}")

    # ---- the split. TIME ordered. the embargo is in SESSIONS. ---------------
    tr, _, te, info = three_way_split(ts, 0.0, C.TEST_FRACTION, C.EMBARGO_SESSIONS)
    print(f"\n  train {int(tr.sum()):>9,}  <= {info['train_end']}")
    print(f"  test  {int(te.sum()):>9,}  >= {info['test_start']}")
    print(f"  thrown away in the {C.EMBARGO_SESSIONS}-session embargo: {info['n_embargoed']:,}")

    # ---- the features, prepared PER MODEL, exactly as train.py does ---------
    # the first version of this harness sentinel-filled NaN for EVERY model and integer-encoded
    # catboost's text columns. production does neither: xgboost/catboost KEEP the NaN (they
    # learn a branch for it) and catboost reads text natively. so the harness was measuring a
    # pipeline that does not exist -- its verdict could not transfer to production. each model
    # now gets the same food it gets in train.py.
    cat_cols = [c for c in feat_cols
                if str(df[c].dtype) in ("object", "str", "string", "category", "bool")]

    X_enc = df[feat_cols].copy()                       # RF + XGB: encoded categoricals
    _, maps = encode_categoricals(X_enc.loc[tr])       # LEARN from train only. never the test.
    X_enc, _ = encode_categoricals(X_enc, mapping=maps)

    X_rf = X_enc.copy()                                # RF ONLY: sentinel-filled NaN
    sentinels = {}
    for c in X_rf.columns:
        if X_rf[c].isna().any():
            sv = compute_sentinel(X_rf.loc[tr, c], C.SENTINEL_MARGIN)
            X_rf[c] = X_rf[c].fillna(sv)
            sentinels[c] = sv

    X_cb = df[feat_cols].copy()                        # CATBOOST: raw text, NaN kept
    for c in cat_cols:
        X_cb[c] = X_cb[c].astype(str).fillna(C.CATEGORICAL_NA_LABEL)
    cat_idx = [X_cb.columns.get_loc(c) for c in cat_cols]

    def slices_for(mtype):
        if mtype == "random_forest":
            return X_rf.loc[tr], X_rf.loc[te], None
        if mtype == "catboost":
            return X_cb.loc[tr], X_cb.loc[te], cat_idx
        return X_enc.loc[tr], X_enc.loc[te], None      # xgboost: encoded cats, NaN KEPT

    ytr, yte = y[tr.to_numpy()], y[te.to_numpy()]
    wtr = w[tr].to_numpy()

    # ---- THE BASELINES. every number below must beat these or it means nothing.
    print(f"\n{'-'*84}\nBASELINES -- anything that does not beat these has learned nothing\n{'-'*84}")
    maj = int(pd.Series(ytr).mode()[0])
    base_pred = np.full(len(yte), maj)
    base_proba = np.zeros((len(yte), len(classes))); base_proba[:, maj] = 1.0
    b = scores(yte, base_pred, base_proba, classes)
    print(f"  always say '{classes[maj]}'      accuracy={(base_pred==yte).mean()*100:5.2f}%   "
          f"macro_f1={b['macro_f1']:.4f}   cost={b['cost']:.2f}")
    print(f"\n  >> accuracy is BANNED from here on. '{classes[maj]}' is "
          f"{(y==maj).mean()*100:.1f}% of the data, so {(base_pred==yte).mean()*100:.1f}% "
          f"accuracy is worth exactly nothing.")

    # ---- the class balance, so the recalls below can be read ---------------
    print(f"\n  class            share    mean weight")
    for i, cl in enumerate(classes):
        share = (y == i).mean() * 100
        mw = float(w[y_raw == cl].mean())
        flag = "   <- WEIGHT 0: contributes NOTHING to the loss" if mw == 0 else ""
        print(f"  {cl:14s} {share:6.2f}%   {mw:6.2f}{flag}")

    results = []
    for mtype in a.models:
        for mode in ("weighted", "unweighted"):
            for labels in ("real", "SHUFFLED"):
                params = {"n_estimators": a.n_estimators, "max_depth": 0,
                          "min_samples_leaf": 50, "learning_rate": 0.05, "seed": 42,
                          "max_features": "sqrt"}
                yy = ytr.copy()
                if labels == "SHUFFLED":
                    # THE CONTROL. destroy the link between X and y, change nothing else.
                    yy = np.random.default_rng(0).permutation(yy)

                sw = wtr if mode == "weighted" else np.ones(len(yy))
                t0 = time.time()
                Xtr_m, Xte_m, ci = slices_for(mtype)
                m = build_model(mtype, len(classes), params, cat_idx=ci)
                m.fit(Xtr_m, yy, sample_weight=sw)
                from trainer.train import full_proba
                pr = full_proba(m, m.predict_proba(Xte_m), len(classes))
                s = scores(yte, pr.argmax(1), pr, classes)
                s.update(model=mtype, mode=mode, labels=labels, secs=time.time() - t0,
                         importances=getattr(m, "feature_importances_", None),
                         fitted=m if labels == "real" else None)
                results.append(s)
                print(f"\n  {mtype:14s} {mode:11s} {labels:9s}  "
                      f"macro_f1={s['macro_f1']:.4f}  pr_auc={s['mean_pr_auc']:.4f}  "
                      f"cost={s['cost']:.2f}  predicts {s['n_predicted']}/{len(classes)} classes  "
                      f"({s['secs']:.0f}s)")

    # =========================================================================
    print(f"\n{'='*84}\nTHE VERDICT\n{'='*84}")
    R = pd.DataFrame(results)

    print("\n  THE CONTROL EXPERIMENT -- real labels vs SHUFFLED labels.")
    print("  if a model scores the same on shuffled labels, it learned NOTHING from the features.\n")
    print(f"  {'model':14s} {'weights':11s} {'REAL f1':>9s} {'SHUFFLED f1':>12s} {'gain':>8s}   verdict")
    for mtype in a.models:
        for mode in ("weighted", "unweighted"):
            real = R[(R.model == mtype) & (R["mode"] == mode) & (R.labels == "real")].iloc[0]
            shuf = R[(R.model == mtype) & (R["mode"] == mode) & (R.labels == "SHUFFLED")].iloc[0]
            gain = real.macro_f1 - shuf.macro_f1
            if gain < 0.005:
                v = "NO SIGNAL. the features tell it nothing."
            elif gain < 0.02:
                v = "very weak. could be noise. treat with suspicion."
            elif gain < 0.10:
                v = "REAL SIGNAL, and it is small -- which is what real edge looks like."
            else:
                v = "*** LARGE. go and look for a leak before you believe it. ***"
            print(f"  {mtype:14s} {mode:11s} {real.macro_f1:>9.4f} {shuf.macro_f1:>12.4f} "
                  f"{gain:>+8.4f}   {v}")

    print("\n  CAN IT STAY OUT OF THE MARKET?  (the NO_TRADE weight-0 problem, measured)\n")
    print(f"  {'model':14s} {'weights':11s} {'minutes it would trade':>24s}")
    for mtype in a.models:
        for mode in ("weighted", "unweighted"):
            r = R[(R.model == mtype) & (R["mode"] == mode) & (R.labels == "real")].iloc[0]
            flag = "  <- it wants to trade EVERY MINUTE" if r.traded_pct > 90 else ""
            print(f"  {mtype:14s} {mode:11s} {r.traded_pct:>23.1f}%{flag}")
    print(f"\n  the honest share of tradeable minutes is {100-(y==classes.index('NO_TRADE')).mean()*100:.1f}%"
          if "NO_TRADE" in classes else "")
    print("  if 'weighted' trades far more than 'unweighted', that is the weight-0 bug, not the")
    print("  features. the fix is upstream in the LABEL POLICY (give NO_TRADE ~0.1-0.2).")

    print("\n  PER-CLASS RECALL on the UNWEIGHTED, REAL run -- can it find the rare signals?\n")
    for mtype in a.models:
        r = R[(R.model == mtype) & (R["mode"] == "unweighted") & (R.labels == "real")].iloc[0]
        print(f"  {mtype}")
        for i, cl in enumerate(classes):
            rec = r["recall"][i]
            bar = "#" * int(rec * 40)
            print(f"      {cl:14s} recall {rec:5.1%}  {bar}")

    best = R[(R.labels == "real") & (R["mode"] == "unweighted")].sort_values("macro_f1").iloc[-1]
    if best["importances"] is not None:
        print(f"\n  WHICH FEATURES CARRY IT?  ({best.model}, unweighted, real labels)\n")
        imp = pd.Series(best["importances"], index=feat_cols).sort_values(ascending=False)
        for f, v in imp.head(15).items():
            print(f"      {v:6.3f}  {'#' * int(v * 120)} {f}")
        dead = [f for f, v in imp.items() if v < 0.001]
        if dead:
            print(f"\n      {len(dead)} feature(s) the model ignored entirely: {dead[:8]}"
                  f"{' ...' if len(dead) > 8 else ''}")

    # ---- save the winner, so you can poke at it without re-training ---------
    import joblib
    out = C.OUT_DIR / "local"
    out.mkdir(parents=True, exist_ok=True)
    bundle = out / f"local_{best.model}_{a.version}.joblib"
    joblib.dump({"model": best["fitted"], "label_encoder": le, "features": feat_cols,
                 "cat_maps": {**maps, "_sentinels": sentinels}, "categorical": cat_cols,
                 "model_type": best.model, "dataset_version": a.version,
                 "split": {"test_start": info["test_start"],
                           "test_fraction": C.TEST_FRACTION,
                           "embargo_sessions": C.EMBARGO_SESSIONS},
                 "mode": "unweighted"}, bundle)
    print(f"\n  model saved -> {bundle}")

    # =========================================================================
    # SHAP -- not "which features are important", but WHY THE EXPENSIVE MISTAKES HAPPEN
    # =========================================================================
    if a.shap:
        shap_report(best, slices_for(best.model)[1], yte, classes, feat_cols, a.shap_rows)

    print(f"\n{'='*84}")
    print("  READ THE 'gain' COLUMN FIRST. everything else is detail.")
    print("  a model that cannot beat its own shuffled control has found nothing, however")
    print("  good its other numbers look.")
    print(f"{'='*84}\n")


def shap_report(best, Xte, yte, classes, feat_cols, n_rows: int):
    """explain the best model, LOCALLY. no ClearML, no GCS.

    trainer/shap_explain.py does this properly, but it fetches the model and the dataset FROM
    ClearML -- so it cannot run until the cloud half is set up. shap_logic.py, where the actual
    maths lives, is pure. So we call that directly.

    WHY WE DO NOT JUST PRINT feature_importances_.
        the tree's built-in importance counts HOW OFTEN a feature was split on. it says nothing
        about direction, nothing about which class, and nothing about what it cost us. SHAP
        attributes each individual prediction, so we can ask the only question that matters:

            when the model made an EXPENSIVE mistake, what was it looking at?

        an expensive mistake is not any mistake. calling an EXIT_SMALL an EXIT_SUB costs little.
        calling a NO_TRADE an ENTRY_SUPER puts real money on a trade that should never have been
        placed -- configs/severity_7class.json prices that at 15. That is the mistake worth
        understanding, and it is invisible to feature_importances_.
    """
    from trainer.shap_logic import (compute_shap, rank_mistakes, worst_case_mistakes,
                                    feature_shares, sample_for_shap)
    import json

    print(f"\n{'='*84}\nSHAP -- why the EXPENSIVE mistakes happen   ({best.model})\n{'='*84}")

    model = best["fitted"]
    sev, default_sev = {}, 1.0
    if C.SEVERITY_FILE.exists():
        cfg = json.loads(C.SEVERITY_FILE.read_text())
        sev = {k: v for k, v in cfg.get("severity", {}).items() if not k.startswith("_")}
        default_sev = float(cfg.get("default", 1))

    y_pred = np.asarray(model.predict_proba(Xte)).argmax(1)

    # ---- 1. rank the mistakes by what they COST, not by how many there are
    rank = rank_mistakes(yte, y_pred, classes, sev, default_sev)
    if rank.empty:
        print("  the model made no mistakes at all. that is not good news -- go and find the leak.")
        return
    print("\n  THE MISTAKES THAT COST THE MOST  (rate x severity, not raw count)\n")
    print(f"  {'true':14s} -> {'predicted':14s} {'of true':>8s} {'severity':>9s} {'cost':>8s}")
    for _, r in rank.head(8).iterrows():
        print(f"  {r['true']:14s} -> {r['pred']:14s} {r['of_true_%']:>7.1f}% "
              f"{r['severity']:>9.0f} {r['importance']:>8.2f}")

    worst = worst_case_mistakes(rank, min_severity=10.0)
    if len(worst):
        print(f"\n  {len(worst)} of these are HIGH-SEVERITY (>=10): a trade placed that should")
        print(f"  never have been placed, or a real signal missed. those are the money ones.")

    # ---- 2. SHAP on a sample, weighted toward the worst mistake
    top = rank.iloc[0]
    ti, pi = classes.index(top["true"]), classes.index(top["pred"])
    idx = sample_for_shap(yte, (ti, pi), n=n_rows, seed=42)
    Xs = Xte.iloc[idx]
    print(f"\n  computing SHAP on {len(Xs):,} test rows "
          f"(over-sampled around {top['true']} -> {top['pred']}, the costliest mistake)...")
    t0 = time.time()
    # compute_shap returns (values, base_values). the base value is what the model predicts
    # before it looks at any feature; the values are what each feature ADDS to that.
    vals, _base = compute_shap(model, Xs, best.model)
    print(f"  done in {time.time()-t0:.0f}s")

    # ---- 3. what drives the model overall, and what drives the costly class
    print(f"\n  WHAT THE MODEL ACTUALLY USES  (mean |SHAP|, all classes)\n")
    sh = feature_shares(vals, feat_cols)
    for _, r in sh.head(15).iterrows():
        print(f"      {r['share_%']:5.1f}%  {'#' * int(r['share_%'] * 1.5)} {r['feature']}")

    print(f"\n  WHAT DRIVES THE '{top['pred']}' CALL -- the one that costs us\n")
    sh_cls = feature_shares(vals, feat_cols, cls=pi)
    for _, r in sh_cls.head(10).iterrows():
        print(f"      {r['share_%']:5.1f}%  {'#' * int(r['share_%'] * 1.5)} {r['feature']}")

    dead = sh[sh["share_%"] < 0.5]["feature"].tolist()
    if dead:
        print(f"\n  {len(dead)} feature(s) the model barely looks at: {dead[:10]}"
              f"{' ...' if len(dead) > 10 else ''}")
        print(f"  they cost you compute and add noise. worth dropping from the next version.")

    print(f"\n  HOW TO READ THIS")
    print(f"    one feature above ~40%   -> the model is a one-trick pony. fragile.")
    print(f"    the top feature also being what drives the costly mistake -> that feature is")
    print(f"    doing the damage. that is the one to fix, drop, or smooth.")


if __name__ == "__main__":
    main()
