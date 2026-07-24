"""
trainer/shap_explain.py -- WHY does the model make its most expensive mistakes?

this runs after a model finishes training. it does what the SHAP script in clearml_tut did, but
for all three model types and for the 7 real classes.

WHAT IT ACTUALLY DOES, IN ORDER
    1. fetch the trained model (the joblib bundle the trainer saved)
    2. fetch the same dataset it was trained on, and rebuild the same test split
    3. rank every kind of mistake by  rate x severity  -- how often it happens, times how much
       it costs. the severity numbers come from configs/severity_7class.json.
    4. ALSO list every high-severity mistake regardless of how rare it is, because a rare
       full reversal can be buried by a common nuisance in a total-damage ranking, and a
       reversal is the kind of mistake that ends a book.
    5. take the single worst mistake pair (true A -> predicted B), compute SHAP for a sample
       of those rows, and show WHICH FEATURES pushed the model into the wrong answer.
    6. send everything to ClearML: the rankings, the feature shares, and waterfall pictures of
       real wrong predictions.

SHAP IS SLOW. we never compute it for all 500,000 rows -- that would take hours and tell us
nothing extra. we sample a few hundred rows of the two classes involved in the worst mistake.
that is enough to see the pattern.

run: never by hand. publish_version.py enqueues it after each model trains.
"""
import argparse
import json
import sys
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # draw straight to files; there is no screen on an agent
import matplotlib.pyplot as plt            # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as C          # noqa: E402
from trainer.shap_logic import (  # noqa: E402
    compute_shap, rank_mistakes, worst_case_mistakes, feature_shares,
    stable_feature_shares, explain_one_row, sample_for_shap, SHAP_SPACE,
)
from trainer.objective import three_way_split   # noqa: E402  (the SAME split as the trainer, or
                                                #             we would explain rows the model
                                                #             actually trained on)

OUT = pathlib.Path("shap_out")


def load_severity() -> tuple[dict, float]:
    """the cost of each kind of mistake. keys look like 'ENTRY_SUPER->EXIT_SUPER'."""
    if not C.SEVERITY_FILE.exists():
        print(f"no severity file at {C.SEVERITY_FILE} -- every mistake will count the same")
        return {}, 1.0
    cfg = json.loads(C.SEVERITY_FILE.read_text())
    sev = {k: v for k, v in cfg.get("severity", {}).items() if not k.startswith("_")}
    return sev, float(cfg.get("default", 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_task_id", default="", help="the training task whose model we explain")
    ap.add_argument("--model_type", default="", help="random_forest | xgboost | catboost")
    ap.add_argument("--dataset_version", default="")
    ap.add_argument("--n_samples", type=int, default=300,
                    help="rows to explain per class of the worst pair. SHAP is slow.")
    ap.add_argument("--n_charts", type=int, default=6, help="waterfall pictures to save")
    from clearml import Dataset, Task   # BEFORE parse_args -- clearml patches argparse at import;
    # parse first and every Args/ override from the clone is silently lost. see train.py.
    a = ap.parse_args()

    import joblib
    import shap

    task = Task.init(project_name=C.CLEARML_PROJECT,
                     task_name=C.BASE_SHAP_NAME,
                     task_type=Task.TaskTypes.qc,
                     output_uri=C.shap_output_uri())   # -> gs://<bucket>/artifacts/shap
    logger = task.get_logger()

    if not a.model_task_id:
        print("no --model_task_id: base-task registration run. exiting cleanly.")
        task.close()
        return

    # ---- 1. the trained model -----------------------------------------------
    print(f"[1/6] fetching the model from task {a.model_task_id}")
    src = Task.get_task(task_id=a.model_task_id)
    if src is None:
        raise SystemExit(f"no such training task: {a.model_task_id}")
    if "model" not in src.artifacts:
        raise SystemExit(f"task {a.model_task_id} has no 'model' artifact -- it either failed "
                         f"or never saved one. nothing to explain.")
    from trainer.train import load_model_bundle   # rebuilds xgboost from portable UBJ (version-safe)
    bundle = load_model_bundle(src.artifacts["model"].get_local_copy())
    model = bundle["model"]
    le = bundle["label_encoder"]
    feats = bundle["features"]
    mtype = bundle.get("model_type", a.model_type)
    classes = list(le.classes_)
    print(f"      {mtype}   {len(feats)} features   {len(classes)} classes")
    print(f"      SHAP values will be in {SHAP_SPACE.get(mtype, '?')} space")

    # ---- 2. the same data, and the SAME test split --------------------------
    print(f"[2/6] fetching dataset {bundle['dataset_id']}")
    ds = Dataset.get(dataset_id=bundle["dataset_id"], alias="shap_data")
    local = pathlib.Path(ds.get_local_copy())
    from trainer.train import find_dataset_parquet
    df = pd.read_parquet(find_dataset_parquet(local, bundle["dataset_id"]))

    ts = pd.to_datetime(df[C.LABEL_TS_COL])
    # THE SPLIT COMES FROM THE BUNDLE, NOT FROM TODAY'S CONFIG.
    # rebuilding it from config assumed config had not moved since training -- and it moves
    # (TEST_FRACTION changed 0.2 -> 0.3 the very day this file was written). explain rows the
    # model TRAINED on and every SHAP number is confidently, silently wrong. the trainer now
    # records its own test cut inside the bundle; that recorded cut is the truth.
    split = bundle.get("split") or {}
    if split.get("test_start"):
        te = ts >= pd.Timestamp(split["test_start"])
        print(f"      test split FROM THE BUNDLE: >= {split['test_start']}")
    else:
        # an old bundle (pre-2026-07-15) without a recorded split. config is the only option
        # left -- say loudly that this is a weaker guarantee.
        print("      !! old bundle: no recorded split. rebuilding from CURRENT config -- if "
              "config changed since training, these SHAP numbers explain the WRONG rows.")
        _, _, te, _ = three_way_split(ts, C.VAL_FRACTION, C.TEST_FRACTION, C.EMBARGO_SESSIONS)
    test = df[te].reset_index(drop=True)
    print(f"      explaining the TEST rows only: {len(test):,}")

    # rebuild the features exactly as the trainer did, or the SHAP values are meaningless
    X = test[feats].copy()
    maps = bundle.get("cat_maps", {}) or {}
    if mtype == "catboost":
        for c in bundle.get("categorical", []):
            X[c] = X[c].astype(str).fillna(C.CATEGORICAL_NA_LABEL)
    else:
        from na_policy import encode_categoricals
        X, _ = encode_categoricals(X, mapping={k: v for k, v in maps.items()
                                               if k != "_sentinels"})
        for c, sv in (maps.get("_sentinels") or {}).items():
            if c in X.columns:
                X[c] = X[c].fillna(sv)

    y_true = le.transform(test[C.LABEL_COL].astype(str))
    y_pred = np.asarray(model.predict_proba(X)).argmax(axis=1)

    # ---- 3. rank the mistakes by rate x severity ----------------------------
    print("[3/6] ranking the mistakes  (importance = rate x severity)")
    sev, default_sev = load_severity()
    rank = rank_mistakes(y_true, y_pred, classes, sev, default_sev)
    if rank.empty:
        print("      no mistakes at all. either the model is perfect or something is wrong.")
        logger.report_text("no misclassifications")
        task.close()
        return

    print("\n--- WHERE THE MONEY GOES (rate x severity) ---")
    print(rank.head(10).to_string(index=False))
    logger.report_table("Mistakes ranked", "by importance (rate x severity)", table_plot=rank)

    # the second view: what could KILL us, however rare
    worst = worst_case_mistakes(rank, min_severity=50)
    if not worst.empty:
        print("\n--- WHAT COULD KILL US (high severity, however rare) ---")
        print(worst.to_string(index=False))
        logger.report_table("Danger list", "high severity, any rate", table_plot=worst)

    # ---- 4. SHAP on the worst pair ------------------------------------------
    A, B = rank.iloc[0]["true"], rank.iloc[0]["pred"]
    Ai, Bi = classes.index(A), classes.index(B)
    print(f"\n[4/6] the worst mistake: TRUE '{A}'  ->  predicted '{B}'  "
          f"(importance {rank.iloc[0]['importance']})")

    picks = sample_for_shap(y_true, (Ai, Bi), a.n_samples)
    print(f"      computing SHAP on {len(picks):,} sampled rows "
          f"(not all {len(X):,} -- SHAP is slow and a sample shows the same pattern)")
    Xs = X.iloc[picks]
    # catboost needs to be TOLD which columns are text, or it tries to turn 'GAP_UP' into a
    # float and dies. the other two never see a text column by this point.
    vals, base = compute_shap(model, Xs, mtype,
                              cat_features=bundle.get("categorical", []))

    # ---- 5. which features drive it -----------------------------------------
    print("[5/6] which features push the model into this mistake")
    # not one ranking, but SEVERAL -- so we can say which parts of it are trustworthy and
    # which are just the luck of the sample. a ranking always LOOKS confident; this one has to
    # earn it. (measured: the top feature is rock-solid; ranks 4-5 wobble ~5%.)
    shares = stable_feature_shares(model, Xs, feats, mtype, cls=Bi, n_boot=5,
                                   cat_features=bundle.get("categorical", []))
    print(shares.head(12).to_string(index=False))
    print("     verdict: 'solid' = in the top 5 of EVERY run. 'noise' = it got lucky once.")
    logger.report_table(f"Feature shares -- pushing towards '{B}'", mtype, table_plot=shares)
    for _, r in shares.head(15).iterrows():
        logger.report_single_value(f"shap_share/{r['feature']}", float(r["share_%"]))

    # a per-row summary: what pushed each sampled prediction
    rows = []
    for k, s in enumerate(picks):
        p = int(y_pred[s])
        rows.append({
            "row": int(s),
            "true": classes[int(y_true[s])],
            "pred": classes[p],
            "result": "CORRECT" if p == int(y_true[s]) else "WRONG",
            "top_features": explain_one_row(vals, feats, k, p, top=3),
        })
    sdf = pd.DataFrame(rows)
    logger.report_table("Sampled rows -- what pushed each one", mtype, table_plot=sdf)
    n_wrong = int((sdf["result"] == "WRONG").sum())
    print(f"      of {len(sdf)} sampled rows, {n_wrong} were wrong")

    # ---- 6. pictures of real wrong predictions ------------------------------
    print("[6/6] drawing waterfalls of actual wrong predictions")
    OUT.mkdir(exist_ok=True)
    drawn = 0
    for k, s in enumerate(picks):
        if drawn >= a.n_charts:
            break
        t, p = int(y_true[s]), int(y_pred[s])
        if t == p or {classes[t], classes[p]} != {A, B}:
            continue                                  # only the A<->B mix-up we are chasing
        e = shap.Explanation(values=vals[k, :, p], base_values=float(base[p]),
                             data=Xs.iloc[k].values, feature_names=feats)
        shap.plots.waterfall(e, show=False, max_display=12)
        plt.title(f"row {s}: true '{classes[t]}' -> predicted '{classes[p]}'  (WRONG)")
        fig = plt.gcf(); fig.set_size_inches(9, 6); fig.tight_layout()
        path = OUT / f"{mtype}_{s}_{classes[t]}_to_{classes[p]}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
        # upload the PNG itself. ClearML converts matplotlib figures to plotly and silently
        # drops parts of them, so a picture that matters must be sent as an image file.
        logger.report_image("Wrong predictions", f"row_{s}", iteration=0, local_path=str(path))
        drawn += 1
    print(f"      saved {drawn} waterfall charts -> ClearML DEBUG SAMPLES")

    task.upload_artifact("mistake_ranking", rank)
    task.upload_artifact("feature_shares", shares)
    task.add_tags([mtype, "shap", f"v{a.dataset_version or bundle.get('dataset_version','')}"])
    print(f"\ndone. worst mistake: {A} -> {B}. "
          f"top feature pushing it: {shares.iloc[0]['feature']} ({shares.iloc[0]['share_%']}%)")
    task.close()


if __name__ == "__main__":
    main()
