"""trainer/export_scored_tables.py -- score a trained model and hand the next team two tables.

this runs AFTER a model trains (same slot as shap_explain). it does one job: take the model
that was just trained on version vN, run it over the WHOLE dataset, and write two tables --

    scored_train_vN.parquet    the rows the model was trained on   (in-sample)
    scored_test_vN.parquet     the rows it was held out on          (out-of-sample)

each row is:

    unique_index      the dataset's OWN unique id column (assigned at dataset-creation time; under
                      the bundle scheme it is two-level, e.g. 1a..1o). positional only as fallback.
    timestamp         the minute
    split             train | val | embargo | test   (so the two files are self-describing)
    true_label        the actual label (trailing spaces stripped)
    predicted_label   the model's argmax class
    correct           YES / no
    proba_<class>     the probability of EACH of the 7 classes for this row   <- what you asked for

it saves both tables to GCS (via the ClearML task's output_uri, exactly like the model and the
SHAP artifacts) so another team can read them and extend the pipeline later.

TWO WAYS TO RUN IT
    1. pipeline mode  (--model_task_id):  fetch the model + dataset FROM GCP, like shap_explain.
                       this is the step that gets wired in after training.
    2. local mode     (--bundle <joblib> --data <parquet>):  no ClearML round-trip. use this to
                       produce the tables for an already-trained model RIGHT NOW, on your machine.

WHY THE SPLIT COMES FROM THE BUNDLE, NOT TODAY'S CONFIG
    same trap shap_explain hit: config.TEST_FRACTION moved after training once, and any consumer
    that rebuilt the split from current config then mislabels which rows were train vs test. the
    bundle records the exact fractions + cut it trained with; we reproduce the split from THOSE.

run (local, on v4.1 -- you run it):
    final_venv/bin/python trainer/export_scored_tables.py \
        --bundle "/home/megaserve/Downloads/clearml_Nifty Production_train_xgboost v4.1.a1a4c4dcf1484ad38f5d06a986e7662d_artifacts_model_model_xgboost.joblib" \
        --data datasets/v4.1/dataset_v4.1.parquet
"""
import argparse
import sys
import pathlib

import numpy as np
import pandas as pd

_here = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_here.parent.parent))          # repo root: config, na_policy, trainer
sys.path.insert(0, str(_here.parent.parent / "scripts"))   # reuse predict.prepare
import config as C                                    # noqa: E402
from trainer.train import load_model_bundle, full_proba, find_dataset_parquet  # noqa: E402
from trainer.objective import three_way_split          # noqa: E402  (the SAME split as training)


def score_dataset(df: pd.DataFrame, bundle: dict) -> tuple:
    """score the WHOLE dataframe with the bundle's model. returns (table, split_counts).

    the table has one row per input row: id, timestamp, the true and predicted label, and one
    probability column per class (confidence is dropped -- it was just a copy of the predicted
    class's proba). the split column marks each row as train / val / embargo / test, rebuilt from
    the fractions the bundle was trained with.
    """
    from predict import prepare                        # reuse the EXACT training preprocessing
    model = bundle["model"]
    le = bundle["label_encoder"]
    classes = list(le.classes_)

    # ---- rebuild the split the model actually trained with, FROM THE BUNDLE ----
    ts = pd.to_datetime(df[C.LABEL_TS_COL])
    sp = bundle.get("split") or {}
    val_frac = float(sp.get("val_fraction", C.VAL_FRACTION))
    test_frac = float(sp.get("test_fraction", C.TEST_FRACTION))
    embargo = int(sp.get("embargo_sessions", C.EMBARGO_SESSIONS))
    if not sp:
        print("      !! old bundle: no recorded split. rebuilding from CURRENT config -- if config "
              "changed since training, the train/test labels below are for the WRONG rows.")
    tr, va, te, info = three_way_split(ts, val_frac, test_frac, embargo)
    split = np.full(len(df), "embargo", dtype=object)  # anything in neither slice is embargoed
    split[tr.to_numpy()] = "train"
    split[va.to_numpy()] = "val"
    split[te.to_numpy()] = "test"
    print(f"      split from bundle: val_frac={val_frac} test_frac={test_frac} "
          f"embargo={embargo} sessions   (test starts {info['test_start']})")

    # ---- score every row in one pass ----
    X = prepare(df, bundle)                            # exact columns, encoding, sentinels
    proba = full_proba(model, model.predict_proba(X), len(classes))   # (n_rows, n_classes)
    pred_idx = proba.argmax(axis=1)
    true = df[C.LABEL_COL].astype(str).str.strip().to_numpy()   # labels carry trailing spaces
    pred = le.inverse_transform(pred_idx)

    # THE UNIQUE INDEX BELONGS TO THE DATASET, NOT TO US.
    # it is assigned at dataset-creation time (the merge step) and travels as a column in the
    # dataset version. under the 15-min BUNDLE scheme it is two-level (bundle id 1,2,3.. + a..o
    # sub-id -> 1a..1o, 2a..2o). so we PASS IT THROUGH untouched whenever it is present, and only
    # fall back to a positional id for old datasets that do not carry it yet.
    if "unique_index" in df.columns:
        uid = df["unique_index"].to_numpy()
        print("      unique_index: using the dataset's own column")
    else:
        uid = np.arange(len(df))
        print("      unique_index: dataset has none yet -> positional 0..n-1 (fallback)")

    tbl = pd.DataFrame({
        "unique_index": uid,
        "timestamp": ts.values,
        "split": split,
        "true_label": true,
        "predicted_label": pred,
        "correct": np.where(pred == true, "YES", "no"),
    })
    for i, cls in enumerate(classes):                  # one probability column per class
        tbl[f"proba_{cls}"] = proba[:, i]

    counts = tbl["split"].value_counts().to_dict()
    return tbl, counts


def split_tables(tbl: pd.DataFrame, strict_train: bool) -> tuple:
    """cut the scored table into the two files. test = the test slice. train = everything else
    (train + val + embargo) so the two files together cover the COMPLETE dataset -- the split
    column still says which is which. --strict_train instead keeps ONLY the true train slice."""
    test_tbl = tbl[tbl["split"] == "test"].reset_index(drop=True)
    if strict_train:
        train_tbl = tbl[tbl["split"] == "train"].reset_index(drop=True)
    else:
        train_tbl = tbl[tbl["split"] != "test"].reset_index(drop=True)
    return train_tbl, test_tbl


def write_and_report(train_tbl, test_tbl, version, out_dir, task=None):
    """write both parquets locally, and (in pipeline mode) upload them to GCS as task artifacts."""
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, t in (("train", train_tbl), ("test", test_tbl)):
        p = out_dir / f"scored_{name}_v{version}.parquet"
        t.to_parquet(p, index=False)
        paths[name] = p
        acc = (t["correct"] == "YES").mean() if len(t) else float("nan")
        print(f"      wrote {p.name}  ({len(t):,} rows, raw-accuracy {acc:.4f})")
        if task is not None:
            # upload the FILE (not the DataFrame) so a 500k-row table goes straight to the bucket
            # as one parquet blob, exactly like the model artifact does.
            task.upload_artifact(f"scored_{name}_v{version}", str(p))
    return paths


def main():
    ap = argparse.ArgumentParser()
    # pipeline mode (fetch from GCP, like shap_explain):
    ap.add_argument("--model_task_id", default="", help="training task whose model we score")
    ap.add_argument("--dataset_version", default="")
    # local mode (no ClearML):
    ap.add_argument("--bundle", default="", help="a model_*.joblib on disk (local mode)")
    ap.add_argument("--data", default="", help="the dataset parquet (local mode)")
    # both:
    ap.add_argument("--out", default="scored_out", help="where to write the two parquets")
    ap.add_argument("--strict_train", action="store_true",
                    help="train table = ONLY the train slice (drop val+embargo). default keeps "
                         "them so the two files cover the complete dataset.")
    a = ap.parse_args()

    # ---------- LOCAL MODE: bundle + data on disk, no ClearML ----------
    if a.bundle and a.data:
        print(f"[local] loading model bundle {pathlib.Path(a.bundle).name}")
        bundle = load_model_bundle(a.bundle)
        version = a.dataset_version or str(bundle.get("dataset_version", "?"))
        print(f"[local] reading {a.data}")
        df = pd.read_parquet(a.data)
        print(f"        {len(df):,} rows   model trained on v{version}   "
              f"{len(bundle['features'])} features")
        tbl, counts = score_dataset(df, bundle)
        print(f"        rows per split: {counts}")
        train_tbl, test_tbl = split_tables(tbl, a.strict_train)
        write_and_report(train_tbl, test_tbl, version, a.out, task=None)
        print(f"\ndone (local). two tables in {a.out}/ . to also push them to GCS, run in pipeline "
              f"mode with --model_task_id, or hand them to the next team as-is.")
        return

    # ---------- PIPELINE MODE: fetch model + data from GCP, upload tables to GCS ----------
    from clearml import Dataset, Task   # imported AFTER parse_args on purpose -- clearml patches
    # argparse at import; parsing first means a cloned task's Args/ overrides are silently lost.
    task = Task.init(project_name=C.CLEARML_PROJECT,
                     task_name=getattr(C, "BASE_EXPORT_NAME", "export_scored_tables (base)"),
                     task_type=Task.TaskTypes.qc,
                     output_uri=C.tables_output_uri())   # gcs mode -> gs://<bucket>/tables
    if not a.model_task_id:
        print("no --model_task_id: base-task registration run. exiting cleanly.")
        task.close()
        return

    print(f"[1/4] fetching the model from task {a.model_task_id}")
    src = Task.get_task(task_id=a.model_task_id)
    if src is None or "model" not in src.artifacts:
        raise SystemExit(f"task {a.model_task_id} has no 'model' artifact -- nothing to score.")
    bundle = load_model_bundle(src.artifacts["model"].get_local_copy())
    version = a.dataset_version or str(bundle.get("dataset_version", "?"))

    print(f"[2/4] fetching dataset {bundle['dataset_id']}")
    ds = Dataset.get(dataset_id=bundle["dataset_id"], alias="scored_data")
    local = pathlib.Path(ds.get_local_copy())
    df = pd.read_parquet(find_dataset_parquet(local, bundle["dataset_id"]))
    print(f"      {len(df):,} rows   model trained on v{version}")

    print("[3/4] scoring the whole dataset")
    tbl, counts = score_dataset(df, bundle)
    print(f"      rows per split: {counts}")
    train_tbl, test_tbl = split_tables(tbl, a.strict_train)

    print("[4/4] writing + uploading the two tables to GCS")
    write_and_report(train_tbl, test_tbl, version, a.out, task=task)
    task.add_tags([bundle.get("model_type", "?"), "scored_tables", f"v{version}"])
    print(f"\ndone. scored_train_v{version} + scored_test_v{version} are artifacts on this task, "
          f"in gs://{C.GCS_BUCKET}/clearml . the next team reads them from there.")
    task.close()


if __name__ == "__main__":
    main()
