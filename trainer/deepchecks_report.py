"""trainer/deepchecks_report.py -- run Deepchecks against a trained model and publish the reports.

this runs AFTER a model trains (same slot as shap_explain and export_scored_tables). it takes the
model that was just trained on version vN, rebuilds the SAME train/test split it trained with, and
runs three suites:

    1_data_integrity          NaNs, duplicates, conflicting labels, outliers        (the data)
    2_train_test_validation   leakage between train and test, drift                 (the split)
    3_model_evaluation        overfit, does it beat a simple baseline               (the model)

each suite is saved as a STATIC html file and uploaded as a task artifact, so the files land in
    gs://<bucket>/artifacts/deepchecks/...
and anyone can download them from the ClearML UI and open them in a browser. no server, no widget.

REPORT ONLY -- it never fails the run. deepchecks conditions will flag things that are true and
deliberate here (53% NO_TRADE is extreme imbalance; a TIME split shows train/test drift BY DESIGN),
so a hard gate would block every run for reasons you already know. read the reports, then decide
which conditions are worth enforcing.

WHY THE SPLIT COMES FROM THE BUNDLE, NOT TODAY'S CONFIG
    same reason as shap_explain / export_scored_tables: config fractions can move after training,
    and rebuilding the split from current config would label the wrong rows as test.

TWO WAYS TO RUN IT
    1. pipeline mode  (--model_task_id):  fetch the model + dataset from GCP. queued after training.
    2. local mode     (--bundle <joblib> --data <parquet>):  no ClearML round trip.

run (local -- you run it):
    final_venv/bin/python trainer/deepchecks_report.py \
        --bundle /path/model_xgboost.joblib \
        --data datasets/v6/dataset_v6.parquet \
        --out /tmp/dc_check
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

_here = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_here.parent.parent))
sys.path.insert(0, str(_here.parent.parent / "scripts"))
import config as C                                    # noqa: E402
from trainer.train import load_model_bundle, find_dataset_parquet   # noqa: E402
from trainer.objective import three_way_split          # noqa: E402

# ---------------------------------------------------------------------------------------------
# TWO COMPATIBILITY SHIMS. deepchecks 0.19.1 is the NEWEST release and it predates both numpy 2
# and sklearn 1.9, so it cannot import against them unaided. these must run BEFORE any deepchecks
# import. downgrading numpy or sklearn instead would drag the TRAINER backwards -- not worth it
# for a reporting step.
#
# 1. numpy 2.0 removed np.Inf / np.NaN / np.NINF. deepchecks still uses them as default arguments
#    (e.g. checks/model_evaluation/performance_bias.py). restoring the aliases is harmless: same
#    values, different spelling.
for _alias, _real in (("Inf", np.inf), ("NaN", np.nan), ("NINF", -np.inf),
                      ("PINF", np.inf), ("infty", np.inf)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _real)

# 2. sklearn 1.9 REMOVED the "max_error" scorer name (it is "neg_max_error" now), but deepchecks
#    calls get_scorer("max_error") at IMPORT time and dies before anything runs. we register it
#    back under the old name. safe: max_error is a REGRESSION scorer, and every suite here is
#    classification -- it only has to exist for the import to succeed, it is never used.
try:
    from sklearn.metrics import _scorer as _sk_scorer, make_scorer as _mk, max_error as _max_err
    if "max_error" not in _sk_scorer._SCORERS:
        _sk_scorer._SCORERS["max_error"] = _mk(_max_err, greater_is_better=False)
except Exception:
    pass          # a future sklearn may restore it, or move _SCORERS -- do not block on this

# 3. setuptools AFTER 81 removed pkg_resources, and deepchecks imports parse_version from it.
#    THIS MUST BE A SHIM, NOT A REQUIREMENTS PIN: a clearml-agent records a task's requirements
#    from the packages the SCRIPT IMPORTS, so "setuptools<82" in requirements.txt never reaches
#    the agent -- it builds a fresh venv with the newest setuptools and fails here every time.
#    packaging.version.parse is the same function pkg_resources.parse_version delegates to.
try:
    import pkg_resources          # noqa: F401  -- present on older setuptools, nothing to do
except ImportError:
    import types as _types
    from packaging.version import parse as _parse_version
    _shim = _types.ModuleType("pkg_resources")
    _shim.parse_version = _parse_version
    sys.modules["pkg_resources"] = _shim
# ---------------------------------------------------------------------------------------------


def build_datasets(df: pd.DataFrame, bundle: dict, sample: int):
    """(train_ds, test_ds, n_train, n_test) as deepchecks Datasets, split like training did."""
    from deepchecks.tabular import Dataset as DcDataset
    from predict import prepare                        # the EXACT training preprocessing
    from trainer.export_scored_tables import normalise_time_column

    # the handed-over matrices (V7/V8/V9) store time as a DatetimeIndex named 'datetime';
    # everything below reads a 'timestamp' COLUMN. same normaliser as the scoring step.
    df = normalise_time_column(df)

    feats = list(bundle["features"])
    le = bundle["label_encoder"]

    ts = pd.to_datetime(df[C.LABEL_TS_COL])
    sp = bundle.get("split") or {}
    tr, va, te, _ = three_way_split(
        ts,
        float(sp.get("val_fraction", C.VAL_FRACTION)),
        float(sp.get("test_fraction", C.TEST_FRACTION)),
        int(sp.get("embargo_sessions", C.EMBARGO_SESSIONS)),
        strategy=sp.get("strategy"), bundle_minutes=sp.get("bundle_minutes"),
        seed=sp.get("seed"),
    )

    X = prepare(df, bundle)                            # same columns/encoding/sentinels as training
    y = pd.Series(le.transform(df[C.LABEL_COL].astype(str).str.strip()), index=df.index)

    def make(mask):
        idx = df.index[mask.to_numpy()]
        if sample and len(idx) > sample:
            # evenly spaced, NOT random: keeps the time ordering deepchecks reasons about
            idx = idx[np.linspace(0, len(idx) - 1, sample).astype(int)]
        frame = X.loc[idx].copy()
        frame["label"] = y.loc[idx].to_numpy()
        # the model's own categorical list -- passing it changes what several checks do
        cats = [c for c in (bundle.get("categorical") or []) if c in frame.columns]
        return DcDataset(frame, label="label", features=feats, cat_features=cats), len(idx)

    train_ds, n_tr = make(tr)
    test_ds, n_te = make(te)
    return train_ds, test_ds, n_tr, n_te


def summarise(name: str, result) -> str:
    """NAME what did not pass, not just how many. the html is 15-50 MB; this is what you read
    first, and what tells you whether opening the html is worth it.

    two different things come back from get_not_passed_checks():
      CheckResult   ran fine, but a CONDITION failed  -> say which condition and why
      CheckFailure  the check itself ERRORED          -> say the exception
    they have different APIs, so each is handled separately. ConditionResult has is_pass and
    details but no .name attribute, which is why the first version of this printed nothing.
    """
    lines = [f"--- {name} ---"]
    try:
        not_passed = result.get_not_passed_checks()
        for r in not_passed:
            header = r.get_header() if hasattr(r, "get_header") else str(r)
            exc = getattr(r, "exception", None)
            if exc is not None:                          # CheckFailure -- the check crashed
                lines.append(f"  ERROR  {header}: {type(exc).__name__}: {exc}")
                continue
            conds = getattr(r, "conditions_results", None) or []
            failed = [c for c in conds if not getattr(c, "is_pass", True)]
            if not failed:
                lines.append(f"  FLAG   {header}")
                continue
            for c in failed:
                detail = getattr(c, "details", "") or ""
                lines.append(f"  FAIL   {header}: {str(detail)[:150]}")
        lines.append(f"  ({len(result.get_passed_checks())} passed, "
                     f"{len(not_passed)} raised something)")
    except Exception as exc:                             # deepchecks APIs move between versions
        lines.append(f"  (could not summarise: {type(exc).__name__}: {exc})")
    return "\n".join(lines)


def run_suites(train_ds, test_ds, model, tag: str, out_dir, task=None, which=None) -> list:
    """run the requested suites, write one static html each, upload them. returns summary lines.

    `which` is a list of suite keys. WHY IT IS A PARAMETER: only model_evaluation depends on the
    MODEL -- data_integrity and train_test_validation look at the dataset and the split alone, so
    running them once per model produces byte-identical reports for xgboost and catboost. today we
    run all three per model (simple); when that duplication starts to cost, move the two data
    suites to publish time with --suites data and leave --suites model here. no rewrite.
    """
    from deepchecks.tabular.suites import (data_integrity, train_test_validation,
                                           model_evaluation)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    ALL = [
        ("1_data_integrity", "data", lambda: data_integrity().run(train_ds)),
        ("2_train_test_validation", "data", lambda: train_test_validation().run(train_ds, test_ds)),
        ("3_model_evaluation", "model", lambda: model_evaluation().run(train_ds, test_ds, model)),
    ]
    want = set(which or ["data", "model"])
    suites = [(n, fn) for n, kind, fn in ALL if kind in want or n in want]
    if not suites:
        raise SystemExit(f"no suites selected from {which!r}. use: data / model / a suite name.")

    for name, run in suites:
        print(f"      running {name} ...", flush=True)
        try:
            result = run()
        except Exception as exc:
            msg = f"--- {name} ---\n  SUITE FAILED: {type(exc).__name__}: {exc}"
            print("   " + msg.replace("\n", "\n   "))
            summaries.append(msg)
            continue                                   # one bad suite must not kill the others
        fname = f"deepchecks_{name}_{tag}.html"
        path = out_dir / fname
        # as_widget=False -> a plain static html file, no anywidget/jupyter dependency
        result.save_as_html(str(path), as_widget=False)
        print(f"      wrote {fname}")
        if task is not None:
            task.upload_artifact(f"deepchecks_{name}", artifact_object=str(path))
        summaries.append(summarise(name, result))

    text = "\n".join(summaries)
    spath = out_dir / f"deepchecks_summary_{tag}.txt"
    spath.write_text(text)
    if task is not None:
        task.upload_artifact("deepchecks_summary", artifact_object=str(spath))
    print("\n" + text)
    return summaries


def main():
    # CLEARML MUST BE IMPORTED BEFORE argparse.parse_args(). it patches argparse at import, and
    # that patch is what feeds a cloned task's Args/* into the parser. importing it afterwards
    # silently loses every override -- that bug cost the scored-tables step every pipeline run.
    try:
        from clearml import Dataset, Task      # noqa: F401
    except ImportError:
        Dataset = Task = None

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_task_id", default="", help="training task whose model we check")
    ap.add_argument("--dataset_version", default="")
    ap.add_argument("--bundle", default="", help="a model_*.joblib on disk (local mode)")
    ap.add_argument("--data", default="", help="the dataset parquet (local mode)")
    ap.add_argument("--sample", type=int, default=C.DEEPCHECKS_SAMPLE,
                    help="rows per split (0 = all). deepchecks is slow on the full 513k.")
    ap.add_argument("--suites", default="data,model",
                    help="which suites: 'data' (integrity + train/test), 'model' (evaluation), "
                         "or both. only 'model' depends on the trained model.")
    ap.add_argument("--out", default="deepchecks_out")
    a = ap.parse_args()

    # ---------- LOCAL MODE ----------
    if a.bundle and a.data:
        print(f"[local] loading {pathlib.Path(a.bundle).name}")
        bundle = load_model_bundle(a.bundle)
        version = a.dataset_version or str(bundle.get("dataset_version", "?"))
        df = pd.read_parquet(a.data)
        print(f"[local] {len(df):,} rows   model trained on v{version}")
        train_ds, test_ds, n_tr, n_te = build_datasets(df, bundle, a.sample)
        print(f"        train {n_tr:,} rows   test {n_te:,} rows"
              f"{'   (sampled)' if a.sample else ''}")
        tag = f"{bundle.get('model_type', 'model')}_v{version}"
        run_suites(train_ds, test_ds, bundle["model"], tag, a.out, task=None,
                   which=[x.strip() for x in a.suites.split(",") if x.strip()])
        print(f"\ndone (local). open the html files in {a.out}/ .")
        return

    # ---------- PIPELINE MODE ----------
    task = Task.init(project_name=C.CLEARML_PROJECT,
                     task_name=getattr(C, "BASE_DEEPCHECKS_NAME", "deepchecks_report (base)"),
                     task_type=Task.TaskTypes.qc,
                     output_uri=C.deepchecks_output_uri())
    if not a.model_task_id:
        print("no --model_task_id: base-task registration run. exiting cleanly.")
        task.close()
        return

    print(f"[1/4] fetching the model from task {a.model_task_id}")
    src = Task.get_task(task_id=a.model_task_id)
    if src is None or "model" not in src.artifacts:
        raise SystemExit(f"task {a.model_task_id} has no 'model' artifact -- nothing to check.")
    bundle = load_model_bundle(src.artifacts["model"].get_local_copy())
    version = a.dataset_version or str(bundle.get("dataset_version", "?"))

    print(f"[2/4] fetching dataset {bundle['dataset_id']}")
    ds = Dataset.get(dataset_id=bundle["dataset_id"], alias="deepchecks_data")
    local = pathlib.Path(ds.get_local_copy())
    df = pd.read_parquet(find_dataset_parquet(local, bundle["dataset_id"]))
    print(f"      {len(df):,} rows   model trained on v{version}")

    print("[3/4] rebuilding the split the model trained with")
    train_ds, test_ds, n_tr, n_te = build_datasets(df, bundle, a.sample)
    print(f"      train {n_tr:,} rows   test {n_te:,} rows"
          f"{'   (sampled from the full split)' if a.sample else ''}")

    mtype = bundle.get("model_type", "model")
    print("[4/4] running the suites")
    run_suites(train_ds, test_ds, bundle["model"], f"{mtype}_v{version}", a.out, task=task,
               which=[x.strip() for x in a.suites.split(",") if x.strip()])
    _lbl = bundle.get("labels_name")
    task.add_tags([mtype, "deepchecks", f"v{version}"] + ([str(_lbl)] if _lbl else []))
    print(f"\ndone. the html reports are artifacts on this task, under "
          f"{C.deepchecks_output_uri()} . download one and open it in a browser.")
    task.close()


if __name__ == "__main__":
    main()
