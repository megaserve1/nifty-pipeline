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
import datetime
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


def normalise_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """make sure the frame has a 'timestamp' column, whatever the feature team called it.

    the pipeline reads C.LABEL_TS_COL ("timestamp") everywhere -- the split, the scored table, the
    signal csv. the handed-over feature files (MASTER_OOS, V7/V8/V9) call it 'datetime' instead, so
    without this every OOS run dies on a bare KeyError: 'timestamp'. renaming here keeps that name
    in exactly one place instead of sprinkling aliases through the scoring code.
    """
    if C.LABEL_TS_COL in df.columns:
        return df

    # THE INDEX CASE, which is what the handed-over files actually do: MASTER_OOS and V7/V8/V9 store
    # the time as a named DatetimeIndex ('datetime'), NOT as a column -- so it never shows up in
    # df.columns and a plain rename finds nothing. reset it into a column first.
    if isinstance(df.index, pd.DatetimeIndex) or str(df.index.name or "").lower() in (
            "datetime", "timestamp", "date", "ts", "time"):
        print(f"      time is the INDEX ({df.index.name!r}) -> moving it to a "
              f"{C.LABEL_TS_COL!r} column")
        out = df.reset_index()
        first = out.columns[0]
        return out.rename(columns={first: C.LABEL_TS_COL}) if first != C.LABEL_TS_COL else out

    for alt in ("datetime", "date", "ts", "time"):
        if alt in df.columns:
            print(f"      time column is {alt!r} -> using it as {C.LABEL_TS_COL!r}")
            return df.rename(columns={alt: C.LABEL_TS_COL})

    raise SystemExit(f"no time column found. expected {C.LABEL_TS_COL!r} or one of "
                     f"datetime/date/ts/time, as a column or the index. "
                     f"got columns: {list(df.columns)[:6]} index: {df.index.name!r}")


def preflight(parquet_path, bundle: dict) -> None:
    """check an OOS parquet against the model BEFORE scoring. reads the file FOOTER only -- no rows,
    so it costs nothing and fails in a second instead of after an hour.

    it catches the two mismatches that would otherwise produce a perfect-looking, meaningless table:
      1. a feature the model needs is missing        -> prepare() would raise anyway, but late
      2. THE SILENT ONE: a column that was TEXT in training arrives as numbers (or the reverse).
         na_policy.encode_categoricals only maps object/str columns, so a flipped column skips the
         saved cat_map entirely and raw numbers go into the model. no error, wrong predictions.
    """
    import pyarrow.parquet as pq
    have = {f.name: str(f.type) for f in pq.ParquetFile(str(parquet_path)).schema_arrow}
    feats = list(bundle.get("features") or [])

    missing = [c for c in feats if c not in have]
    if missing:
        raise SystemExit(
            f"OOS set is missing {len(missing)} feature column(s) this model needs -- refusing to "
            f"score. first few: {missing[:10]}\n"
            f"(the OOS file must carry the SAME feature columns the model trained on.)")

    TEXTY = ("string", "large_string", "binary", "large_binary", "bool")
    cats = set(bundle.get("categorical") or [])
    flips = []
    for c in feats:
        is_text_now = str(have[c]).startswith(TEXTY)
        if (c in cats) != is_text_now:
            flips.append(f"{c} (training={'text' if c in cats else 'numeric'}, oos={have[c]})")
    if flips:
        raise SystemExit(
            f"dtype flip vs training on {len(flips)} column(s) -- refusing to score, because the "
            f"saved category mapping would NOT be applied and the predictions would be silently "
            f"wrong. first few: {flips[:10]}")

    extra = [c for c in have if c not in feats]
    print(f"      preflight OK: {len(feats)} feature columns match "
          f"({len(extra)} extra column(s) in the file, ignored)")


def score_dataset(df: pd.DataFrame, bundle: dict, oos: bool = False, meta: dict = None) -> tuple:
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

    ts = pd.to_datetime(df[C.LABEL_TS_COL])
    if oos:
        # OOS ROWS THE MODEL HAS NEVER SEEN -> the split is the CONSTANT "oos". never rebuild the
        # training split here: the bundle's fractions would carve a meaningless "train" slice out
        # of unseen data and the backtest would believe it.
        split = np.full(len(df), "oos", dtype=object)
        print("      split: oos (constant -- these rows were never part of training)")
    else:
        # ---- rebuild the split the model actually trained with, FROM THE BUNDLE ----
        sp = bundle.get("split") or {}
        val_frac = float(sp.get("val_fraction", C.VAL_FRACTION))
        test_frac = float(sp.get("test_fraction", C.TEST_FRACTION))
        embargo = int(sp.get("embargo_sessions", C.EMBARGO_SESSIONS))
        if not sp:
            print("      !! old bundle: no recorded split. rebuilding from CURRENT config -- if config "
                  "changed since training, the train/test labels below are for the WRONG rows.")
        tr, va, te, info = three_way_split(
            ts, val_frac, test_frac, embargo,
            strategy=sp.get("strategy"), bundle_minutes=sp.get("bundle_minutes"),
            seed=sp.get("seed"))
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
    pred = le.inverse_transform(pred_idx)

    # A REAL OOS SET USUALLY HAS NO LABELS YET -- the future has not happened. so the label column
    # is OPTIONAL: without it true_label and correct come back null and the backtest works off
    # predicted_label + the probabilities. (this used to be a bare KeyError, raised AFTER the whole
    # dataset had already been scored.)
    #
    # WHICH truth column. an OOS file prepared by scripts/attach_oos_labels.py carries ONE PER
    # LABEL SET -- primary_label_L1 / _L2 / _L3 -- because truth depends on which set the model
    # trained under. L3 is SIX classes, L1 and L2 are seven. so we pick by the bundle's own
    # labels_name. a training-era dataset has the plain column and falls straight through.
    want = str(bundle.get("labels_name") or "")
    tagged = sorted(c for c in df.columns if c.startswith(C.LABEL_COL + "_"))
    if want and f"{C.LABEL_COL}_{want}" in df.columns:
        label_col = f"{C.LABEL_COL}_{want}"
        print(f"      truth column: {label_col}  (this model trained on {want})")
    elif want and tagged:
        # DO NOT fall back to another set. a 6-class truth scored against a 7-class model just
        # looks like a model that makes a lot of mistakes -- nothing downstream can tell the
        # difference between wrong labels and a bad model. refuse instead.
        raise SystemExit(f"this model trained on {want}, but the file only carries {tagged}. "
                         f"rebuild it:  scripts/attach_oos_labels.py --sets {want}")
    else:
        label_col = C.LABEL_COL if C.LABEL_COL in df.columns else None

    has_label = label_col is not None
    if has_label:
        true = df[label_col].astype(str).str.strip().to_numpy()   # labels carry trailing spaces
        true_col = pd.array(true, dtype="string")
        correct_col = pd.array(np.where(pred == true, "YES", "no"), dtype="string")
    else:
        true_col = pd.array([None] * len(df), dtype="string")       # nullable STRING, not object:
        correct_col = pd.array([None] * len(df), dtype="string")    # object-None -> arrow 'null'
        print("      no label column -> true_label / correct are null (scoring does not need them)")

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
        "true_label": true_col,
        "predicted_label": pred,
        "correct": correct_col,
    })
    for i, cls in enumerate(classes):                  # one probability column per class
        tbl[f"proba_{cls}"] = proba[:, i]

    # ---- traceability: WHICH MODEL produced this table, and on WHICH rows ----------------------
    # constant per file, so parquet dictionary-encodes them for almost nothing. without these you
    # cannot tell two tables apart once they are concatenated, and "one table per model" is a
    # filename convention instead of a fact in the data. emitted in BOTH modes -> ONE contract.
    m = meta or {}
    tbl["model_task_id"] = str(m.get("model_task_id", ""))
    tbl["model_type"] = str(bundle.get("model_type", ""))
    tbl["train_dataset_id"] = str(bundle.get("dataset_id", ""))
    tbl["train_dataset_version"] = str(bundle.get("dataset_version", ""))
    tbl["scored_dataset_id"] = str(m.get("scored_dataset_id", bundle.get("dataset_id", "")))
    tbl["scored_dataset_version"] = str(m.get("scored_dataset_version", ""))
    tbl["scored_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

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
        if len(t) and t["correct"].notna().any():
            acc = (t["correct"] == "YES").mean()
            print(f"      wrote {p.name}  ({len(t):,} rows, raw-accuracy {acc:.4f})")
        else:
            print(f"      wrote {p.name}  ({len(t):,} rows, unlabelled -- no accuracy)")
        if task is not None:
            # upload the FILE (not the DataFrame) so a 500k-row table goes straight to the bucket
            # as one parquet blob, exactly like the model artifact does.
            task.upload_artifact(f"scored_{name}_v{version}", str(p))
    return paths


def write_oos(tbl, tag, model_type, version, out_dir, task=None):
    """ONE table for an OOS run (there is no train/test to cut -- every row is 'oos')."""
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    name = f"scored_oos_{tag or 'set'}_{model_type or 'model'}_v{version}"
    p = out_dir / f"{name}.parquet"
    tbl.to_parquet(p, index=False)
    if len(tbl) and tbl["correct"].notna().any():
        print(f"      wrote {p.name}  ({len(tbl):,} rows, raw-accuracy "
              f"{(tbl['correct'] == 'YES').mean():.4f})")
    else:
        print(f"      wrote {p.name}  ({len(tbl):,} rows, unlabelled -- no accuracy)")
    if task is not None:
        task.upload_artifact(name, str(p))       # the FILE, straight to the bucket
    return p


def run_backtest(script: str, table_path, price: str, out_dir, task=None) -> int:
    """run the backtest script on the table we just wrote, and PRINT everything it says.

    THE CALLING CONVENTION IS THEIRS, not ours (backtest_single.py, main()):
        python <script>  <signal_file>  <price_parquet>  <output_dir>
    three POSITIONAL arguments -- no flags.

    IT CANNOT READ PARQUET FOR THE SIGNAL. load_signal() only special-cases .xlsx/.xls and sends
    everything else to pd.read_csv, so we hand it a CSV. it needs exactly two of our columns:
      timestamp        (first hit in its TIME_COLS)
      predicted_label  (first hit in its SIG_COLS)
    our 7 class names are already what it validates against, and it upper-cases/strips anyway.

    printing is the whole trick: ClearML captures a task's stdout, so their report lands in this
    task's CONSOLE tab with no extra wiring. we never parse it -- we just show it.
    """
    import subprocess
    script_path = pathlib.Path(script)
    if not script_path.is_absolute():
        script_path = C.ROOT / script
    if not script_path.exists():
        print(f"      !! backtest script not found: {script_path} -- table is written, skipping "
              f"the backtest. (it must be committed in the repo so agents get it.)")
        return 1
    if not price:
        print("      !! no --price given -- the backtest needs the OHLC parquet. skipping.")
        return 1
    price_path = pathlib.Path(price)
    if not price_path.is_absolute():
        price_path = C.ROOT / price
    if not price_path.exists():
        print(f"      !! price file not found: {price_path} -- skipping the backtest.")
        return 1

    # parquet -> CSV, because their loader cannot read parquet signals.
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    signal_csv = out_dir / (pathlib.Path(table_path).stem + "_signal.csv")
    # HAND OVER THE TRUTH TOO, when we have it. the 2026-08 backtest measures churn, missed
    # opportunity, wrong direction and delay -- all of which need the true label. it finds it via
    # its own TRUTH_COLS list, which includes "true_label", and reports n/a for every one of them
    # if the column is absent. it also filters on "split" itself. both are optional: an OOS table
    # carries true_label all-null today, and the engine treats that as no truth.
    import pyarrow.parquet as _pq
    have = {f.name for f in _pq.ParquetFile(str(table_path)).schema_arrow}   # footer only, no rows
    extra = [c for c in ("true_label", "split") if c in have]
    sig = pd.read_parquet(table_path, columns=["timestamp", "predicted_label"] + extra)
    # an all-null true_label is worse than none: the engine would see the column, switch the cost
    # decomposition on, and report zeros as if they were measurements. drop it unless it has values.
    if "true_label" in sig.columns and not sig["true_label"].notna().any():
        sig = sig.drop(columns=["true_label"])
        print("      true_label is entirely null -> dropped. churn / missed / delay will read n/a.")
    sig.to_csv(signal_csv, index=False)
    print(f"      signal csv for the backtest: {signal_csv.name}  ({len(sig):,} rows, "
          f"columns {list(sig.columns)})")

    # COVERAGE CHECK. their engine inner-merges signal x price and says NOTHING when only part of
    # the window matches -- measured on the real files: the price parquet ends 2025-07-31 while
    # the master OOS runs to 2026-02-24, so HALF the OOS would be silently ignored and the report
    # would read as if it covered everything. refuse the nothing case, shout about the partial.
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(price_path))
        tcol = next((f.name for f in pf.schema_arrow
                     if f.name.lower() in ("timestamp", "datetime", "date", "ts")), None)
        if tcol:
            pt = pd.to_datetime(pd.read_parquet(price_path, columns=[tcol])[tcol])
            st = pd.to_datetime(sig["timestamp"])
            covered = int(((st >= pt.min()) & (st <= pt.max())).sum())
            pct = covered / max(len(st), 1) * 100
            if covered == 0:
                print(f"      !! ZERO overlap: signals {st.min()} -> {st.max()}, prices "
                      f"{pt.min()} -> {pt.max()}. skipping the backtest -- get a price file "
                      f"that covers the signal window.")
                return 1
            if pct < 90:
                print(f"      !! PARTIAL PRICE COVERAGE: only {covered:,} of {len(st):,} signal "
                      f"rows ({pct:.0f}%) fall inside the price file "
                      f"({pt.min().date()} -> {pt.max().date()}).")
                print(f"      !! the report below covers ONLY that slice -- do not read it as "
                      f"the full period. update the OHLCV file to fix this.")
            else:
                print(f"      price coverage: {covered:,}/{len(st):,} signal rows ({pct:.0f}%)")
    except Exception as exc:
        print(f"      (coverage check skipped: {exc})")

    bt_out = out_dir / "backtest"
    cmd = [sys.executable, "-u", str(script_path), str(signal_csv), str(price_path), str(bt_out)]
    print(f"\n{'=' * 70}\nBACKTEST  {' '.join(cmd[1:])}\n{'=' * 70}", flush=True)
    r = subprocess.run(cmd, cwd=str(C.ROOT), capture_output=True, text=True)
    out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
    print(out, flush=True)                      # -> ClearML CONSOLE tab
    print("=" * 70, flush=True)
    if task is not None:
        if out.strip():
            # park it as a report too, so it survives console truncation on a long run
            task.get_logger().report_text(out, print_console=False)
        # their output lands in a timestamped folder -- upload it or it dies with the worker.
        # RGLOB, not glob: the 3-version suite writes at the top AND one SUBFOLDER per version
        # (V1_.../metrics.csv, trades.csv, ...). a flat glob would drop most of it.
        # .XLSX IS NOT OPTIONAL: the 2026-08 engine writes ONE workbook and a full_report.txt, so
        # a csv+txt glob would upload the txt and silently lose every number.
        n_up = 0
        for d in sorted(bt_out.glob("backtest_*")):
            for f in (sorted(d.rglob("*.csv")) + sorted(d.rglob("*.txt"))
                      + sorted(d.rglob("*.xlsx"))):
                # name the artifact by its path inside the run folder, so V1/metrics and
                # V2/metrics do not overwrite each other under one name.
                rel = f.relative_to(d).with_suffix("")
                task.upload_artifact(f"backtest_{str(rel).replace('/', '_')}", str(f))
                print(f"      uploaded {rel}")
                n_up += 1
        # SAY SO WHEN NOTHING CAME BACK. on 2026-08-06 the backtest computed every number, printed
        # the whole report, then died writing the workbook (no openpyxl on the agent) -- exit 1,
        # zero files, and the task still finished COMPLETED with one lonely parquet on it. the tag
        # is what makes that visible in the task LIST, without failing a run whose table is fine.
        if n_up == 0:
            task.add_tags(["backtest_failed"])
            print(f"      !! the backtest produced NO files (exit {r.returncode}). tagged "
                  f"'backtest_failed'. the numbers are in the console above, but nothing was "
                  f"saved -- read the traceback at the end of its output.")
    if r.returncode != 0:
        print(f"      !! backtest exited {r.returncode} -- output above.")
    return r.returncode


def resolve_price(price: str, price_dataset_id: str, Dataset=None) -> str:
    """the OHLCV the backtest prices trades against. either a local path (--price) or a ClearML
    dataset on GCP (--price_dataset_id), so a worker needs nothing pre-staged on its disk."""
    if price_dataset_id and Dataset is not None:
        print(f"      fetching the OHLC prices {price_dataset_id}")
        px_local = pathlib.Path(Dataset.get(dataset_id=price_dataset_id,
                                            alias="price_data").get_local_copy())
        p = str(find_dataset_parquet(px_local, price_dataset_id))
        print(f"      prices: {pathlib.Path(p).name}")
        return p
    return price


def main():
    # CLEARML MUST BE IMPORTED BEFORE argparse.parse_args() RUNS. THIS IS NOT STYLE.
    # clearml patches argparse AT IMPORT; that patch is what injects a cloned task's Args/* into
    # the parser. import it afterwards and the injection never happens -- every Arg keeps its
    # argparse default, so a queued run saw model_task_id="" and exited as a "base registration
    # run", reporting COMPLETED while producing nothing. that is exactly what happened to
    # scored_tables_xgboost v6. train.py carries the same warning at its parser.
    # local mode does not need clearml, but importing it is harmless (no Task is created).
    try:
        from clearml import Dataset, Task      # noqa: F401  -- BEFORE the parser. see above.
    except ImportError:
        Dataset = Task = None                  # local mode still works without clearml installed

    ap = argparse.ArgumentParser()
    # pipeline mode (fetch from GCP, like shap_explain):
    ap.add_argument("--model_task_id", default="", help="training task whose model we score")
    ap.add_argument("--dataset_version", default="")
    # local mode (no ClearML):
    ap.add_argument("--bundle", default="", help="a model_*.joblib on disk (local mode)")
    ap.add_argument("--data", default="", help="the dataset parquet (local mode)")
    # OOS mode -- score a set the model has NEVER seen, for the backtest:
    # a STRING, not store_true: a cloned ClearML task hands every Arg back as a string, and a
    # store_true round-trips badly ("False" is truthy). a string compare is unambiguous.
    ap.add_argument("--mode", default="insample", choices=["insample", "oos"])
    ap.add_argument("--oos_dataset_id", default="", help="ClearML dataset id of the OOS set")
    ap.add_argument("--oos_data", default="", help="an OOS parquet on disk (local OOS mode)")
    ap.add_argument("--oos_tag", default="", help="short name for the file, e.g. 2025h2")
    ap.add_argument("--backtest", default="",
                    help="path to the backtest script to run on the table. its output is PRINTED, "
                         "so it appears in this task's ClearML console.")
    ap.add_argument("--price", default="",
                    help="the OHLC futures parquet the backtest prices trades against "
                         "(its 2nd positional argument). a LOCAL path.")
    ap.add_argument("--price_dataset_id", default="",
                    help="same file, but registered on GCP as a ClearML dataset -- the worker "
                         "fetches it. use this in the pipeline; --price is for local runs.")
    # both:
    ap.add_argument("--out", default="scored_out", help="where to write the two parquets")
    ap.add_argument("--strict_train", action="store_true",
                    help="train table = ONLY the train slice (drop val+embargo). default keeps "
                         "them so the two files cover the complete dataset.")
    a = ap.parse_args()

    # ---------- LOCAL OOS: bundle + an OOS parquet on disk, no ClearML ----------
    if a.mode == "oos" and a.bundle and a.oos_data:
        print(f"[local-oos] loading model bundle {pathlib.Path(a.bundle).name}")
        bundle = load_model_bundle(a.bundle)
        version = a.dataset_version or str(bundle.get("dataset_version", "?"))
        print(f"[local-oos] checking {a.oos_data} against the model")
        preflight(a.oos_data, bundle)
        df = normalise_time_column(pd.read_parquet(a.oos_data))
        print(f"        {len(df):,} OOS rows   model trained on v{version}")
        tbl, _ = score_dataset(df, bundle, oos=True,
                               meta={"scored_dataset_version": a.oos_tag or "oos"})
        p = write_oos(tbl, a.oos_tag, bundle.get("model_type", ""), version, a.out, task=None)
        if a.backtest:
            # same resolution as pipeline mode: an explicit --price wins; else the price dataset
            # (fetched via ClearML if importable); else config's PRICE_FILE. --price_dataset_id
            # used to be ACCEPTED here and silently ignored.
            price = resolve_price(a.price or getattr(C, "PRICE_FILE", ""),
                                  a.price_dataset_id or getattr(C, "PRICE_DATASET_ID", ""),
                                  Dataset)
            run_backtest(a.backtest, p, price, a.out, task=None)
        print(f"\ndone (local oos). table: {p}")
        return

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
        paths = write_and_report(train_tbl, test_tbl, version, a.out, task=None)
        if a.backtest:
            # the TEST table -- backtesting rows the model was fitted on is meaningless.
            # local mode has no ClearML, so the price file is a plain path (--price).
            print("\n[backtest] on the TEST split")
            run_backtest(a.backtest, paths["test"], a.price, a.out, task=None)
        print(f"\ndone (local). two tables in {a.out}/ . to also push them to GCS, run in pipeline "
              f"mode with --model_task_id, or hand them to the next team as-is.")
        return

    # ---------- PIPELINE MODE: fetch model + data from GCP, upload tables to GCS ----------
    from clearml import Dataset, Task   # imported AFTER parse_args on purpose -- clearml patches
    # argparse at import; parsing first means a cloned task's Args/ overrides are silently lost.
    _base_name = (getattr(C, "BASE_OOS_NAME", "score_oos (base)") if a.mode == "oos"
                  else getattr(C, "BASE_EXPORT_NAME", "export_scored_tables (base)"))

    # THE AGENT INSTALLS THE TASK'S RECORDED PACKAGES, NOT requirements.txt. clearml builds that
    # list by scanning what THIS SCRIPT IMPORTS -- and the backtest runs in a SUBPROCESS, so
    # nothing here ever imports its excel writer and clearml never recorded it. result on
    # 2026-08-06: the backtest computed every number, printed the whole report, then died on
    # "No module named 'openpyxl'" -- on two different agents, twice, while requirements.txt had
    # carried both since that morning. declaring them here is the only thing that reaches an agent.
    # MUST be before Task.init.
    for _pkg in ("openpyxl", "xlsxwriter"):
        try:
            Task.add_requirements(_pkg)
        except Exception:
            pass                            # older clearml, or a local run with no task
    task = Task.init(project_name=C.CLEARML_PROJECT,
                     task_name=_base_name,
                     task_type=Task.TaskTypes.qc,
                     output_uri=C.tables_output_uri())   # gcs mode -> gs://<bucket>/tables
    if not a.model_task_id:
        print("no --model_task_id: base-task registration run. exiting cleanly.")
        task.close()
        return

    # ---------- PIPELINE OOS: model from a training task, rows from the OOS dataset ----------
    if a.mode == "oos":
        if not a.oos_dataset_id:
            raise SystemExit("--mode oos needs --oos_dataset_id (the ClearML dataset id of the "
                             "OOS set). it must NOT be the training dataset.")
        print(f"[1/5] fetching the model from task {a.model_task_id}")
        src = Task.get_task(task_id=a.model_task_id)
        if src is None or "model" not in src.artifacts:
            raise SystemExit(f"task {a.model_task_id} has no 'model' artifact -- nothing to score.")
        bundle = load_model_bundle(src.artifacts["model"].get_local_copy())
        version = a.dataset_version or str(bundle.get("dataset_version", "?"))

        print(f"[2/5] fetching the OOS dataset {a.oos_dataset_id}")
        oos_ds = Dataset.get(dataset_id=a.oos_dataset_id, alias="oos_data")
        oos_local = pathlib.Path(oos_ds.get_local_copy())
        oos_parquet = find_dataset_parquet(oos_local, a.oos_dataset_id)

        print("[3/5] preflight -- do the OOS columns match what the model trained on?")
        preflight(oos_parquet, bundle)          # cheap footer read; refuses on a mismatch
        df = normalise_time_column(pd.read_parquet(oos_parquet))
        print(f"      {len(df):,} OOS rows   model trained on v{version}")

        print("[4/5] scoring")
        tbl, _ = score_dataset(df, bundle, oos=True, meta={
            "model_task_id": a.model_task_id,
            "scored_dataset_id": a.oos_dataset_id,
            "scored_dataset_version": a.oos_tag or getattr(oos_ds, "version", "") or "oos",
        })
        mtype = bundle.get("model_type", "")
        p = write_oos(tbl, a.oos_tag, mtype, version, a.out, task=task)
        _lbl = bundle.get("labels_name")
        task.add_tags([mtype or "?", "scored_oos", a.oos_tag or "oos", f"v{version}"]
                      + ([str(_lbl)] if _lbl else []))

        print("[5/5] backtest")
        # the OHLC prices live on GCP as their own ClearML dataset -- pull them the same way we
        # pulled the OOS rows, so any worker can run this with nothing pre-staged on its disk.
        price_path = resolve_price(a.price, a.price_dataset_id, Dataset)
        if a.backtest:
            run_backtest(a.backtest, p, price_path, a.out, task=task)  # output -> CONSOLE tab
        else:
            print("      no --backtest given -- table written, nothing to run.")
        print(f"\ndone. table '{p.name}' is an artifact on this task, under "
              f"{C.tables_output_uri()} .")
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
    paths = write_and_report(train_tbl, test_tbl, version, a.out, task=task)
    _lbl = bundle.get("labels_name")
    task.add_tags([bundle.get("model_type", "?"), "scored_tables", f"v{version}"]
                  + ([str(_lbl)] if _lbl else []))

    # BACKTEST THE TEST TABLE, not the train one -- backtesting rows the model was fitted on is
    # meaningless. its report prints straight into this task's CONSOLE tab.
    if a.backtest:
        print("[5/5] backtest (on the TEST split)")
        run_backtest(a.backtest, paths["test"],
                     resolve_price(a.price, a.price_dataset_id, Dataset), a.out, task=task)
    print(f"\ndone. scored_train_v{version} + scored_test_v{version} are artifacts on this task, "
          f"in gs://{C.GCS_BUCKET}/clearml . the next team reads them from there.")
    task.close()


if __name__ == "__main__":
    main()
