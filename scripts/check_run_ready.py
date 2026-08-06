"""scripts/check_run_ready.py -- is THIS run wired up? read-only, spends nothing.

preflight_check.sh checks the MACHINE (clearml.conf, GCS auth, libraries, queue). This checks the
RUN: that the code you edited is the code the workers will execute, that the hyperparameters you
uploaded are the ones that will be used, and that the OOS set carries truth.

Every check here exists because its failure is SILENT -- the run goes green and produces the wrong
thing. Stale base tasks are the worst of them: the model trains fine and every follow-on task
quietly executes last week's code.

    final_venv/bin/python scripts/check_run_ready.py
    final_venv/bin/python scripts/check_run_ready.py --model catboost
"""
import argparse
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as C   # noqa: E402

P, F, W = [], [], []


def ok(m): P.append(m); print(f"  PASS  {m}")
def no(m): F.append(m); print(f"  FAIL  {m}")
def warn(m): W.append(m); print(f"  warn  {m}")
def head(m): print(f"\n== {m} ==")


def git(*a):
    return subprocess.run(["git", *a], cwd=str(C.ROOT), capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="catboost")
    a = ap.parse_args()

    head("1. the code you are about to run is pushed")
    dirty = git("status", "--porcelain")
    if dirty:
        no(f"{len(dirty.splitlines())} uncommitted change(s) -- workers clone from git, "
           f"they will NOT see them")
        for l in dirty.splitlines()[:6]:
            print(f"          {l}")
    else:
        ok("working tree clean")
    hd, up = git("rev-parse", "HEAD"), git("rev-parse", "@{u}")
    if not up:
        warn("no upstream branch set -- cannot tell if HEAD is pushed")
    elif hd == up:
        ok(f"HEAD {hd[:12]} is pushed")
    else:
        no(f"HEAD {hd[:12]} != origin {up[:12]} -- git push first")

    head("2. the new backtest is installed")
    bt = C.ROOT / C.BACKTEST_SCRIPT
    if not bt.exists():
        no(f"{C.BACKTEST_SCRIPT} is missing")
    else:
        src = bt.read_text()
        markers = ["TRUTH_COLS", "COMPUTE_COST_DECOMP", "METRIC_DOC", "_wilder_atr",
                   "cost_decomposition"]
        gone = [m for m in markers if m not in src]
        if gone:
            no(f"still the OLD backtest -- missing {gone}. "
               f"cp /home/megaserve/Downloads/backtest_single.py {C.BACKTEST_SCRIPT}")
        else:
            ok(f"new backtest in place ({len(src.splitlines()):,} lines)")
    try:
        import xlsxwriter  # noqa: F401
        ok("xlsxwriter -> formatted workbook")
    except ImportError:
        warn("no xlsxwriter -> openpyxl fallback. workbook is written, just unformatted. "
             "final_venv/bin/pip install xlsxwriter")

    head("3. our side is wired to it")
    ex = (C.ROOT / "trainer/export_scored_tables.py").read_text()
    ok('signal csv passes true_label + split') if '"true_label", "split"' in ex \
        else no("signal csv still 2 columns -- churn/missed/delay will read n/a")
    ok("xlsx is uploaded as an artifact") if 'rglob("*.xlsx")' in ex \
        else no("upload glob has no *.xlsx -- the workbook would die with the worker")

    head(f"4. the hyperparameters {a.model} will actually train on")
    try:
        from trainer import hyperparams as H
        import yaml as _y
        raw = (_y.safe_load(H.HP_FILE.read_text()) or {}).get(a.model, {}).get("default") or {}
        eff = H.defaults(a.model)
        diff = {k: (raw.get(k), v) for k, v in eff.items() if raw.get(k) != v}
        if diff:
            ok(f"{len(diff)} tuned value(s) in effect from configs/tuned/{a.model}.json")
            for k, (was, now) in sorted(diff.items()):
                print(f"          {k}: yaml {was!r} -> {now!r}")
        else:
            warn(f"NO tuned overlay -- {a.model} will train on the plain yaml. if you meant to "
                 f"upload params, do it on the Build Dataset page step 4.")
        esr = eff.get("early_stopping_rounds")
        if esr in (None, 0, "0"):
            warn(f"early_stopping_rounds={esr!r} -- 0 means OFF, all {eff.get('n_estimators')} "
                 f"iterations will be built whatever validation does")
        else:
            ok(f"early stopping armed at {esr}")
    except Exception as exc:
        no(f"could not resolve hyperparameters: {exc}")

    head("5. base tasks -- what the WORKERS will run")
    try:
        from clearml import Task
        names = [getattr(C, n) for n in dir(C) if n.startswith("BASE_") and n.endswith("NAME")]
        names += [f"train_{m} (base)" for m in ("xgboost", "catboost")]
        for n in sorted(set(names)):
            t = Task.get_task(project_name=C.CLEARML_PROJECT, task_name=n)
            if t is None:
                warn(f"{n}: not registered")
                continue
            c = str(t.data.script.version_num or "")
            if c[:12] == hd[:12]:
                ok(f"{n}  {c[:12]}")
            else:
                no(f"{n}  {c[:12] or '(none)'} != HEAD {hd[:12]} -- run "
                   f"trainer/register_base_trainer.py --force")
    except Exception as exc:
        no(f"could not reach ClearML: {exc}")

    head("6. the OOS set carries truth")
    try:
        import pyarrow.parquet as pq
        from clearml import Dataset
        ds = sorted(Dataset.list_datasets(dataset_project=C.CLEARML_PROJECT, partial_name="oos"),
                    key=lambda x: str(x.get("created")))
        if not ds:
            no("no OOS dataset registered")
        else:
            d = Dataset.get(dataset_id=ds[-1]["id"])
            print(f"        newest: {ds[-1]['name']} v{ds[-1]['version']}")
            found = False
            for p in pathlib.Path(d.get_local_copy()).rglob("*"):
                if not p.is_file():
                    continue
                try:
                    cols = [f.name for f in pq.ParquetFile(str(p)).schema_arrow]
                except Exception:
                    continue
                found = True
                lab = sorted(c for c in cols if c.startswith(C.LABEL_COL))
                if lab:
                    ok(f"truth columns present: {lab}")
                else:
                    warn("no label column -> true_label stays null and churn / missed "
                         "opportunity / wrong direction / delay all read n/a. fix: "
                         "scripts/attach_oos_labels.py then upload_reference.py --go")
            if not found:
                no("no parquet inside the OOS dataset")
    except Exception as exc:
        warn(f"OOS check skipped: {exc}")

    head("7. is anyone listening?")
    try:
        from clearml.backend_api.session.client import APIClient
        ws = APIClient().workers.get_all()
        live = [w for w in ws if C.TRAIN_QUEUE in
                [q.name for q in (getattr(w, "queues", None) or [])]]
        if live:
            ok(f"{len(live)} agent(s) on '{C.TRAIN_QUEUE}': "
               f"{', '.join(str(w.id)[:22] for w in live)}")
        else:
            no(f"NO agent on '{C.TRAIN_QUEUE}' -- everything queues and nothing runs. "
               f"final_venv/bin/clearml-agent daemon --queue {C.TRAIN_QUEUE}")
    except Exception as exc:
        warn(f"worker check skipped: {exc}")

    print(f"\n{'=' * 60}")
    print(f"  {len(P)} passed   {len(F)} FAILED   {len(W)} warning(s)")
    if F:
        print("\n  fix these before spending money:")
        for m in F:
            print(f"    - {m}")
    print("=" * 60)
    return 1 if F else 0


if __name__ == "__main__":
    raise SystemExit(main())
