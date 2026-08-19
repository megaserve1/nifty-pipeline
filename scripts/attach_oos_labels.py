"""scripts/attach_oos_labels.py -- put the target labels INTO the OOS parquet.

WHY THIS EXISTS
    The OOS file is 558 feature columns and no target. So score_dataset() takes the
    `has_label = "primary_label" in df.columns` branch as False and writes true_label=None for
    every row -- measured on scored_oos_2025jf_catboost v7: 106,500 of 106,500 null. Everything
    that needs truth then reports nothing: accuracy, and in the 2026-08 backtest also churn,
    missed opportunity, wrong direction and delay.

    The labels are NOT the problem. L1/L2/L3 run 2015-01-09 -> 2026-02-24 and cover 106,500 of
    106,500 OOS rows -- 100%. They were simply never merged in.

    A worker cannot fix it at score time either: data/labels/*.csv is gitignored (.gitignore:41),
    so an agent that clones the repo has no L1.csv to join against. The labels have to travel
    INSIDE the parquet.

WHY ALL THREE SETS AT ONCE
    Truth depends on which set the model trained under. L3 is SIX classes, L1/L2 are seven. Bake
    in L1 only, score an L3 model, and true_label would be quietly wrong -- and it would read as
    an ordinary pile of mistakes, not as a bug. So we write one column per set and let
    score_dataset pick by the bundle's labels_name.

    Three string columns on 106,500 rows is well under a megabyte on a 150 MB file.

RUN -- dry run first, it writes nothing:
    final_venv/bin/python scripts/attach_oos_labels.py \
        --oos "/home/megaserve/Downloads/VERSIONS/OOS_01012025_24022025_DATASET.parquet"

then, to actually write:
    final_venv/bin/python scripts/attach_oos_labels.py \
        --oos "/home/megaserve/Downloads/VERSIONS/OOS_01012025_24022025_DATASET.parquet" \
        --out data/oos/OOS_labelled.parquet
"""
import argparse
import hashlib
import json
import pathlib
import sys

import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as C   # noqa: E402


def label_time(s: pd.Series) -> pd.Series:
    """the label CSVs are ISO 'YYYY-MM-DD HH:MM:SS'.

    PARSE THEM AS ISO, NEVER dayfirst. with dayfirst=True pandas reads 2026-02-24 as 2026-12-02,
    the join then misses almost everything, and the only symptom is a low match count -- which is
    easy to read as "the labels do not cover this period". that exact misread happened on
    2026-08-06.
    """
    return pd.to_datetime(s, format="ISO8601")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos", required=True, help="the OOS parquet to attach labels to")
    ap.add_argument("--out", default="", help="where to write. omit = dry run, writes nothing")
    ap.add_argument("--sets", default="L1,L2,L3", help="which label handles to attach")
    a = ap.parse_args()

    src = pathlib.Path(a.oos).expanduser()
    if not src.exists():
        raise SystemExit(f"not found: {src}")

    reg = yaml.safe_load((C.LABELS_DIR / "registry.yaml").read_text())
    handles = [h.strip() for h in a.sets.split(",") if h.strip()]
    for h in handles:
        if h not in reg:
            raise SystemExit(f"{h} is not in {C.LABELS_DIR/'registry.yaml'} -- have {list(reg)}")

    print(f"oos file : {src}")
    df = pd.read_parquet(src)

    # the handed-over files store the time as a named INDEX ('datetime'), not a column -- same
    # trap normalise_time_column() exists for. reset it or the join has nothing to join on.
    if not any(c.lower() in ("timestamp", "datetime", "ts") for c in df.columns):
        df = df.reset_index()
    tcol = next((c for c in df.columns if c.lower() in ("timestamp", "datetime", "ts")), None)
    if tcol is None:
        raise SystemExit(f"no time column or index found. columns: {list(df.columns)[:8]}")
    ts = pd.to_datetime(df[tcol])
    print(f"           {len(df):,} rows, {len(df.columns)} cols, time in {tcol!r}")
    print(f"           {ts.min()}  ->  {ts.max()}")

    dup = int(ts.duplicated().sum())
    if dup:
        raise SystemExit(f"{dup:,} duplicate timestamps in the OOS file -- the join would fan out. "
                         f"de-duplicate it first.")

    existing = [c for c in df.columns if c == C.LABEL_COL or c.startswith(C.LABEL_COL + "_")]
    if existing:
        print(f"           already carries {existing} -- REPLACING them")

    out = df.copy()
    ok = True
    used_shas = {}                 # handle -> sha256, stamped into the output. see label_guard.
    for h in handles:
        meta = reg[h]
        lf = C.LABELS_DIR / meta["file"]
        if not lf.exists():
            raise SystemExit(f"{h}: {lf} is missing")

        # the sha256 in the registry is the REAL identity of a label set -- names get reused, files
        # get swapped. if it does not match, the labels we are about to bake in are not the ones
        # anything was certified against, and every downstream number would be quietly off.
        got, wantsha = sha256(lf), str(meta.get("sha256", ""))
        if wantsha and got != wantsha:
            raise SystemExit(f"{h}: {lf.name} sha256 {got[:12]} != registry {wantsha[:12]} -- the "
                             f"file on disk is NOT the label set the registry describes. stop.")

        used_shas[h] = got
        lab = pd.read_csv(lf, usecols=["timestamp", C.LABEL_COL])
        lab["timestamp"] = label_time(lab["timestamp"])
        # 6 of the 7 class strings carry a trailing space in the raw csv. strip once, here.
        lab[C.LABEL_COL] = lab[C.LABEL_COL].astype(str).str.strip()

        m = pd.Series(ts).map(lab.set_index("timestamp")[C.LABEL_COL])
        hit = m.notna()
        out[f"{C.LABEL_COL}_{h}"] = pd.array(m.to_numpy(), dtype="string")

        classes = sorted(m.dropna().unique())
        flag = "" if hit.all() else "   <-- NOT FULL COVERAGE"
        print(f"  {h:<3} {lf.name:<8} matched {int(hit.sum()):>7,}/{len(ts):,} "
              f"({hit.mean()*100:5.1f}%)  {len(classes)} classes{flag}")
        if not hit.all():
            miss = ts[~hit.to_numpy()]
            print(f"      unmatched span {miss.min()} -> {miss.max()}")
            ok = False
        n_want = int(meta.get("n_classes", len(classes)))
        if len(classes) != n_want:
            print(f"      !! {len(classes)} classes found, registry says {n_want}: {classes}")
            ok = False

    if not a.out:
        print("\nDRY RUN -- nothing written. add --out <path> to write the file.")
        return 0 if ok else 1

    dst = pathlib.Path(a.out).expanduser()
    dst.parent.mkdir(parents=True, exist_ok=True)
    # STAMP THE SHAS INTO THE FILE. a label column is just words -- there is no way to look at
    # "EXIT_SUPER" and tell which L1 wrote it. without this the file cannot be checked later, and
    # that is exactly how a replaced L1 went unnoticed for ten days (2026-08-18).
    import datetime as _dt
    import pyarrow as _pa, pyarrow.parquet as _pq
    from trainer.label_guard import STAMP_KEY, BUILT_KEY
    _t = _pa.Table.from_pandas(out, preserve_index=False)
    _md = dict(_t.schema.metadata or {})
    _md[STAMP_KEY] = json.dumps(used_shas).encode()
    _md[BUILT_KEY] = _dt.datetime.now().isoformat(timespec="seconds").encode()
    _pq.write_table(_t.replace_schema_metadata(_md), dst)
    grew = (dst.stat().st_size - src.stat().st_size) / 1e6
    print(f"\nwrote {dst}")
    print(f"      {len(out):,} rows, {len(out.columns)} cols, "
          f"{dst.stat().st_size/1e6:.0f} MB  ({grew:+.1f} MB)")
    print(f"\nnext -- THIS UPLOADS TO GCS AND COSTS STORAGE:")
    print(f"  final_venv/bin/python scripts/upload_reference.py \\")
    print(f"      --file {dst} --name {C.CLEARML_OOS_DATASET} --go")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
