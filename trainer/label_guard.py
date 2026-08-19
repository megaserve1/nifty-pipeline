"""trainer/label_guard.py -- refuse to score an OOS file whose truth column is out of date.

THE INCIDENT THIS EXISTS FOR (2026-08-18)
    the OOS set was labelled on 06 Aug. L1.csv was REPLACED on 08 Aug -- same handle, different
    file, NO_TRADE went from 78% to 88%. nothing re-checked the OOS afterwards, so v7.4, v10 and
    v11 were all trained on the new L1 and then scored against the old one. 10.3% of OOS minutes
    carried the wrong truth. every OOS backtest, drawdown and losing-streak number from those
    three runs was measured against labels the models had never seen.

    it went unnoticed for ten days because the OOS file looks perfect from the outside: the
    column is there, it is full, it has seven classes, and 90% of it is even correct.

WHY A STAMP AND NOT A RE-CHECK
    the file cannot be checked by looking at it -- a label column is just words, and there is no
    way to tell "EXIT_SUPER" written from the old L1 from one written from the new one. so the
    builder RECORDS the sha256 of every label file it used, into the parquet's own metadata, and
    this reads it back and compares against the registry as it stands today.

    an OOS file with no stamp is REFUSED, not warned about. an unverifiable file is exactly what
    caused the incident, and a warning in a log is what let it run for ten days.
"""
from __future__ import annotations

import json
import pathlib

STAMP_KEY = b"label_shas"
BUILT_KEY = b"labels_attached_at"


def read_stamp(parquet_path) -> dict:
    """{handle: sha256} the file was built from, or {} if it carries no stamp."""
    import pyarrow.parquet as pq
    md = pq.ParquetFile(str(parquet_path)).schema_arrow.metadata or {}
    try:
        return json.loads(md.get(STAMP_KEY, b"{}").decode())
    except Exception:
        return {}


def registry_sha(handle: str) -> str:
    import yaml
    import config as C
    reg = yaml.safe_load((C.LABELS_DIR / "registry.yaml").read_text()) or {}
    return str((reg.get(handle) or {}).get("sha256", ""))


def check(parquet_path, handle: str, strict: bool = True) -> None:
    """raise unless the OOS file's truth column for `handle` came from today's label file."""
    import config as C
    p = pathlib.Path(parquet_path)
    want = registry_sha(handle)
    stamp = read_stamp(p)
    rebuild = (f"  rebuild it:\n"
               f"    final_venv/bin/python scripts/attach_oos_labels.py \\\n"
               f"        --oos {p} --sets {handle} --out <new path>\n"
               f"  then upload it as a new {C.CLEARML_OOS_DATASET} version.")

    if not stamp:
        msg = (f"the OOS file carries NO label stamp, so there is no way to tell which label "
               f"file its truth column was built from.\n"
               f"  {p}\n"
               f"  a file built before 2026-08-18 has no stamp. one of those was scored against "
               f"a replaced L1 for ten days without anyone noticing.\n{rebuild}")
        if strict:
            raise SystemExit(msg)
        print(f"      !! {msg}")
        return

    got = str(stamp.get(handle, ""))
    if not got:
        raise SystemExit(f"the OOS file was stamped, but not for {handle} -- it carries "
                         f"{sorted(stamp)}.\n{rebuild}")
    if want and got != want:
        raise SystemExit(
            f"STALE OOS LABELS. the truth column for {handle} was built from label file "
            f"{got[:12]}, but the registry now says {handle} is {want[:12]}.\n"
            f"  {p}\n"
            f"  the file was replaced under the same handle, so the column looks fine and is "
            f"quietly wrong. scoring against it would measure the model on labels it never "
            f"trained on.\n{rebuild}")
    print(f"      OOS labels verified: {handle} sha {got[:12]} matches the registry")
