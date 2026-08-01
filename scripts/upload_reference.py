"""scripts/upload_reference.py -- put a REFERENCE dataset (OOS, OHLCV) on GCS + ClearML.

Reference data is not built by the pipeline -- it arrives as a file, gets a name, and every
worker resolves it by that name. That is why it does not go through publish_version.py: there is
no recipe, no manifest and no training to trigger.

    final_venv/bin/python scripts/upload_reference.py --file <path> --name nifty_oos_dataset
    final_venv/bin/python scripts/upload_reference.py --file <path> --name nifty_oos_dataset --go

Without --go it only inspects and reports. --go copies to GCS and publishes.

THE ORDER IS NOT NEGOTIABLE: add_external_files -> upload -> finalize -> PUBLISH.
Stopping at finalize() leaves the dataset 'completed', and anything waiting on 'published' waits
for ever with no error. (clearml/automation/trigger.py filters on status='published'.)

THE BYTES GO TO OUR BUCKET, never files.clear.ml. output_uri points at gs://<bucket>/datasets and
ClearML stores the file there -- the SAME layout the existing nifty_ohlcv and nifty_oos_dataset
already use, so get_local_copy() on a worker behaves identically to what is working today.
"""
import argparse
import pathlib
import sys

import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as C   # noqa: E402


def next_version(project: str, name: str) -> str:
    """1.0 -> 1.1 -> 1.2. numeric strings only: ClearML orders versions with PEP440, so 'v2'
    is not a version and '10' sorts after '9' only if both are numeric."""
    from clearml import Dataset
    try:
        cur = Dataset.get(dataset_project=project, dataset_name=name).version or "1.0"
    except Exception:
        return "1.0"
    try:
        maj, mn = str(cur).split(".")[:2]
        return f"{int(maj)}.{int(mn) + 1}"
    except Exception:
        return "1.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="the parquet to upload")
    ap.add_argument("--name", required=True, help="ClearML dataset name, e.g. nifty_oos_dataset")
    ap.add_argument("--project", default="", help=f"default: {C.CLEARML_PROJECT}")
    ap.add_argument("--go", action="store_true", help="actually copy + publish (costs storage)")
    a = ap.parse_args()

    src = pathlib.Path(a.file).expanduser()
    if not src.exists():
        raise SystemExit(f"not found: {src}")
    project = a.project or C.CLEARML_PROJECT

    pf = pq.ParquetFile(str(src))
    cols = [c.name for c in pf.schema_arrow]
    print(f"file    : {src}")
    print(f"          {src.stat().st_size/1e6:.1f} MB   {pf.metadata.num_rows:,} rows   "
          f"{len(cols)} columns")

    # COVERAGE CHECK, before anything is uploaded. an OOS file that is missing even one column a
    # model trained on cannot score that model -- and finding that out after the upload, on an
    # agent, is a wasted round trip.
    import yaml
    reg = yaml.safe_load(C.REGISTRY.read_text()) or {}
    have = set(cols)
    print(f"\ncovers the registered feature sets:")
    for name, meta in reg.items():
        want = set(meta.get("columns") or [])
        miss = want - have
        print(f"   {name:<12} {len(want):>3} features   missing {len(miss)}"
              + ("" if not miss else f"   <-- {sorted(miss)[:4]}"))

    version = next_version(project, a.name)
    dest = f"gs://{C.GCS_BUCKET}/datasets/{project}/.datasets/{a.name}/"
    print(f"\nproject : {project}")
    print(f"name    : {a.name}   version {version}")
    print(f"bucket  : {dest}")

    if not a.go:
        print("\nDRY RUN -- nothing copied, nothing published. add --go to do it.")
        return

    print(f"\nClearML: create -> add file -> upload -> finalize -> PUBLISH")
    from clearml import Dataset
    # output_uri is REQUIRED. without it ClearML uploads to files.clear.ml and the data leaves
    # our GCP -- the one thing that must never happen.
    ds = Dataset.create(dataset_project=project, dataset_name=a.name, dataset_version=version,
                        output_uri=f"gs://{C.GCS_BUCKET}/datasets")
    ds.add_files(path=str(src))
    ds.set_metadata({"source_file": src.name, "rows": pf.metadata.num_rows,
                     "columns": len(cols)}, metadata_name="summary")
    ds.upload()      # clears the dirty flag; sends no bytes for an external link
    ds.finalize()    # -> 'completed'.  NOT enough on its own.
    ds.publish()     # -> 'published'.  the state everything else waits for.
    print(f"\npublished  {a.name} v{version}   id {ds.id}")
    print(f"resolve it by NAME, never by id -- config.resolve_dataset_id('{a.name}')")


if __name__ == "__main__":
    main()
