"""
contract.py -- THE BOUNDARY between the two teams.

The bridge (feature/dataset production) hands the core exactly two files per version:

    datasets/vN/dataset_vN.parquet     the materialized training table
    datasets/vN/manifest.json          this contract: what it is, and proof of what it is

The core NEVER imports bridge code, never opens the labels CSV, and never re-derives
anything. It reads the manifest, verifies it, and publishes. If the manifest does not
validate, publishing STOPS -- a bad dataset can never reach the model.

This is the ONE module both sides import. On handover day the bridge takes a copy.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import hashes

MANIFEST_SCHEMA_VERSION = 1

# every key a valid manifest must carry
REQUIRED = [
    "manifest_schema_version", "version", "recipe_sha256", "parquet_sha256",
    "rows", "n_features", "schema", "feature_columns", "label_col", "weight_cols",
    "per_feature", "labels_name", "labels_sha256", "class_distribution",
    "no_peek", "built_at", "built_by", "status",
]


class ContractError(Exception):
    """Raised when a manifest is missing, malformed, or disagrees with the files."""


# ------------------------------------------------------------------ write side (bridge)
def write_manifest(path, **fields) -> dict:
    doc = {"manifest_schema_version": MANIFEST_SCHEMA_VERSION,
           "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           **fields}
    missing = [k for k in REQUIRED if k not in doc]
    if missing:
        raise ContractError(f"manifest is missing required keys: {missing}")
    Path(path).write_text(json.dumps(doc, indent=2, default=str))
    return doc


def load(path) -> dict:
    p = Path(path)
    if not p.exists():
        raise ContractError(f"no manifest at {p} -- the bridge must produce one "
                            f"(run: python bridge/build_dataset.py --version ...)")
    return json.loads(p.read_text())


# ------------------------------------------------------------------ read side (core)
def validate_manifest(man: dict, recipe_path, parquet_path) -> None:
    """The publish GATE. Raises ContractError on any disagreement.

    Checks, in order of what would hurt most:
      1. schema version we understand
      2. the manifest describes THIS recipe (recipe_sha256), so you cannot publish
         a parquet that was built from a different selection
      3. the parquet on disk is byte-identical to the one that was certified
      4. the no-peek rule was actually applied (not merely claimed absent)
      5. the build finished ('ready'), not half-written
    """
    recipe_path, parquet_path = Path(recipe_path), Path(parquet_path)

    v = man.get("manifest_schema_version")
    if v != MANIFEST_SCHEMA_VERSION:
        raise ContractError(f"manifest schema v{v}, this core understands "
                            f"v{MANIFEST_SCHEMA_VERSION}")

    missing = [k for k in REQUIRED if k not in man]
    if missing:
        raise ContractError(f"manifest missing keys: {missing}")

    if man["status"] != "ready":
        raise ContractError(f"manifest status is '{man['status']}', not 'ready' "
                            f"-- the build did not finish")

    if not recipe_path.exists():
        raise ContractError(f"recipe not found: {recipe_path}")
    got = hashes.sha256_file(recipe_path)
    if got != man["recipe_sha256"]:
        raise ContractError(
            "RECIPE MISMATCH -- this parquet was NOT built from this recipe.\n"
            f"  recipe on disk : {got}\n  manifest says  : {man['recipe_sha256']}\n"
            "  (rebuild: python bridge/build_dataset.py --version "
            f"{man['version']})")

    if not parquet_path.exists():
        raise ContractError(f"parquet not found: {parquet_path}")
    got = hashes.sha256_file(parquet_path)
    if got != man["parquet_sha256"]:
        raise ContractError(
            "PARQUET MISMATCH -- the file changed after it was certified.\n"
            f"  file on disk : {got}\n  manifest says: {man['parquet_sha256']}")

    np_ = man["no_peek"]
    if not np_.get("applied"):
        raise ContractError("manifest says no_peek.applied=false -- refusing to publish "
                            "a dataset that may contain lookahead")
    if np_.get("rule") != "bar_close":
        raise ContractError(f"unknown no_peek rule {np_.get('rule')!r}; this core only "
                            f"trusts 'bar_close'")

    if not man["feature_columns"]:
        raise ContractError("manifest lists zero feature columns")

    # THE CHEAP TRUTHS THE CERTIFICATE USED TO MERELY *CARRY*.
    # rows / n_features / column names were required keys but never VERIFIED against the
    # parquet -- a certificate that records claims without checking any of them. parquet
    # metadata gives all three without reading the data (pyarrow reads the footer only).
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(parquet_path)
    got_rows = int(meta.metadata.num_rows)
    if got_rows != int(man["rows"]):
        raise ContractError(f"ROWS MISMATCH: parquet has {got_rows:,}, manifest says "
                            f"{int(man['rows']):,}")
    got_cols = list(meta.schema_arrow.names)
    want_cols = [c["col"] for c in man["schema"]]
    if got_cols != want_cols:
        raise ContractError(
            "COLUMN MISMATCH between the parquet and its certificate:\n"
            f"  only in parquet : {[c for c in got_cols if c not in want_cols][:6]}\n"
            f"  only in manifest: {[c for c in want_cols if c not in got_cols][:6]}")
    n_feat = len(man["feature_columns"])
    if int(man["n_features"]) != n_feat:
        raise ContractError(f"n_features says {man['n_features']} but feature_columns lists "
                            f"{n_feat}")


def assert_schema(df, man: dict) -> None:
    """Trainer-side guard: the dataframe we loaded IS the one that was certified.

    Catches a feature notebook that silently renamed a column or changed a dtype
    between the build and the training run.
    """
    want = [(c["col"], c["dtype"]) for c in man["schema"]]
    got = [(c, str(t)) for c, t in zip(df.columns, df.dtypes)]
    if got != want:
        w, g = dict(want), dict(got)
        added = sorted(set(g) - set(w))
        removed = sorted(set(w) - set(g))
        retyped = sorted(c for c in set(w) & set(g) if w[c] != g[c])
        raise ContractError(
            "SCHEMA DRIFT between manifest and parquet:\n"
            f"  added  : {added}\n  removed: {removed}\n"
            f"  retyped: {[(c, w[c], '->', g[c]) for c in retyped]}")


# ------------------------------------------------------------------ the lock (core)
def write_lock(path, man: dict, clearml_dataset_id: str, semver: str,
               gcs_url: str, git_sha: str) -> dict:
    """The lock-back receipt, written AFTER a successful publish.

    It lives in its OWN file, never inside the recipe -- because the manifest
    certifies sha256(recipe), and writing into the recipe would break that hash
    on the very next read.
    """
    import yaml
    doc = {
        "version": man["version"],
        "clearml_dataset_id": clearml_dataset_id,
        "clearml_version": semver,
        "gcs_url": gcs_url,
        "git_sha": git_sha,
        "parquet_sha256": man["parquet_sha256"],
        "recipe_sha256": man["recipe_sha256"],
        "labels_sha256": man["labels_sha256"],
        "rows": man["rows"],
        "n_features": man["n_features"],
        "per_feature": man["per_feature"],
        "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    Path(path).write_text(
        "# LOCK -- written by core/publish_version.py after a successful publish.\n"
        "# This is what makes 'which exact data did model X train on?' answerable.\n"
        + yaml.safe_dump(doc, sort_keys=False))
    return doc
