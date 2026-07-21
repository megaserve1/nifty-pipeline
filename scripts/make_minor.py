"""scripts/make_minor.py -- make a MINOR dataset version by DROPPING columns from a built one.

    v4  -> v4.1   (v4's parquet, minus the columns listed in configs/drops/v4.1.csv)
    v5  -> v5.1                                          "        configs/drops/v5.1.csv

THE DROP LIST IS A CONFIG FILE, NOT CODE. you edit configs/drops/<minor>.csv -- the feature team's
own format, `action,dropped_column,kept_instead`. to make a new minor version, drop a new csv in
that folder and run this. no python is touched. (same idea as the version ballot: the decision
lives in a file you can read and diff, not in a script.)

recipe-not-copy still holds: we do NOT rebuild from feature sources (v4/v5 arrived as pre-combined
parquets). we take the CERTIFIED base parquet, drop the listed columns, and regenerate a valid
recipe + parquet + manifest triple so `publish_version` accepts it.

THE NAME PROBLEM THIS SOLVES. your drop list is BARE names (atr14, vwap, PCR_OI) but the parquet
columns are PREFIXED (bucket_bucket_raw_v4__atr14). so each bare name D is matched to a real column
C when  C == D  or  C ends with "__D". every resolution is PRINTED, and a name that matches 0 or
>1 columns is FLAGGED, not guessed -- a wrong drop silently corrupts the dataset.

run:
    # 1. DRY: resolve + show exactly what would be dropped, write nothing
    final_venv/bin/python scripts/make_minor.py --from v4 --to v4.1
    # 2. real
    final_venv/bin/python scripts/make_minor.py --from v4 --to v4.1 --write
    # then --from v5 --to v5.1
"""
import sys, json, argparse, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd, yaml
import config as C
import contract
import hashlib

DROPS_DIR = C.CONFIGS_DIR / "drops"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_drops(text):
    names, kept = [], []
    for ln in text.strip().splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if not parts or parts[0] == "action" or len(parts) < 2 or not parts[1]:
            continue
        names.append(parts[1])
        if len(parts) >= 3 and parts[2]:
            kept.append(parts[2])
    return names, kept


def resolve(dropname, cols):
    """bare name -> the ACTUAL parquet column(s). exact, else endswith '__name'."""
    exact = [c for c in cols if c == dropname]
    if exact:
        return exact
    return [c for c in cols if c.endswith("__" + dropname)]


def main():
    ap = argparse.ArgumentParser(description="make a minor dataset version by dropping columns")
    ap.add_argument("--from", dest="base", required=True, help="the built base version, e.g. v4")
    ap.add_argument("--to", dest="minor", required=True, help="the new minor version, e.g. v4.1")
    ap.add_argument("--drops", default=None,
                    help="the drop-list csv. default: configs/drops/<minor>.csv")
    ap.add_argument("--write", action="store_true", help="actually write (default is a dry run)")
    a = ap.parse_args()
    base, minor, write = a.base, a.minor, a.write

    drops_csv = pathlib.Path(a.drops) if a.drops else DROPS_DIR / f"{minor}.csv"
    if not drops_csv.exists():
        raise SystemExit(f"no drop list at {drops_csv}\n"
                         f"  create it (feature-team format: action,dropped_column,kept_instead) "
                         f"or pass --drops <path>.")

    base_dir = C.DATASETS_DIR / base
    base_pq = base_dir / f"dataset_{base}.parquet"
    base_man = json.load(open(base_dir / "manifest.json"))
    base_recipe = C.VERSIONS_DIR / f"dataset_{base}.yaml"

    print(f"drop list: {drops_csv}")
    print(f"reading {base_pq} ...")
    df = pd.read_parquet(base_pq)
    cols = list(df.columns)
    print(f"  {len(df):,} rows x {len(cols)} columns")

    names, kept = parse_drops(drops_csv.read_text())
    print(f"\nresolving {len(names)} drop names against the parquet:")
    to_drop, missing, ambiguous = [], [], []
    for d in names:
        hit = resolve(d, cols)
        if len(hit) == 0:
            missing.append(d);  print(f"  MISSING  {d:<45} -> no column (already gone / typo)")
        elif len(hit) > 1:
            ambiguous.append((d, hit)); print(f"  AMBIG    {d:<45} -> {hit}  !! not dropping")
        else:
            to_drop.append(hit[0])
    # kept columns should survive -- warn if any is absent
    for k in set(kept):
        if not resolve(k, cols):
            print(f"  note: kept-instead '{k}' not found in parquet (ok if it never existed here)")

    print(f"\n  will drop {len(to_drop)} columns; {len(missing)} missing; {len(ambiguous)} ambiguous")
    remaining_feats = [c for c in base_man["feature_columns"] if c not in to_drop]
    print(f"  features {base_man['n_features']} -> {len(remaining_feats)}")

    if ambiguous:
        print("\n  STOP: ambiguous names above match >1 column. tighten the list before --write.")
        raise SystemExit(1)
    if not write:
        print("\nDRY RUN. nothing written. re-run with --write to build the minor version.")
        return

    # ---- write the minor parquet ----
    out_dir = C.DATASETS_DIR / minor
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pq = out_dir / f"dataset_{minor}.parquet"
    df.drop(columns=to_drop).to_parquet(out_pq, index=False)
    print(f"\nwrote {out_pq}  ({out_pq.stat().st_size/1e6:.0f} MB)")

    # ---- the recipe (documented; its sha must match the manifest) ----
    rec = yaml.safe_load(base_recipe.read_text())
    rec["name"] = f"dataset_{minor}"
    rec["parent"] = base
    rec["derived_by"] = f"drop {len(to_drop)} columns from {base}"
    rec["dropped_columns"] = sorted(to_drop)
    out_recipe = C.VERSIONS_DIR / f"dataset_{minor}.yaml"
    out_recipe.write_text(yaml.safe_dump(rec, sort_keys=False))
    print(f"wrote {out_recipe}")

    # ---- the manifest (v4 manifest, minus dropped cols, fresh checksums) ----
    keep = set(df.drop(columns=to_drop).columns)
    new_schema = [s for s in base_man["schema"] if s["col"] in keep]
    per_feature = []
    for pf in base_man["per_feature"]:
        cc = {k: v for k, v in (pf.get("column_clocks") or {}).items() if k in keep}
        per_feature.append({**pf, "column_clocks": cc})
    fields = dict(base_man)
    fields.update({
        "version": minor,
        "recipe_sha256": sha256_file(out_recipe),
        "parquet_sha256": sha256_file(out_pq),
        "n_features": len(remaining_feats),
        "schema": new_schema,
        "feature_columns": remaining_feats,
        "categorical_columns": [c for c in base_man["categorical_columns"] if c in keep],
        "per_feature": per_feature,
        "status": "ready",
    })
    fields.pop("manifest_schema_version", None)   # write_manifest stamps it
    fields.pop("built_at", None)
    contract.write_manifest(out_dir / "manifest.json", **fields)
    print(f"wrote {out_dir/'manifest.json'}")

    # ---- self-check: the publish gate must accept it ----
    man = contract.load(out_dir / "manifest.json")
    contract.validate_manifest(man, out_recipe, out_pq)
    print(f"\nOK  {minor} validates against the publish gate. "
          f"{man['rows']:,} rows x {man['n_features']} features.")
    print(f"next:  final_venv/bin/python core/publish_version.py --version {minor} "
          f"--models xgboost,catboost --queue lossbased --no-champion --dry-run")


if __name__ == "__main__":
    main()
