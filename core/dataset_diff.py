"""
dataset_diff.py -- compare dataset versions by what they ACTUALLY BUILT
===================================================================
Reads the manifests (datasets/vN/manifest.json), not the giant parquets, so
comparing 2 or 20 versions is instant. It does NOT touch DVC or the bucket --
every file it reads is a tiny local one that is committed to git.

WHY MANIFEST COLUMNS AND NOT THE RECIPE'S `features:` LIST.
    the recipe names feature SOURCES (one entry, `bucket_bucket_raw_v4`), and one
    source expands into 300+ COLUMNS at build time. an ablation like v4.1 ("drop 18
    columns from v4") changes the COLUMNS but not the source list -- so diffing the
    `features:` list reported "kept: 1, nothing changed" for two datasets that differ
    by 18 features. it compared what was ASKED at source level, not what was GIVEN.
    so we diff the manifest's feature_columns -- the columns that were really built.

Run:
    python dataset_diff.py v1 v2          # pairwise column diff
    python dataset_diff.py v1 v2 v3 v4    # matrix of the columns that DIFFER
    python dataset_diff.py --all          # matrix of every version
"""
import argparse

import json
import yaml

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402


def _recipe_files():
    """the frozen recipes -- and ONLY the recipes. dataset_v2.lock.yaml matches the same glob,
    and its stem parsed as int('2.lock') crashed --all the moment anything was ever published.
    make_version._version_files() got this right; this file had forgotten the filter."""
    return [p for p in C.VERSIONS_DIR.glob("dataset_v*.yaml")
            if not p.name.endswith(".lock.yaml")]


def _sort_key(v: str) -> tuple:
    """'v2' -> (2, 0), 'v2.1' -> (2, 1). int(v.lstrip('v')) crashed on every minor version --
    int('2.1') is a ValueError, and minor versions are the whole point of the ablation design."""
    parts = v.lstrip("v").split(".")
    return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)


def load(v: str) -> dict:
    p = C.VERSIONS_DIR / f"dataset_{v}.yaml"
    if not p.exists():
        have = sorted(x.stem.replace("dataset_", "") for x in _recipe_files())
        raise SystemExit(f"unknown version '{v}'. existing: {have or 'none'}")
    return yaml.safe_load(p.read_text())


def columns_of(v: str, recipe: dict) -> tuple:
    """the columns this version actually built -- the GIVEN, read from its manifest.

    returns (set_of_names, granularity). granularity is 'columns' when the manifest was
    found (the real, built columns) and 'sources' when it was not (a version that has been
    frozen but not built yet -- all we can show then is the recipe's source list). the caller
    says which one it got, so a column diff and a source diff are never silently mixed.
    """
    man = C.DATASETS_DIR / v / "manifest.json"
    if man.exists():
        cols = json.loads(man.read_text()).get("feature_columns")
        if cols:
            return set(cols), "columns"
    return set(recipe.get("features", [])), "sources"


def _print_names(names: list, cap: int = 60):
    for n in names[:cap]:
        print(f"      - {n}")
    if len(names) > cap:
        print(f"      ... (+{len(names) - cap} more)")


def pairwise(a: str, b: str):
    da, db = load(a), load(b)
    ca, ka = columns_of(a, da)
    cb, kb = columns_of(b, db)
    gran = "columns" if ka == "columns" and kb == "columns" else "sources"
    print(f"diff {a} -> {b}   (comparing built {gran})")
    if gran == "sources":
        print(f"  NOTE: {a if ka!='columns' else b} has no manifest yet, so this compares recipe")
        print(f"        SOURCES, not the real columns. build the dataset to diff at column level.")
    added, removed = sorted(cb - ca), sorted(ca - cb)
    print(f"  {gran} added   : {len(added) or '-'}")
    _print_names(added)
    print(f"  {gran} removed : {len(removed) or '-'}")
    _print_names(removed)
    print(f"  {gran} kept    : {len(ca & cb)}")
    if db.get("derived_by"):
        print(f"  recipe note      : {db['derived_by']}")
    for key in ("labels_name", "date_range"):        # was 'labels_file' -- a key the recipe never
        va, vb = da.get(key), db.get(key)             # has, so it read None==None and always said
        print(f"  {key:16s} : {'unchanged' if va == vb else f'{va}  ->  {vb}'}")   # 'unchanged'
    print(f"  created          : {da.get('created')}  vs  {db.get('created')}")
    print(f"  author           : {da.get('author')}  vs  {db.get('author')}")


def matrix(versions: list):
    """show ONLY the columns that differ across the versions -- the ablation, not all 315 rows."""
    docs = {v: load(v) for v in versions}
    cols = {v: columns_of(v, docs[v])[0] for v in versions}
    everywhere = set.intersection(*cols.values()) if cols else set()
    varying = sorted(set().union(*cols.values()) - everywhere)
    w = max((len(f) for f in varying), default=10) + 2
    print(f"columns that DIFFER across {', '.join(versions)}  "
          f"({len(everywhere)} column(s) are in ALL of them, not shown)")
    print(f"{'COLUMN':{w}s} " + " ".join(f"{v:>7s}" for v in versions))
    for f in varying:
        row = " ".join(f"{'x' if f in cols[v] else '-':>7s}" for v in versions)
        print(f"{f:{w}s} {row}")
    print(f"\n{'built total':{w}s} " +
          " ".join(f"{len(cols[v]):>7d}" for v in versions))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("versions", nargs="*", help="e.g. v1 v2 [v3 ...]")
    ap.add_argument("--all", action="store_true", help="compare every version")
    a = ap.parse_args()

    if a.all:
        vs = sorted((p.stem.replace("dataset_", "") for p in _recipe_files()),
                    key=_sort_key)
        if len(vs) < 2:
            raise SystemExit("need at least 2 frozen versions")
        matrix(vs)
    elif len(a.versions) == 2:
        pairwise(*a.versions)
    elif len(a.versions) > 2:
        matrix(a.versions)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
